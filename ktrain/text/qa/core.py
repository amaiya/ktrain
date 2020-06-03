from ...imports import *
from ... import utils as U
from .. import textutils as TU
from .. import preprocessor as tpp


from whoosh import index
from whoosh.fields import *
from whoosh import qparser
from whoosh.qparser import QueryParser


from transformers import TFBertForQuestionAnswering
from transformers import BertTokenizer
LOWCONF = -10000


class QA(ABC):
    """
    Base class for QA
    """

    def __init__(self, bert_squad_model='bert-large-uncased-whole-word-masking-finetuned-squad',
                 bert_emb_model='bert-base-uncased'):
        self.model_name = bert_squad_model
        self.model = TFBertForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.maxlen = 512
        self.te = tpp.TransformerEmbedding(bert_emb_model, layers=[-2])


    @abstractmethod
    def search(self, query):
        pass

    def predict_squad(self, document, question):
        input_ids = self.tokenizer.encode(question, document)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        sep_index = input_ids.index(self.tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)
        n_ids = len(segment_ids)
        if n_ids < self.maxlen:
            start_scores, end_scores = self.model(np.array([input_ids]), 
                                             token_type_ids=np.array([segment_ids]))
        else:
            #TODO: use different truncation strategies or run multiple inferences
            start_scores, end_scores = self.model(np.array([input_ids[:self.maxlen]]), 
                                             token_type_ids=np.array([segment_ids[:self.maxlen]]))
        start_scores = start_scores[:,1:-1]
        end_scores = end_scores[:,1:-1]
        answer_start = np.argmax(start_scores)
        answer_end = np.argmax(end_scores)
        answer = self._reconstruct_text(tokens, answer_start, answer_end+2)
        if answer.startswith('. ') or answer.startswith(', '):
            answer = answer[2:]  
        sep_index = tokens.index('[SEP]')
        full_txt_tokens = tokens[sep_index+1:]
        paragraph_bert = self._reconstruct_text(full_txt_tokens)

        ans={}
        ans['answer'] = answer
        if answer.startswith('[CLS]') or answer_end < sep_index or answer.endswith('[SEP]'):
            ans['confidence'] = LOWCONF
        else:
            #confidence = torch.max(start_scores) + torch.max(end_scores)
            #confidence = np.log(confidence.item())
            ans['confidence'] = start_scores[0,answer_start]+end_scores[0,answer_end]
        ans['start'] = answer_start
        ans['end'] = answer_end
        ans['context'] = paragraph_bert
        return ans


    def _reconstruct_text(self, tokens, start=0, stop=-1):
        tokens = tokens[start: stop]
        if '[SEP]' in tokens:
            sepind = tokens.index('[SEP]')
            tokens = tokens[sepind+1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        length = len(txt_list)
        if length == 1:
            return txt_list[0]
        new_list =[]
        for i,t in enumerate(txt_list):
            if i < length -1:
                if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                    new_list += [t,',']
                else:
                    new_list += [t, ', ']
            else:
                new_list += [t]
        return ''.join(new_list)



class SimpleQA(QA):
    """
    SimpleQA: Question-Answering on a list of texts
    """
    def __init__(self, index_dir, 
                 bert_squad_model='bert-large-uncased-whole-word-masking-finetuned-squad',
                 bert_emb_model='bert-base-uncased'):
        """
        SimpleQA constructor
        Args:
          index_dir(str):  path to index directory created by SimpleQA.initialze_index
          bert_squad_model(str): name of BERT SQUAD model to use
          bert_emb_model(str): BERT model to use to generate embeddings for semantic similarity

        """

        self.index_dir = index_dir
        try:
            ix = index.open_dir(self.index_dir)
        except:
            raise ValueError('index_dir has not yet been created - please call SimpleQA.initialize_index("%s")' % (self.index_dir))
        super().__init__(bert_squad_model=bert_squad_model, bert_emb_model=bert_emb_model)


    def _open_ix(self):
        return index.open_dir(self.index_dir)


    @classmethod
    def initialize_index(cls, index_dir):
        schema = Schema(reference=ID(stored=True), content=TEXT, rawtext=TEXT(stored=True))
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        else:
            raise ValueError('There is already an existing directory or file with path %s' % (index_dir))
        ix = index.create_in(index_dir, schema)
        return ix

    @classmethod
    def index_from_list(cls, docs, index_dir, commit_every=1024,
                        procs=1, limitmb=256, multisegment=False):
        """
        index documents from list.
        The procs, limitmb, and especially multisegment arguments can be used to 
        speed up indexing, if it is too slow.  Please see the whoosh documentation
        for more information on these parameters:  https://whoosh.readthedocs.io/en/latest/batch.html
        Args:
          docs(list): list of strings representing documents
          commit_every(int): commet after adding this many documents
          procs(int): number of processors
          limitmb(int): memory limit in MB for each process
          multisegment(bool): new segments written instead of merging
        """
        if not isinstance(docs, (np.ndarray, list)): raise ValueError('docs must be a list of strings')
        ix = index.open_dir(index_dir)
        writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)
        mb = master_bar(range(1))
        for i in mb:
            for idx, doc in enumerate(progress_bar(docs, parent=mb)):
                reference = "%s" % (idx)
                content = doc 
                writer.add_document(reference=reference, content=content, rawtext=content)
                idx +=1
                if idx % commit_every == 0:
                    writer.commit()
                    #writer = ix.writer()
                    writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)
            writer.commit()
        return


    @classmethod
    def index_from_folder(cls, folder_path, index_dir,  commit_every=1024, verbose=1, encoding='utf-8',
                          procs=1, limitmb=256, multisegment=False):
        """
        index all plain text documents within a folder.
        The procs, limitmb, and especially multisegment arguments can be used to 
        speed up indexing, if it is too slow.  Please see the whoosh documentation
        for more information on these parameters:  https://whoosh.readthedocs.io/en/latest/batch.html

        Args:
          folder_path(str): path to folder containing plain text documents
          commit_every(int): commet after adding this many documents
          procs(int): number of processors
          limitmb(int): memory limit in MB for each process
          multisegment(bool): new segments written instead of merging

        """
        if not os.path.isdir(folder_path): raise ValueError('folder_path is not a valid folder')
        if folder_path[-1] != os.sep: folder_path += os.sep
        ix = index.open_dir(index_dir)
        writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)
        for idx, fpath in enumerate(TU.extract_filenames(folder_path)):
            if not TU.is_txt(fpath): continue
            reference = "%s" % (fpath.join(fpath.split(folder_path)[1:]))
            with open(fpath, 'r', encoding=encoding) as f:
                doc = f.read()
            content = doc
            writer.add_document(reference=reference, content=content, rawtext=content)
            idx +=1
            if idx % commit_every == 0:
                writer.commit()
                #writer = ix.writer()
                writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)
                if verbose: print("%s docs indexed" % (idx))
        writer.commit()
        return


    def search(self, query, limit=10):
        """
        search index for query
        Args:
          query(str): search query
          limit(int):  number of top search results to return
        Returns:
          list of dicts with keys: reference, rawtext
        """
        ix = self._open_ix()
        with ix.searcher() as searcher:
            query_obj = QueryParser("content", ix.schema, group=qparser.OrGroup).parse(query)
            results = searcher.search(query_obj, limit=limit)
            docs = []
            output = [dict(r) for r in results]
            return output


    def _expand_answer(self, answer):
        """
        expand answer to include more of the context
        """
        full_abs = answer['context']
        bert_ans = answer['answer']
        split_abs = full_abs.split(bert_ans)
        sent_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
        if len(split_abs) == 1:
            sent_end_pos = len(full_abs)
            sent_end =''
        else:
            sent_end_pos = split_abs[1].find('. ')+1
            if sent_end_pos == 0:
                sent_end = split_abs[1]
            else:
                sent_end = split_abs[1][:sent_end_pos]
            
        answer['full_answer'] = sent_beginning+bert_ans+sent_end
        answer['full_answer'] = answer['full_answer'].strip()
        answer['sentence_beginning'] = sent_beginning
        answer['sentence_end'] = sent_end
        return answer



    def ask(self, question, n_docs_considered=10, n_answers=50, rerank_threshold=0.015):
        """
        submit question to obtain candidate answers

        Args:
          question(str): question in the form of a string
          n_docs_considered(int): number of top search results that will
                                  be searched for answer
                                  default:10
          n_answers(int): maximum number of candidate answers to return
                          default:50
          rerank_threshold(int): rerank top answers with confidence >= rerank_threshold
                                 based on semantic similarity between question and answer.
                                 This can help bump the correct answer closer to the top.
                                 default:0.015.
                                 If None, no re-ranking is performed.
        Returns:
          list
        """
        # locate candidate document contexts
        doc_results = self.search(question, limit=n_docs_considered)
        paragraphs = []
        refs = []
        for doc_result in doc_results:
            rawtext = doc_result.get('rawtext', '')
            reference = doc_result.get('reference', '')
            if len(self.tokenizer.tokenize(rawtext)) < self.maxlen:
                paragraphs.append(rawtext)
                refs.append(reference)
                continue
            plist = TU.paragraph_tokenize(rawtext, join_sentences=True)
            paragraphs.extend(plist)
            refs.extend([reference]*len(plist))

        # locate candidate answers
        answers = []
        mb = master_bar(range(1))
        for i in mb:
            for idx, paragraph in enumerate(progress_bar(paragraphs, parent=mb)):
                answer = self.predict_squad(paragraph, question)
                if not answer['answer'] or answer['confidence'] <0: continue
                answer['confidence'] = answer['confidence'].numpy()
                answer['reference'] = refs[idx]
                answer = self._expand_answer(answer)
                answers.append(answer)
        answers = sorted(answers, key = lambda k:k['confidence'], reverse=True)
        if n_answers is not None:
            answers = answers[:n_answers]

        # transform confidence scores
        confidences = [a['confidence'] for a in answers]
        max_conf = max(confidences)
        total = 0.0
        exp_scores = []
        for c in confidences:
            s = np.exp(c-max_conf)
            exp_scores.append(s)
        total = sum(exp_scores)
        for idx,c in enumerate(confidences):
            answers[idx]['confidence'] = exp_scores[idx]/total

        if rerank_threshold is None:
            return answers

        # re-rank
        top_confidences = [a['confidence'] for idx, a in enumerate(answers) if a['confidence']> rerank_threshold]
        v1 = self.te.embed(question, word_level=False)
        for idx, answer in enumerate(answers):
            #if idx >= rerank_top_n: 
            if answer['confidence'] <= rerank_threshold:
                answer['similarity_score'] = 0.0
                continue
            v2 = self.te.embed(answer['full_answer'], word_level=False)
            score = v1 @ v2.T / (np.linalg.norm(v1)*np.linalg.norm(v2))
            answer['similarity_score'] = float(np.squeeze(score))
            answer['confidence'] = top_confidences[idx]
        answers = sorted(answers, key = lambda k:(k['similarity_score'], k['confidence']), reverse=True)
        for idx, confidence in enumerate(top_confidences):
            answers[idx]['confidence'] = confidence


        return answers


    def answers2df(self, answers):
        dfdata = []
        for a in answers:
            answer_text = a['answer']
            snippet_html = '<div>' +a['sentence_beginning'] + " <font color='red'>"+a['answer']+"</font> "+a['sentence_end']+'</div>'
            confidence = a['confidence']
            doc_key = a['reference']
            dfdata.append([answer_text, snippet_html, confidence, doc_key])
        df = pd.DataFrame(dfdata, columns = ['Candidate Answer', 'Context',  'Confidence', 'Document Reference'])
        return df


    def display_answers(self, answers):
        df = self.answers2df(answers)
        from IPython.core.display import display, HTML
        display(HTML(df.to_html(render_links=True, escape=False)))


