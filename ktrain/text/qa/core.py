from ...imports import *
from ... import utils as U
from .. import textutils as TU
from .. import preprocessor as tpp


from whoosh import index
from whoosh.fields import *
from whoosh import qparser
from whoosh.qparser import QueryParser


#from transformers import TFBertForQuestionAnswering
#from transformers import BertTokenizer
from transformers import TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer
LOWCONF = -10000

def _answers2df(answers):
    dfdata = []
    for a in answers:
        answer_text = a['answer']
        snippet_html = '<div>' +a['sentence_beginning'] + " <font color='red'>"+a['answer']+"</font> "+a['sentence_end']+'</div>'
        confidence = a['confidence']
        doc_key = a['reference']
        dfdata.append([answer_text, snippet_html, confidence, doc_key])
    df = pd.DataFrame(dfdata, columns = ['Candidate Answer', 'Context',  'Confidence', 'Document Reference'])
    if "\t" in answers[0]['reference']:
        df['Document Reference'] = df['Document Reference'].apply(lambda x: '<a href="{}">{}</a>'.format(x.split('\t')[1], x.split('\t')[0]))
    return df



def display_answers(answers):
    if not answers: return
    df = _answers2df(answers)
    from IPython.core.display import display, HTML
    return display(HTML(df.to_html(render_links=True, escape=False)))


def _process_question(question, include_np=False):
    if include_np:
        try:
            # attempt to use extract_noun_phrases first if textblob is installed
            np_list = ['"%s"' % (np) for np in TU.extract_noun_phrases(question) if len(np.split()) > 1]
            q_tokens = TU.tokenize(question, join_tokens=False)
            q_tokens.extend(np_list)
            return " ".join(q_tokens)
        except:
            import warnings
            warnings.warn('TextBlob is not currently installed, so falling back to include_np=False with no extra question processing. '+\
                          'To install: pip install textblob')
            return TU.tokenize(question, join_tokens=True)
    else:
        return TU.tokenize(question, join_tokens=True)



class QA(ABC):
    """
    Base class for QA
    """

    def __init__(self, bert_squad_model='bert-large-uncased-whole-word-masking-finetuned-squad',
                 bert_emb_model='bert-base-uncased'):
        self.model_name = bert_squad_model
        try:
            self.model = TFAutoModelForQuestionAnswering.from_pretrained(self.model_name)
        except:
            self.model = TFAutoModelForQuestionAnswering.from_pretrained(self.model_name, from_pt=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.maxlen = 512
        self.te = tpp.TransformerEmbedding(bert_emb_model, layers=[-2])


    @abstractmethod
    def search(self, query):
        pass

    def predict_squad(self, documents, question):
        if isinstance(documents, str): documents = [documents]
        sequences = [[question, d] for d in documents]
        batch = self.tokenizer.batch_encode_plus(sequences, return_tensors='tf', max_length=512, truncation='only_second', padding=True)
        tokens_batch = list( map(self.tokenizer.convert_ids_to_tokens, batch['input_ids']))

        # Added from: https://github.com/huggingface/transformers/commit/16ce15ed4bd0865d24a94aa839a44cf0f400ef50
        if U.get_hf_model_name(self.model_name) in  ['xlm', 'roberta', 'distilbert']:
           start_scores, end_scores = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
        else:
           start_scores, end_scores = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], 
                                                 token_type_ids=batch['token_type_ids'])
        start_scores = start_scores[:,1:-1]
        end_scores = end_scores[:,1:-1]
        answer_starts = np.argmax(start_scores, axis=1)
        answer_ends = np.argmax(end_scores, axis=1)

        answers = []
        for i, tokens in enumerate(tokens_batch):
            answer_start = answer_starts[i]
            answer_end = answer_ends[i]
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
                ans['confidence'] = start_scores[i,answer_start]+end_scores[i,answer_end]
            ans['start'] = answer_start
            ans['end'] = answer_end
            ans['context'] = paragraph_bert
            answers.append(ans)
        #if len(answers) == 1: answers = answers[0]
        return answers




    def _reconstruct_text(self, tokens, start=0, stop=-1):
        """
        Reconstruct text of *either* question or answer
        """
        tokens = tokens[start: stop]
        #if '[SEP]' in tokens:
            #sepind = tokens.index('[SEP]')
            #tokens = tokens[sepind+1:]
        txt = ' '.join(tokens)
        txt = txt.replace('[SEP]', '') # added for batch_encode_plus - removes [SEP] before [PAD]
        txt = txt.replace('[PAD]', '') # added for batch_encode_plus - removes [PAD]
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



    def ask(self, question, batch_size=8, n_docs_considered=10, n_answers=50, 
            rerank_threshold=0.015, include_np=False):
        """
        submit question to obtain candidate answers

        Args:
          question(str): question in the form of a string
          batch_size(int):  number of question-context pairs fed to model at each iteration
                            Default:8
                            Increase for faster answer-retrieval.
                            Decrease to reduce memory (if out-of-memory errors occur).
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
          include_np(bool):  If True, noun phrases will be extracted from question and included
                             in query that retrieves documents likely to contain candidate answers.
                             This may be useful if you ask a question about artificial intelligence
                             and the answers returned pertain just to intelligence, for example.
                             Note: include_np=True requires textblob be installed.
                             Default:False
        Returns:
          list
        """
        # locate candidate document contexts
        paragraphs = []
        refs = []
        #doc_results = self.search(question, limit=n_docs_considered)
        doc_results = self.search(_process_question(question, include_np=include_np), limit=n_docs_considered)
        if not doc_results: 
            warnings.warn('No documents matched words in question')
            return []
        # extract paragraphs as contexts
        contexts = []
        refs = []
        for doc_result in doc_results:
            rawtext = doc_result.get('rawtext', '')
            reference = doc_result.get('reference', '')
            if len(self.tokenizer.tokenize(rawtext)) < self.maxlen:
                contexts.append(rawtext)
                refs.append(reference)
            else:
                paragraphs = TU.paragraph_tokenize(rawtext, join_sentences=True)
                contexts.extend(paragraphs)
                refs.extend([reference] * len(paragraphs))


        #for doc_result in doc_results:
            #rawtext = doc_result.get('rawtext', '')
            #reference = doc_result.get('reference', '')
            #if len(self.tokenizer.tokenize(rawtext)) < self.maxlen:
                #paragraphs.append(rawtext)
                #refs.append(reference)
                #continue
            #plist = TU.paragraph_tokenize(rawtext, join_sentences=True)
            #paragraphs.extend(plist)
            #refs.extend([reference]*len(plist))


        # batchify contexts
        #return contexts
        if batch_size  > len(contexts): batch_size = len(contexts)
        #if len(contexts) >= 100 and batch_size==8:
            #warnings.warn('TIP: Try increasing batch_size to speedup ask predictions')
        num_chunks = math.ceil(len(contexts)/batch_size)
        context_batches = list( U.list2chunks(contexts, n=num_chunks) )


        # locate candidate answers
        answers = []
        mb = master_bar(range(1))
        answer_batches = []
        for i in mb:
            idx = 0
            for batch_id, contexts in enumerate(progress_bar(context_batches, parent=mb)):
                answer_batch = self.predict_squad(contexts, question)
                answer_batches.extend(answer_batch)
                for answer in answer_batch:
                    idx+=1
                    if not answer['answer'] or answer['confidence'] <-100: continue
                    answer['confidence'] = answer['confidence'].numpy()
                    answer['reference'] = refs[idx-1]
                    answer = self._expand_answer(answer)
                    answers.append(answer)

                mb.child.comment = f'generating candidate answers'


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


    def display_answers(self, answers):
        return display_answers(answers)



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
    def index_from_list(cls, docs, index_dir, commit_every=1024, breakup_docs=False,
                        procs=1, limitmb=256, multisegment=False, min_words=20, references=None):
        """
        index documents from list.
        The procs, limitmb, and especially multisegment arguments can be used to 
        speed up indexing, if it is too slow.  Please see the whoosh documentation
        for more information on these parameters:  https://whoosh.readthedocs.io/en/latest/batch.html
        Args:
          docs(list): list of strings representing documents
          index_dir(str): path to index directory (see initialize_index)
          commit_every(int): commet after adding this many documents
          breakup_docs(bool): break up documents into smaller paragraphs and treat those as the documents.
                              This can potentially improve the speed at which answers are returned by the ask method
                              when documents being searched are longer.
          procs(int): number of processors
          limitmb(int): memory limit in MB for each process
          multisegment(bool): new segments written instead of merging
          min_words(int):  minimum words for a document (or paragraph extracted from document when breakup_docs=True) to be included in index.
                           Useful for pruning contexts that are unlikely to contain useful answers
          references(list): List of strings containing a reference (e.g., file name) for each document in docs.
                            Each string is treated as a label for the document (e.g., file name, MD5 hash, etc.):
                               Example:  ['some_file.pdf', 'some_other_file,pdf', ...]
                            Strings can also be hyperlinks in which case the label and URL should be separated by a single tab character:
                               Example: ['ktrain_article\thttps://arxiv.org/pdf/2004.10703v4.pdf', ...]

                            These references will be returned in the output of the ask method.
                            If strings are  hyperlinks, then they will automatically be made clickable when the display_answers function
                            displays candidate answers in a pandas DataFRame.

                            If references is None, the index of element in docs is used as reference.
        """
        if not isinstance(docs, (np.ndarray, list)): raise ValueError('docs must be a list of strings')
        if references is not None and not isinstance(references, (np.ndarray, list)): raise ValueError('references must be a list of strings')
        if references is not None and len(references) != len(docs): raise ValueError('lengths of docs and references must be equal')

        ix = index.open_dir(index_dir)
        writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)

        mb = master_bar(range(1))
        for i in mb:
            for idx, doc in enumerate(progress_bar(docs, parent=mb)):
                reference = "%s" % (idx) if references is None else references[idx]

                if breakup_docs:
                    small_docs = TU.paragraph_tokenize(doc, join_sentences=True, lang='en')
                    refs = [reference] * len(small_docs)
                    for i, small_doc in enumerate(small_docs):
                        if len(small_doc.split()) < min_words: continue
                        content = small_doc
                        reference = refs[i]
                        writer.add_document(reference=reference, content=content, rawtext=content)
                else:
                    if len(doc.split()) < min_words: continue
                    content = doc 
                    writer.add_document(reference=reference, content=content, rawtext=content)

                idx +=1
                if idx % commit_every == 0:
                    writer.commit()
                    #writer = ix.writer()
                    writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)
                mb.child.comment = f'indexing documents'
            writer.commit()
            #mb.write(f'Finished indexing documents')
        return


    @classmethod
    def index_from_folder(cls, folder_path, index_dir,  commit_every=1024, breakup_docs=False, min_words=20,
                          encoding='utf-8', procs=1, limitmb=256, multisegment=False, verbose=1):
        """
        index all plain text documents within a folder.
        The procs, limitmb, and especially multisegment arguments can be used to 
        speed up indexing, if it is too slow.  Please see the whoosh documentation
        for more information on these parameters:  https://whoosh.readthedocs.io/en/latest/batch.html

        Args:
          folder_path(str): path to folder containing plain text documents (e.g., .txt files)
          index_dir(str): path to index directory (see initialize_index)
          commit_every(int): commet after adding this many documents
          breakup_docs(bool): break up documents into smaller paragraphs and treat those as the documents.
                              This can potentially improve the speed at which answers are returned by the ask method
                              when documents being searched are longer.
          min_words(int):  minimum words for a document (or paragraph extracted from document when breakup_docs=True) to be included in index.
                           Useful for pruning contexts that are unlikely to contain useful answers
          encoding(str): encoding to use when reading document files from disk
          procs(int): number of processors
          limitmb(int): memory limit in MB for each process
          multisegment(bool): new segments written instead of merging
          verbose(bool): verbosity

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

            if breakup_docs:
                small_docs = TU.paragraph_tokenize(doc, join_sentences=True, lang='en')
                refs = [reference] * len(small_docs)
                for i, small_doc in enumerate(small_docs):
                    if len(small_doc.split()) < min_words: continue
                    content = small_doc
                    reference = refs[i]
                    writer.add_document(reference=reference, content=content, rawtext=content)
            else:
                if len(doc.split()) < min_words: continue
                content = doc
                writer.add_document(reference=reference, content=content, rawtext=content)

            idx +=1
            if idx % commit_every == 0:
                writer.commit()
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



