# from transformers import TFBertForQuestionAnswering
# from transformers import BertTokenizer
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TFAutoModelForQuestionAnswering,
)
from whoosh import index, qparser
from whoosh.fields import *
from whoosh.qparser import QueryParser

from ... import utils as U
from ...imports import *
from ...torch_base import TorchBase
from .. import preprocessor as tpp
from .. import textutils as TU

LOWCONF = -10000

DEFAULT_MODEL = "bert-large-uncased-whole-word-masking-finetuned-squad"
DEFAULT_MIN_CONF = 6

from itertools import chain, zip_longest


def twolists(l1, l2):
    return [x for x in chain(*zip_longest(l1, l2)) if x is not None]


def _answers2df(answers):
    dfdata = []
    for a in answers:
        answer_text = a["answer"]
        snippet_html = (
            "<div>"
            + a["sentence_beginning"]
            + " <font color='red'>"
            + a["answer"]
            + "</font> "
            + a["sentence_end"]
            + "</div>"
        )
        confidence = a["confidence"]
        doc_key = a["reference"]
        dfdata.append([answer_text, snippet_html, confidence, doc_key])
    df = pd.DataFrame(
        dfdata,
        columns=["Candidate Answer", "Context", "Confidence", "Document Reference"],
    )
    if "\t" in answers[0]["reference"]:
        df["Document Reference"] = df["Document Reference"].apply(
            lambda x: '<a href="{}" target="_blank">{}</a>'.format(
                x.split("\t")[1], x.split("\t")[0]
            )
        )
    return df


def display_answers(answers):
    if not answers:
        return
    df = _answers2df(answers)
    from IPython.core.display import HTML, display

    return display(HTML(df.to_html(render_links=True, escape=False)))


def process_question(
    question, include_np=False, and_np=False, remove_english_stopwords=False
):
    result = None
    np_list = []
    if include_np:
        try:
            # np_list = ['"%s"' % (np) for np in TU.extract_noun_phrases(question) if len(np.split()) > 1]
            raw_np_list = [
                np for np in TU.extract_noun_phrases(question) if len(np.split()) > 1
            ]
            np_list = []
            for np in raw_np_list:
                N = 2
                sentence = np.split()
                np_list.extend(
                    [
                        '"%s"' % (" ".join(sentence[i : i + N]))
                        for i in range(len(sentence) - N + 1)
                    ]
                )
            np_list = list(set(np_list))
        except:
            import warnings

            warnings.warn(
                "TextBlob is not currently installed, so falling back to include_np=False with no extra question processing. "
                + "To install: pip install textblob"
            )
    result = TU.tokenize(question, join_tokens=False)
    if remove_english_stopwords:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        result = [
            term
            for term in result
            if term.lower().strip() not in list(ENGLISH_STOP_WORDS) + ["?"]
        ]
    if np_list and and_np:
        return f'( {" ".join(result)} ) AND ({" ".join(np_list)})'
    else:
        return " ".join(result + np_list)


_process_question = process_question  # for backwards compatibility

# def _process_question(question, include_np=False):
#    if include_np:
#        try:
#            # attempt to use extract_noun_phrases first if textblob is installed
#            np_list = ['"%s"' % (np) for np in TU.extract_noun_phrases(question) if len(np.split()) > 1]
#            q_tokens = TU.tokenize(question, join_tokens=False)
#            q_tokens.extend(np_list)
#            return " ".join(q_tokens)
#        except:
#            import warnings
#            warnings.warn('TextBlob is not currently installed, so falling back to include_np=False with no extra question processing. '+\
#                          'To install: pip install textblob')
#            return TU.tokenize(question, join_tokens=True)
#    else:
#        return TU.tokenize(question, join_tokens=True)


class ExtractiveQABase(ABC, TorchBase):
    """
    Base class for QA
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        bert_squad_model=None,
        bert_emb_model="bert-base-uncased",
        framework="tf",
        device=None,
        quantize=False,
    ):
        model_name = bert_squad_model if bert_squad_model is not None else model_name
        if bert_squad_model:
            warnings.warn(
                "The bert_squad_model argument is deprecated - please use model_name instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.model_name = model_name
        self.framework = framework
        if framework == "tf":
            try:
                import tensorflow as tf
            except ImportError:
                raise Exception('If framework=="tf", TensorFlow must be installed.')
            try:
                self.model = TFAutoModelForQuestionAnswering.from_pretrained(
                    self.model_name
                )
            except:
                warnings.warn(
                    "Could not load supplied model as TensorFlow checkpoint - attempting to load using from_pt=True"
                )
                self.model = TFAutoModelForQuestionAnswering.from_pretrained(
                    self.model_name, from_pt=True
                )
        else:
            bert_emb_model = (
                None  # set to None and ignore since we only want to use PyTorch
            )
            super().__init__(device=device, quantize=quantize)
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name
            ).to(self.torch_device)
            if quantize:
                self.model = self.quantize_model(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.maxlen = 512
        self.te = (
            tpp.TransformerEmbedding(bert_emb_model, layers=[-2])
            if bert_emb_model is not None
            else None
        )

    @abstractmethod
    def search(self, query):
        pass

    def predict_squad(self, documents, question):
        """
        Generates candidate answers to the <question> provided given <documents> as contexts.
        """
        if isinstance(documents, str):
            documents = [documents]
        sequences = [[question, d] for d in documents]
        batch = self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors=self.framework,
            max_length=self.maxlen,
            truncation="only_second",
            padding=True,
        )
        batch = batch.to(self.torch_device) if self.framework == "pt" else batch
        tokens_batch = list(
            map(self.tokenizer.convert_ids_to_tokens, batch["input_ids"])
        )

        # Added from: https://github.com/huggingface/transformers/commit/16ce15ed4bd0865d24a94aa839a44cf0f400ef50
        if U.get_hf_model_name(self.model_name) in ["xlm", "roberta", "distilbert"]:
            start_scores, end_scores = self.model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=False,
            )
        else:
            start_scores, end_scores = self.model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                return_dict=False,
            )
        start_scores = (
            start_scores.cpu().detach().numpy()
            if self.framework == "pt"
            else start_scores.numpy()
        )
        end_scores = (
            end_scores.cpu().detach().numpy()
            if self.framework == "pt"
            else end_scores.numpy()
        )
        start_scores = start_scores[:, 1:-1]
        end_scores = end_scores[:, 1:-1]

        # normalize logits and spans to retrieve the answer
        # start_scores = np.exp(start_scores - np.log(np.sum(np.exp(start_scores), axis=-1, keepdims=True))) # from HF pipeline
        # end_scores = np.exp(end_scores - np.log(np.sum(np.exp(end_scores), axis=-1, keepdims=True)))             # from HF pipeline
        answer_starts = np.argmax(start_scores, axis=1)
        answer_ends = np.argmax(end_scores, axis=1)

        answers = []
        for i, tokens in enumerate(tokens_batch):
            answer_start = answer_starts[i]
            answer_end = answer_ends[i]
            answer = self._reconstruct_text(tokens, answer_start, answer_end + 2)
            if answer.startswith(". ") or answer.startswith(", "):
                answer = answer[2:]
            sep_index = tokens.index("[SEP]")
            full_txt_tokens = tokens[sep_index + 1 :]
            paragraph_bert = self._reconstruct_text(full_txt_tokens)

            ans = {}
            ans["answer"] = answer
            if (
                answer.startswith("[CLS]")
                or answer_end < sep_index
                or answer.endswith("[SEP]")
            ):
                ans["confidence"] = LOWCONF
            else:
                # confidence = torch.max(start_scores) + torch.max(end_scores)
                # confidence = np.log(confidence.item())
                # ans['confidence'] = start_scores[i,answer_start]*end_scores[i,answer_end]
                ans["confidence"] = (
                    start_scores[i, answer_start] + end_scores[i, answer_end]
                )

            ans["start"] = answer_start
            ans["end"] = answer_end
            ans["context"] = paragraph_bert
            answers.append(ans)
        # if len(answers) == 1: answers = answers[0]
        return answers

    def _clean_answer(self, answer):
        import string

        if not answer:
            return answer
        remove_list = [
            "is ",
            "are ",
            "was ",
            "were ",
            "of ",
            "include ",
            "including ",
            "in ",
            "of ",
            "the ",
            "for ",
            "on ",
            "to ",
            "-",
            ":",
            "/",
            "and ",
        ]
        for w in remove_list:
            if answer.startswith(w):
                answer = answer.replace(w, "", 1)
        answer = answer.replace(" . ", ".")
        answer = answer.replace(" / ", "/")
        answer = answer.replace(" :// ", "://")
        answer = answer.strip()
        if answer and answer[0] in string.punctuation:
            answer = answer[1:]
        if answer and answer[-1] in string.punctuation:
            answer = answer[:-1]
        return answer

    def _reconstruct_text(self, tokens, start=0, stop=-1):
        """
        Reconstruct text of *either* question or answer
        """
        tokens = tokens[start:stop]
        # if '[SEP]' in tokens:
        # sepind = tokens.index('[SEP]')
        # tokens = tokens[sepind+1:]
        txt = " ".join(tokens)
        txt = txt.replace(
            "[SEP]", ""
        )  # added for batch_encode_plus - removes [SEP] before [PAD]
        txt = txt.replace("[PAD]", "")  # added for batch_encode_plus - removes [PAD]
        txt = txt.replace(" ##", "")
        txt = txt.replace("##", "")
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(" .", ".")
        txt = txt.replace("( ", "(")
        txt = txt.replace(" )", ")")
        txt = txt.replace(" - ", "-")
        txt_list = txt.split(" , ")
        txt = ""
        length = len(txt_list)
        if length == 1:
            return txt_list[0]
        new_list = []
        for i, t in enumerate(txt_list):
            if i < length - 1:
                if t[-1].isdigit() and txt_list[i + 1][0].isdigit():
                    new_list += [t, ","]
                else:
                    new_list += [t, ", "]
            else:
                new_list += [t]
        return "".join(new_list)

    def _expand_answer(self, answer):
        """
        expand answer to include more of the context
        """
        full_abs = answer["context"]
        bert_ans = answer["answer"]
        split_abs = full_abs.split(bert_ans)
        sent_beginning = split_abs[0][split_abs[0].rfind(".") + 1 :]
        if len(split_abs) == 1:
            sent_end_pos = len(full_abs)
            sent_end = ""
        else:
            sent_end_pos = split_abs[1].find(". ") + 1
            if sent_end_pos == 0:
                sent_end = split_abs[1]
            else:
                sent_end = split_abs[1][:sent_end_pos]

        answer["full_answer"] = sent_beginning + bert_ans + sent_end
        answer["full_answer"] = answer["full_answer"].strip()
        answer["sentence_beginning"] = sent_beginning
        answer["sentence_end"] = sent_end
        return answer

    def _span_to_answer(self, question, text, start, end):
        """
        ```
        This method maps token indexes to actual word in the initial context.

        Args:
            text (str): The actual context to extract the answer from.
            start (int): The answer starting token index.
            end (int): The answer end token index.

        Returns:
            dct:  `{'answer': str, 'start': int, 'end': int}`
        ```
        """
        all_tokens = self.tokenizer.tokenize(
            text=question, pair=text, add_special_tokens=True
        )
        sep_idxs = [i for i, x in enumerate(all_tokens) if x == "[SEP]"]
        start = start - sep_idxs[0]
        end = end - sep_idxs[0]

        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0
        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }

    def _batchify(self, contexts, batch_size=8):
        """
        batchify contexts
        """
        if batch_size > len(contexts):
            batch_size = len(contexts)
        num_chunks = math.ceil(len(contexts) / batch_size)
        return list(U.list2chunks(contexts, n=num_chunks))

    def _split_contexts(self, doc_results):
        """
        ```
        splitup contexts into a manageable size
        Args:
          doc_results(list):  list of dicts with keys: rawtext and reference
        ```
        """
        # extract paragraphs as contexts
        contexts = []
        refs = []
        for doc_result in doc_results:
            rawtext = doc_result.get("rawtext", "")
            reference = doc_result.get("reference", "")
            if len(self.tokenizer.tokenize(rawtext)) < self.maxlen:
                contexts.append(rawtext)
                refs.append(reference)
            else:
                paragraphs = TU.paragraph_tokenize(rawtext, join_sentences=True)
                contexts.extend(paragraphs)
                refs.extend([reference] * len(paragraphs))
        return (contexts, refs)

    def ask(
        self,
        question,
        query=None,
        batch_size=8,
        n_docs_considered=10,
        n_answers=50,
        raw_confidence=False,
        rerank_threshold=0.015,
        include_np=False,
    ):
        """
        ```
        submit question to obtain candidate answers

        Args:
          question(str): question in the form of a string
          query(str): Optional. If not None, words in query will be used to retrieve contexts instead of words in question
          batch_size(int):  number of question-context pairs fed to model at each iteration
                            Default:8
                            Increase for faster answer-retrieval.
                            Decrease to reduce memory (if out-of-memory errors occur).
          n_docs_considered(int): number of top search results that will
                                  be searched for answer
                                  Default:10
          n_answers(int): maximum number of candidate answers to return
                          Default:50
          raw_confidence(bool): If True, show raw confidence score of each answer. It could be used to
                                mitigate very high confidence on first answer when softmax is used.
                                If False, perform softmax on raw confidence scores.
                                Default: False
          rerank_threshold(int): rerank top answers with confidence >= rerank_threshold
                                 based on semantic similarity between question and answer.
                                 This can help bump the correct answer closer to the top.
                                 Default:0.015. This should be changed to somethink like 6.0
                                 if raw_confidence=True.
                                 If None, no re-ranking is performed.
          include_np(bool):  If True, noun phrases will be extracted from question and included
                             in query that retrieves documents likely to contain candidate answers.
                             This may be useful if you ask a question about artificial intelligence
                             and the answers returned pertain just to intelligence, for example.
                             Note: include_np=True requires textblob be installed.
                             Default:False
        Returns:
          list
        ```
        """
        # sanity check
        if raw_confidence and rerank_threshold is not None and rerank_threshold < 1.00:
            warnings.warn(
                "Raw confidence is used, but rerank_threshold value is below 1.00: are you sure you this is what you wanted?"
            )

        # locate candidate document contexts
        doc_results = self.search(
            process_question(
                query if query is not None else question, include_np=include_np
            ),
            limit=n_docs_considered,
        )
        if not doc_results:
            warnings.warn(
                "No documents matched words in question (or query if supplied)"
            )
            return []

        # extract paragraphs as contexts
        contexts, refs = self._split_contexts(doc_results)

        # batchify contexts
        context_batches = self._batchify(contexts, batch_size=batch_size)

        # locate candidate answers
        answers = []
        mb = master_bar(range(1))
        answer_batches = []
        for i in mb:
            idx = 0
            for batch_id, contexts in enumerate(
                progress_bar(context_batches, parent=mb)
            ):
                answer_batch = self.predict_squad(contexts, question)
                answer_batches.extend(answer_batch)
                for answer in answer_batch:
                    idx += 1
                    if not answer["answer"] or answer["confidence"] < -100:
                        continue
                    answer["confidence"] = answer["confidence"]
                    answer["reference"] = refs[idx - 1]
                    answer = self._expand_answer(answer)
                    answers.append(answer)

                mb.child.comment = f"generating candidate answers"

        if not answers:
            return answers  # fix for #307
        answers = sorted(answers, key=lambda k: k["confidence"], reverse=True)
        if n_answers is not None:
            answers = answers[:n_answers]

        # transform confidence scores
        if not raw_confidence:
            confidences = [a["confidence"] for a in answers]
            max_conf = max(confidences)
            total = 0.0
            exp_scores = []
            for c in confidences:
                s = np.exp(c - max_conf)
                exp_scores.append(s)
            total = sum(exp_scores)
            for idx, c in enumerate(confidences):
                answers[idx]["confidence"] = exp_scores[idx] / total

        if rerank_threshold is None or self.te is None:
            return answers

        # re-rank
        top_confidences = [
            a["confidence"]
            for idx, a in enumerate(answers)
            if a["confidence"] > rerank_threshold
        ]
        v1 = self.te.embed(question, word_level=False)
        for idx, answer in enumerate(answers):
            # if idx >= rerank_top_n:
            if answer["confidence"] <= rerank_threshold:
                answer["similarity_score"] = 0.0
                continue
            v2 = self.te.embed(answer["full_answer"], word_level=False)
            score = v1 @ v2.T / (np.linalg.norm(v1) * np.linalg.norm(v2))
            answer["similarity_score"] = float(np.squeeze(score))
            answer["confidence"] = top_confidences[idx]
        answers = sorted(
            answers,
            key=lambda k: (k["similarity_score"], k["confidence"]),
            reverse=True,
        )
        for idx, confidence in enumerate(top_confidences):
            answers[idx]["confidence"] = confidence

        return answers

    def display_answers(self, answers):
        return display_answers(answers)


class SimpleQA(ExtractiveQABase):
    """
    SimpleQA: Question-Answering on a list of texts
    """

    def __init__(
        self,
        index_dir,
        model_name=DEFAULT_MODEL,
        bert_squad_model=None,  # deprecated
        bert_emb_model="bert-base-uncased",
        framework="tf",
        device=None,
        quantize=False,
    ):
        """
        ```
        SimpleQA constructor
        Args:
          index_dir(str):  path to index directory created by SimpleQA.initialze_index
          model_name(str): name of Question-Answering model (e.g., BERT SQUAD) to use
          bert_squad_model(str): alias for model_name (deprecated)
          bert_emb_model(str): BERT model to use to generate embeddings for semantic similarity
          framework(str): 'tf' for TensorFlow or 'pt' for PyTorch
          device(str): Torch device to use (e.g., 'cuda', 'cpu'). Ignored if framework=='tf'.
                       If framework=='tf', use CUDA_VISIBLE_DEVICES environment variable
                       to select device.
          quantize(bool): If True and framework=='pt' and device != 'cpu', then faster quantized inference is used.
                      Ignored if framework=="tf".
        ```
        """

        self.index_dir = index_dir
        try:
            ix = index.open_dir(self.index_dir)
        except:
            raise ValueError(
                'index_dir has not yet been created - please call SimpleQA.initialize_index("%s")'
                % (self.index_dir)
            )
        super().__init__(
            model_name=model_name,
            bert_squad_model=bert_squad_model,
            bert_emb_model=bert_emb_model,
            framework=framework,
            device=device,
            quantize=quantize,
        )

    def _open_ix(self):
        return index.open_dir(self.index_dir)

    @classmethod
    def initialize_index(cls, index_dir):
        schema = Schema(
            reference=ID(stored=True), content=TEXT, rawtext=TEXT(stored=True)
        )
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        else:
            raise ValueError(
                "There is already an existing directory or file with path %s"
                % (index_dir)
            )
        ix = index.create_in(index_dir, schema)
        return ix

    @classmethod
    def index_from_list(
        cls,
        docs,
        index_dir,
        commit_every=1024,
        breakup_docs=True,
        procs=1,
        limitmb=256,
        multisegment=False,
        min_words=20,
        references=None,
    ):
        """
        ```
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
        ```
        """
        if not isinstance(docs, (np.ndarray, list)):
            raise ValueError("docs must be a list of strings")
        if references is not None and not isinstance(references, (np.ndarray, list)):
            raise ValueError("references must be a list of strings")
        if references is not None and len(references) != len(docs):
            raise ValueError("lengths of docs and references must be equal")

        ix = index.open_dir(index_dir)
        writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)

        mb = master_bar(range(1))
        for i in mb:
            for idx, doc in enumerate(progress_bar(docs, parent=mb)):
                reference = "%s" % (idx) if references is None else references[idx]

                if breakup_docs:
                    small_docs = TU.paragraph_tokenize(
                        doc, join_sentences=True, lang="en"
                    )
                    refs = [reference] * len(small_docs)
                    for i, small_doc in enumerate(small_docs):
                        if len(small_doc.split()) < min_words:
                            continue
                        content = small_doc
                        reference = refs[i]
                        writer.add_document(
                            reference=reference, content=content, rawtext=content
                        )
                else:
                    if len(doc.split()) < min_words:
                        continue
                    content = doc
                    writer.add_document(
                        reference=reference, content=content, rawtext=content
                    )

                idx += 1
                if idx % commit_every == 0:
                    writer.commit()
                    # writer = ix.writer()
                    writer = ix.writer(
                        procs=procs, limitmb=limitmb, multisegment=multisegment
                    )
                mb.child.comment = f"indexing documents"
            writer.commit()
            # mb.write(f'Finished indexing documents')
        return

    @classmethod
    def index_from_folder(
        cls,
        folder_path,
        index_dir,
        use_text_extraction=False,
        commit_every=1024,
        breakup_docs=True,
        min_words=20,
        encoding="utf-8",
        procs=1,
        limitmb=256,
        multisegment=False,
        verbose=1,
    ):
        """
        ```
        index all plain text documents within a folder.
        The procs, limitmb, and especially multisegment arguments can be used to
        speed up indexing, if it is too slow.  Please see the whoosh documentation
        for more information on these parameters:  https://whoosh.readthedocs.io/en/latest/batch.html

        Args:
          folder_path(str): path to folder containing plain text documents (e.g., .txt files)
          index_dir(str): path to index directory (see initialize_index)
          use_text_extraction(bool): If True, the  `textract` package will be used to index text from various
                                     file types including PDF, MS Word, and MS PowerPoint (in addition to plain text files).
                                     If False, only plain text files will be indexed.
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
        ```
        """
        if use_text_extraction:
            # TODO:  change this to use TextExtractor
            try:
                import textract
            except ImportError:
                raise Exception(
                    "use_text_extraction=True requires textract:   pip install textract"
                )

        if not os.path.isdir(folder_path):
            raise ValueError("folder_path is not a valid folder")
        if folder_path[-1] != os.sep:
            folder_path += os.sep
        ix = index.open_dir(index_dir)
        writer = ix.writer(procs=procs, limitmb=limitmb, multisegment=multisegment)
        for idx, fpath in enumerate(TU.extract_filenames(folder_path)):
            reference = "%s" % (fpath.join(fpath.split(folder_path)[1:]))
            if TU.is_txt(fpath):
                with open(fpath, "r", encoding=encoding) as f:
                    doc = f.read()
            else:
                if use_text_extraction:
                    try:
                        doc = textract.process(fpath)
                        doc = doc.decode("utf-8", "ignore")
                    except:
                        if verbose:
                            warnings.warn("Could not extract text from %s" % (fpath))
                        continue
                else:
                    continue

            if breakup_docs:
                small_docs = TU.paragraph_tokenize(doc, join_sentences=True, lang="en")
                refs = [reference] * len(small_docs)
                for i, small_doc in enumerate(small_docs):
                    if len(small_doc.split()) < min_words:
                        continue
                    content = small_doc
                    reference = refs[i]
                    writer.add_document(
                        reference=reference, content=content, rawtext=content
                    )
            else:
                if len(doc.split()) < min_words:
                    continue
                content = doc
                writer.add_document(
                    reference=reference, content=content, rawtext=content
                )

            idx += 1
            if idx % commit_every == 0:
                writer.commit()
                writer = ix.writer(
                    procs=procs, limitmb=limitmb, multisegment=multisegment
                )
                if verbose:
                    print("%s docs indexed" % (idx))
        writer.commit()
        return

    def search(self, query, limit=10):
        """
        ```
        search index for query
        Args:
          query(str): search query
          limit(int):  number of top search results to return
        Returns:
          list of dicts with keys: reference, rawtext
        ```
        """
        ix = self._open_ix()
        with ix.searcher() as searcher:
            query_obj = QueryParser("content", ix.schema, group=qparser.OrGroup).parse(
                query
            )
            results = searcher.search(query_obj, limit=limit)
            docs = []
            output = [dict(r) for r in results]
            return output


class _QAExtractor(ExtractiveQABase):
    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        bert_squad_model=None,
        framework="tf",
        device=None,
        quantize=False,
    ):
        """
        ```
        QAExtractor is a convenience class for extracting answers from contexts
        Args:
          model_name(str): name of Question-Answering model (e.g., BERT SQUAD) to use
          bert_squad_model(str): alias for model_name (deprecated)
          framework(str): 'tf' for TensorFlow or 'pt' for PyTorch
          device(str): Torch device to use (e.g., 'cuda', 'cpu'). Ignored if framework=='tf'.
                       If framework=='tf', use CUDA_VISIBLE_DEVICES environment variable
                       to select device.
          quantize(bool): If True and framework=='pt' and device != 'cpu', then faster quantized inference is used.
                      Ignored if framework=="tf".
        ```
        """
        super().__init__(
            model_name=model_name,
            bert_squad_model=bert_squad_model,
            framework=framework,
            device=device,
            quantize=quantize,
        )

    def search(self, query):
        raise NotImplemented(
            "This method is not used or needed for extraction QA-based extraction."
        )

    def ask(self, question, batch_size=8, **kwargs):
        # locate candidate document contexts
        doc_results = kwargs.get("doc_results", [])
        if not doc_results:
            return []

        # extract paragraphs as contexts
        contexts, refs = self._split_contexts(doc_results)
        contexts = [c.replace("\n", " ") for c in contexts]

        # batchify contexts
        context_batches = self._batchify(contexts, batch_size=batch_size)

        # locate candidate answers
        answers = []
        mb = master_bar(range(1))
        answer_batches = []
        for i in mb:
            idx = 0
            for batch_id, contexts in enumerate(
                progress_bar(context_batches, parent=mb)
            ):
                answer_batch = self.predict_squad(contexts, question)
                answer_batches.extend(answer_batch)
                for i, answer in enumerate(answer_batch):
                    idx += 1
                    if not answer["answer"]:
                        answer["answer"] = None
                    answer["confidence"] = (
                        answer["confidence"]
                        if isinstance(
                            answer["confidence"], (int, float, np.float32, np.float16)
                        )
                        else answer["confidence"].numpy()
                    )
                    answer["reference"] = refs[idx - 1]
                    if answer["answer"] is not None:
                        formatted_answer = self._span_to_answer(
                            question, contexts[i], answer["start"], answer["end"]
                        )["answer"].strip()
                        if formatted_answer:
                            answer["answer"] = formatted_answer
                    answer["answer"] = self._clean_answer(answer["answer"])
                    answers.append(answer)
                mb.child.comment = f"extracting information"
        return answers


class AnswerExtractor:
    """
    Question-Answering-based Information Extraction
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        bert_squad_model=None,
        framework="tf",
        device=None,
        quantize=False,
    ):
        """
        ```
        Extracts information from documents using Question-Answering.

          model_name(str): name of Question-Answering model (e.g., BERT SQUAD) to use
          bert_squad_model(str): alias for model_name (deprecated)
          framework(str): 'tf' for TensorFlow or 'pt' for PyTorch
          device(str): Torch device to use (e.g., 'cuda', 'cpu'). Ignored if framework=='tf'.
                       If framework=='tf', use CUDA_VISIBLE_DEVICES environment variable
                       to select device.
          quantize(bool): If True and framework=='pt' and device != 'cpu', then faster quantized inference is used.
                      Ignored if framework=="tf".
        ```
        """
        self.qa = _QAExtractor(
            model_name=model_name,
            bert_squad_model=bert_squad_model,
            framework=framework,
            device=device,
            quantize=quantize,
        )
        return

    def _check_columns(self, labels, df):
        """check columns"""
        cols = df.columns.values
        for l in labels:
            if l in cols:
                raise ValueError(
                    "There is already a column named %s in your DataFrame." % (l)
                )

    def _extract(
        self,
        questions,
        contexts,
        min_conf=DEFAULT_MIN_CONF,
        return_conf=False,
        batch_size=8,
    ):
        """
        ```
        Extracts answers
        ```
        """
        num_rows = len(contexts)
        doc_results = [
            {"rawtext": rawtext, "reference": row}
            for row, rawtext in enumerate(contexts)
        ]
        cols = []
        for q in questions:
            result_dict = {}
            conf_dict = {}
            answers = self.qa.ask(q, doc_results=doc_results, batch_size=batch_size)
            for a in answers:
                answer = a["answer"] if a["confidence"] > min_conf else None
                lst = result_dict.get(a["reference"], [])
                lst.append(answer)
                result_dict[a["reference"]] = lst
                lst = conf_dict.get(a["reference"], [])
                lst.append(a["confidence"])
                conf_dict[a["reference"]] = lst

            results = []
            for i in range(num_rows):
                ans = [a for a in result_dict[i] if a is not None]
                results.append(None if not ans else " | ".join(ans))
            cols.append(results)
            if return_conf:
                confs = []
                for i in range(num_rows):
                    conf = [str(round(c, 2)) for c in conf_dict[i] if c is not None]
                    confs.append(None if not conf else " | ".join(conf))
                cols.append(confs)
        return cols

    def extract(
        self,
        texts,
        df,
        question_label_pairs,
        min_conf=DEFAULT_MIN_CONF,
        return_conf=False,
        batch_size=8,
    ):
        """
        ```
        Extracts answers from texts

        Args:
          texts(list): list of strings
          df(pd.DataFrame): original DataFrame to which columns need to be added
          question_label_pairs(list):  A list of tuples of the form (question, label).
                                     Extracted ansewrs to the question will be added as new columns with the
                                     specified labels.
                                     Example: ('What are the risk factors?', 'Risk Factors')
          min_conf(float):  Answers at or below this confidence value will be set to None in the results
                            Default: 5.0
                            Lower this value to reduce false negatives.
                            Raise this value to reduce false positives.
          return_conf(bool): If True, confidence score of each extraction is included in results
          batch_size(int): batch size. Default: 8
        ```
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if len(texts) != df.shape[0]:
            raise ValueError(
                "Number of texts is not equal to the number of rows in the DataFrame."
            )
        # texts = [t.replace("\n", " ").replace("\t", " ") for t in texts]
        texts = [t.replace("\t", " ") for t in texts]
        questions = [q for q, l in question_label_pairs]
        labels = [l for q, l in question_label_pairs]
        self._check_columns(labels, df)
        cols = self._extract(
            questions,
            texts,
            min_conf=min_conf,
            return_conf=return_conf,
            batch_size=batch_size,
        )
        data = list(zip(*cols)) if len(cols) > 1 else cols[0]
        if return_conf:
            labels = twolists(labels, [l + " CONF" for l in labels])
        return df.join(pd.DataFrame(data, columns=labels, index=df.index))

    def finetune(
        self, data, epochs=3, learning_rate=2e-5, batch_size=8, max_seq_length=512
    ):
        """
        ```
        Finetune a QA model.

        Args:
          data(list): list of dictionaries of the form:
                      [{'question': 'What is ktrain?'
                       'context': 'ktrain is a low-code library for augmented machine learning.'
                       'answer': 'ktrain'}]
          epochs(int): number of epochs.  Default:3
          learning_rate(float): learning rate.  Default: 2e-5
          batch_size(int): batch size. Default:8
          max_seq_length(int): maximum sequence length.  Default:512
        Returns:
          None
        ```
        """
        if self.qa.framework != "tf":
            raise ValueError(
                'The finetune method does not currently support the framework="pt" option. Please use framework="tf" to finetune.'
            )
        from .qa_finetuner import QAFineTuner

        ft = QAFineTuner(self.qa.model, self.qa.tokenizer)
        model = ft.finetune(
            data, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size
        )
        return
