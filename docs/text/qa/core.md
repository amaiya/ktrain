Module ktrain.text.qa.core
==========================

Functions
---------

    
`display_answers(answers)`
:   

    
`pack_byte(...)`
:   S.pack(v1, v2, ...) -> bytes
    
    Return a bytes object containing values v1, v2, ... packed according
    to the format string S.format.  See help(struct) for more on format
    strings.

    
`unpack_byte(...)`
:   S.unpack(buffer) -> (v1, v2, ...)
    
    Return a tuple containing values unpacked according to the format
    string S.format.  The buffer's size in bytes must be S.size.  See
    help(struct) for more on format strings.

Classes
-------

`QA(bert_squad_model='bert-large-uncased-whole-word-masking-finetuned-squad', bert_emb_model='bert-base-uncased')`
:   Base class for QA

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * ktrain.text.qa.core.SimpleQA

    ### Methods

    `ask(self, question, batch_size=8, n_docs_considered=10, n_answers=50, rerank_threshold=0.015, include_np=False)`
    :   submit question to obtain candidate answers
        
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

    `display_answers(self, answers)`
    :

    `predict_squad(self, documents, question)`
    :   Generates candidate answers to the <question> provided given <documents> as contexts.

    `search(self, query)`
    :

`SimpleQA(index_dir, bert_squad_model='bert-large-uncased-whole-word-masking-finetuned-squad', bert_emb_model='bert-base-uncased')`
:   SimpleQA: Question-Answering on a list of texts
    
    SimpleQA constructor
    Args:
      index_dir(str):  path to index directory created by SimpleQA.initialze_index
      bert_squad_model(str): name of BERT SQUAD model to use
      bert_emb_model(str): BERT model to use to generate embeddings for semantic similarity

    ### Ancestors (in MRO)

    * ktrain.text.qa.core.QA
    * abc.ABC

    ### Static methods

    `index_from_folder(folder_path, index_dir, use_text_extraction=False, commit_every=1024, breakup_docs=True, min_words=20, encoding='utf-8', procs=1, limitmb=256, multisegment=False, verbose=1)`
    :   index all plain text documents within a folder.
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

    `index_from_list(docs, index_dir, commit_every=1024, breakup_docs=True, procs=1, limitmb=256, multisegment=False, min_words=20, references=None)`
    :   index documents from list.
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
                               Example: ['ktrain_article        https://arxiv.org/pdf/2004.10703v4.pdf', ...]
        
                            These references will be returned in the output of the ask method.
                            If strings are  hyperlinks, then they will automatically be made clickable when the display_answers function
                            displays candidate answers in a pandas DataFRame.
        
                            If references is None, the index of element in docs is used as reference.

    `initialize_index(index_dir)`
    :

    ### Methods

    `search(self, query, limit=10)`
    :   search index for query
        Args:
          query(str): search query
          limit(int):  number of top search results to return
        Returns:
          list of dicts with keys: reference, rawtext