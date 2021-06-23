Module ktrain.text
==================

Sub-modules
-----------
* ktrain.text.data
* ktrain.text.eda
* ktrain.text.learner
* ktrain.text.models
* ktrain.text.ner
* ktrain.text.predictor
* ktrain.text.preprocessor
* ktrain.text.qa
* ktrain.text.shallownlp
* ktrain.text.summarization
* ktrain.text.textutils
* ktrain.text.translation
* ktrain.text.zsl

Functions
---------

    
`entities_from_array(x_train, y_train, x_test=None, y_test=None, use_char=False, val_pct=0.1, verbose=1)`
:   Load entities from arrays
    Args:
      x_train(list): list of list of entity tokens for training
                     Example: x_train = [['Hello', 'world'], ['Hello', 'Cher'], ['I', 'love', 'Chicago']]
      y_train(list): list of list of tokens representing entity labels
                     Example:  y_train = [['O', 'O'], ['O', 'B-PER'], ['O', 'O', 'B-LOC']]
      x_test(list): list of list of entity tokens for validation 
                     Example: x_train = [['Hello', 'world'], ['Hello', 'Cher'], ['I', 'love', 'Chicago']]
      y_test(list): list of list of tokens representing entity labels
                     Example:  y_train = [['O', 'O'], ['O', 'B-PER'], ['O', 'O', 'B-LOC']]
     use_char(bool):    If True, data will be preprocessed to use character embeddings  in addition to word embeddings
     val_pct(float):  percentage of training to use for validation if no validation data is supplied
     verbose (boolean): verbosity

    
`entities_from_conll2003(train_filepath, val_filepath=None, use_char=False, encoding=None, val_pct=0.1, verbose=1)`
:   Loads sequence-labeled data from a file in CoNLL2003 format.

    
`entities_from_df(train_df, val_df=None, word_column='Word', tag_column='Tag', sentence_column='SentenceID', use_char=False, val_pct=0.1, verbose=1)`
:   Load entities from pandas DataFrame
    Args:
      train_df(pd.DataFrame): training data
      val_df(pdf.DataFrame): validation data
      word_column(str): name of column containing the text
      tag_column(str): name of column containing lael
      sentence_column(str): name of column containing Sentence IDs
      use_char(bool):    If True, data will be preprocessed to use character embeddings  in addition to word embeddings
      verbose (boolean): verbosity

    
`entities_from_gmb(train_filepath, val_filepath=None, use_char=False, word_column='Word', tag_column='Tag', sentence_column='SentenceID', encoding=None, val_pct=0.1, verbose=1)`
:   Loads sequence-labeled data from text file in the  Groningen
    Meaning Bank  (GMB) format.

    
`entities_from_txt(train_filepath, val_filepath=None, use_char=False, word_column='Word', tag_column='Tag', sentence_column='SentenceID', data_format='conll2003', encoding=None, val_pct=0.1, verbose=1)`
:   Loads sequence-labeled data from comma or tab-delmited text file.
    Format of file is either the CoNLL2003 format or Groningen Meaning
    Bank (GMB) format - specified with data_format parameter.
    
    In both formats, each word appars on a separate line along with
    its associated tag (or label).  
    The last item on each line should be the tag or label assigned to word.
    
    In the CoNLL2003 format, there is an empty line after
    each sentence.  In the GMB format, sentences are deliniated
    with a third column denoting the Sentence ID.
    
    
    
    More information on CoNLL2003 format: 
       https://www.aclweb.org/anthology/W03-0419
    
    CoNLL Example (each column is typically separated by space or tab)
    and  no column headings:
    
       Paul     B-PER
       Newman   I-PER
       is       O
       a        O
       great    O
       actor    O
       !        O
    
    More information on GMB format:
    Refer to ner_dataset.csv on Kaggle here:
       https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/2
    
    GMB example (each column separated by comma or tab)
    with column headings:
    
      SentenceID   Word     Tag    
      1            Paul     B-PER
      1            Newman   I-PER
      1            is       O
      1            a        O
      1            great    O
      1            actor    O
      1            !        O
    
    
    Args:
        train_filepath(str): file path to training CSV
        val_filepath (str): file path to validation dataset
        use_char(bool):    If True, data will be preprocessed to use character embeddings in addition to word embeddings
        word_column(str): name of column containing the text
        tag_column(str): name of column containing lael
        sentence_column(str): name of column containing Sentence IDs
        data_format(str): one of colnll2003 or gmb
                          word_column, tag_column, and sentence_column
                          ignored if 'conll2003'
        encoding(str): the encoding to use.  If None, encoding is discovered automatically
        val_pct(float): Proportion of training to use for validation.
        verbose (boolean): verbosity

    
`extract_filenames(corpus_path, follow_links=False)`
:   

    
`load_text_files(corpus_path, truncate_len=None, clean=True, return_fnames=False)`
:   load text files

    
`print_sequence_taggers()`
:   

    
`print_text_classifiers()`
:   

    
`print_text_regression_models()`
:   

    
`sequence_tagger(name, preproc, wv_path_or_url=None, bert_model='bert-base-multilingual-cased', bert_layers_to_use=[-2], word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, verbose=1)`
:   Build and return a sequence tagger (i.e., named entity recognizer).
    
    Args:
        name (string): one of:
                      - 'bilstm-crf' for Bidirectional LSTM-CRF model
                      - 'bilstm' for Bidirectional LSTM (no CRF layer)
        preproc(NERPreprocessor):  an instance of NERPreprocessor
        wv_path_or_url(str): either a URL or file path toa fasttext word vector file (.vec or .vec.zip or .vec.gz)
                             Example valid values for wv_path_or_url:
    
                               Randomly-initialized word embeeddings:
                                 set wv_path_or_url=None
                               English pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
                               Chinese pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
                               Russian pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
                               Dutch pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz
    
    
                             See these two Web pages for a full list of URLs to word vector files for 
                             different languages:
                                1.  https://fasttext.cc/docs/en/english-vectors.html (for English)
                                2.  https://fasttext.cc/docs/en/crawl-vectors.html (for non-English langages)
    
                            Default:None (randomly-initialized word embeddings are used)
    
        bert_model_name(str):  the name of the BERT model.  default: 'bert-base-multilingual-cased'
                               This parameter is only used if bilstm-bert is selected for name parameter.
                               The value of this parameter is a name of BERT model from here:
                                        https://huggingface.co/transformers/pretrained_models.html
                               or a community-uploaded BERT model from here:
                                        https://huggingface.co/models
                               Example values:
                                 bert-base-multilingual-cased:  Multilingual BERT (157 languages) - this is the default
                                 bert-base-cased:  English BERT
                                 bert-base-chinese: Chinese BERT
                                 distilbert-base-german-cased: German DistilBert
                                 albert-base-v2: English ALBERT model
                                 monologg/biobert_v1.1_pubmed: community uploaded BioBERT (pretrained on PubMed)
    
        bert_layers_to_use(list): indices of hidden layers to use.  default:[-2] # second-to-last layer
                                  To use the concatenation of last 4 layers: use [-1, -2, -3, -4]
        word_embedding_dim (int): word embedding dimensions.
        char_embedding_dim (int): character embedding dimensions.
        word_lstm_size (int): character LSTM feature extractor output dimensions.
        char_lstm_size (int): word tagger LSTM output dimensions.
        fc_dim (int): output fully-connected layer size.
        dropout (float): dropout rate.
    
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`text_classifier(name, train_data, preproc=None, multilabel=None, metrics=['accuracy'], verbose=1)`
:   Build and return a text classification model.
    
    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'logreg' for logistic regression using embedding layers
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model
    
        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train) or ktrain.Dataset instance
                            returned from one of the texts_from_* functions
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        metrics(list): metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`text_regression_model(name, train_data, preproc=None, metrics=['mae'], verbose=1)`
:   Build and return a text regression model.
    
    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'linreg' for linear regression using embedding layers
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model
    
        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train)
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        metrics(list): metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`texts_from_array(x_train, y_train, x_test=None, y_test=None, class_names=[], max_features=20000, maxlen=400, val_pct=0.1, ngram_range=1, preprocess_mode='standard', lang=None, random_state=None, verbose=1)`
:   Loads and preprocesses text data from arrays.
    texts_from_array can handle data for both text classification
    and text regression.  If class_names is empty, a regression task is assumed.
    Args:
        x_train(list): list of training texts 
        y_train(list): labels in one of the following forms:
                       1. list of integers representing classes (class_names is required)
                       2. list of strings representing classes (class_names is not needed and ignored.)
                       3. a one or multi hot encoded array representing classes (class_names is required)
                       4. numerical values for text regresssion (class_names should be left empty)
        x_test(list): list of training texts 
        y_test(list): labels in one of the following forms:
                       1. list of integers representing classes (class_names is required)
                       2. list of strings representing classes (class_names is not needed and ignored.)
                       3. a one or multi hot encoded array representing classes (class_names is required)
                       4. numerical values for text regresssion (class_names should be left empty)
        class_names (list): list of strings representing class labels
                            shape should be (num_examples,1) or (num_examples,)
        max_features(int): max num of words to consider in vocabulary
                           Note: This is only used for preprocess_mode='standard'.
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if x_val and  y_val is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        lang (str):            language.  Auto-detected if None.
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random.
        verbose (boolean): verbosity

    
`texts_from_csv(train_filepath, text_column, label_columns=[], val_filepath=None, max_features=20000, maxlen=400, val_pct=0.1, ngram_range=1, preprocess_mode='standard', encoding=None, lang=None, sep=',', is_regression=False, random_state=None, verbose=1)`
:   Loads text data from CSV or TSV file. Class labels are assumed to be
    one of the following formats:
        1. one-hot-encoded or multi-hot-encoded arrays representing classes:
              Example with label_columns=['positive', 'negative'] and text_column='text':
                text|positive|negative
                I like this movie.|1|0
                I hated this movie.|0|1
            Classification will have a single one in each row: [[1,0,0], [0,1,0]]]
            Multi-label classification will have one more ones in each row: [[1,1,0], [0,1,1]]
        2. labels are in a single column of string or integer values representing classs labels
               Example with label_columns=['label'] and text_column='text':
                 text|label
                 I like this movie.|positive
                 I hated this movie.|negative
       3. labels are a single column of numerical values for text regression
          NOTE: Must supply is_regression=True for labels to be treated as numerical targets
                 wine_description|wine_price
                 Exquisite wine!|100
                 Wine for budget shoppers|8
    
    Args:
        train_filepath(str): file path to training CSV
        text_column(str): name of column containing the text
        label_column(list): list of columns that are to be treated as labels
        val_filepath(string): file path to test CSV.  If not supplied,
                               10% of documents in training CSV will be
                               used for testing/validation.
        max_features(int): max num of words to consider in vocabulary
                           Note: This is only used for preprocess_mode='standard'.
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if val_filepath is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        encoding (str):        character encoding to use. Auto-detected if None
        lang (str):            language.  Auto-detected if None.
        sep(str):              delimiter for CSV (comma is default)
        is_regression(bool):  If True, integer targets will be treated as numerical targets instead of class IDs
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random
        verbose (boolean): verbosity

    
`texts_from_df(train_df, text_column, label_columns=[], val_df=None, max_features=20000, maxlen=400, val_pct=0.1, ngram_range=1, preprocess_mode='standard', lang=None, is_regression=False, random_state=None, verbose=1)`
:   Loads text data from Pandas dataframe file. Class labels are assumed to be
    one of the following formats:
        1. one-hot-encoded or multi-hot-encoded arrays representing classes:
              Example with label_columns=['positive', 'negative'] and text_column='text':
                text|positive|negative
                I like this movie.|1|0
                I hated this movie.|0|1
            Classification will have a single one in each row: [[1,0,0], [0,1,0]]]
            Multi-label classification will have one more ones in each row: [[1,1,0], [0,1,1]]
        2. labels are in a single column of string or integer values representing class labels
               Example with label_columns=['label'] and text_column='text':
                 text|label
                 I like this movie.|positive
                 I hated this movie.|negative
       3. labels are a single column of numerical values for text regression
          NOTE: Must supply is_regression=True for integer labels to be treated as numerical targets
                 wine_description|wine_price
                 Exquisite wine!|100
                 Wine for budget shoppers|8
    
    Args:
        train_df(dataframe): Pandas dataframe
        text_column(str): name of column containing the text
        label_columns(list): list of columns that are to be treated as labels
        val_df(dataframe): file path to test dataframe.  If not supplied,
                               10% of documents in training df will be
                               used for testing/validation.
        max_features(int): max num of words to consider in vocabulary.
                           Note: This is only used for preprocess_mode='standard'.
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if val_filepath is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        lang (str):            language.  Auto-detected if None.
        is_regression(bool):  If True, integer targets will be treated as numerical targets instead of class IDs
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random
        verbose (boolean): verbosity

    
`texts_from_folder(datadir, classes=None, max_features=20000, maxlen=400, ngram_range=1, train_test_names=['train', 'test'], preprocess_mode='standard', encoding=None, lang=None, val_pct=0.1, random_state=None, verbose=1)`
:   Returns corpus as sequence of word IDs.
    Assumes corpus is in the following folder structure:
    ├── datadir
    │   ├── train
    │   │   ├── class0       # folder containing documents of class 0
    │   │   ├── class1       # folder containing documents of class 1
    │   │   ├── class2       # folder containing documents of class 2
    │   │   └── classN       # folder containing documents of class N
    │   └── test 
    │       ├── class0       # folder containing documents of class 0
    │       ├── class1       # folder containing documents of class 1
    │       ├── class2       # folder containing documents of class 2
    │       └── classN       # folder containing documents of class N
    
    Each subfolder should contain documents in plain text format.
    If train and test contain additional subfolders that do not represent
    classes, they can be ignored by explicitly listing the subfolders of
    interest using the classes argument.
    Args:
        datadir (str): path to folder
        classes (list): list of classes (subfolders to consider).
                        This is simply supplied as the categories argument
                        to sklearn's load_files function.
        max_features (int):  maximum number of unigrams to consider
                             Note: This is only used for preprocess_mode='standard'.
        maxlen (int):  maximum length of tokens in document
        ngram_range (int):  If > 1, will include 2=bigrams, 3=trigrams and bigrams
        train_test_names (list):  list of strings represnting the subfolder
                                 name for train and validation sets
                                 if test name is missing, <val_pct> of training
                                 will be used for validation
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        encoding (str):        character encoding to use. Auto-detected if None
        lang (str):            language.  Auto-detected if None.
        val_pct(float):        Onlyl used if train_test_names  has 1 and not 2 names
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                IF None, train/test split will be random
        verbose (bool):         verbosity

Classes
-------

`EnglishTranslator(src_lang=None, device=None)`
:   Class to translate text in various languages to English.
    
    Constructor for English translator
    
    Args:
      src_lang(str): language code of source language.
                     Must be one of SUPPORTED_SRC_LANGS:
                       'zh': Chinese (either tradtional or simplified)
                       'ar': Arabic
                       'ru' : Russian
                       'de': German
                       'af': Afrikaans
                       'es': Spanish
                       'fr': French
                       'it': Italian
                       'pt': Portuguese
      device(str): device to use (e.g., 'cuda', 'cpu')

    ### Methods

    `translate(self, src_text, join_with='\n', num_beams=None, early_stopping=None)`
    :   Translate source document to English.
        To speed up translations, you can set num_beams and early_stopping (e.g., num_beams=4, early_stopping=True).
        
        Args:
          src_text(str): source text. Must be in language specified by src_lang (language code) supplied to constructor
                         The source text can either be a single sentence or an entire document with multiple sentences
                         and paragraphs. 
                         IMPORTANT NOTE: Sentences are joined together and fed to model as single batch.
                                         If the input text is very large (e.g., an entire book), you should
                                         break it up into reasonbly-sized chunks (e.g., pages, paragraphs, or sentences) and 
                                         feed each chunk separately into translate to avoid out-of-memory issues.
          join_with(str):  list of translated sentences will be delimited with this character.
                           default: each sentence on separate line
          num_beams(int): Number of beams for beam search. Defaults to None.  If None, the transformers library defaults this to 1, 
                          whicn means no beam search.
          early_stopping(bool):  Whether to stop the beam search when at least ``num_beams`` sentences 
                                 are finished per batch or not. Defaults to None.  If None, the transformers library
                                 sets this to False.
        Returns:
          str: translated text

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

`get_topic_model(texts=None, n_topics=None, n_features=10000, min_df=5, max_df=0.5, stop_words='english', model_type='lda', lda_max_iter=5, lda_mode='online', token_pattern=None, verbose=1, hyperparam_kwargs=None)`
:   Fits a topic model to documents in <texts>.
    Example:
        tm = ktrain.text.get_topic_model(docs, n_topics=20, 
                                        n_features=1000, min_df=2, max_df=0.95)
    Args:
        texts (list of str): list of texts
        n_topics (int): number of topics.
                        If None, n_topics = min{400, sqrt[# documents/2]})
        n_features (int):  maximum words to consider
        max_df (float): words in more than max_df proportion of docs discarded
        stop_words (str or list): either 'english' for built-in stop words or
                                  a list of stop words to ignore
        model_type(str): type of topic model to fit. One of {'lda', 'nmf'}.  Default:'lda'
        lda_max_iter (int): maximum iterations for 'lda'.  5 is default if using lda_mode='online'.
                            If lda_mode='batch', this should be increased (e.g., 1500).
                            Ignored if model_type != 'lda'
        lda_mode (str):  one of {'online', 'batch'}. Ignored if model_type !='lda'
        token_pattern(str): regex pattern to use to tokenize documents. 
        verbose(bool): verbosity

    ### Instance variables

    `topics`
    :   convenience method/property

    ### Methods

    `build(self, texts, threshold=None)`
    :   Builds the document-topic distribution showing the topic probability distirbution
        for each document in <texts> with respect to the learned topic space.
        Args:
            texts (list of str): list of text documents
            threshold (float): If not None, documents with whose highest topic probability
                               is less than threshold are filtered out.

    `filter(self, lst)`
    :   The build method may prune documents based on threshold.
        This method prunes other lists based on how build pruned documents.
        This is useful to filter lists containing metadata associated with documents
        for use with visualize_documents.
        Args:
            lst(list): a list of data
        Returns:
            list:  a filtered list of data based on how build filtered the documents

    `get_docs(self, topic_ids=[], doc_ids=[], rank=False)`
    :   Returns document entries for supplied topic_ids.
        Documents returned are those whose primary topic is topic with given topic_id
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            rank(bool): If True, the list is sorted first by topic_id (ascending)
                        and then ty topic probability (descending).
                        Otherwise, list is sorted by doc_id (i.e., the order
                        of texts supplied to self.build (which is the order of self.doc_topics).
        
        Returns:
            list of dicts:  list of dicts with keys:
                            'text': text of document
                            'doc_id': ID of document
                            'topic_proba': topic probability (or score)
                            'topic_id': ID of topic

    `get_doctopics(self, topic_ids=[], doc_ids=[])`
    :   Returns a topic probability distribution for documents
        with primary topic that is one of <topic_ids> and with doc_id in <doc_ids>.
        
        If no topic_ids or doc_ids are provided, then topic distributions for all documents
        are returned (which equivalent to the output of get_document_topic_distribution).
        
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
        Returns:
            np.ndarray: Each row is the topic probability distribution of a document.
                        Array is sorted in the order returned by self.get_docs.

    `get_document_topic_distribution(self)`
    :   Gets the document-topic distribution.
        Each row is a document and each column is a topic
        The output of this method is equivalent to invoking get_doctopics with no arguments.

    `get_sorted_docs(self, topic_id)`
    :   Returns all docs sorted by relevance to <topic_id>.
        Unlike get_docs, this ranks documents by the supplied topic_id rather
        than the topic_id to which document is most relevant.

    `get_texts(self, topic_ids=[])`
    :   Returns texts for documents
        with primary topic that is one of <topic_ids>
        Args:
            topic_ids(list of ints): list of topic IDs
        Returns:
            list of str

    `get_topics(self, n_words=10, as_string=True)`
    :   Returns a list of discovered topics
        Args:
            n_words(int): number of words to use in topic summary
            as_string(bool): If True, each summary is a space-delimited string instead of list of words

    `get_word_weights(self, topic_id, n_words=100)`
    :   Returns a list tuples of the form: (word, weight) for given topic_id.
        The weight can be interpreted as the number of times word was assigned to topic with given topic_id.
        REFERENCE: https://stackoverflow.com/a/48890889/13550699
        Args:
            topic_id(int): topic ID
            n_words=int): number of top words

    `predict(self, texts, threshold=None, harden=False)`
    :   Args:
            texts (list of str): list of texts
            threshold (float): If not None, documents with maximum topic scores
                                less than <threshold> are filtered out
            harden(bool): If True, each document is assigned to a single topic for which
                          it has the highest score
        Returns:
            if threshold is None:
                np.ndarray: topic distribution for each text document
            else:
                (np.ndarray, np.ndarray): topic distribution and boolean array

    `print_topics(self, n_words=10, show_counts=False)`
    :   print topics
        n_words(int): number of words to describe each topic
        show_counts(bool): If True, print topics with document counts, where
                           the count is the number of documents with that topic as primary.

    `recommend(self, text=None, doc_topic=None, n=5, n_neighbors=100)`
    :   Given an example document, recommends documents similar to it
        from the set of documents supplied to build().
        
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            n (int): number of recommendations to return
        Returns:
            list of tuples: each tuple is of the form:
                            (text, doc_id, topic_probability, topic_id)

    `save(self, fname)`
    :   save TopicModel object

    `score(self, texts=None, doc_topics=None)`
    :   Given a new set of documents (supplied as texts or doc_topics), the score method
        uses a One-Class classifier to score documents based on similarity to a
        seed set of documents (where seed set is computed by train_scorer() method).
        
        Higher scores indicate a higher degree of similarity.
        Positive values represent a binary decision of similar.
        Negative values represent a binary decision of dissimlar.
        In practice, negative scores closer to zer will also be simlar as One-Class
        classifiers are more strict than traditional binary classifiers.
        Documents with negative scores closer to zero are good candidates for
        inclusion in a training set for binary classification (e.g., active labeling).
        
        NOTE: The score method currently employs the use of LocalOutLierFactor, which
        means you should not try to score documents that were used in training. Only
        new, unseen documents should be scored for similarity.
        
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
        Returns:
            list of floats:  larger values indicate higher degree of similarity
                             positive values indicate a binary decision of similar
                             negative values indicate binary decision of dissimilar
                             In practice, negative scores closer to zero will also 
                             be similar as One-class classifiers are more strict
                             than traditional binary classifiers.

    `search(self, query, topic_ids=[], doc_ids=[], case_sensitive=False)`
    :   search documents for query string.
        Args:
            query(str):  the word or phrase to search
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics
            case_sensitive(bool):  If True, case sensitive search

    `train(self, texts, model_type='lda', n_topics=None, n_features=10000, min_df=5, max_df=0.5, stop_words='english', lda_max_iter=5, lda_mode='online', token_pattern=None, hyperparam_kwargs=None)`
    :   Fits a topic model to documents in <texts>.
        Example:
            tm = ktrain.text.get_topic_model(docs, n_topics=20, 
                                            n_features=1000, min_df=2, max_df=0.95)
        Args:
            texts (list of str): list of texts
            n_topics (int): number of topics.
                            If None, n_topics = min{400, sqrt[# documents/2]})
            n_features (int):  maximum words to consider
            max_df (float): words in more than max_df proportion of docs discarded
            stop_words (str or list): either 'english' for built-in stop words or
                                      a list of stop words to ignore
            lda_max_iter (int): maximum iterations for 'lda'.  5 is default if using lda_mode='online'.
                                If lda_mode='batch', this should be increased (e.g., 1500).
                                Ignored if model_type != 'lda'
            lda_mode (str):  one of {'online', 'batch'}. Ignored of model_type !='lda'
            token_pattern(str): regex pattern to use to tokenize documents. 
                                If None, a default tokenizer will be used
            hyperparam_kwargs(dict): hyperparameters for LDA/NMF
                                     Keys in this dict can be any of the following:
                                         alpha: alpha for LDA  default: 5./n_topics
                                         beta: beta for LDA.  default:0.01
                                         nmf_alpha: alpha for NMF.  default:0
                                         l1_ratio: l1_ratio for NMF. default: 0
                                         ngram_range:  whether to consider bigrams, trigrams. default: (1,1) 
                                    
        Returns:
            tuple: (model, vectorizer)

    `train_recommender(self, n_neighbors=20, metric='minkowski', p=2)`
    :   Trains a recommender that, given a single document, will return
        documents in the corpus that are semantically similar to it.
        
        Args:
            n_neighbors (int): 
        Returns:
            None

    `train_scorer(self, topic_ids=[], doc_ids=[], n_neighbors=20)`
    :   Trains a scorer that can score documents based on similarity to a
        seed set of documents represented by topic_ids and doc_ids.
        
        NOTE: The score method currently employs the use of LocalOutLierFactor, which
        means you should not try to score documents that were used in training. Only
        new, unseen documents should be scored for similarity. 
        REFERENCE: 
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
        
        Args:
            topic_ids(list of ints): list of topid IDs where each id is in the range
                                     of range(self.n_topics).  Documents associated
                                     with these topic_ids will be used as seed set.
            doc_ids (list of ints): list of document IDs where each id is an index
                                    into self.doctopics.  Documents associated 
                                    with these doc_ids will be used as seed set.
        Returns:
            None

    `visualize_documents(self, texts=None, doc_topics=None, width=700, height=700, point_size=5, title='Document Visualization', extra_info={}, colors=None, filepath=None)`
    :   Generates a visualization of a set of documents based on model.
        If <texts> is supplied, raw documents will be first transformed into document-topic
        matrix.  If <doc_topics> is supplied, then this will be used for visualization instead.
        Args:
            texts(list of str): list of document texts.  Mutually-exclusive with <doc_topics>
            doc_topics(ndarray): pre-computed topic distribution for each document in texts.
                                 Mutually-exclusive with <texts>.
            width(int): width of image
            height(int): height of image
            point_size(int): size of circles in plot
            title(str):  title of visualization
            extra_info(dict of lists): A user-supplied information for each datapoint (attributes of the datapoint).
                                       The keys are field names.  The values are lists - each of which must
                                       be the same number of elements as <texts> or <doc_topics>. These fields are displayed
                                       when hovering over datapoints in the visualization.
            colors(list of str):  list of Hex color codes for each datapoint.
                                  Length of list must match either len(texts) or doc_topics.shape[0]
            filepath(str):             Optional filepath to save the interactive visualization

`Transformer(model_name, maxlen=128, class_names=[], classes=[], batch_size=None, use_with_learner=True)`
:   convenience class for text classification Hugging Face transformers 
    Usage:
       t = Transformer('distilbert-base-uncased', maxlen=128, classes=['neg', 'pos'], batch_size=16)
       train_dataset = t.preprocess_train(train_texts, train_labels)
       model = t.get_classifier()
       model.fit(train_dataset)
    
    Args:
        model_name (str):  name of Hugging Face pretrained model
        maxlen (int):  sequence length
        class_names(list):  list of strings of class names (e.g., 'positive', 'negative').
                            The index position of string is the class ID.
                            Not required for:
                              - regression problems
                              - binary/multi classification problems where
                                labels in y_train/y_test are in string format.
                                In this case, classes will be populated automatically.
                                get_classes() can be called to view discovered class labels.
                            The class_names argument replaces the old classes argument.
        classes(list):  alias for class_names.  Included for backwards-compatiblity.
    
        use_with_learner(bool):  If False, preprocess_train and preprocess_test
                                 will return tf.Datasets for direct use with model.fit
                                 in tf.Keras.
                                 If True, preprocess_train and preprocess_test will
                                 return a ktrain TransformerDataset object for use with
                                 ktrain.get_learner.
        batch_size (int): batch_size - only required if use_with_learner=False

    ### Ancestors (in MRO)

    * ktrain.text.preprocessor.TransformersPreprocessor
    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `preprocess_test(self, texts, y=None, verbose=1)`
    :   Preprocess the validation or test set for a Transformer model
        Y values can be in one of the following forms:
        1) integers representing the class (index into array returned by get_classes)
           for binary and multiclass text classification.
           If labels are integers, class_names argument to Transformer constructor is required.
        2) strings representing the class (e.g., 'negative', 'positive').
           If labels are strings, class_names argument to Transformer constructor is ignored,
           as class labels will be extracted from y.
        3) multi-hot-encoded vector for multilabel text classification problems
           If labels are multi-hot-encoded, class_names argument to Transformer constructor is requird.
        4) Numerical values for regression problems.
           <class_names> argument to Transformer constructor should NOT be supplied
        
        Args:
            texts (list of strings): text of documents
            y: labels
            verbose(bool): verbosity
        Returns:
            TransformerDataset if self.use_with_learner = True else tf.Dataset

    `preprocess_train(self, texts, y=None, mode='train', verbose=1)`
    :   Preprocess training set for A Transformer model
        
        Y values can be in one of the following forms:
        1) integers representing the class (index into array returned by get_classes)
           for binary and multiclass text classification.
           If labels are integers, class_names argument to Transformer constructor is required.
        2) strings representing the class (e.g., 'negative', 'positive').
           If labels are strings, class_names argument to Transformer constructor is ignored,
           as class labels will be extracted from y.
        3) multi-hot-encoded vector for multilabel text classification problems
           If labels are multi-hot-encoded, class_names argument to Transformer constructor is requird.
        4) Numerical values for regression problems.
           <class_names> argument to Transformer constructor should NOT be supplied
        
        Args:
            texts (list of strings): text of documents
            y: labels
            mode (str):  If 'train' and prepare_for_learner=False,
                         a tf.Dataset will be returned with repeat enabled
                         for training with fit_generator
            verbose(bool): verbosity
        Returns:
          TransformerDataset if self.use_with_learner = True else tf.Dataset

`TransformerEmbedding(model_name, layers=[-2])`
:   Args:
        model_name (str):  name of Hugging Face pretrained model.
                           Choose from here: https://huggingface.co/transformers/pretrained_models.html
        layers(list): list of indexes indicating which hidden layers to use when
                      constructing the embedding (e.g., last=[-1])

    ### Methods

    `embed(self, texts, word_level=True, max_length=512)`
    :   get embedding for word, phrase, or sentence
        Args:
          text(str|list): word, phrase, or sentence or list of them representing a batch
          word_level(bool): If True, returns embedding for each token in supplied texts.
                            If False, returns embedding for each text in texts
          max_length(int): max length of tokens
        Returns:
            np.ndarray : embeddings

`TransformerSummarizer(model_name='facebook/bart-large-cnn', device=None)`
:   interface to Transformer-based text summarization
    
    interface to BART-based text summarization using transformers library
    
    Args:
      model_name(str): name of BART model for summarization
      device(str): device to use (e.g., 'cuda', 'cpu')

    ### Methods

    `summarize(self, doc)`
    :   summarize document text
        Args:
          doc(str): text of document
        Returns:
          str: summary text

`Translator(model_name=None, device=None, half=False)`
:   Translator: basic wrapper around MarianMT model for language translation
    
    basic wrapper around MarianMT model for language translation
    
    Args:
      model_name(str): Helsinki-NLP model
      device(str): device to use (e.g., 'cuda', 'cpu')
      half(bool): If True, use half precision.

    ### Methods

    `translate(self, src_text, join_with='\n', num_beams=None, early_stopping=None)`
    :   Translate document (src_text).
        To speed up translations, you can set num_beams and early_stopping (e.g., num_beams=4, early_stopping=True).
        Args:
          src_text(str): source text.
                         The source text can either be a single sentence or an entire document with multiple sentences
                         and paragraphs. 
                         IMPORTANT NOTE: Sentences are joined together and fed to model as single batch.
                                         If the input text is very large (e.g., an entire book), you should
                                         break it up into reasonbly-sized chunks (e.g., pages, paragraphs, or sentences) and 
                                         feed each chunk separately into translate to avoid out-of-memory issues.
          join_with(str):  list of translated sentences will be delimited with this character.
                           default: each sentence on separate line
          num_beams(int): Number of beams for beam search. Defaults to None.  If None, the transformers library defaults this to 1, 
                          whicn means no beam search.
          early_stopping(bool):  Whether to stop the beam search when at least ``num_beams`` sentences 
                                 are finished per batch or not. Defaults to None.  If None, the transformers library
                                 sets this to False.
        Returns:
          str: translated text

    `translate_sentences(self, sentences, num_beams=None, early_stopping=None)`
    :   Translate sentences using model_name as model.
        To speed up translations, you can set num_beams and early_stopping (e.g., num_beams=4, early_stopping=True).
        Args:
          sentences(list): list of strings representing sentences that need to be translated
                         IMPORTANT NOTE: Sentences are joined together and fed to model as single batch.
                                         If the input text is very large (e.g., an entire book), you should
                                         break it up into reasonbly-sized chunks (e.g., pages, paragraphs, or sentences) and 
                                         feed each chunk separately into translate to avoid out-of-memory issues.
          num_beams(int): Number of beams for beam search. Defaults to None.  If None, the transformers library defaults this to 1, 
                          whicn means no beam search.
          early_stopping(bool):  Whether to stop the beam search when at least ``num_beams`` sentences 
                                 are finished per batch or not. Defaults to None.  If None, the transformers library
                                 sets this to False.
        Returns:
          str: translated sentences

`ZeroShotClassifier(model_name='facebook/bart-large-mnli', device=None)`
:   interface to Zero Shot Topic Classifier
    
    ZeroShotClassifier constructor
    
    Args:
      model_name(str): name of a BART NLI model
      device(str): device to use (e.g., 'cuda', 'cpu')

    ### Methods

    `predict(self, docs, labels=[], include_labels=False, multilabel=True, max_length=512, batch_size=8, nli_template='This text is about {}.', topic_strings=[])`
    :   This method performs zero-shot text classification using Natural Language Inference (NLI).
        Args:
          docs(list|str): text of document or list of texts
          labels(list): a list of strings representing topics of your choice
                        Example:
                          labels=['political science', 'sports', 'science']
          include_labels(bool): If True, will return topic labels along with topic probabilities
          multilabel(bool): If True, labels are considered independent and multiple labels can predicted true for document and be close to 1.
                            If False, scores are normalized such that probabilities sum to 1.
          max_length(int): truncate long documents to this many tokens
          batch_size(int): batch_size to use. default:8
                           Increase this value to speed up predictions - especially
                           if len(topic_strings) is large.
          nli_template(str): labels are inserted into this template for use as hypotheses in natural language inference
          topic_strings(list): alias for labels parameter for backwards compatibility
        Returns:
          inferred probabilities or list of inferred probabilities if doc is list