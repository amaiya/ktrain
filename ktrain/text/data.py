from ..imports import *
from .. import utils as U
from  . import preprocessor as tpp



MAX_FEATURES = 20000
MAXLEN = 400


def texts_from_folder(datadir, classes=None, 
                      max_features=MAX_FEATURES, maxlen=MAXLEN,
                      ngram_range=1,
                      train_test_names=['train', 'test'],
                      preprocess_mode='standard',
                      encoding=None, # detected automatically
                      lang=None, # detected automatically
                      val_pct=0.1, random_state=None,
                      verbose=1):
    """
    Returns corpus as sequence of word IDs.
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

    If train and test contain additional subfolders that do not represent
    classes, they can be ignored by explicitly listing the subfolders of
    interest using the classes argument.
    Args:
        datadir (str): path to folder
        classes (list): list of classes (subfolders to consider)
        max_features (int):  maximum number of unigrams to consider
        maxlen (int):  maximum length of tokens in document
        ngram_range (int):  If > 1, will include 2=bigrams, 3=trigrams and bigrams
        train_test_names (list):  list of strings represnting the subfolder
                                 name for train and validation sets
                                 if test name is missing, <val_pct> of training
                                 will be used for validation
        preprocess_mode (str):  Either 'standard' (normal tokenization) or 'bert'
                                tokenization and preprocessing for use with 
                                BERT text classification model.
        encoding (str):        character encoding to use. Auto-detected if None
        lang (str):            language.  Auto-detected if None.
        val_pct(float):        Onlyl used if train_test_names  has 1 and not 2 names
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                IF None, train/test split will be random
        verbose (bool):         verbosity
        
    """
    # check is the given path is a valid filepath
    if(not os.path.isdir(datadir)):
        raise Exception("Given datadir is not a valid directory path")

    # check train_test_names
    if len(train_test_names) < 1 or len(train_test_names) > 2:
        raise ValueError('train_test_names must have 1 or two elements for train and optionally validation')

    # read in training and test corpora
    train_str = train_test_names[0]
    train_b = load_files(os.path.join(datadir, train_str), shuffle=True, categories=classes)
    if len(train_test_names) > 1:
        test_str = train_test_names[1]
        test_b = load_files(os.path.join(datadir,  test_str), shuffle=False, categories=classes)
        x_train = train_b.data
        y_train = train_b.target
        x_test = test_b.data
        y_test = test_b.target
    else:
        x_train, x_test, y_train, y_test = train_test_split(train_b.data, 
                                                            train_b.target, 
                                                            test_size=val_pct,
                                                            random_state=random_state)

    # decode based on supplied encoding
    if encoding is None:
        # detect encoding from first training example
        lst = [chardet.detect(doc)['encoding'] for doc in x_train[:32]]
        encoding = max(set(lst), key=lst.count)
        encoding = standardize_to_utf8(encoding)
        U.vprint('detected encoding: %s' % (encoding), verbose=verbose)
    
    try:
        x_train = [x.decode(encoding) for x in x_train]
        x_test = [x.decode(encoding) for x in x_test]
    except:
        U.vprint('Decoding with %s failed 1st attempt - using %s with skips' % (encoding, 
                                                                                encoding),
                                                                                verbose=verbose)
        x_train = tpp.decode_by_line(x_train, encoding=encoding, verbose=verbose)
        x_test = tpp.decode_by_line(x_test, encoding=encoding, verbose=verbose)


    # detect language
    if lang is None: lang = tpp.detect_lang(x_train)
    check_unsupported_lang(lang, preprocess_mode)



    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None: raise ValueError('unsupported preprocess_mode')
    preproc = preproc_type(maxlen,
                           max_features,
                           classes = train_b.target_names,
                           lang=lang, ngram_range=ngram_range)
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test,  y_test, verbose=verbose)
    return (trn, val, preproc)





def texts_from_csv(train_filepath, 
                   text_column,
                   label_columns = [],
                   val_filepath=None,
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, preprocess_mode='standard', 
                   encoding=None,  # auto-detected
                   lang=None,      # auto-detected
                   sep=',', random_state=None,       
                   verbose=1):
    """
    Loads text data from CSV file. Class labels are assumed to be
    one-hot-encoded arrays representing classes:
         classification (a single one in each array): [[1,0,0], [0,1,0]]]
         multi-label classification (one more ones in each array): [[1,1,0], [0,1,1]]
    This treats task as classification problem.  
    If data is for a text regression task, use texts_from_array.

    Args:
        train_filepath(str): file path to training CSV
        text_column(str): name of column containing the text
        label_column(list): list of columns that are to be treated as labels
        val_filepath(string): file path to test CSV.  If not supplied,
                               10% of documents in training CSV will be
                               used for testing/validation.
        max_features(int): max num of words to consider in vocabulary
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if val_filepath is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or 'bert'
                                tokenization and preprocessing for use with 
                                BERT text classification model.
        encoding (str):        character encoding to use. Auto-detected if None
        lang (str):            language.  Auto-detected if None.
        sep(str):              delimiter for CSV (comma is default)
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random
        verbose (boolean): verbosity
    """

    # check is the given path is a valid filepath
    if(not os.path.isfile(train_filepath)):
        raise Exception("Given train_filepath is not a valid file path")


    if encoding is None:
        with open(train_filepath, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            encoding = standardize_to_utf8(encoding)
            U.vprint('detected encoding: %s (if wrong, set manually)' % (encoding), verbose=verbose)

    train_df = pd.read_csv(train_filepath, encoding=encoding,sep=sep)
    val_df = pd.read_csv(val_filepath, encoding=encoding,sep=sep) if val_filepath is not None else None
    return texts_from_df(train_df,
                         text_column,
                         label_columns=label_columns,
                         val_df = val_df,
                         max_features=max_features,
                         maxlen=maxlen,
                         val_pct=val_pct,
                         ngram_range=ngram_range, 
                         preprocess_mode=preprocess_mode,
                         lang=lang, random_state=random_state,
                         verbose=verbose)





def texts_from_df(train_df, 
                   text_column,
                   label_columns = [],
                   val_df=None,
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, preprocess_mode='standard', 
                   lang=None, # auto-detected
                   random_state=None,
                   verbose=1):
    """
    Loads text data from Pandas dataframe file. Class labels are assumed to be
    one-hot-encoded arrays representing classes:
         classification (a single one in each array): [[1,0,0], [0,1,0]]]
         multi-label classification (one more ones in each array): [[1,1,0], [0,1,1]]
    This treats task as classification problem.  
    If this is a text regression task, use texts_from_array.
    Args:
        train_df(dataframe): Pandas dataframe
        text_column(str): name of column containing the text
        label_column(list): list of columns that are to be treated as labels
        val_df(dataframe): file path to test dataframe.  If not supplied,
                               10% of documents in training df will be
                               used for testing/validation.
        max_features(int): max num of words to consider in vocabulary
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if val_filepath is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or 'bert'
                                tokenization and preprocessing for use with 
                                BERT text classification model.
        lang (str):            language.  Auto-detected if None.
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random
        verbose (boolean): verbosity
    """

    #checks if the dtpye of input data is appropriate

    if(type(train_df) is not pd.DataFrame()):
        dtype_df =  type(train_df)
        raise Exception(f"Data type of train_df is {dtype}, make sure it is pandas DataFrame")


    # read in train and test data
    train = train_df

    x = train[text_column].fillna('fillna').values
    y = train[label_columns].values
    if val_df is not None:
        test = val_df
        x_test = test[text_column].fillna('fillna').values
        y_test = test[label_columns].values
        x_train = x
        y_train = y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size=val_pct,
                                                            random_state=random_state)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    # detect language
    if lang is None: lang = tpp.detect_lang(x_train)
    check_unsupported_lang(lang, preprocess_mode)


    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None: raise ValueError('unsupported preprocess_mode')
    preproc = preproc_type(maxlen,
                           max_features,
                           classes = label_columns,
                           lang=lang, ngram_range=ngram_range)
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test,  y_test, verbose=verbose)
    return (trn, val, preproc)



def texts_from_array(x_train, y_train, x_test=None, y_test=None, 
                   class_names = [],
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, preprocess_mode='standard', 
                   lang=None, # auto-detected
                   random_state=None,
                   verbose=1):
    """
    Loads and preprocesses text data from arrays.
    texts_from_array can handle data for both text classification
    and text regression.  If class_names is empty, a regression task is assumed.
    Args:
        x_train(list): list of training texts 
        y_train(list): labels in one of the following forms:
                       1. list of integers representing classes (class_names is required)
                       2. list of strings representing classes (class_names is required)
                       3. a one or multi hot encoded array representing classes (class_names is required)
                       4. numerical values for text regresssion (class_names should be left empty)
        x_val(list): list of training texts 
        y_val(list): labels in one of the following forms:
                      1. list of integers representing classes (class_names is required)
                      2. list of strings representing classes (class_names is required)
                      3. a one or multi hot encoded array representing classes (class_names is required)
                      4. numerical values for text regression (class_names should be left empty)
        class_names (list): list of strings representing class labels
                            shape should be (num_examples,1) or (num_examples,)
        max_features(int): max num of words to consider in vocabulary
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if x_val and  y_val is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or 'bert'
                                tokenization and preprocessing for use with 
                                BERT text classification model.
        lang (str):            language.  Auto-detected if None.
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random.
        verbose (boolean): verbosity
    """

    #checks if the dtpye of input data is appropriate
    if(not (type(x_train)==list )):
        try:
            x_train = list(x_train)
            y_train = list(y_train)
            if(x_test):
                x_test=list(x_test)
                y_test = list(y_test)
        except:
            dtype = str(type(x_train))
            raise Exception(f"Data type of the input data is {dtype}, make sure it is list")

    if not class_names and verbose:
        #classes =  list(set(y_train))
        #classes.sort()
        #class_names = ["%s" % (c) for c in classes]
        print('task: text regression (supply class_names argument if this is supposed to be classification task)')
    else:
        print('task: text classification')

    if x_test is None or y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
                                                            test_size=val_pct,
                                                            random_state=random_state)

    # convert string labels to integers, if necessary
    if isinstance(y_train[0], str):
        if not isinstance(y_test[0], str): 
            raise ValueError('y_train contains strings, but y_test does not')
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)


    # detect language
    if lang is None: lang = tpp.detect_lang(x_train)
    check_unsupported_lang(lang, preprocess_mode)

    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None: raise ValueError('unsupported preprocess_mode')
    preproc = preproc_type(maxlen,
                           max_features,
                           classes = class_names,
                           lang=lang, ngram_range=ngram_range)
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test,  y_test, verbose=verbose)
    return (trn, val, preproc)



def standardize_to_utf8(encoding):
    """
    standardize to utf-8 if necessary.
    NOTE: mainly used to use utf-8 if ASCII is detected, as
    BERT performance suffers otherwise.
    """
    encoding = 'utf-8' if encoding.lower() in ['ascii', 'utf8', 'utf-8'] else encoding
    return encoding



def check_unsupported_lang(lang, preprocess_mode):
    """
    check for unsupported language (e.g., nospace langs no supported by Jieba)
    """
    unsupported = preprocess_mode=='standard' and tpp.is_nospace_lang(lang) and not tpp.is_chinese(lang)
    if unsupported:
        raise ValueError('language %s is currently only supported by the BERT model. ' % (lang) +
                         'Please select preprocess_mode="bert"')

