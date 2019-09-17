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
        preprocess_mode (str):  Either 'standard' (normal tokenization) or 'bert'
                                tokenization and preprocessing for use with 
                                BERT text classification model.
        verbose (bool):         verbosity
        
    """

    # read in training and test corpora
    train_str = train_test_names[0]
    test_str = train_test_names[1]
    train_b = load_files(os.path.join(datadir, train_str), shuffle=True, categories=classes)
    test_b = load_files(os.path.join(datadir,  test_str), shuffle=False, categories=classes)
    x_train = [x.decode('utf-8') for x in train_b.data]
    x_test = [x.decode('utf-8') for x in test_b.data]
    y_train = train_b.target
    y_test = test_b.target


    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None: raise ValueError('unsupported preprocess_mode')
    preproc = preproc_type(maxlen,
                           max_features,
                           classes = train_b.target_names,
                           ngram_range=ngram_range)
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test,  y_test, verbose=verbose)
    return (trn, val, preproc)



def texts_from_csv(train_filepath, 
                   text_column,
                   label_columns = [],
                   val_filepath=None,
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, preprocess_mode='standard', verbose=1):
    """
    Loads text data from CSV file. Class labels are assumed to one of following:
      1. integers representing classes (e.g., 1,2,3,4)
      2. one-hot-encoded arrays representing classes
         classification (a single one in each array): [[1,0,0], [0,1,0]]]
         multi-label classification (one more ones in each array): [[1,1,0], [0,1,1]]
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
        verbose (boolean): verbosity
    """
    train_df = pd.read_csv(train_filepath)
    val_df = pd.read_csv(val_filepath) if val_filepath is not None else None
    return texts_from_df(train_df,
                         text_column,
                         label_columns=label_columns,
                         val_df = val_df,
                         max_features=max_features,
                         maxlen=maxlen,
                         val_pct=val_pct,
                         ngram_range=ngram_range, 
                         preprocess_mode=preprocess_mode,
                         verbose=verbose)





def texts_from_df(train_df, 
                   text_column,
                   label_columns = [],
                   val_df=None,
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, preprocess_mode='standard', verbose=1):
    """
    Loads text data from Pandas dataframe file. Class labels are assumed to one of following:
      1. integers representing classes (e.g., 1,2,3,4)
      2. one-hot-encoded arrays representing classes
         classification (a single one in each array): [[1,0,0], [0,1,0]]]
         multi-label classification (one more ones in each array): [[1,1,0], [0,1,1]]
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
        verbose (boolean): verbosity
    """

    # read in train and test data
    train = train_df

    x = train[text_column].fillna('fillna').values
    y = train[label_columns].values
    if val_df is not None:
        test = val_df
        x_test = train[text_column].fillna('fillna').values
        y_test = train[label_columns].values
        x_train = x
        y_train = y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val_pct)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None: raise ValueError('unsupported preprocess_mode')
    preproc = preproc_type(maxlen,
                           max_features,
                           classes = label_columns,
                           ngram_range=ngram_range)
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test,  y_test, verbose=verbose)
    return (trn, val, preproc)



def texts_from_array(x_train, y_train, x_test=None, y_test=None, 
                   class_names = [],
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, preprocess_mode='standard', verbose=1):
    """
    Loads and preprocesses text data from arrays.
    Args:
        x_train(list): list of training texts 
        y_train(list): list of integers representing classes
        x_val(list): list of training texts 
        y_val(list): list of integers representing classes
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
        verbose (boolean): verbosity
    """

    if not class_names:
        classes =  list(set(y_train))
        classes.sort()
        class_names = ["%s" % (c) for c in classes]
    if x_test is None or y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=val_pct)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None: raise ValueError('unsupported preprocess_mode')
    preproc = preproc_type(maxlen,
                           max_features,
                           classes = class_names,
                           ngram_range=ngram_range)
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test,  y_test, verbose=verbose)
    return (trn, val, preproc)



