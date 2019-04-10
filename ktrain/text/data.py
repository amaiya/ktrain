from keras.utils import to_categorical
import operator
import numpy as np
import os.path
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
from .. import utils as U
from .preprocessor import TextPreprocessor

MAX_FEATURES = 20000
MAXLEN = 400



def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    # TODO:  There is currently a copy of this function in TextPrepreprocessor
    # These must be merged.
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def process_text( train_data, val_data, 
                  ngram_range=1, max_features=MAX_FEATURES, maxlen=MAXLEN,
                  verbose=1):

    # check args
    U.data_arg_check(train_data=train_data, val_data=val_data, ndarray_only=True,
                   train_required=True, val_required=True)

    # setup data
    train_text = train_data[0]
    test_text = val_data[0]
    y_train = train_data[1]
    y_test = val_data[1]


    # tokenize and convert to sequences of word IDs
    t = Tokenizer(num_words=max_features)
    t.fit_on_texts(train_text)
    U.vprint('Word Counts: {}'.format(len(t.word_counts)), verbose=verbose)
    U.vprint('Nrows: {}'.format(len(train_text)), verbose=verbose)
    x_train = t.texts_to_sequences(train_text)
    x_test = t.texts_to_sequences(test_text)
    U.vprint('{} train sequences'.format(len(x_train)), verbose=verbose)
    U.vprint('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), 
                                                             dtype=int)), verbose=verbose)
    U.vprint('{} test sequences'.format(len(x_test)), verbose=verbose)
    U.vprint('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), 
                                                             dtype=int)), verbose=verbose)
    # optionally add ngrams
    token_indice = {}
    if ngram_range > 1:
        U.vprint('Adding {}-gram features'.format(ngram_range), verbose=verbose)
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        U.vprint('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)), verbose=verbose)
        U.vprint('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)), verbose=verbose)    
    
    # pad sequences
    U.vprint('Pad sequences (samples x time)', verbose=verbose)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    if len(y_train.shape) == 1:
        y_train = to_categorical(y_train)
    if len(y_test.shape) == 1:
        y_test = to_categorical(y_test)

    U.vprint('x_train shape: ({},{})'.format(x_train.shape[0], x_train.shape[1]), verbose=verbose)
    U.vprint('x_test shape: ({},{})'.format(x_test.shape[0], x_test.shape[1]), verbose=verbose)
    U.vprint('y_train shape: ({},{})'.format(y_train.shape[0], y_train.shape[1]), verbose=verbose)
    U.vprint('y_test shape: ({},{})'.format(y_test.shape[0], y_test.shape[1]), verbose=verbose)
    out = tuple([(x_train, y_train), (x_test, y_test), t, token_indice])
    return out


def texts_from_folder(datadir, classes=None, 
                      max_features=MAX_FEATURES, maxlen=MAXLEN,
                      ngram_range=1,
                      train_test_names=['train', 'test'],
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
    """

    # read in training and test corpora
    train_str = train_test_names[0]
    test_str = train_test_names[1]
    train_b = load_files(os.path.join(datadir, train_str), shuffle=True)
    test_b = load_files(os.path.join(datadir,  test_str), shuffle=False)
    x_train = [x.decode('utf-8') for x in train_b.data]
    x_test = [x.decode('utf-8') for x in test_b.data]
    y_train = train_b.target
    y_test = test_b.target

    (trn, val, tok, tok_dct) =  process_text((x_train, y_train), 
                                            (x_test, y_test),
                                            max_features=max_features,
                                            maxlen=maxlen,
                                            ngram_range=ngram_range, verbose=verbose)
    preproc = TextPreprocessor(tok, tok_dct, train_b.target_names, maxlen,
                               ngram_range=ngram_range)
    return (trn, val, preproc)
 


def texts_from_csv(train_filepath, 
                   text_column,
                   label_columns = [],
                   val_filepath=None,
                   max_features=MAX_FEATURES, maxlen=MAXLEN, 
                   val_pct=0.1, ngram_range=1, verbose=1):
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
        verbose (boolean): verbosity
    """

    # read in train and test data
    train = pd.read_csv(train_filepath)

    x = train[text_column].fillna('fillna').values
    y = train[label_columns].values
    if val_filepath is not None:
        test = pd.read_csv(val_filepath)
        x_test = train[text_column].fillna('fillna').values
        y_test = train[label_columns].values
        x_train = x
        y_train = y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val_pct)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    (trn, val, tok, tok_dct) =  process_text((x_train, y_train), 
                                            (x_test, y_test),
                                            max_features=max_features,
                                            maxlen=maxlen,
                                            ngram_range=ngram_range, verbose=verbose)
    preproc = TextPreprocessor(tok, tok_dct, label_columns, maxlen,
                               ngram_range=ngram_range)
    return (trn, val, preproc)

