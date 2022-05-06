from .. import utils as U
from ..imports import *
from . import preprocessor as tpp
from . import textutils as TU

MAX_FEATURES = 20000
MAXLEN = 400


def texts_from_folder(
    datadir,
    classes=None,
    max_features=MAX_FEATURES,
    maxlen=MAXLEN,
    ngram_range=1,
    train_test_names=["train", "test"],
    preprocess_mode="standard",
    encoding=None,  # detected automatically
    lang=None,  # detected automatically
    val_pct=0.1,
    random_state=None,
    verbose=1,
):
    """
    ```
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

    ```
    """

    # check train_test_names
    if len(train_test_names) < 1 or len(train_test_names) > 2:
        raise ValueError(
            "train_test_names must have 1 or two elements for train and optionally validation"
        )

    # read in training and test corpora
    train_str = train_test_names[0]
    train_b = load_files(
        os.path.join(datadir, train_str), shuffle=True, categories=classes
    )
    if len(train_test_names) > 1:
        test_str = train_test_names[1]
        test_b = load_files(
            os.path.join(datadir, test_str), shuffle=False, categories=classes
        )
        x_train = train_b.data
        y_train = train_b.target
        x_test = test_b.data
        y_test = test_b.target
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            train_b.data, train_b.target, test_size=val_pct, random_state=random_state
        )

    # decode based on supplied encoding
    if encoding is None:
        encoding = TU.detect_encoding(x_train)
        U.vprint("detected encoding: %s" % (encoding), verbose=verbose)

    try:
        x_train = [x.decode(encoding) for x in x_train]
        x_test = [x.decode(encoding) for x in x_test]
    except:
        U.vprint(
            "Decoding with %s failed 1st attempt - using %s with skips"
            % (encoding, encoding),
            verbose=verbose,
        )
        x_train = TU.decode_by_line(x_train, encoding=encoding, verbose=verbose)
        x_test = TU.decode_by_line(x_test, encoding=encoding, verbose=verbose)

    # detect language
    if lang is None:
        lang = TU.detect_lang(x_train)
    check_unsupported_lang(lang, preprocess_mode)

    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None:
        raise ValueError("unsupported preprocess_mode")
    preproc = preproc_type(
        maxlen,
        max_features,
        class_names=train_b.target_names,
        lang=lang,
        ngram_range=ngram_range,
    )
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test, y_test, verbose=verbose)
    return (trn, val, preproc)


def texts_from_csv(
    train_filepath,
    text_column,
    label_columns=[],
    val_filepath=None,
    max_features=MAX_FEATURES,
    maxlen=MAXLEN,
    val_pct=0.1,
    ngram_range=1,
    preprocess_mode="standard",
    encoding=None,  # auto-detected
    lang=None,  # auto-detected
    sep=",",
    is_regression=False,
    random_state=None,
    verbose=1,
):
    """
    ```
    Loads text data from CSV or TSV file. Class labels are assumed to be
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
    ```
    """
    if encoding is None:
        with open(train_filepath, "rb") as f:
            # encoding = chardet.detect(f.read())['encoding']
            # encoding = 'utf-8' if encoding.lower() in ['ascii', 'utf8', 'utf-8'] else encoding
            encoding = TU.detect_encoding(f.read())
            U.vprint(
                "detected encoding: %s (if wrong, set manually)" % (encoding),
                verbose=verbose,
            )

    train_df = pd.read_csv(train_filepath, encoding=encoding, sep=sep)
    val_df = (
        pd.read_csv(val_filepath, encoding=encoding, sep=sep)
        if val_filepath is not None
        else None
    )
    return texts_from_df(
        train_df,
        text_column,
        label_columns=label_columns,
        val_df=val_df,
        max_features=max_features,
        maxlen=maxlen,
        val_pct=val_pct,
        ngram_range=ngram_range,
        preprocess_mode=preprocess_mode,
        lang=lang,
        is_regression=is_regression,
        random_state=random_state,
        verbose=verbose,
    )


def texts_from_df(
    train_df,
    text_column,
    label_columns=[],
    val_df=None,
    max_features=MAX_FEATURES,
    maxlen=MAXLEN,
    val_pct=0.1,
    ngram_range=1,
    preprocess_mode="standard",
    lang=None,  # auto-detected
    is_regression=False,
    random_state=None,
    verbose=1,
):
    """
    ```
    Loads text data from Pandas dataframe file. Class labels are assumed to be
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
    ```
    """

    # read in train and test data
    train_df = train_df.copy()
    train_df[text_column].fillna("fillna", inplace=True)
    if val_df is not None:
        val_df = val_df.copy()
        val_df[text_column].fillna("fillna", inplace=True)
    else:
        train_df, val_df = train_test_split(
            train_df, test_size=val_pct, random_state=random_state
        )

    # transform labels
    ytransdf = U.YTransformDataFrame(label_columns, is_regression=is_regression)
    t_df = ytransdf.apply_train(train_df)
    v_df = ytransdf.apply_test(val_df)
    class_names = ytransdf.get_classes()
    new_lab_cols = ytransdf.get_label_columns(squeeze=True)
    x_train = t_df[text_column].values
    y_train = t_df[new_lab_cols].values
    x_test = v_df[text_column].values
    y_test = v_df[new_lab_cols].values

    # detect language
    if lang is None:
        lang = TU.detect_lang(x_train)
    check_unsupported_lang(lang, preprocess_mode)

    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None:
        raise ValueError("unsupported preprocess_mode")
    preproc = preproc_type(
        maxlen,
        max_features,
        class_names=class_names,
        lang=lang,
        ngram_range=ngram_range,
    )
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test, y_test, verbose=verbose)
    # QUICKFIX for #314
    preproc.ytransform.le = ytransdf.le
    return (trn, val, preproc)


def texts_from_array(
    x_train,
    y_train,
    x_test=None,
    y_test=None,
    class_names=[],
    max_features=MAX_FEATURES,
    maxlen=MAXLEN,
    val_pct=0.1,
    ngram_range=1,
    preprocess_mode="standard",
    lang=None,  # auto-detected
    random_state=None,
    verbose=1,
):
    """
    ```
    Loads and preprocesses text data from arrays.
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
    ```
    """
    U.check_array(x_train, y=y_train, X_name="x_train", y_name="y_train")

    if x_test is None or y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=val_pct, random_state=random_state
        )
    else:
        U.check_array(x_test, y=y_test, X_name="x_test", y_name="y_test")

    # removed as TextPreprocessor now handles this.
    # if isinstance(y_train[0], str):
    # if not isinstance(y_test[0], str):
    # raise ValueError('y_train contains strings, but y_test does not')
    # encoder = LabelEncoder()
    # encoder.fit(y_train)
    # y_train = encoder.transform(y_train)
    # y_test = encoder.transform(y_test)

    # detect language
    if lang is None:
        lang = TU.detect_lang(x_train)
    check_unsupported_lang(lang, preprocess_mode)

    # return preprocessed the texts
    preproc_type = tpp.TEXT_PREPROCESSORS.get(preprocess_mode, None)
    if None:
        raise ValueError("unsupported preprocess_mode")
    preproc = preproc_type(
        maxlen,
        max_features,
        class_names=class_names,
        lang=lang,
        ngram_range=ngram_range,
    )
    trn = preproc.preprocess_train(x_train, y_train, verbose=verbose)
    val = preproc.preprocess_test(x_test, y_test, verbose=verbose)
    if not preproc.get_classes() and verbose:
        print(
            "task: text regression (supply class_names argument if this is supposed to be classification task)"
        )
    else:
        print("task: text classification")
    return (trn, val, preproc)


def check_unsupported_lang(lang, preprocess_mode):
    """
    ```
    check for unsupported language (e.g., nospace langs not supported by Jieba)
    ```
    """
    unsupported = (
        preprocess_mode == "standard"
        and TU.is_nospace_lang(lang)
        and not TU.is_chinese(lang)
    )
    if unsupported:
        raise ValueError(
            "language %s is currently only supported by the BERT model. " % (lang)
            + 'Please select preprocess_mode="bert"'
        )
