from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModel,
    TFAutoModelForSequenceClassification,
)

from .. import utils as U
from ..imports import *
from ..preprocessor import Preprocessor
from . import textutils as TU

DISTILBERT = "distilbert"

NOSPACE_LANGS = ["zh-cn", "zh-tw", "ja"]


def is_nospace_lang(lang):
    return lang in NOSPACE_LANGS


def fname_from_url(url):
    return os.path.split(url)[-1]


# ------------------------------------------------------------------------------
# Word Vectors
# ------------------------------------------------------------------------------
WV_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
# WV_URL = 'http://nlp.stanford.edu/data/glove.6B.zip


def get_wv_path(wv_path_or_url=WV_URL):
    # process if file path given
    if os.path.isfile(wv_path_or_url) and wv_path_or_url.endswith("vec"):
        return wv_path_or_url
    elif os.path.isfile(wv_path_or_url):
        raise ValueError(
            "wv_path_or_url must either be URL .vec.zip or .vec.gz file or file path to .vec file"
        )

    # process if URL is given
    fasttext_url = "https://dl.fbaipublicfiles.com/fasttext"
    if not wv_path_or_url.startswith(fasttext_url):
        raise ValueError("selected word vector file must be from %s" % (fasttext_url))
    if not wv_path_or_url.endswith(".vec.zip") and not wv_path_or_url.endswith(
        "vec.gz"
    ):
        raise ValueError(
            "If wv_path_or_url is URL, must be .vec.zip filea from Facebook fasttext site."
        )

    ktrain_data = U.get_ktrain_data()
    zip_fpath = os.path.join(ktrain_data, fname_from_url(wv_path_or_url))
    wv_path = os.path.join(
        ktrain_data, os.path.splitext(fname_from_url(wv_path_or_url))[0]
    )
    if not os.path.isfile(wv_path):
        # download zip
        print("downloading pretrained word vectors to %s ..." % (ktrain_data))
        U.download(wv_path_or_url, zip_fpath)

        # unzip
        print("\nextracting pretrained word vectors...")
        if wv_path_or_url.endswith(".vec.zip"):
            with zipfile.ZipFile(zip_fpath, "r") as zip_ref:
                zip_ref.extractall(ktrain_data)
        else:  # .vec.gz
            with gzip.open(zip_fpath, "rb") as f_in:
                with open(wv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        print("done.\n")

        # cleanup
        print("cleanup downloaded zip...")
        try:
            os.remove(zip_fpath)
            print("done.\n")
        except OSError:
            print("failed to cleanup/remove %s" % (zip_fpath))
    return wv_path


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype="float32")


# def load_wv(wv_path=None, verbose=1):
# if verbose: print('Loading pretrained word vectors...this may take a few moments...')
# if wv_path is None: wv_path = get_wv_path()
# embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(wv_path, encoding='utf-8'))
# if verbose: print('Done.')
# return embeddings_index


def file_len(fname):
    with open(fname, encoding="utf-8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def load_wv(wv_path_or_url=WV_URL, verbose=1):
    wv_path = get_wv_path(wv_path_or_url)
    if verbose:
        print("loading pretrained word vectors...this may take a few moments...")
    length = file_len(wv_path)
    tups = []
    mb = master_bar(range(1))
    for i in mb:
        f = open(wv_path, encoding="utf-8")
        for o in progress_bar(range(length), parent=mb):
            o = f.readline()
            tups.append(get_coefs(*o.rstrip().rsplit(" ")))
        f.close()
        # if verbose: mb.write('done.')
    return dict(tups)


# ------------------------------------------------------------------------------
# BERT
# ------------------------------------------------------------------------------

# BERT_PATH = os.path.join(os.path.dirname(os.path.abspath(localbert.__file__)), 'uncased_L-12_H-768_A-12')
BERT_URL = (
    "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
)
BERT_URL_MULTI = "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip"
BERT_URL_CN = (
    "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"
)


def get_bert_path(lang="en"):
    if lang == "en":
        bert_url = BERT_URL
    elif lang.startswith("zh-"):
        bert_url = BERT_URL_CN
    else:
        bert_url = BERT_URL_MULTI
    ktrain_data = U.get_ktrain_data()
    zip_fpath = os.path.join(ktrain_data, fname_from_url(bert_url))
    bert_path = os.path.join(ktrain_data, os.path.splitext(fname_from_url(bert_url))[0])
    if (
        not os.path.isdir(bert_path)
        or not os.path.isfile(os.path.join(bert_path, "bert_config.json"))
        or not os.path.isfile(
            os.path.join(bert_path, "bert_model.ckpt.data-00000-of-00001")
        )
        or not os.path.isfile(os.path.join(bert_path, "bert_model.ckpt.index"))
        or not os.path.isfile(os.path.join(bert_path, "bert_model.ckpt.meta"))
        or not os.path.isfile(os.path.join(bert_path, "vocab.txt"))
    ):
        # download zip
        print("downloading pretrained BERT model (%s)..." % (fname_from_url(bert_url)))
        U.download(bert_url, zip_fpath)

        # unzip
        print("\nextracting pretrained BERT model...")
        with zipfile.ZipFile(zip_fpath, "r") as zip_ref:
            zip_ref.extractall(ktrain_data)
        print("done.\n")

        # cleanup
        print("cleanup downloaded zip...")
        try:
            os.remove(zip_fpath)
            print("done.\n")
        except OSError:
            print("failed to cleanup/remove %s" % (zip_fpath))
    return bert_path


def bert_tokenize(docs, tokenizer, max_length, verbose=1):
    if verbose:
        mb = master_bar(range(1))
        pb = progress_bar(docs, parent=mb)
    else:
        mb = range(1)
        pb = docs

    indices = []
    for i in mb:
        for doc in pb:
            # https://stackoverflow.com/questions/67360987/bert-model-bug-encountered-during-training/67375675#67375675
            doc = str(doc) if isinstance(doc, (float, int)) else doc
            ids, segments = tokenizer.encode(doc, max_len=max_length)
            indices.append(ids)
        if verbose:
            mb.write("done.")
    zeros = np.zeros_like(indices)
    return [np.array(indices), np.array(zeros)]


# ------------------------------------------------------------------------------
# Transformers UTILITIES
# ------------------------------------------------------------------------------

# def convert_to_tfdataset(csv):
# def gen():
# for ex in csv:
# yield  {'idx': ex[0],
#'sentence': ex[1],
#'label': str(ex[2])}
# return tf.data.Dataset.from_generator(gen,
# {'idx': tf.int64,
#'sentence': tf.string,
#'label': tf.int64})


# def features_to_tfdataset(features):

#    def gen():
#        for ex in features:
#            yield ({'input_ids': ex.input_ids,
#                     'attention_mask': ex.attention_mask,
#                     'token_type_ids': ex.token_type_ids},
#                    ex.label)

#    return tf.data.Dataset.from_generator(gen,
#        ({'input_ids': tf.int32,
#          'attention_mask': tf.int32,
#          'token_type_ids': tf.int32},
#         tf.int64),
#        ({'input_ids': tf.TensorShape([None]),
#          'attention_mask': tf.TensorShape([None]),
#          'token_type_ids': tf.TensorShape([None])},
#         tf.TensorShape([None])))
#         #tf.TensorShape(])))


def _is_sentence_pair(tup):
    if (
        isinstance(tup, (tuple))
        and len(tup) == 2
        and isinstance(tup[0], str)
        and isinstance(tup[1], str)
    ):
        return True
    else:
        # if (
        # isinstance(tup, (list, np.ndarray))
        # and len(tup) == 2
        # and isinstance(tup[0], str)
        # and isinstance(tup[1], str)
        # ):
        # warnings.warn(
        # "List or array of two texts supplied, so task being treated as text classification. "
        # + "If this is a sentence pair classification task, please cast to tuple."
        # )
        return False


def detect_text_format(texts):
    is_pair = False
    is_array = False
    err_msg = "invalid text format: texts should be list of strings or list of sentence pairs in form of tuples (str, str)"
    if _is_sentence_pair(texts):
        is_pair = True
        is_array = False
    elif isinstance(texts, (tuple, list, np.ndarray)):
        is_array = True
        if len(texts) == 0:
            raise ValueError("texts is empty")
        peek = texts[0]
        is_pair = _is_sentence_pair(peek)
        if not is_pair and not isinstance(peek, str):
            raise ValueError(err_msg)
    return is_array, is_pair


def hf_features_to_tfdataset(features_list, labels):
    features_list = np.array(features_list)
    labels = np.array(labels) if labels is not None else None
    tfdataset = tf.data.Dataset.from_tensor_slices((features_list, labels))
    tfdataset = tfdataset.map(
        lambda x, y: (
            {"input_ids": x[0], "attention_mask": x[1], "token_type_ids": x[2]},
            y,
        )
    )

    return tfdataset


def hf_convert_example(
    text_a,
    text_b=None,
    tokenizer=None,
    max_length=512,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    ```
    convert InputExample to InputFeature for Hugging Face transformer
    ```
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        return_token_type_ids=True,
        max_length=max_length,
        truncation="longest_first",
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
        len(input_ids), max_length
    )
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
    )

    # if ex_index < 1:
    # print("*** Example ***")
    # print("guid: %s" % (example.guid))
    # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
    # print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
    # print("label: %s (id = %d)" % (example.label, label))

    return [input_ids, attention_mask, token_type_ids]


# ------------------------------------------------------------------------------


class TextPreprocessor(Preprocessor):
    """
    ```
    Text preprocessing base class
    ```
    """

    def __init__(self, maxlen, class_names, lang="en", multilabel=None):
        self.set_classes(class_names)  # converts to list of necessary
        self.maxlen = maxlen
        self.lang = lang
        self.multilabel = multilabel  # currently, this is always initially set None until set by set_multilabel
        self.preprocess_train_called = False
        # self.label_encoder = None # only set if y is in string format
        self.ytransform = None
        self.c = self.c.tolist() if isinstance(self.c, np.ndarray) else self.c

    def migrate_classes(self, class_names, classes):
        # NOTE: this method transforms to np.ndarray to list.
        # If removed and "if class_names" is issued prior to set_classes(), an error will occur.
        class_names = (
            class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
        )
        classes = classes.tolist() if isinstance(classes, np.ndarray) else classes

        if not class_names and classes:
            class_names = classes
            warnings.warn(
                "The class_names argument is replacing the classes argument. Please update your code."
            )
        return class_names

    def get_tokenizer(self):
        raise NotImplementedError("This method was not overridden in subclass")

    def check_trained(self):
        if not self.preprocess_train_called:
            warnings.warn(
                "The method preprocess_train was never called. You can disable this warning by setting preprocess_train_called=True."
            )
            # raise Exception('preprocess_train must be called')

    def get_preprocessor(self):
        raise NotImplementedError

    def get_classes(self):
        return self.c

    def set_classes(self, class_names):
        self.c = (
            class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
        )

    def preprocess(self, texts):
        raise NotImplementedError

    def set_multilabel(self, data, mode, verbose=1):
        if mode == "train" and self.get_classes():
            original_multilabel = self.multilabel
            discovered_multilabel = U.is_multilabel(data)
            if original_multilabel is None:
                self.multilabel = discovered_multilabel
            elif original_multilabel is True and discovered_multilabel is False:
                warnings.warn(
                    "The multilabel=True argument was supplied, but labels do not indicate "
                    + "a multilabel problem (labels appear to be mutually-exclusive).  Using multilabel=True anyways."
                )
            elif original_multilabel is False and discovered_multilabel is True:
                warnings.warn(
                    "The multilabel=False argument was supplied, but labels inidcate that  "
                    + "this is a multilabel problem (labels are not mutually-exclusive).  Using multilabel=False anyways."
                )
            U.vprint("Is Multi-Label? %s" % (self.multilabel), verbose=verbose)

    def undo(self, doc):
        """
        ```
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        ```
        """
        raise NotImplementedError

    def is_chinese(self):
        return TU.is_chinese(self.lang)

    def is_nospace_lang(self):
        return TU.is_nospace_lang(self.lang)

    def process_chinese(self, texts, lang=None):
        # if lang is None: lang = langdetect.detect(texts[0])
        if lang is None:
            lang = TU.detect_lang(texts)
        if not TU.is_nospace_lang(lang):
            return texts
        return TU.split_chinese(texts)

    @classmethod
    def seqlen_stats(cls, list_of_texts):
        """
        ```
        compute sequence length stats from
        list of texts in any spaces-segmented language
        Args:
            list_of_texts: list of strings
        Returns:
            dict: dictionary with keys: mean, 95percentile, 99percentile
        ```
        """
        counts = []
        for text in list_of_texts:
            if isinstance(text, (list, np.ndarray)):
                lst = text
            else:
                lst = text.split()
            counts.append(len(lst))
        p95 = np.percentile(counts, 95)
        p99 = np.percentile(counts, 99)
        avg = sum(counts) / len(counts)
        return {"mean": avg, "95percentile": p95, "99percentile": p99}

    def print_seqlen_stats(self, texts, mode, verbose=1):
        """
        ```
        prints stats about sequence lengths
        ```
        """
        if verbose and not self.is_nospace_lang():
            stat_dict = TextPreprocessor.seqlen_stats(texts)
            print("%s sequence lengths:" % mode)
            for k in stat_dict:
                print("\t%s : %s" % (k, int(round(stat_dict[k]))))

    def _transform_y(self, y_data, train=False, verbose=1):
        """
        ```
        preprocess y
        If shape of y is 1, then task is considered classification if self.c exists
        or regression if not.
        ```
        """
        if self.ytransform is None:
            self.ytransform = U.YTransform(class_names=self.get_classes())
        y = self.ytransform.apply(y_data, train=train)
        if train:
            self.c = self.ytransform.get_classes()
        return y


class StandardTextPreprocessor(TextPreprocessor):
    """
    ```
    Standard text preprocessing
    ```
    """

    def __init__(
        self,
        maxlen,
        max_features,
        class_names=[],
        classes=[],
        lang="en",
        ngram_range=1,
        multilabel=None,
    ):
        class_names = self.migrate_classes(class_names, classes)
        super().__init__(maxlen, class_names, lang=lang, multilabel=multilabel)
        self.tok = None
        self.tok_dct = {}
        self.max_features = max_features
        self.ngram_range = ngram_range

    def get_tokenizer(self):
        return self.tok

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        """
        ```
        For backwards compatibility with pre-ytransform versions
        ```
        """
        self.__dict__.update(state)
        if not hasattr(self, "ytransform"):
            le = self.label_encoder if hasattr(self, "label_encoder") else None
            self.ytransform = U.YTransform(
                class_names=self.get_classes(), label_encoder=le
            )

    def get_preprocessor(self):
        return (self.tok, self.tok_dct)

    def preprocess(self, texts):
        return self.preprocess_test(texts, verbose=0)[0]

    def undo(self, doc):
        """
        ```
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        ```
        """
        dct = self.tok.index_word
        return " ".join([dct[wid] for wid in doc if wid != 0 and wid in dct])

    def preprocess_train(self, train_text, y_train, verbose=1):
        """
        ```
        preprocess training set
        ```
        """
        if self.lang is None:
            self.lang = TU.detect_lang(train_text)

        U.vprint("language: %s" % (self.lang), verbose=verbose)

        # special processing if Chinese
        train_text = self.process_chinese(train_text, lang=self.lang)

        # extract vocabulary
        self.tok = keras.preprocessing.text.Tokenizer(num_words=self.max_features)
        self.tok.fit_on_texts(train_text)
        U.vprint("Word Counts: {}".format(len(self.tok.word_counts)), verbose=verbose)
        U.vprint("Nrows: {}".format(len(train_text)), verbose=verbose)

        # convert to word IDs
        x_train = self.tok.texts_to_sequences(train_text)
        U.vprint("{} train sequences".format(len(x_train)), verbose=verbose)
        self.print_seqlen_stats(x_train, "train", verbose=verbose)

        # add ngrams
        x_train = self._fit_ngrams(x_train, verbose=verbose)

        # pad sequences
        x_train = keras.preprocessing.sequence.pad_sequences(
            x_train, maxlen=self.maxlen
        )
        U.vprint(
            "x_train shape: ({},{})".format(x_train.shape[0], x_train.shape[1]),
            verbose=verbose,
        )

        # transform y
        y_train = self._transform_y(y_train, train=True, verbose=verbose)
        if y_train is not None and verbose:
            print("y_train shape: %s" % (y_train.shape,))

        # return
        result = (x_train, y_train)
        self.set_multilabel(result, "train")
        self.preprocess_train_called = True
        return result

    def preprocess_test(self, test_text, y_test=None, verbose=1):
        """
        ```
        preprocess validation or test dataset
        ```
        """
        self.check_trained()
        if self.tok is None or self.lang is None:
            raise Exception(
                "Unfitted tokenizer or missing language. Did you run preprocess_train first?"
            )

        # check for and process chinese
        test_text = self.process_chinese(test_text, self.lang)

        # convert to word IDs
        x_test = self.tok.texts_to_sequences(test_text)
        U.vprint("{} test sequences".format(len(x_test)), verbose=verbose)
        self.print_seqlen_stats(x_test, "test", verbose=verbose)

        # add n-grams
        x_test = self._add_ngrams(x_test, mode="test", verbose=verbose)

        # pad sequences
        x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.maxlen)
        U.vprint(
            "x_test shape: ({},{})".format(x_test.shape[0], x_test.shape[1]),
            verbose=verbose,
        )

        # transform y
        y_test = self._transform_y(y_test, train=False, verbose=verbose)
        if y_test is not None and verbose:
            print("y_test shape: %s" % (y_test.shape,))

        # return
        return (x_test, y_test)

    def _fit_ngrams(self, x_train, verbose=1):
        self.tok_dct = {}
        if self.ngram_range < 2:
            return x_train
        U.vprint("Adding {}-gram features".format(self.ngram_range), verbose=verbose)
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, self.ngram_range + 1):
                set_of_ngram = self._create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = self.max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        self.tok_dct = token_indice

        # max_features is the highest integer that could be found in the dataset.
        self.max_features = np.max(list(indice_token.keys())) + 1
        U.vprint(
            "max_features changed to %s with addition of ngrams" % (self.max_features),
            verbose=verbose,
        )

        # Augmenting x_train with n-grams features
        x_train = self._add_ngrams(x_train, verbose=verbose, mode="train")
        return x_train

    def _add_ngrams(self, sequences, verbose=1, mode="test"):
        """
        ```
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        ```
        """
        token_indice = self.tok_dct
        if self.ngram_range < 2:
            return sequences
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, self.ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i : i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)
        U.vprint(
            "Average {} sequence length with ngrams: {}".format(
                mode, np.mean(list(map(len, new_sequences)), dtype=int)
            ),
            verbose=verbose,
        )
        self.print_seqlen_stats(new_sequences, "%s (w/ngrams)" % mode, verbose=verbose)
        return new_sequences

    def _create_ngram_set(self, input_list, ngram_value=2):
        """
        ```
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]

        ```
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def ngram_count(self):
        if not self.tok_dct:
            return 1
        s = set()
        for k in self.tok_dct.keys():
            s.add(len(k))
        return max(list(s))


class BERTPreprocessor(TextPreprocessor):
    """
    ```
    text preprocessing for BERT model
    ```
    """

    def __init__(
        self,
        maxlen,
        max_features,
        class_names=[],
        classes=[],
        lang="en",
        ngram_range=1,
        multilabel=None,
    ):
        class_names = self.migrate_classes(class_names, classes)

        if maxlen > 512:
            raise ValueError("BERT only supports maxlen <= 512")

        super().__init__(maxlen, class_names, lang=lang, multilabel=multilabel)
        vocab_path = os.path.join(get_bert_path(lang=lang), "vocab.txt")
        token_dict = {}
        with codecs.open(vocab_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        check_keras_bert()
        tokenizer = BERT_Tokenizer(token_dict)
        self.tok = tokenizer
        self.tok_dct = dict((v, k) for k, v in token_dict.items())
        self.max_features = max_features  # ignored
        self.ngram_range = 1  # ignored

    def get_tokenizer(self):
        return self.tok

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        """
        ```
        For backwards compatibility with pre-ytransform versions
        ```
        """
        self.__dict__.update(state)
        if not hasattr(self, "ytransform"):
            le = self.label_encoder if hasattr(self, "label_encoder") else None
            self.ytransform = U.YTransform(
                class_names=self.get_classes(), label_encoder=le
            )

    def get_preprocessor(self):
        return (self.tok, self.tok_dct)

    def preprocess(self, texts):
        return self.preprocess_test(texts, verbose=0)[0]

    def undo(self, doc):
        """
        ```
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        ```
        """
        dct = self.tok_dct
        return " ".join([dct[wid] for wid in doc if wid != 0 and wid in dct])

    def preprocess_train(self, texts, y=None, mode="train", verbose=1):
        """
        ```
        preprocess training set
        ```
        """
        if mode == "train" and y is None:
            raise ValueError("y is required when mode=train")
        if self.lang is None and mode == "train":
            self.lang = TU.detect_lang(texts)
        U.vprint("preprocessing %s..." % (mode), verbose=verbose)
        U.vprint("language: %s" % (self.lang), verbose=verbose)

        x = bert_tokenize(texts, self.tok, self.maxlen, verbose=verbose)

        # transform y
        y = self._transform_y(y, train=mode == "train", verbose=verbose)
        result = (x, y)
        self.set_multilabel(result, mode)
        if mode == "train":
            self.preprocess_train_called = True
        return result

    def preprocess_test(self, texts, y=None, mode="test", verbose=1):
        self.check_trained()
        return self.preprocess_train(texts, y=y, mode=mode, verbose=verbose)


class TransformersPreprocessor(TextPreprocessor):
    """
    ```
    text preprocessing for Hugging Face Transformer models
    ```
    """

    def __init__(
        self,
        model_name,
        maxlen,
        max_features,
        class_names=[],
        classes=[],
        lang="en",
        ngram_range=1,
        multilabel=None,
    ):
        class_names = self.migrate_classes(class_names, classes)

        if maxlen > 512:
            warnings.warn(
                "Transformer models typically only support maxlen <= 512, unless you are using certain models like the Longformer."
            )

        super().__init__(maxlen, class_names, lang=lang, multilabel=multilabel)

        self.model_name = model_name
        self.name = model_name.split("-")[0]
        if model_name.startswith("xlm-roberta"):
            self.name = "xlm_roberta"
            self.model_name = "jplu/tf-" + self.model_name
        else:
            self.name = model_name.split("-")[0]
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_type = TFAutoModelForSequenceClassification
        self.tokenizer_type = AutoTokenizer

        # DistilBert call method no longer accepts **kwargs, so we must avoid including token_type_ids parameter
        # reference: https://github.com/huggingface/transformers/issues/2702
        try:
            self.use_token_type_ids = (
                "token_type_ids"
                in self.model_type.from_pretrained(
                    self.model_name
                ).call.__code__.co_varnames
            )
        except:
            try:
                self.use_token_type_ids = (
                    "token_type_ids"
                    in self.model_type.from_pretrained(
                        self.model_name, from_pt=True
                    ).call.__code__.co_varnames
                )
            except:
                # load model as normal to expose error to user
                self.use_token_type_ids = (
                    "token_type_ids"
                    in self.model_type.from_pretrained(
                        self.model_name
                    ).call.__code__.co_varnames
                )

        if "bert-base-japanese" in model_name:
            self.tokenizer_type = transformers.BertJapaneseTokenizer

        # NOTE: As of v0.16.1, do not unnecessarily instantiate tokenizer
        # as it will be saved/pickled along with Preprocessor, which causes
        # problems for some community-uploaded models like bert-base-japanse-whole-word-masking.
        # tokenizer = self.tokenizer_type.from_pretrained(model_name)
        # self.tok = tokenizer
        self.tok = None  # not pickled,  see __getstate__

        self.tok_dct = None
        self.max_features = max_features  # ignored
        self.ngram_range = 1  # ignored

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ["tok"]}

    def __setstate__(self, state):
        """
        ```
        For backwards compatibility with previous versions of ktrain
        that saved tokenizer and did not use ytransform
        ```
        """
        self.__dict__.update(state)
        if not hasattr(self, "tok"):
            self.tok = None
        if not hasattr(self, "ytransform"):
            le = self.label_encoder if hasattr(self, "label_encoder") else None
            self.ytransform = U.YTransform(
                class_names=self.get_classes(), label_encoder=le
            )
        if not hasattr(self, "use_token_type_ids"):
            # As a shortcut, we simply set use_token_type_ids to False if model is distilbert,
            # as most models use token_type_ids (e.g., bert, deberta, etc.) in their call method
            self.use_token_type_ids = self.name != "distilbert"

    def set_config(self, config):
        self.config = config

    def get_config(self):
        return self.config

    def set_tokenizer(self, tokenizer):
        self.tok = tokenizer

    def get_tokenizer(self, fpath=None):
        model_name = self.model_name if fpath is None else fpath
        if self.tok is None:
            try:
                # use fast tokenizer if possible
                if self.name == "bert" and "japanese" not in model_name:
                    from transformers import BertTokenizerFast

                    self.tok = BertTokenizerFast.from_pretrained(model_name)
                elif self.name == "distilbert":
                    from transformers import DistilBertTokenizerFast

                    self.tok = DistilBertTokenizerFast.from_pretrained(model_name)
                elif self.name == "roberta":
                    from transformers import RobertaTokenizerFast

                    self.tok = RobertaTokenizerFast.from_pretrained(model_name)
                else:
                    self.tok = self.tokenizer_type.from_pretrained(model_name)
            except:
                error_msg = (
                    f"Could not load tokenizer from model_name: {model_name}. "
                    + f"If {model_name} is a local path, please make sure it exists and contains tokenizer files from Hugging Face. "
                    + f"You can also reset model_name with preproc.model_name = '/your/new/path'."
                )
                raise ValueError(error_msg)
        return self.tok

    def save_tokenizer(self, fpath):
        if os.path.isfile(fpath):
            raise ValueError(
                f"There is an existing file named {fpath}. "
                + "Please use dfferent value for fpath."
            )
        elif os.path.exists(fpath):
            pass
        elif not os.path.exists(fpath):
            os.makedirs(fpath)
        tok = self.get_tokenizer()
        tok.save_pretrained(fpath)
        return

    def get_preprocessor(self):
        return (self.get_tokenizer(), self.tok_dct)

    def preprocess(self, texts):
        tseq = self.preprocess_test(texts, verbose=0)
        return tseq.to_tfdataset(train=False)

    def undo(self, doc):
        """
        ```
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        ```
        """
        tok, _ = self.get_preprocessor()
        return self.tok.convert_ids_to_tokens(doc)
        # raise Exception('currently_unsupported: Transformers.Preprocessor.undo is not yet supported')

    def preprocess_train(self, texts, y=None, mode="train", verbose=1):
        """
        ```
        preprocess training set
        ```
        """

        U.vprint("preprocessing %s..." % (mode), verbose=verbose)
        U.check_array(texts, y=y, X_name="texts")

        # detect sentence pairs
        is_array, is_pair = detect_text_format(texts)
        if not is_array:
            raise ValueError(
                "texts must be a list of strings or a list of sentence pairs"
            )

        # detect language
        if self.lang is None and mode == "train":
            self.lang = TU.detect_lang(texts)
        U.vprint("language: %s" % (self.lang), verbose=verbose)

        # print stats
        if not is_pair:
            self.print_seqlen_stats(texts, mode, verbose=verbose)
        if is_pair:
            U.vprint("sentence pairs detected", verbose=verbose)

        # transform y
        if y is None and mode == "train":
            raise ValueError("y is required for training sets")
        elif y is None:
            y = np.array([1] * len(texts))
        y = self._transform_y(y, train=mode == "train", verbose=verbose)

        # convert examples
        tok, _ = self.get_preprocessor()
        dataset = self.hf_convert_examples(
            texts,
            y=y,
            tokenizer=tok,
            max_length=self.maxlen,
            pad_on_left=bool(self.name in ["xlnet"]),
            pad_token=tok.convert_tokens_to_ids([tok.pad_token][0]),
            pad_token_segment_id=4 if self.name in ["xlnet"] else 0,
            use_dynamic_shape=False if mode == "train" else True,
            verbose=verbose,
        )
        self.set_multilabel(dataset, mode, verbose=verbose)
        if mode == "train":
            self.preprocess_train_called = True
        return dataset

    def preprocess_test(self, texts, y=None, mode="test", verbose=1):
        self.check_trained()
        return self.preprocess_train(texts, y=y, mode=mode, verbose=verbose)

    @classmethod
    def load_model_and_configure_from_data(cls, fpath, transformer_ds):
        """
        ```
        loads model from file path and configures loss function and metrics automatically
        based on inspecting data
        Args:
          fpath(str): path to model folder
          transformer_ds(TransformerDataset): an instance of TransformerDataset
        ```
        """
        is_regression = U.is_regression_from_data(transformer_ds)
        multilabel = U.is_multilabel(transformer_ds)
        model = TFAutoModelForSequenceClassification.from_pretrained(fpath)
        if is_regression:
            metrics = ["mae"]
            loss_fn = "mse"
        else:
            metrics = ["accuracy"]
            if multilabel:
                loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn, optimizer=U.DEFAULT_OPT, metrics=metrics)
        return model

    def _load_pretrained(self, mname, num_labels):
        """
        ```
        load pretrained model
        ```
        """
        if self.config is not None:
            self.config.num_labels = num_labels
            try:
                model = self.model_type.from_pretrained(mname, config=self.config)
            except:
                warnings.warn(
                    "Could not load a Tensorflow version of model. (If this worked before, it might be an out-of-memory issue.) "
                    + "Attempting to download/load PyTorch version as TensorFlow model using from_pt=True. You will need PyTorch installed for this."
                )
                try:
                    model = self.model_type.from_pretrained(
                        mname, config=self.config, from_pt=True
                    )
                except:
                    # load model as normal to expose error to user
                    model = self.model_type.from_pretrained(mname, config=self.config)
                    # raise ValueError('could not load pretrained model %s using both from_pt=False and from_pt=True' % (mname))
        else:
            model = self.model_type.from_pretrained(mname, num_labels=num_labels)
        # ISSUE 416: mname is either model name (e.g., bert-base-uncased) or path to folder with tokenizer files
        self.get_tokenizer(mname)
        return model

    def get_classifier(self, fpath=None, multilabel=None, metrics=None):
        """
        ```
        creates a model for text classification
        Args:
          fpath(str): optional path to saved pretrained model. Typically left as None.
          multilabel(bool): If None, multilabel status is discovered from data [recommended].
                            If True, model will be forcibly configured for multilabel task.
                            If False, model will be forcibly configured for non-multilabel task.
                            It is recommended to leave this as None.
          metrics(list): Metrics to use.  If None, 'binary_accuracy' will be used if multilabel is True
                         and 'accuracy' is used otherwise.
        ```
        """
        self.check_trained()
        if not self.get_classes():
            warnings.warn("no class labels were provided - treating as regression")
            return self.get_regression_model()

        # process multilabel task
        multilabel = self.multilabel if multilabel is None else multilabel
        if multilabel is True and self.multilabel is False:
            warnings.warn(
                "The multilabel=True argument was supplied, but labels do not indicate "
                + "a multilabel problem (labels appear to be mutually-exclusive).  Using multilabel=True anyways."
            )
        elif multilabel is False and self.multilabel is True:
            warnings.warn(
                "The multilabel=False argument was supplied, but labels inidcate that  "
                + "this is a multilabel problem (labels are not mutually-exclusive).  Using multilabel=False anyways."
            )

        if multilabel and metrics is None:
            metrics = ["binary_accuracy"]
        elif metrics is None:
            metrics = ["accuracy"]

        if multilabel and metrics == ["accuracy"]:
            warnings.warn(
                'For multilabel problems, we recommend you supply the following argument to this method: metrics=["binary_accuracy"]. '
                + "Otherwise, a low accuracy score will be displayed by TensorFlow (https://github.com/tensorflow/tensorflow/issues/41114)."
            )

        # setup model
        num_labels = len(self.get_classes())
        mname = fpath if fpath is not None else self.model_name
        model = self._load_pretrained(mname, num_labels)
        if multilabel:
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn, optimizer=U.DEFAULT_OPT, metrics=metrics)
        return model

    def get_regression_model(self, fpath=None, metrics=["mae"]):
        """
        ```
        creates a model for text regression
        Args:
          fpath(str): optional path to saved pretrained model. Typically left as None.
          metrics(list): metrics to use
        ```
        """
        self.check_trained()
        if self.get_classes():
            warnings.warn(
                "class labels were provided - treating as classification problem"
            )
            return self.get_classifier()
        num_labels = 1
        mname = fpath if fpath is not None else self.model_name
        model = self._load_pretrained(mname, num_labels)
        loss_fn = "mse"
        model.compile(loss=loss_fn, optimizer=U.DEFAULT_OPT, metrics=metrics)
        return model

    def get_model(self, fpath=None):
        self.check_trained()
        if not self.get_classes():
            return self.get_regression_model(fpath=fpath)
        else:
            return self.get_classifier(fpath=fpath)

    def hf_convert_examples(
        self,
        texts,
        y=None,
        tokenizer=None,
        max_length=512,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        use_dynamic_shape=False,
        verbose=1,
    ):
        """
        ```
        Loads a data file into a list of ``InputFeatures``
        Args:
            texts: texts of documents or sentence pairs
            y:  labels for documents
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)
            use_dynamic_shape(bool):  If True, supplied max_length will be ignored and will be computed
                                      based on provided texts instead.
            verbose(bool): verbosity
        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        ```
        """

        is_array, is_pair = detect_text_format(texts)

        if use_dynamic_shape:
            sentences = []
            for text in texts:
                if is_pair:
                    text_a = text[0]
                    text_b = text[1]
                else:
                    text_a = text
                    text_b = None
                sentences.append(
                    tokenizer.convert_ids_to_tokens(tokenizer.encode(text_a, text_b))
                )
                # sentences.append(tokenizer.tokenize(text_a, text_b)) # only works for Fast tokenizers
            maxlen = (
                len(
                    max(
                        [tokens for tokens in sentences],
                        key=len,
                    )
                )
                + 2
            )

            if maxlen < max_length:
                max_length = maxlen

        data = []
        features_list = []
        labels = []
        if verbose:
            mb = master_bar(range(1))
            pb = progress_bar(texts, parent=mb)
        else:
            mb = range(1)
            pb = texts
        for i in mb:
            # for (idx, text) in enumerate(progress_bar(texts, parent=mb)):
            for idx, text in enumerate(pb):
                if is_pair:
                    text_a = text[0]
                    text_b = text[1]
                else:
                    text_a = text
                    text_b = None
                features = hf_convert_example(
                    text_a,
                    text_b=text_b,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    pad_on_left=pad_on_left,
                    pad_token=pad_token,
                    pad_token_segment_id=pad_token_segment_id,
                    mask_padding_with_zero=mask_padding_with_zero,
                )
                features_list.append(features)
                labels.append(y[idx] if y is not None else None)
        # tfdataset = hf_features_to_tfdataset(features_list, labels)
        # return tfdataset
        # return (features_list, labels)
        # HF_EXCEPTION
        # due to issues in transormers library and TF2 tf.Datasets, arrays are converted
        # to iterators on-the-fly
        # return  TransformerSequence(np.array(features_list), np.array(labels))
        from .dataset import TransformerDataset

        return TransformerDataset(
            np.array(features_list),
            np.array(labels),
            use_token_type_ids=self.use_token_type_ids,
        )


class DistilBertPreprocessor(TransformersPreprocessor):
    """
    ```
    text preprocessing for Hugging Face DistlBert model
    ```
    """

    def __init__(
        self, maxlen, max_features, class_names=[], classes=[], lang="en", ngram_range=1
    ):
        class_names = self.migrate_classes(class_names, classes)
        name = DISTILBERT
        if lang == "en":
            model_name = "distilbert-base-uncased"
        else:
            model_name = "distilbert-base-multilingual-cased"

        super().__init__(
            model_name,
            maxlen,
            max_features,
            class_names=class_names,
            lang=lang,
            ngram_range=ngram_range,
        )


class Transformer(TransformersPreprocessor):
    """
    ```
    convenience class for text classification Hugging Face transformers
    Usage:
       t = Transformer('distilbert-base-uncased', maxlen=128, classes=['neg', 'pos'], batch_size=16)
       train_dataset = t.preprocess_train(train_texts, train_labels)
       model = t.get_classifier()
       model.fit(train_dataset)
    ```
    """

    def __init__(
        self,
        model_name,
        maxlen=128,
        class_names=[],
        classes=[],
        batch_size=None,
        use_with_learner=True,
    ):
        """
        ```
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



        ```
        """
        multilabel = None  # force discovery of multilabel task from data in preprocess_train->set_multilabel
        class_names = self.migrate_classes(class_names, classes)
        if not use_with_learner and batch_size is None:
            raise ValueError("batch_size is required when use_with_learner=False")
        if multilabel and (class_names is None or not class_names):
            raise ValueError("classes argument is required when multilabel=True")
        super().__init__(
            model_name,
            maxlen,
            max_features=10000,
            class_names=class_names,
            multilabel=multilabel,
        )
        self.batch_size = batch_size
        self.use_with_learner = use_with_learner
        self.lang = None

    def preprocess_train(self, texts, y=None, mode="train", verbose=1):
        """
        ```
        Preprocess training set for A Transformer model

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
        ```
        """
        tseq = super().preprocess_train(texts, y=y, mode=mode, verbose=verbose)
        if self.use_with_learner:
            return tseq
        tseq.batch_size = self.batch_size
        train = mode == "train"
        return tseq.to_tfdataset(train=train)

    def preprocess_test(self, texts, y=None, verbose=1):
        """
        ```
        Preprocess the validation or test set for a Transformer model
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
        ```
        """
        self.check_trained()
        return self.preprocess_train(texts, y=y, mode="test", verbose=verbose)


class TransformerEmbedding:
    def __init__(self, model_name, layers=U.DEFAULT_TRANSFORMER_LAYERS):
        """
        ```
        Args:
            model_name (str):  name of Hugging Face pretrained model.
                               Choose from here: https://huggingface.co/transformers/pretrained_models.html
            layers(list): list of indexes indicating which hidden layers to use when
                          constructing the embedding (e.g., last=[-1])

        ```
        """
        self.layers = layers
        self.model_name = model_name
        if model_name.startswith("xlm-roberta"):
            self.name = "xlm_roberta"
        else:
            self.name = model_name.split("-")[0]

        self.config = AutoConfig.from_pretrained(model_name)
        self.model_type = TFAutoModel
        self.tokenizer_type = AutoTokenizer

        if "bert-base-japanese" in model_name:
            self.tokenizer_type = transformers.BertJapaneseTokenizer

        self.tokenizer = self.tokenizer_type.from_pretrained(model_name)
        self.model = self._load_pretrained(model_name)
        try:
            self.embsize = self.embed("ktrain", word_level=False).shape[
                1
            ]  # (batch_size, embsize)
        except:
            warnings.warn("could not determine Embedding size")
        # if type(self.model).__name__ not in [
        # "TFBertModel",
        # "TFDistilBertModel",
        # "TFAlbertModel",
        # "TFRobertaModel",
        # ]:
        # raise ValueError(
        # "TransformerEmbedding class currently only supports BERT-style models: "
        # + "Bert, DistilBert, RoBERTa and Albert and variants like BioBERT and SciBERT\n\n"
        # + "model received: %s (%s))" % (type(self.model).__name__, model_name)
        # )

    def _load_pretrained(self, model_name):
        """
        ```
        load pretrained model
        ```
        """
        if self.config is not None:
            self.config.output_hidden_states = True
            try:
                model = self.model_type.from_pretrained(model_name, config=self.config)
            except:
                warnings.warn(
                    "Could not load a Tensorflow version of model. (If this worked before, it might be an out-of-memory issue.) "
                    + "Attempting to download/load PyTorch version as TensorFlow model using from_pt=True. You will need PyTorch installed for this."
                )
                model = self.model_type.from_pretrained(
                    model_name, config=self.config, from_pt=True
                )
        else:
            model = self.model_type.from_pretrained(
                model_name, output_hidden_states=True
            )
        return model

    def _reconstruct_word_ids(self, offsets):
        """
        ```
        Reverse engineer the word_ids.
        ```
        """
        word_ids = []
        last_word_id = -1
        last_offset = (-1, -1)
        for o in offsets:
            if o == (0, 0):
                word_ids.append(None)
                continue
            # must test to see if start is same as last offset start due to xml-roberta quirk with tokens like 070
            if o[0] == last_offset[0] or o[0] == last_offset[1]:
                word_ids.append(last_word_id)
            elif o[0] > last_offset[1]:
                last_word_id += 1
                word_ids.append(last_word_id)
            last_offset = o
        return word_ids

    def embed(
        self,
        texts,
        word_level=True,
        max_length=512,
        aggregation_strategy="first",
        layers=U.DEFAULT_TRANSFORMER_LAYERS,
    ):
        """
        ```
        Get embedding for word, phrase, or sentence.

        Args:
          text(str|list): word, phrase, or sentence or list of them representing a batch
          word_level(bool): If True, returns embedding for each token in supplied texts.
                            If False, returns embedding for each text in texts
          max_length(int): max length of tokens
          aggregation_strategy(str): If 'first', vector of first subword is used as representation.
                                     If 'average', mean of all subword vectors is used.
          layers(list): list of indexes indicating which hidden layers to use when
                        constructing the embedding (e.g., last hidden state is [-1])
        Returns:
            np.ndarray : embeddings
        ```
        """
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts[0], str):
            texts = [" ".join(text) for text in texts]

        sentences = []
        for text in texts:
            sentences.append(self.tokenizer.tokenize(text))
        maxlen = (
            len(
                max(
                    [tokens for tokens in sentences],
                    key=len,
                )
            )
            + 2
        )
        if max_length is not None and maxlen > max_length:
            maxlen = max_length  # added due to issue #270
        sentences = []

        all_input_ids = []
        all_input_masks = []
        all_word_ids = []
        all_offsets = []  # retained but not currently used as of v0.36.1 (#492)
        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text, max_length=maxlen, truncation=True, return_offsets_mapping=True
            )
            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            del encoded["offset_mapping"]
            inp = encoded["input_ids"][:]
            inp = inp[1:] if inp[0] == self.tokenizer.cls_token_id else inp
            inp = inp[:-1] if inp[-1] == self.tokenizer.sep_token_id else inp
            tokens = self.tokenizer.convert_ids_to_tokens(inp)
            if len(tokens) > maxlen - 2:
                tokens = tokens[0 : (maxlen - 2)]
            sentences.append(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < maxlen:
                input_ids.append(0)
                input_mask.append(0)
            all_input_ids.append(input_ids)
            all_input_masks.append(input_mask)
            # Note about Issue #492:
            # deberta includes preceding space in offfset_mapping (https://www.kaggle.com/code/junkoda/be-aware-of-white-space-deberta-roberta)
            # models like bert-base-case produce word_ids that do not correspond to whitespace tokenization (e.g.,"score 99.9%", "BRUSSELS 1996-08-22")
            # Therefore, we use offset_mappings unless the model is deberta for now.
            word_ids = (
                encoded.word_ids()
                if "deberta" in self.model_name
                else self._reconstruct_word_ids(offsets)
            )
            all_word_ids.append(word_ids)
            all_offsets.append(offsets)

        all_input_ids = np.array(all_input_ids)
        all_input_masks = np.array(all_input_masks)
        outputs = self.model(all_input_ids, attention_mask=all_input_masks)
        hidden_states = outputs[-1]  # output_hidden_states=True

        # compile raw embeddings
        if len(self.layers) == 1:
            # raw_embeddings = hidden_states[-1].numpy()
            raw_embeddings = hidden_states[self.layers[0]].numpy()
        else:
            raw_embeddings = []
            for batch_id in range(hidden_states[0].shape[0]):
                token_embeddings = []
                for token_id in range(hidden_states[0].shape[1]):
                    all_layers = []
                    for layer_id in self.layers:
                        all_layers.append(
                            hidden_states[layer_id][batch_id][token_id].numpy()
                        )
                    token_embeddings.append(np.concatenate(all_layers))
                raw_embeddings.append(token_embeddings)
            raw_embeddings = np.array(raw_embeddings)

        if not word_level:  # sentence-level embedding
            return np.mean(raw_embeddings, axis=1)

        # all space-separate tokens in input should be assigned a single embedding vector
        # example: If 99.9% is a token, then it gets a single embedding.
        # example: If input is pre-tokenized (i.e., 99 . 9 %), then there are four embedding vectors
        filtered_embeddings = []
        for i in range(len(raw_embeddings)):
            filtered_embedding = []
            raw_embedding = raw_embeddings[i]
            subvectors = []
            last_word_id = -1
            for j in range(len(all_offsets[i])):
                word_id = all_word_ids[i][j]
                if word_id is None:
                    continue
                if word_id == last_word_id:
                    subvectors.append(raw_embedding[j])
                if word_id > last_word_id:
                    if len(subvectors) > 0:
                        if aggregation_strategy == "average":
                            filtered_embedding.append(np.mean(subvectors, axis=0))
                        else:
                            filtered_embedding.append(subvectors[0])
                        subvectors = []
                    subvectors.append(raw_embedding[j])
                    last_word_id = word_id
            if len(subvectors) > 0:
                if aggregation_strategy == "average":
                    filtered_embedding.append(np.mean(subvectors, axis=0))
                else:
                    filtered_embedding.append(subvectors[0])
                subvectors = []
            filtered_embeddings.append(filtered_embedding)

        # pad embeddings with zeros
        max_length = max([len(e) for e in filtered_embeddings])
        embeddings = []
        for e in filtered_embeddings:
            for i in range(max_length - len(e)):
                e.append(np.zeros((self.embsize,)))
            embeddings.append(np.array(e))
        return np.array(embeddings)


# preprocessors
TEXT_PREPROCESSORS = {
    "standard": StandardTextPreprocessor,
    "bert": BERTPreprocessor,
    "distilbert": DistilBertPreprocessor,
}
