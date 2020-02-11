from ..imports import *
from .. import utils as U
from ..preprocessor import Preprocessor
from ..data import Dataset

DistilBertTokenizer = transformers.DistilBertTokenizer
DISTILBERT= 'distilbert'

from transformers import BertConfig, TFBertForSequenceClassification, BertTokenizer
from transformers import XLNetConfig, TFXLNetForSequenceClassification, XLNetTokenizer
from transformers import XLMConfig, TFXLMForSequenceClassification, XLMTokenizer
from transformers import RobertaConfig, TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AlbertConfig, TFAlbertForSequenceClassification, AlbertTokenizer
#from transformers import CamembertConfig, TFCamembertForSequenceClassification, CamembertTokenizer

TRANSFORMER_MODELS = {
    'bert':       (BertConfig, TFBertForSequenceClassification, BertTokenizer),
    'xlnet':      (XLNetConfig, TFXLNetForSequenceClassification, XLNetTokenizer),
    'xlm':        (XLMConfig, TFXLMForSequenceClassification, XLMTokenizer),
    'roberta':    (RobertaConfig, TFRobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer),
    'albert':     (AlbertConfig, TFAlbertForSequenceClassification, AlbertTokenizer),
    #'camembert':  (CamembertConfig, TFCamembertForSequenceClassification, CamembertTokenizer)
}


NOSPACE_LANGS = ['zh-cn', 'zh-tw', 'ja']

def is_chinese(lang):
    return lang is not None and lang.startswith('zh-')


def is_nospace_lang(lang):
    return lang in NOSPACE_LANGS


def fname_from_url(url):
    return os.path.split(url)[-1]


#------------------------------------------------------------------------------
# Word Vectors
#------------------------------------------------------------------------------
WV_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
#WV_URL = 'http://nlp.stanford.edu/data/glove.6B.zip


def get_wv_path():
    ktrain_data = U.get_ktrain_data()
    zip_fpath = os.path.join(ktrain_data, fname_from_url(WV_URL))
    wv_path =  os.path.join(ktrain_data, os.path.splitext(fname_from_url(WV_URL))[0])
    if not os.path.isfile(wv_path):
        # download zip
        print('downloading pretrained word vectors (~1.5G) ...')
        U.download(WV_URL, zip_fpath)

        # unzip
        print('\nextracting pretrained word vectors...')
        with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
            zip_ref.extractall(ktrain_data)
        print('done.\n')

        # cleanup
        print('cleanup downloaded zip...')
        try:
            os.remove(zip_fpath)
            print('done.\n')
        except OSError:
            print('failed to cleanup/remove %s' % (zip_fpath))
    return wv_path


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def load_wv(wv_path=None, verbose=1):
    if verbose: print('Loading pretrained word vectors...this may take a few moments...')
    if wv_path is None: wv_path = get_wv_path()
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(wv_path, encoding='utf-8'))
    if verbose: print('Done.')
    return embeddings_index



#------------------------------------------------------------------------------
# BERT
#------------------------------------------------------------------------------

#BERT_PATH = os.path.join(os.path.dirname(os.path.abspath(localbert.__file__)), 'uncased_L-12_H-768_A-12')
BERT_URL = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
BERT_URL_MULTI = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'
BERT_URL_CN = 'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip'

def get_bert_path(lang='en'):
    if lang == 'en':
        bert_url = BERT_URL
    elif lang.startswith('zh-'):
        bert_url = BERT_URL_CN
    else:
        bert_url = BERT_URL_MULTI
    ktrain_data = U.get_ktrain_data()
    zip_fpath = os.path.join(ktrain_data, fname_from_url(bert_url))
    bert_path = os.path.join( ktrain_data, os.path.splitext(fname_from_url(bert_url))[0] )
    if not os.path.isdir(bert_path) or \
       not os.path.isfile(os.path.join(bert_path, 'bert_config.json')) or\
       not os.path.isfile(os.path.join(bert_path, 'bert_model.ckpt.data-00000-of-00001')) or\
       not os.path.isfile(os.path.join(bert_path, 'bert_model.ckpt.index')) or\
       not os.path.isfile(os.path.join(bert_path, 'bert_model.ckpt.meta')) or\
       not os.path.isfile(os.path.join(bert_path, 'vocab.txt')):
        # download zip
        print('downloading pretrained BERT model (%s)...' % (fname_from_url(bert_url)))
        U.download(bert_url, zip_fpath)

        # unzip
        print('\nextracting pretrained BERT model...')
        with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
            zip_ref.extractall(ktrain_data)
        print('done.\n')

        # cleanup
        print('cleanup downloaded zip...')
        try:
            os.remove(zip_fpath)
            print('done.\n')
        except OSError:
            print('failed to cleanup/remove %s' % (zip_fpath))
    return bert_path



def bert_tokenize(docs, tokenizer, maxlen, verbose=1):
    indices = []
    mb = master_bar(range(1))
    for i in mb:
        for doc in progress_bar(docs, parent=mb):
            ids, segments = tokenizer.encode(doc, max_len=maxlen)
            indices.append(ids)
        if verbose: mb.write('done.')
    zeros = np.zeros_like(indices)
    return [np.array(indices), np.array(zeros)]

#------------------------------------------------------------------------------
# Transformers UTILITIES
#------------------------------------------------------------------------------

#def convert_to_tfdataset(csv):
    #def gen():
        #for ex in csv:
            #yield  {'idx': ex[0],
                     #'sentence': ex[1],
                     #'label': str(ex[2])}
    #return tf.data.Dataset.from_generator(gen,
        #{'idx': tf.int64,
          #'sentence': tf.string,
          #'label': tf.int64})


#def features_to_tfdataset(features):

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



def hf_features_to_tfdataset(features_list, labels):
    features_list = np.array(features_list)
    labels = np.array(labels) if labels is not None else None
    tfdataset = tf.data.Dataset.from_tensor_slices((features_list, labels))
    tfdataset = tfdataset.map(lambda x,y: ({'input_ids': x[0], 
                                            'attention_mask': x[1], 
                                             'token_type_ids': x[2]}, y))

    return tfdataset



def hf_convert_example(text, tokenizer=None,
                       max_length=512,
                       pad_on_left=False,
                       pad_token=0,
                       pad_token_segment_id=0,
                       mask_padding_with_zero=True):
    """
    convert InputExample to InputFeature for Hugging Face transformer
    """
    if tokenizer is None: raise ValueError('tokenizer is required')

    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)


    #if ex_index < 1:
        #print("*** Example ***")
        #print("guid: %s" % (example.guid))
        #print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #print("label: %s (id = %d)" % (example.label, label))

    return [input_ids, attention_mask, token_type_ids]




def hf_convert_examples(texts, y=None, tokenizer=None,
                        max_length=512,
                        pad_on_left=False,
                        pad_token=0,
                        pad_token_segment_id=0,
                        mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        texts: texts of documents
        y:  labels for documents
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """



    data = []
    mb = master_bar(range(1))
    features_list = []
    labels = []
    for i in mb:
        for (idx, text) in enumerate(progress_bar(texts, parent=mb)):

            features = hf_convert_example(text, tokenizer=tokenizer,
                                          max_length=max_length,
                                          pad_on_left=pad_on_left,
                                          pad_token=pad_token,
                                          pad_token_segment_id=pad_token_segment_id,
                                          mask_padding_with_zero=mask_padding_with_zero)
            features_list.append(features)
            labels.append(y[idx] if y is not None else None)
    #tfdataset = hf_features_to_tfdataset(features_list, labels)
    #return tfdataset
    #return (features_list, labels)
    # HF_EXCEPTION
    # due to issues in transormers library and TF2 tf.Datasets, arrays are converted
    # to iterators on-the-fly
    #return  TransformerSequence(np.array(features_list), np.array(labels))
    return  TransformerDataset(np.array(features_list), np.array(labels))


#------------------------------------------------------------------------------
# MISC UTILITIES
#------------------------------------------------------------------------------

def decode_by_line(texts, encoding='utf-8', verbose=1):
    """
    Decode text line by line and skip over errors.
    """
    new_texts = []
    skips=0
    num_lines = 0
    for doc in texts:
        text = ""
        for line in doc.splitlines():
            num_lines +=1
            try:
                line = line.decode(encoding)
            except:
                skips +=1
                continue
            text += line
        new_texts.append(text)
    pct = round((skips*1./num_lines) * 100, 1)
    if verbose: 
        print('skipped %s lines (%s%%) due to character decoding errors' % (skips, pct))
        if pct > 10:
            print('If this is too many, try a different encoding')
    return new_texts



def detect_lang(texts, sample_size=32):
    """
    detect language
    """
    if isinstance(texts, (pd.Series, pd.DataFrame)):
        texts = texts.values
    if isinstance(texts, str): texts = [texts]
    if not isinstance(texts, (list, np.ndarray)):
        raise ValueError('texts must be a list or NumPy array of strings')
    lst = []
    for doc in texts[:sample_size]:
        try:
            lst.append(langdetect.detect(doc))
        except:
            continue
    if len(lst) == 0: 
        raise Exception('could not detect language in random sample of %s docs.'  % (sample_size))
    return max(set(lst), key=lst.count)





#------------------------------------------------------------------------------


class TextPreprocessor(Preprocessor):
    """
    Text preprocessing base class
    """

    def __init__(self, maxlen, classes, lang='en', multilabel=False):

        self.c = classes
        self.maxlen = maxlen
        self.lang = lang
        self.multilabel = multilabel

    
    def get_preprocessor(self):
        raise NotImplementedError


    def get_classes(self):
        return self.c


    def preprocess(self, texts):
        raise NotImplementedError


    def set_multilabel(self, data, mode):
        if mode == 'train' and self.get_classes():
            self.multilabel = U.is_multilabel(data)


    def undo(self, doc):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        raise NotImplementedError


    def is_chinese(self):
        return is_chinese(self.lang)


    def is_nospace_lang(self):
        return is_nospace_lang(self.lang)


    def process_chinese(self, texts, lang=None):
        #if lang is None: lang = langdetect.detect(texts[0])
        if lang is None: lang = detect_lang(texts)
        if not is_nospace_lang(lang): return texts
        split_texts = []
        for doc in texts:
            seg_list = jieba.cut(doc, cut_all=False)
            seg_list = list(seg_list)
            split_texts.append(seg_list)
        return [" ".join(tokens) for tokens in split_texts] 


    @classmethod
    def seqlen_stats(cls, list_of_texts):
        """
        compute sequence length stats from
        list of texts in any spaces-segmented language
        Args:
            list_of_texts: list of strings
        Returns:
            dict: dictionary with keys: mean, 95percentile, 99percentile
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
        avg = sum(counts)/len(counts)
        return {'mean':avg, '95percentile': p95, '99percentile':p99}


    def print_seqlen_stats(self, texts, mode, verbose=1):
        """
        prints stats about sequence lengths
        """
        if verbose and not self.is_nospace_lang():
            stat_dict = TextPreprocessor.seqlen_stats(texts)
            print( "%s sequence lengths:" % mode)
            for k in stat_dict:
                print("\t%s : %s" % (k, int(round(stat_dict[k]))))


    def _transform_y(self, y_data):
        """
        preprocess y
        If shape of y is 1, then task is considered classification if self.c exists
        or regression if not.
        """
        if y_data is None: return y_data
        # if shape is 1, this is either a classification or regression task 
        # depending on class_names existing
        y_data = np.array(y_data) if type(y_data) == list else y_data
        y_data = to_categorical(y_data) if len(y_data.shape) == 1 and self.get_classes() else y_data
        return y_data






class StandardTextPreprocessor(TextPreprocessor):
    """
    Standard text preprocessing
    """

    def __init__(self, maxlen, max_features, classes=[], 
                 lang='en', ngram_range=1, multilabel=False):
        super().__init__(maxlen, classes, lang=lang, multilabel=multilabel)
        self.tok = None
        self.tok_dct = {}
        self.max_features = max_features
        self.ngram_range = ngram_range


    def get_preprocessor(self):
        return (self.tok, self.tok_dct)


    def preprocess(self, texts):
        return self.preprocess_test(texts, verbose=0)[0]


    def undo(self, doc):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        dct = self.tok.index_word
        return " ".join([dct[wid] for wid in doc if wid != 0 and wid in dct])


    def preprocess_train(self, train_text, y_train, verbose=1):
        """
        preprocess training set
        """
        if self.lang is None: self.lang = detect_lang(train_text)


        U.vprint('language: %s' % (self.lang), verbose=verbose)

        # special processing if Chinese
        train_text = self.process_chinese(train_text, lang=self.lang)

        # extract vocabulary
        self.tok = Tokenizer(num_words=self.max_features)
        self.tok.fit_on_texts(train_text)
        U.vprint('Word Counts: {}'.format(len(self.tok.word_counts)), verbose=verbose)
        U.vprint('Nrows: {}'.format(len(train_text)), verbose=verbose)

        # convert to word IDs
        x_train = self.tok.texts_to_sequences(train_text)
        U.vprint('{} train sequences'.format(len(x_train)), verbose=verbose)
        self.print_seqlen_stats(x_train, 'train', verbose=verbose)

        # add ngrams
        x_train = self._fit_ngrams(x_train, verbose=verbose)

        # pad sequences
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        U.vprint('x_train shape: ({},{})'.format(x_train.shape[0], x_train.shape[1]), verbose=verbose)

        # transform y
        y_train = self._transform_y(y_train)
        if y_train is not None and verbose:
            print('y_train shape: %s' % (y_train.shape,))

        # return
        result =  (x_train, y_train)
        self.set_multilabel(result, 'train')
        return result


    def preprocess_test(self, test_text, y_test=None, verbose=1):
        """
        preprocess validation or test dataset
        """
        if self.tok is None or self.lang is None:
            raise Exception('Unfitted tokenizer or missing language. Did you run preprocess_train first?')

        # check for and process chinese
        test_text = self.process_chinese(test_text, self.lang)

        # convert to word IDs
        x_test = self.tok.texts_to_sequences(test_text)
        U.vprint('{} test sequences'.format(len(x_test)), verbose=verbose)
        self.print_seqlen_stats(x_test, 'test', verbose=verbose)

        # add n-grams
        x_test = self._add_ngrams(x_test, mode='test', verbose=verbose)


        # pad sequences
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        U.vprint('x_test shape: ({},{})'.format(x_test.shape[0], x_test.shape[1]), verbose=verbose)

        # transform y
        y_test = self._transform_y(y_test)
        if y_test is not None and verbose:
            print('y_test shape: %s' % (y_test.shape,))


        # return
        return (x_test, y_test)



    def _fit_ngrams(self, x_train, verbose=1):
        self.tok_dct = {}
        if self.ngram_range < 2: return x_train
        U.vprint('Adding {}-gram features'.format(self.ngram_range), verbose=verbose)
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
        U.vprint('max_features changed to %s with addition of ngrams' % (self.max_features), verbose=verbose)

        # Augmenting x_train with n-grams features
        x_train = self._add_ngrams(x_train, verbose=verbose, mode='train')
        return x_train


    def _add_ngrams(self, sequences, verbose=1, mode='test'):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        """
        token_indice = self.tok_dct
        if self.ngram_range < 2: return sequences
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, self.ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)
        U.vprint('Average {} sequence length with ngrams: {}'.format(mode,
            np.mean(list(map(len, new_sequences)), dtype=int)), verbose=verbose)    
        self.print_seqlen_stats(new_sequences, '%s (w/ngrams)' % mode, verbose=verbose)
        return new_sequences



    def _create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))


    def ngram_count(self):
        if not self.tok_dct: return 1
        s = set()
        for k in self.tok_dct.keys():
            s.add(len(k))
        return max(list(s))


class BERTPreprocessor(TextPreprocessor):
    """
    text preprocessing for BERT model
    """

    def __init__(self, maxlen, max_features, classes=[], 
                lang='en', ngram_range=1, multilabel=False):

        if maxlen > 512: raise ValueError('BERT only supports maxlen <= 512')

        super().__init__(maxlen, classes, lang=lang, multilabel=multilabel)
        vocab_path = os.path.join(get_bert_path(lang=lang), 'vocab.txt')
        token_dict = {}
        with codecs.open(vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = BERT_Tokenizer(token_dict)
        self.tok = tokenizer
        self.tok_dct = dict((v,k) for k,v in token_dict.items())
        self.max_features = max_features # ignored
        self.ngram_range = 1 # ignored


    def get_preprocessor(self):
        return (self.tok, self.tok_dct)



    def preprocess(self, texts):
        return self.preprocess_test(texts, verbose=0)[0]


    def undo(self, doc):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        dct = self.tok_dct
        return " ".join([dct[wid] for wid in doc if wid != 0 and wid in dct])


    def preprocess_train(self, texts, y=None, mode='train', verbose=1):
        """
        preprocess training set
        """
        if mode == 'train' and y is None:
            raise ValueError('y is required when mode=train')
        if self.lang is None and mode=='train': self.lang = detect_lang(texts)
        U.vprint('preprocessing %s...' % (mode), verbose=verbose)
        U.vprint('language: %s' % (self.lang), verbose=verbose)

        x = bert_tokenize(texts, self.tok, self.maxlen, verbose=verbose)

        # transform y
        y = self._transform_y(y)
        result = (x, y)
        self.set_multilabel(result, mode)
        return result



    def preprocess_test(self, texts, y=None, mode='test', verbose=1):
        return self.preprocess_train(texts, y=y, mode=mode, verbose=verbose)


class TransformersPreprocessor(TextPreprocessor):
    """
    text preprocessing for Hugging Face Transformer models
    """

    def __init__(self,  model_name,
                maxlen, max_features, classes=[], 
                lang='en', ngram_range=1, multilabel=False):

        if maxlen > 512: raise ValueError('Transformer models only supports maxlen <= 512')

        super().__init__(maxlen, classes, lang=lang, multilabel=multilabel)

        self.model_name = model_name
        self.name = model_name.split('-')[0]
        if self.name not in TRANSFORMER_MODELS:
            raise ValueError('uknown model name %s' % (model_name))
        self.model_type = TRANSFORMER_MODELS[self.name][1]
        self.tokenizer_type = TRANSFORMER_MODELS[self.name][2]
        if "bert-base-japanese" in model_name:
            self.tokenizer_type = transformers.BertJapaneseTokenizer

        tokenizer = self.tokenizer_type.from_pretrained(model_name)

        self.tok = tokenizer
        self.tok_dct = None
        self.max_features = max_features # ignored
        self.ngram_range = 1 # ignored




    def get_preprocessor(self):
        return (self.tok, self.tok_dct)



    def preprocess(self, texts):
        tseq = self.preprocess_test(texts, verbose=0)
        return tseq.to_tfdataset(shuffle=False, repeat=False)


    def undo(self, doc):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        print(doc)
        print(type(doc))
        return self.tok.convert_ids_to_tokens(doc)
        #raise Exception('currently_unsupported: Transformers.Preprocessor.undo is not yet supported')


    def preprocess_train(self, texts, y=None, mode='train', verbose=1):
        """
        preprocess training set
        """
        U.vprint('preprocessing %s...' % (mode), verbose=verbose)
        if self.lang is None and mode=='train': self.lang = detect_lang(texts)
        U.vprint('language: %s' % (self.lang), verbose=verbose)
        self.print_seqlen_stats(texts, mode, verbose=verbose)

        # transform y
        if y is None and mode == 'train':
            raise ValueError('y is required for training sets')
        elif y is None:
            y = np.array([1] * len(texts))
        y = self._transform_y(y)
        dataset = hf_convert_examples(texts, y=y, tokenizer=self.tok, max_length=self.maxlen,
                                      pad_on_left=bool(self.name in ['xlnet']),
                                      pad_token=self.tok.convert_tokens_to_ids([self.tok.pad_token][0]),
                                      pad_token_segment_id=4 if self.name in ['xlnet'] else 0)
        self.set_multilabel(dataset, mode)
        return dataset



    def preprocess_test(self, texts, y=None, mode='test', verbose=1):
        return self.preprocess_train(texts, y=y, mode=mode, verbose=verbose)



class DistilBertPreprocessor(TransformersPreprocessor):
    """
    text preprocessing for Hugging Face DistlBert model
    """

    def __init__(self, maxlen, max_features, classes=[], 
                lang='en', ngram_range=1):

        name = DISTILBERT
        if lang == 'en':
            model_name = 'distilbert-base-uncased'
        else:
            model_name = 'distilbert-base-multilingual-cased'

        super().__init__(model_name,
                         maxlen, max_features, classes=classes, 
                         lang=lang, ngram_range=ngram_range)


class Transformer(TransformersPreprocessor):
    """
    convenience class for text classification Hugging Face transformers 
    Usage:
       t = Transformer('distilbert-base-uncased', maxlen=128, classes=['neg', 'pos'], batch_size=16)
       train_dataset = t.preprocess_train(train_texts, train_labels)
       model = t.get_classifier()
       model.fit(train_dataset)
    """

    def __init__(self, model_name, maxlen=128, classes=[], 
                 batch_size=None, multilabel=False,
                 use_with_learner=True):
        """
        Args:
            model_name (str):  name of Hugging Face pretrained model
            maxlen (int):  sequence length
            classes(list):  list of strings of class names (e.g., 'positive', 'negative')
            use_with_learner(bool):  If False, preprocess_train and preprocess_test
                                     will return tf.Datasets for direct use with model.fit
                                     in tf.Keras.
                                     If True, preprocess_train and preprocess_test will
                                     return a ktrain TransformerSequence object for use with
                                     ktrain.get_learner.
            batch_size (int): batch_size - only required if use_with_learner=False
            multilabel (int):  if True, classifier will be configured for
                                  multilabel classification.

        """
        if not use_with_learner and batch_size is None:
            raise ValueError('batch_size is required when use_with_learner=False')
        if multilabel and (classes is None or not classes):
            raise ValueError('classes argument is required when multilabel=True')
        super().__init__(model_name,
                         maxlen, max_features=10000, classes=classes, multilabel=multilabel)
        self.batch_size = batch_size
        self.use_with_learner = use_with_learner
        self.lang = None


    def preprocess_train(self, texts, y=None, mode='train', verbose=1):
        """
        Preprocess training set for A Transformer model

        Each label can be in the form of either:
        1) integer representing the class (index into array returned by get_classes)
           for binary and multiclass text classification
        2) multi-hot-encoded vector for multilabel text classification problems

        Args:
            texts (list of strings): text of documents
            y: labels
            mode (str):  If 'train' and prepare_for_learner=False,
                         a tf.Dataset will be returned with repeat enabled
                         for training with fit_generator
            verbose(bool): verbosity
        """
        tseq = super().preprocess_train(texts, y=y, mode=mode, verbose=verbose)
        if self.use_with_learner: return tseq
        tseq.batch_size = self.batch_size
        shuffle=True if mode=='train' else False
        repeat=True if mode=='train' else False
        return tseq.to_tfdataset(shuffle=shuffle, repeat=repeat)


    def preprocess_test(self, texts, y=None,  verbose=1):
        """
        Preprocess the validation or test set for a Transformer model

        Each label can be in the form of either:
        1) integer representing the class (index into array returned by get_classes)
           for binary and multiclass text classification
        2) multi-hot-encoded vector for multilabel text classification problems

        Args:
            texts (list of strings): text of documents
            y: labels
            verbose(bool): verbosity
        """
        return self.preprocess_train(texts, y=y, mode='test', verbose=verbose)



    def get_classifier(self):
        if not self.get_classes():
            warnings.warn('no class labels were provided - treating as regression')
            return self.get_regression_model()
        num_labels = len(self.get_classes())
        model = self.model_type.from_pretrained(self.model_name, num_labels=num_labels)
        if self.multilabel:
            loss_fn =  keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn,
                      optimizer=keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08),
                      metrics=['accuracy'])
        return model


    def get_regression_model(self):
        if self.get_classes():
            warnings.warn('class labels were provided - treating as classification problem')
            return self.get_classifier()
        model = self.model_type.from_pretrained(self.model_name, num_labels=1)
        loss_fn = 'mse'
        model.compile(loss=loss_fn,
                      optimizer=keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08),
                      metrics=['mae'])
        return model


    def get_model(self):
        if not self.get_classes():
            return self.get_regression_model()
        else:
            return self.get_classifier()



class TransformerDataset(Dataset):
    """
    Wrapper for Transformer datasets.
    """

    def __init__(self, x, y, batch_size=1):
        if type(x) not in [list, np.ndarray]: raise ValueError('x must be list or np.ndarray')
        if type(y) not in [list, np.ndarray]: raise ValueError('y must be list or np.ndarray')
        if type(x) == list: x = np.array(x)
        if type(y) == list: y = np.array(y)
        self.x = x
        self.y = y
        self.batch_size = batch_size


    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        return (batch_x, batch_y)


    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def to_tfdataset(self, shuffle=True, repeat=True):
        """
        convert transformer features to tf.Dataset
        """
        if len(self.y.shape) == 1:
            yshape = []
        else:
            yshape = [None]

        def gen():
            for idx, data in enumerate(self.x):
                yield ({'input_ids': data[0],
                         'attention_mask': data[1],
                         'token_type_ids': data[2]},
                        self.y[idx])

        tfdataset= tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape(yshape)))

        if shuffle:
            tfdataset = tfdataset.shuffle(self.x.shape[0])
        tfdataset = tfdataset.batch(self.batch_size)
        if repeat:
            tfdataset = tfdataset.repeat(-1)
        return tfdataset


    def get_y(self):
        return self.y

    def nsamples(self):
        return len(self.x)

    def nclasses(self):
        return self.y.shape[1]

    def xshape(self):
        return (len(self.x), self.x[0].shape[1])


# preprocessors
TEXT_PREPROCESSORS = {'standard': StandardTextPreprocessor,
                      'bert': BERTPreprocessor,
                      'distilbert': DistilBertPreprocessor}
