from ..imports import *
from .. import utils as U
from ..preprocessor import Preprocessor


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
        print('\nextracting pretrained pretrained word vectors...')
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
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(wv_path))
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
    if not isinstance(texts, (list, np.ndarray)): texts = [texts]
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

    def __init__(self, maxlen, classes, lang='en'):

        self.c = classes
        self.maxlen = maxlen
        self.lang = lang


    def get_preprocessor(self):
        raise NotImplementedError


    def get_classes(self):
        return self.c


    def preprocess(self, texts):
        raise NotImplementedError


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




class StandardTextPreprocessor(TextPreprocessor):
    """
    Standard text preprocessing
    """

    def __init__(self, maxlen, max_features, classes=[], 
                 lang='en', ngram_range=1):
        super().__init__(maxlen, classes, lang=lang)
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
        U.vprint('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)), verbose=verbose)

        # add ngrams
        x_train = self._fit_ngrams(x_train, verbose=verbose)

        # pad sequences
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)

        # transform y
        if len(y_train.shape) == 1:
            y_train = to_categorical(y_train)
        U.vprint('x_train shape: ({},{})'.format(x_train.shape[0], x_train.shape[1]), verbose=verbose)
        U.vprint('y_train shape: ({},{})'.format(y_train.shape[0], y_train.shape[1]), verbose=verbose)

        # return
        return (x_train, y_train)


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
        U.vprint('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), 
                                                                 dtype=int)), verbose=verbose)

        # add n-grams
        x_test = self._add_ngrams(x_test, mode='test', verbose=verbose)


        # pad sequences
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        U.vprint('x_test shape: ({},{})'.format(x_test.shape[0], x_test.shape[1]), verbose=verbose)

        # convert y
        if y_test is not None:
            if len(y_test.shape) == 1:
                y_test = to_categorical(y_test)
            U.vprint('y_test shape: ({},{})'.format(y_test.shape[0], y_test.shape[1]), verbose=verbose)

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
                lang='en', ngram_range=1):

        if maxlen > 512: raise ValueError('BERT only supports maxlen <= 512')

        super().__init__(maxlen, classes, lang=lang)
        vocab_path = os.path.join(get_bert_path(lang=lang), 'vocab.txt')
        token_dict = {}
        with codecs.open(vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = BERT_Tokenizer(token_dict)
        self.tok = tokenizer
        self.tok_dct = dict((v,k) for k,v in token_dict.items())
        self.max_features = max_features
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
        U.vprint('preprocessing %s...' % (mode), verbose=verbose)
        U.vprint('language: %s' % (self.lang), verbose=verbose)

        x = bert_tokenize(texts, self.tok, self.maxlen, verbose=verbose)
        if y is not None:
            if len(y.shape) == 1:
                y = to_categorical(y)
            #U.vprint('\ty shape: ({},{})'.format(y.shape[0], y.shape[1]), verbose=verbose)
        return (x, y)



    def preprocess_test(self, texts, y=None, mode='test', verbose=1):
        return self.preprocess_train(texts, y=y, mode=mode, verbose=verbose)





# preprocessors
TEXT_PREPROCESSORS = {'standard': StandardTextPreprocessor,
                      'bert': BERTPreprocessor}

