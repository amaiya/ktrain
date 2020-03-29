from ...imports import *
from ... import utils as U
from ...preprocessor import Preprocessor
from ...data import Dataset
from .. import textutils as TU
from .. import preprocessor as tpp
from .anago.utils import filter_embeddings

OTHER = 'O'
W2V = 'word2vec'
SUPPORTED_EMBEDDINGS = [W2V]

#tokenizer_filter = rs='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


class NERPreprocessor(Preprocessor):
    """
    NER preprocessing base class
    """

    def __init__(self, p):
        self.p = p
        self.c = p._label_vocab._id2token

    def get_preprocessor(self):
        return self.p


    def get_classes(self):
        return self.c


    def filter_embeddings(self, embeddings, vocab, dim):
        """Loads word vectors in numpy array.

        Args:
            embeddings (dict or TransformerEmbedding): a dictionary of numpy array or Transformer Embedding instance
            vocab (dict): word_index lookup table.

        Returns:
            numpy array: an array of word embeddings.
        """
        if not isinstance(embeddings, dict):
            return
        _embeddings = np.zeros([len(vocab), dim])
        for word in vocab:
            if word in embeddings:
                word_idx = vocab[word]
                _embeddings[word_idx] = embeddings[word]
        return _embeddings



    def get_wv_model(self, wv_path_or_url, verbose=1):
        if wv_path_or_url is None: 
            raise ValueError('wordvector_path_or_url is empty: supply a file path or '+\
                             'URL to fasttext word vector file')
        if verbose: print('pretrained word embeddings will be loaded from:\n\t%s' % (wv_path_or_url))
        word_embedding_dim = 300 # all fasttext word vectors are of dim=300
        embs = tpp.load_wv(wv_path_or_url, verbose=verbose)
        wv_model = self.filter_embeddings(embs, self.p._word_vocab.vocab, word_embedding_dim)
        return (wv_model, word_embedding_dim)



    def preprocess(self, sentences):
        if type(sentences) != list:
            raise ValueError('Param sentences must be a list of strings')

        # language detection
        lang = TU.detect_lang(sentences)
        X = []
        y = []
        for s in sentences:
            if TU.is_chinese(lang, strict=False): # strict=False: workaround for langdetect bug on short chinese texts
                tokenize_chinese = lambda text:[c for c in text]
                tokens = tokenize_chinese(s)
            else:
                tokens = tokenize(s)
            X.append(tokens)
            y.append([OTHER] * len(tokens))
        nerseq = NERSequence(X, y, p=self.p)
        return nerseq

    def preprocess_test(self, x_test, y_test, verbose=1):
        """
        Args:
          x_test(list of lists of str): lists of token lists
          x_test (list of lists of str):  lists of tag lists
          verbose(bool): verbosity
        Returns:
          NERSequence:  can be used as argument to NERLearner.validate() to evaluate test sets
        """
        # array > df > array in order to print statistics more easily
        from .data import array_to_df
        test_df = array_to_df(x_test, y_test) 
        (x_list, y_list)  = process_df(test_df, verbose=verbose) 
        return NERSequence(x_list, y_list, batch_size=U.DEFAULT_BS, p=self.p)


    def undo(self, nerseq):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        return [" ".join(e) for e in nerseq.x]




    def fit(self, X, y):
        """
        Learn vocabulary from training set
        """
        self.p.fit(X, y)
        return


    def transform(self, X, y=None):
        """
        Transform documents to sequences of word IDs
        """
        return self.p.transform(X, y=y)




def process_df(df, 
               sentence_column='SentenceID', 
               word_column='Word', 
               tag_column='Tag',
               verbose=1):
    """
    Extract words, tags, and sentences from dataframe
    """


    # get words and tags
    words = list(set(df[word_column].values))
    n_words = len(words)
    tags = list(set(df[tag_column].values))
    n_tags = len(tags)
    if verbose:
        print("Number of sentences: ", len(df.groupby([sentence_column])))
        print("Number of words in the dataset: ", n_words)
        print("Tags:", tags)
        print("Number of Labels: ", n_tags)


    # retrieve all sentences
    getter = SentenceGetter(df, word_column, tag_column, sentence_column)
    sentences = getter.sentences
    largest_sen = max(len(sen) for sen in sentences)
    if verbose: print('Longest sentence: {} words'.format(largest_sen))
    data = [list(zip(*s)) for s in sentences]
    X = [list(e[0]) for e in data]
    y = [list(e[1]) for e in data]
    return (X, y)




class SentenceGetter(object):
    """Class to Get the sentence in this format:
    [(Token_1, Part_of_Speech_1, Tag_1), ..., (Token_n, Part_of_Speech_1, Tag_1)]"""
    def __init__(self, data, word_column, tag_column, sentence_column):
        """Args:
            data is the pandas.DataFrame which contains the above dataset"""
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s[word_column].values.tolist(),
                                                           s[tag_column].values.tolist())]
        self.grouped = self.data.groupby(sentence_column).apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        """Return one sentence"""
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None




class NERSequence(Dataset):

    def __init__(self, x, y, batch_size=1, p=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.p = p
        self.prepare_called = False

    def prepare(self):
        if self.p is not None and not self.prepare_called:
            self.x, self.y = self.p.fix_tokenization(self.x, self.y)
        self.prepare_called = True
        return


    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.p.transform(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def get_lengths(self, idx):
        x_true, y_true = self[idx]
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def nsamples(self):
        return len(self.x)   


    def get_y(self):
        return self.y


    def xshape(self):
        return (len(self.x), self[0][0][0].shape[1]) 


    def nclasses(self):
        return len(self.p._label_vocab._id2token) 














