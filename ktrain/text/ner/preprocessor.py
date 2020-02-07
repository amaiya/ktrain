from ...imports import *
from ... import utils as U
from ...preprocessor import Preprocessor
from ...data import Dataset

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

    def __init__(self, p, embeddings=None):
        self.p = p
        self.c = p._label_vocab._id2token
        self.e = embeddings
        if embeddings is not None and embeddings not in SUPPORTED_EMBEDDINGS:
            raise ValueError('%s is not a supported word embedding type' % (embeddings))



    def get_preprocessor(self):
        return self.p


    def get_classes(self):
        return self.c

    def get_embedding_name(self):
        return self.e


    def preprocess(self, sentences):
        if type(sentences) != list:
            raise ValueError('Param sentences must be a list of strings')
        X = []
        y = []
        for s in sentences:
            tokens = tokenize(s)
            X.append(tokens)
            y.append([OTHER] * len(tokens))
        nerseq = NERSequence(X, y, p=self.p)
        return nerseq


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














