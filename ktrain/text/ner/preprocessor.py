from ...imports import *
from ... import utils as U
from ...preprocessor import Preprocessor

OTHER = 'O'

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


    def preprocess(self, sentence, filter_padding=True):

        #s = tokenize(sentence)
        #x = [self.w2idx.get(w, unk_id) for w in s]
        #x = sequence.pad_sequences(maxlen=self.maxlen, sequences=[x],
                                   #padding='post', value=pad_id)
        #x = np.array(x)
        #return x
        return


    def undo(self, seq):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        #x = [self.idx2w.get(wid, UNK) for wid in seq if wid != pad_id]
        #return x
        return None


    def undo_val(self, val_data, val_id=0):
        """
        undoes preprocessing for a particular entry 
        in a validatio nset.
        """
        #sentence = self.undo(val_data[0][val_id])
        #tags = [self.c[tid] for tid in np.argmax(val_data[1][val_id], axis=-1) if tid != pad_id]
        #return list(zip(sentence, tags))
        return None



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






