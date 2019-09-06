
from ...imports import *
from ... import utils as U
from .preprocessor import NERPreprocessor, PAD, UNK



MAXLEN = 128

def entities_from_csv(train_filepath, 
                      word_column='Word',
                      tag_column='Tag',
                      sentence_column='SentenceID',
                      val_filepath=None,
                       maxlen=MAXLEN, 
                       val_pct=0.1, verbose=1):
    """
    Loads sequence-labeled data from CSV file. 
    Args:
        train_filepath(str): file path to training CSV
        word_column(str): name of column containing the text
        tag_column(list): name of column containing lael
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        val_pct(float): Proportion of training to use for validation.
        verbose (boolean): verbosity
    """


    # read data
    data = pd.read_csv(train_filepath, encoding="latin1")
    data = data.fillna(method="ffill")
    words = list(set(data[word_column].values))
    n_words = len(words)
    tags = list(set(data[tag_column].values))
    n_tags = len(tags)

    # print some stats
    if verbose:
        print("Number of sentences: ", len(data.groupby([sentence_column])))
        print("Number of words in the dataset: ", n_words)
        print("Tags:", tags)
        print("Number of Labels: ", n_tags)

    # retrieve all sentences
    getter = SentenceGetter(data, word_column, tag_column, sentence_column)
    #sent = getter.get_next()

    # Get all the sentences
    sentences = getter.sentences
    largest_sen = max(len(sen) for sen in sentences)
    if verbose: print('Longest sentence: {} words'.format(largest_sen))

    # convert words to IDs
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx[UNK] = 1 # Unknown words
    word2idx[PAD] = 0 # Padding
    # Vocabulary Key:token_index -> Value:word
    idx2word = {i: w for w, i in word2idx.items()}

    # convert tags to IDs
    tags.sort()
    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx[PAD] = 0
    # Vocabulary Key:tag_index -> Value:Label/Tag
    idx2tag = {i: w for w, i in tag2idx.items()}


    # preprocess words and tags
    pad = sequence.pad_sequences
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad(maxlen=maxlen, sequences=X, padding="post", value=word2idx[PAD])

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad(maxlen=maxlen, sequences=y, padding="post", value=tag2idx[PAD])
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]  # n_tags+1(PAD)
    y = np.array(y)

    # split into training and validation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=val_pct)

    # instantiate Preprocessor
    class_names =  [k for k,v in sorted(tag2idx.items(), key=lambda kv: kv[1])]
    preproc = NERPreprocessor(maxlen, len(word2idx), class_names, word2idx)

    return (X_tr, y_tr), (X_te, y_te), preproc



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





