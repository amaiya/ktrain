from ...imports import *
from ... import utils as U
from ...preprocessor import Preprocessor

PAD='[PAD]'
UNK = '[UNK]'
OTHER = 'O'

#tokenizer_filter = rs='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


class NERPreprocessor(Preprocessor):
    """
    NER preprocessing base class
    """

    def __init__(self, maxlen, word2idx, tag2idx):
        self.maxlen = maxlen
        self.w2idx = word2idx
        self.t2idx = tag2idx
        self.idx2w = {i: w for w, i in word2idx.items()}
        self.c = [k for k,v in sorted(tag2idx.items(), key=lambda kv: kv[1])]
        self.max_features = len(word2idx)


    def check(self):
        if self.w2idx is None: raise Exception('perprocess_train must be called')


    def get_preprocessor(self):
        return self.w2idx


    def get_classes(self):
        return self.c


    def preprocess(self, sentence, filter_padding=True):
        self.check()

        unk_id = self.w2idx[UNK]
        pad_id = self.w2idx[PAD]
        s = tokenize(sentence)
        x = [self.w2idx.get(w, unk_id) for w in s]
        x = sequence.pad_sequences(maxlen=self.maxlen, sequences=[x],
                                   padding='post', value=pad_id)
        x = np.array(x)
        return x


    def undo(self, seq):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        self.check()

        pad_id = self.w2idx[PAD]
        #s = text_to_word_sequence(sentence, lower=False)
        x = [self.idx2w.get(wid, UNK) for wid in seq if wid != pad_id]
        return x


    def undo_val(self, val_data, val_id=0):
        """
        undoes preprocessing for a particular entry 
        in a validatio nset.
        """
        self.check()
        pad_id = self.w2idx[PAD]
        sentence = self.undo(val_data[0][val_id])
        tags = [self.c[tid] for tid in np.argmax(val_data[1][val_id], axis=-1) if tid != pad_id]
        return list(zip(sentence, tags))



    def transform(self, sentences_with_labels):
        """
        process tagged-sentences into sequences of word IDs and sequences of tag IDs
        """


        unk_id = self.w2idx[UNK]
        pad_id = self.w2idx[PAD]
        o_id = self.t2idx[OTHER]

        # preprocess words and tags
        pad = sequence.pad_sequences
        X = [[self.w2idx.get(w[0], unk_id) for w in s] for s in sentences_with_labels]
        X = pad(maxlen=self.maxlen, sequences=X, padding="post", value=self.w2idx[PAD])
        y = [[self.t2idx.get(w[1], o_id) for w in s] for s in sentences_with_labels]
        y = pad(maxlen=self.maxlen, sequences=y, padding="post", value=self.t2idx[PAD])
        y = [to_categorical(i, num_classes=len(self.c)) for i in y]  # n_tags+1(PAD)
        y = np.array(y)

        # split into training and validation
        #X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=val_pct)
        return (X, y)




def process_df(df, maxlen, 
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

    # convert words to IDs
    w2idx = {w: i + 2 for i, w in enumerate(words)}
    w2idx[UNK] = 1 # Unknown words
    w2idx[PAD] = 0 # Padding

    # convert tags to IDs
    tags.sort()
    t2idx = {t: i+1 for i, t in enumerate(tags)}
    t2idx[PAD] = 0

    # retrieve all sentences
    getter = SentenceGetter(df, word_column, tag_column, sentence_column)
    sentences = getter.sentences
    largest_sen = max(len(sen) for sen in sentences)
    if verbose: print('Longest sentence: {} words'.format(largest_sen))

    return (w2idx, t2idx, sentences)




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






