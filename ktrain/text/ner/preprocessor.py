from ...imports import *
from ... import utils as U
from ...preprocessor import Preprocessor

PAD='[PAD]'
UNK = '[UNK]'
#tokenizer_filter = rs='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


class NERPreprocessor(Preprocessor):
    """
    NER preprocessing base class
    """

    def __init__(self, maxlen, max_features, classes, word2idx):
        self.max_features = max_features
        self.maxlen = maxlen
        self.c = classes
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}


    def get_preprocessor(self):
        return self.word2idx


    def get_classes(self):
        return self.c


    def preprocess(self, sentence, filter_padding=True):
        unk_id = self.word2idx[UNK]
        pad_id = self.word2idx[PAD]
        s = tokenize(sentence)
        x = [self.word2idx.get(w, unk_id) for w in s]
        x = sequence.pad_sequences(maxlen=self.maxlen, sequences=[x],
                                   padding='post', value=pad_id)
        x = np.array(x)
        return x


    def undo(self, seq):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        pad_id = self.word2idx[PAD]
        #s = text_to_word_sequence(sentence, lower=False)
        x = [self.idx2word.get(wid, UNK) for wid in seq if wid != pad_id]
        return x


    def undo_val(self, val_data, val_id=0):
        """
        undoes preprocessing for a particular entry 
        in a validatio nset.
        """
        pad_id = self.word2idx[PAD]
        sentence = self.undo(val_data[0][val_id])
        tags = [self.c[tid] for tid in np.argmax(val_data[1][val_id], axis=-1) if tid != pad_id]
        return list(zip(sentence, tags))

