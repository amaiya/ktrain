from ..preprocessor import Preprocessor
from keras.preprocessing import image
from keras.preprocessing import sequence
class TextPreprocessor(Preprocessor):
    """
    Text preprocessing
    """

    def __init__(self, tok, tok_dct, classes, maxlen, ngram_range=1):

        self.tok = tok
        self.tok_dct = tok_dct
        self.c = classes
        self.maxlen = maxlen
        self.ngram_range = ngram_range


    def get_preprocessor(self):
        return (self.tok, self.tok_dct)


    def get_classes(self):
        return self.c


    def preprocess(self, texts):
        texts = self.tok.texts_to_sequences(texts)
        texts = self.add_ngram(texts)
        texts = sequence.pad_sequences(texts, maxlen=self.maxlen)
        return texts


    def undo(self, doc):
        """
        undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words
        """
        dct = self.tok.index_word
        return " ".join([dct[wid] for wid in doc if wid != 0 and wid in dct])


    def ngram_count(self):
        if not self.tok_dct: return 1
        s = set()
        for k in self.tok_dct.keys():
            s.add(len(k))
        return max(list(s))


    def add_ngram(self, sequences):
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
        return new_sequences




