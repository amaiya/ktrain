
from ...imports import *
from ... import utils as U
from . import preprocessor as pp
from .preprocessor import NERPreprocessor



MAXLEN = 128
WORD_COL = 'Word'
TAG_COL = 'Tag'
SENT_COL = 'SentenceID'

def entities_from_gmb(train_filepath, 
                      word_column=WORD_COL,
                      tag_column=TAG_COL,
                      sentence_column=SENT_COL,
                      val_filepath=None,
                       maxlen=MAXLEN, 
                       encoding='latin1',
                       val_pct=0.1, verbose=1):
    """
    Loads sequence-labeled data from a CSV (or character-delmited) text file in GMB format.
    Format of file is that of the Groningen Meaning Bank (GMB) corpus
    available on Kaggle here: 
    https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/2

    Text file should have the following three columns representing:
    1) Word in a sentence
    2) Tag assigned to Word
    3) The ID of the Sentence to which the word belongs.
    The names of these columns can be specified with the
    word_column, tag_column, and sent_column fields.

    Example:

      SentenceID   Word     Tag    
      1            Paul     B-PER
      1            Newman   I-PER
      1            is       O
      1            a        O
      1            great    O
      1            actor    O
      1            !        O


    Args:
        train_filepath(str): file path to training CSV
        word_column(str): name of column containing the text
        tag_column(str): name of column containing lael
        sentence_column(str): name of column containing Sentence IDs
        val_filepath (str): file path to validation dataset
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        encoding(str): the encoding to use
        val_pct(float): Proportion of training to use for validation.
        verbose (boolean): verbosity
    """


    # create dataframe
    df = pd.read_csv(train_filepath, encoding=encoding)
    df = df.fillna(method="ffill")

    # process dataframe and instantiate NERPreprocessor
    (word2idx, tag2idx, trn_sentences) = pp.process_df(df, maxlen,
                                                       word_column=word_column,
                                                       tag_column=tag_column,
                                                       sentence_column=sentence_column,
                                                       verbose=verbose)
    preproc = NERPreprocessor(maxlen, word2idx, tag2idx)


    # preprocess train and validation sets
    if val_filepath is None:
        random.shuffle(trn_sentences)
        k = round(len(trn_sentences) * val_pct)
        val_sentences = trn_sentences[:k]
        trn_sentences = trn_sentences[k:]
    else:
        val_df = pd.read_csv(train_filepath, encoding=encoding)
        val_df = val_df.fillna(method="ffill")
        (_, _, val_sentences) = pp.process_df(val_df, maxlen,
                                              word_column=word_column,
                                              tag_column=tag_column,
                                              sentence_column=sentence_column,
                                              verbose=0)
    X_trn, y_trn = preproc.transform(trn_sentences)
    X_val, y_val = preproc.transform(val_sentences)
    return ( (X_trn, y_trn), (X_val, y_val), preproc)

        


def entities_from_conll2003(train_filepath, 
                            val_filepath=None,
                            maxlen=MAXLEN, 
                            encoding='latin1',
                            val_pct=0.1, verbose=1):
    """
    Loads sequence-labeled data from CSV file (no headers).
    Format of CSV file is that of the CoNLL 2003 shared NER task.
    Each word appars on a separate line and there is an empty line after
    each sentence.  The first item on each line is the word.  The 
    last item on each line is the tag or label assigned to word.
    Additional columns between the first and last items are ignored.
    More information: https://www.aclweb.org/anthology/W03-0419
    Example (each column is separated by a tab):

       Paul	B-PER
       Newman	I-PER
       is	O
       a	O
       great	O
       actor	O
       !	O

    Args:
        train_filepath(str): file path to training CSV
        val_filepath (str): file path to validation dataset
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        encoding(str): the encoding to use
        val_pct(float): Proportion of training to use for validation.
        verbose (boolean): verbosity
    """

    IndexTransformer = anago.preprocessing.IndexTransformer


    # set dataframe converter
    data_to_df = conll2003_to_df

    # create dataframe
    df = data_to_df(train_filepath, encoding=encoding)


    # process dataframe and instantiate NERPreprocessor
    x, y  = pp.process_df(df, 
                          word_column=WORD_COL,
                          tag_column=TAG_COL,
                          sentence_column=SENT_COL,
                          verbose=verbose)


    # get validation set
    if val_filepath is None:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=val_pct)
    else:
        val_df = data_to_df(val_filepath, encoding=encoding)
        x_train, y_train = x, y
        (x_valid, y_valid)  = pp.process_df(val_df,
                                            word_column=WORD_COL,
                                            tag_column=TAG_COL,
                                            sentence_column=SENT_COL,
                                            verbose=0)

    # preprocess and convert to generator
    p = IndexTransformer(use_char=True)
    preproc = NERPreprocessor(p)
    preproc.fit(x_train, y_train)
    trn = NERSequence(x_train, y_train, batch_size=U.DEFAULT_BS, p=p)
    val = NERSequence(x_valid, y_valid, batch_size=U.DEFAULT_BS, p=p)

    return ( trn, val, preproc)




def conll2003_to_df(filepath, encoding='latin1'):
    # read data and convert to dataframe
    sents, words, tags = [],  [], []
    sent_id = 0
    docstart = False
    with open(filepath, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                if line.startswith('-DOCSTART-'): 
                    docstart=True
                    continue
                else:
                    docstart=False
                    parts = line.split()
                    words.append(parts[0])
                    tags.append(parts[-1])
                    sents.append(sent_id)
            else:
                if not docstart:
                    sent_id +=1
    df = pd.DataFrame({SENT_COL: sents, WORD_COL : words, TAG_COL:tags})
    df = df.fillna(method="ffill")
    return df



class NERSequence(Sequence):

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

