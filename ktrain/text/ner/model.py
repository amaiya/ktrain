
from ...imports import *
from ... import utils as U
from .. import preprocessor as tpp
from ...keras_contrib.layers import CRF
from ...keras_contrib.metrics import crf_accuracy
from ...keras_contrib.losses import crf_loss


BILSTM_CRF = 'bilstm-crf'
SEQUENCE_TAGGERS = {
                     BILSTM_CRF: 'Bidirectional LSTM-CRF  (https://arxiv.org/abs/1603.01360)',
                     }

def print_sequence_taggers():
    for k,v in SEQUENCE_TAGGERS.items():
        print("%s: %s" % (k,v))


def sequence_tagger(name, preproc, verbose=1):
    """
    Build and return a sequence tagger (i.e., named entity recognizer).

    Args:
        name (string): one of:
                      - 'bilstm-crf' for Bidirectional LSTM-CRF model
        preproc(NERPreprocessor):  an instance of NERPreprocessor
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    
    BiLSTMCRF =  anago.models.BiLSTMCRF

    if name == BILSTM_CRF:
        model = BiLSTMCRF(char_embedding_dim=25,
                          word_embedding_dim=100,
                          char_lstm_size=25,
                          word_lstm_size=100,
                          char_vocab_size=preproc.p.char_vocab_size,
                          word_vocab_size=preproc.p.word_vocab_size,
                          num_labels=preproc.p.label_size,
                          dropout=0.5,
                          use_char=preproc.p._use_char,
                          use_crf=True)
        model, loss = model.build()
        model.compile(loss=loss, optimizer='adam')
        return model
    else:
        raise ValueError('Invalid value for name:  %s' % (name))





