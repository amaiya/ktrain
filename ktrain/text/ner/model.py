
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
        loss = crf_loss
        model.compile(loss=loss, optimizer='adam')
        return model
    else:
        raise ValueError('Invalid value for name:  %s' % (name))





def crf_nll(y_true, y_pred):
    """The negative log-likelihood for linear chain Conditional Random Field (CRF).

    This loss function is only used when the `layers.CRF` layer
    is trained in the "join" mode.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        A scalar representing corresponding to the negative log-likelihood.

    # Raises
        TypeError: If CRF is not the last layer.

    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.
    """

    crf, idx = y_pred._keras_history[:2]
    if crf._outbound_nodes:
        raise TypeError('When learn_model="join", CRF must be the last layer.')
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):
    """General CRF loss function depending on the learning mode.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        If the CRF layer is being trained in the join mode, returns the negative
        log-likelihood. Otherwise returns the categorical crossentropy implemented
        by the underlying Keras backend.

    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)
