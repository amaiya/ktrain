
from ...imports import *
from ... import utils as U
from .. import preprocessor as tpp
from . import preprocessor as pp
from .anago.models import BiLSTMCRF
from .anago.utils import filter_embeddings

BILSTM_CRF = 'bilstm-crf'
SEQUENCE_TAGGERS = {
                     BILSTM_CRF: 'Bidirectional LSTM-CRF  (https://arxiv.org/abs/1603.01360)',
                     }

def print_sequence_taggers():
    for k,v in SEQUENCE_TAGGERS.items():
        print("%s: %s" % (k,v))


def sequence_tagger(name, preproc, 
                    word_embedding_dim=100,
                    char_embedding_dim=25,
                    word_lstm_size=100,
                    char_lstm_size=25,
                    fc_dim=100,
                    dropout=0.5,
                    verbose=1):
    """
    Build and return a sequence tagger (i.e., named entity recognizer).

    Args:
        name (string): one of:
                      - 'bilstm-crf' for Bidirectional LSTM-CRF model
        preproc(NERPreprocessor):  an instance of NERPreprocessor
        embeddings(str): Currently, either None or 'cbow' is supported
                         If 'cbow' is specified, pretrained word vectors
                         are automatically downloaded to <home>/ktran_data
                         and used as weights in the Embedding layer.
                         If None, random embeddings used.
        word_embedding_dim (int): word embedding dimensions.
        char_embedding_dim (int): character embedding dimensions.
        word_lstm_size (int): character LSTM feature extractor output dimensions.
        char_lstm_size (int): word tagger LSTM output dimensions.
        fc_dim (int): output fully-connected layer size.
        dropout (float): dropout rate.

        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    



    emb_model = None
    if preproc.e == pp.W2V:
        if verbose: print('pretrained %s word embeddings will be used with bilstm-crf' % (preproc.e))
        word_embedding_dim = 300
        emb_dict = tpp.load_wv(verbose=verbose)
        emb_model = filter_embeddings(emb_dict, preproc.p._word_vocab.vocab, word_embedding_dim)

    if name == BILSTM_CRF:
        model = BiLSTMCRF(char_embedding_dim=char_embedding_dim,
                          word_embedding_dim=word_embedding_dim,
                          char_lstm_size=char_lstm_size,
                          word_lstm_size=word_lstm_size,
                          fc_dim=fc_dim,
                          char_vocab_size=preproc.p.char_vocab_size,
                          word_vocab_size=preproc.p.word_vocab_size,
                          num_labels=preproc.p.label_size,
                          dropout=dropout,
                          use_char=preproc.p._use_char,
                          use_crf=True,
                          embeddings=emb_model)
        model, loss = model.build()
        #loss = crf_loss
        model.compile(loss=loss, optimizer=U.DEFAULT_OPT)
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
