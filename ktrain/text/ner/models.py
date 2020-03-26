
from ...imports import *
from ... import utils as U
from . import preprocessor as pp
from .anago.models import BiLSTMCRF

BILSTM_CRF = 'bilstm-crf'
BILSTM = 'bilstm'
SEQUENCE_TAGGERS = {
                     BILSTM_CRF: 'Bidirectional LSTM-CRF  (https://arxiv.org/abs/1603.01360)',
                     BILSTM: 'Bidirectional LSTM (no CRF layer)  (https://arxiv.org/abs/1603.01360)',
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
                      - 'bilstm' for Bidirectional LSTM (no CRF layer)
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
    
    if name not in SEQUENCE_TAGGERS:
        raise ValueError('invalid name: %s' % (name))

    if not DISABLE_V2_BEHAVIOR:
       warnings.warn("Please add os.environ['DISABLE_V2_BEHAVIOR'] = '1' at top of your script or notebook")
       msg = "\nktrain uses the CRF module from keras_contrib, which is not yet\n" +\
             "fully compatible with TensorFlow 2. You can still use the BiLSTM-CRF model\n" +\
             "in ktrain for sequence tagging with TensorFlow 2, but you must add the\n" +\
             "following to the top of your script or notebook BEFORE you import ktrain:\n\n" +\
             "import os\n" +\
             "os.environ['DISABLE_V2_BEHAVIOR'] = '1'\n"
       raise Exception(msg)

    # setup embedding
    if preproc.wv_path_or_url is not None:
        emb_model, word_embedding_dim = preproc.get_embed_model(verbose=verbose)
    else:
        emb_model = None
    #mask_zero = False if not DISABLE_V2_BEHAVIOR else True  # https://github.com/tensorflow/tensorflow/issues/33148
    mask_zero = True
    if name == BILSTM_CRF:
        use_crf = True
        if not DISABLE_V2_BEHAVIOR:
            use_crf = False  
            warnings.warn('Falling back to BiLSTM because DISABLE_V2_BEHAVIOR=False')
            msg = "\nktrain uses the CRF module from keras_contrib, which is not yet\n" +\
                  "fully compatible with TensorFlow 2. You can still use the BiLSTM-CRF model\n" +\
                  "in ktrain for sequence tagging with TensorFlow 2, but you must add the\n" +\
                  "following to the top of your script or notebook BEFORE you import ktrain:\n\n" +\
                  "import os\n" +\
                  "os.environ['DISABLE_V2_BEHAVIOR'] = '1'\n\n" +\
                  "For this run, a vanilla BiLSTM model (with no CRF layer) will be used."
            print(msg)
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
                          use_crf=use_crf,
                          embeddings=emb_model,
                          mask_zero=mask_zero,
                          use_elmo=preproc.use_elmo)
    else:
        use_crf = False
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
                          use_crf=use_crf,
                          embeddings=emb_model,
                          mask_zero=mask_zero,
                          use_elmo=preproc.use_elmo)

    model, loss = model.build()
    model.compile(loss=loss, optimizer=U.DEFAULT_OPT)
    return model

