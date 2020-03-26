
from ...imports import *
from ... import utils as U
from . import preprocessor as pp
from .anago.models import BiLSTMCRF

BILSTM_CRF = 'bilstm-crf'
BILSTM = 'bilstm'
BILSTM_ELMO = 'bilstm-elmo'
BILSTM_CRF_ELMO = 'bilstm-crf-elmo'
SEQUENCE_TAGGERS = {
                     BILSTM_CRF: 'Bidirectional LSTM-CRF  (https://arxiv.org/abs/1603.01360)',
                     BILSTM: 'Bidirectional LSTM (no CRF layer)  (https://arxiv.org/abs/1603.01360)',
                     BILSTM_CRF: 'Bidirectional LSTM-CRF w/ Elmo embeddings (English only)',
                     BILSTM_ELMO: 'Bidirectional LSTM w/ Elmo embeddings (English only)'
                     }
V1_ONLY_MODELS = [BILSTM_CRF, BILSTM_CRF_ELMO]

def print_sequence_taggers():
    for k,v in SEQUENCE_TAGGERS.items():
        print("%s: %s" % (k,v))


def sequence_tagger(name, preproc, 
                    wv_path_or_url=None,
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
        wv_path_or_url(str): either a URL or file path toa fasttext word vector file (.vec or .vec.zip or .vec.gz)
                             Example valid values for wv_path_or_url:

                               Randomly-initialized word embeeddings:
                                 set wv_path_or_url=None
                               English pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
                               Chinese pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
                               Russian pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
                               Dutch pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz


                             See these two Web pages for a full list of URLs to word vector files for 
                             different languages:
                                1.  https://fasttext.cc/docs/en/english-vectors.html (for English)
                                2.  https://fasttext.cc/docs/en/crawl-vectors.html (for non-English langages)

                            Default:None (randomly-initialized word embeddings are used)

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

    # check CRF
    if not DISABLE_V2_BEHAVIOR and name in V1_ONLY_MODELS:
        warnings.warn('Falling back to BiLSTM (no CRF) because DISABLE_V2_BEHAVIOR=False')
        msg = "\nIMPORTANT NOTE: ktrain uses the CRF module from keras_contrib, which is not yet\n" +\
              "fully compatible with TensorFlow 2. You can still use the BiLSTM-CRF model\n" +\
              "in ktrain for sequence tagging with TensorFlow 2, but you must add the\n" +\
              "following to the top of your script or notebook BEFORE you import ktrain:\n\n" +\
              "import os\n" +\
              "os.environ['DISABLE_V2_BEHAVIOR'] = '1'\n\n" +\
              "For this run, a vanilla BiLSTM model (with no CRF layer) will be used.\n"
        print(msg)
        name = BILSTM if name == BILSTM_CRF else BILSTM_ELMO

    # check use_char=True
    if not DISABLE_V2_BEHAVIOR and preproc.p._use_char:
        warnings.warn('Disabling character embeddings:  use_char=True changed to use_char=False')
        msg = '\nIMPORTANT NOTE:  Due to an open TensorFlow 2 issue (#33148), character-level embeddings fail \n' +\
                'when mask_zero=True in model\n' +\
                'Since mask_zero=True is important for NER, we are setting use_char to False for this run.\n '  +\
                'See this for more information on this issue: https://github.com/tensorflow/tensorflow/issues/33148\n'
        print(msg)
        preproc.p._use_char = False


    # setup embedding
    if wv_path_or_url is not None:
        wv_model, word_embedding_dim = preproc.get_wv_model(wv_path_or_url, verbose=verbose)
    else:
        wv_model = None
    mask_zero = True
    if name == BILSTM_CRF:
        use_crf = False if not DISABLE_V2_BEHAVIOR else True # fallback to bilstm 
        use_elmo = False
    elif name == BILSTM_CRF_ELMO:
        use_crf = False if not DISABLE_V2_BEHAVIOR else True # fallback to bilstm
        use_elmo = True 
    elif name == BILSTM:
        use_crf = False
        use_elmo = False
    elif name == BILSTM_ELMO:
        use_crf = False
        use_elmo = True
    else:
        raise ValueError('Unsupported model name')
    model = BiLSTMCRF(char_embedding_dim=char_embedding_dim,
                      word_embedding_dim=word_embedding_dim,
                      char_lstm_size=char_lstm_size,
                      word_lstm_size=word_lstm_size,
                      fc_dim=fc_dim,
                      char_vocab_size=preproc.p.char_vocab_size,
                      word_vocab_size=preproc.p.word_vocab_size,
                      num_labels=preproc.p.label_size,
                      dropout=dropout,
                      use_crf=use_crf,
                      mask_zero=mask_zero,
                      use_char=preproc.p._use_char,
                      embeddings=wv_model,
                      use_elmo=use_elmo)
    model, loss = model.build()
    model.compile(loss=loss, optimizer=U.DEFAULT_OPT)
    return model

