
from ...imports import *
from ... import utils as U
from . import preprocessor as pp
from .anago.models import BiLSTMCRF

BILSTM_CRF = 'bilstm-crf'
BILSTM = 'bilstm'
BILSTM_ELMO = 'bilstm-elmo'
BILSTM_CRF_ELMO = 'bilstm-crf-elmo'
BILSTM_TRANSFORMER = 'bilstm-bert'
SEQUENCE_TAGGERS = {
                     BILSTM: 'Bidirectional LSTM (https://arxiv.org/abs/1603.01360)',
                     BILSTM_TRANSFORMER: 'Bidirectional LSTM w/ BERT embeddings',
                     BILSTM_CRF: 'Bidirectional LSTM-CRF  (https://arxiv.org/abs/1603.01360)',
                     BILSTM_ELMO: 'Bidirectional LSTM w/ Elmo embeddings [English only]',
                     BILSTM_CRF_ELMO: 'Bidirectional LSTM-CRF w/ Elmo embeddings [English only]',
                     }
V1_ONLY_MODELS = [BILSTM_CRF, BILSTM_CRF_ELMO]
TRANSFORMER_MODELS = [BILSTM_TRANSFORMER]
ELMO_MODELS = [BILSTM_ELMO, BILSTM_CRF_ELMO]

def print_sequence_taggers():
    for k,v in SEQUENCE_TAGGERS.items():
        print("%s: %s" % (k,v))


def sequence_tagger(name, preproc, 
                    wv_path_or_url=None,
                    bert_model = 'bert-base-multilingual-cased',
                    bert_layers_to_use = [-1], # last hidden layer
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

        bert_model_name(str):  the name of the BERT model.  default: 'bert-base-multilingual-cased'
                               This parameter is only used if bilstm-bert is selected for name parameter.
                               The value of this parameter is a name of BERT model from here:
                                        https://huggingface.co/transformers/pretrained_models.html
                               or a community-uploaded BERT model from here:
                                        https://huggingface.co/models
                               Example values:
                                 bert-base-multilingual-cased:  Multilingual BERT (157 languages) - this is the default
                                 bert-base-cased:  English BERT
                                 bert-base-chinese: Chinese BERT
                                 distilbert-base-german-cased: German DistilBert
                                 albert-base-v2: English ALBERT model
                                 monologg/biobert_v1.1_pubmed: community uploaded BioBERT (pretrained on PubMed)

        bert_layers_to_use(list): indices of hidden layers to use.  default:[-1] # last layer
                                  To use the last 4 layers: use [-1, -2, -3, -4]
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

    # check BERT
    if name in TRANSFORMER_MODELS and not bert_model:
        raise ValueError('bert_model is required for bilstm-bert models')
    if name in TRANSFORMER_MODELS and DISABLE_V2_BEHAVIOR:
        raise ValueError('BERT and other transformer models cannot be used with DISABLE_v2_BEHAVIOR')

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

    # check for use_char=True
    if not DISABLE_V2_BEHAVIOR and preproc.p._use_char:
        # turn off masking due to open TF2 issue ##33148: https://github.com/tensorflow/tensorflow/issues/33148
        warnings.warn('Setting use_char=False:  character embeddings cannot be used in TF2 due to open TensorFlow 2 bug (#33148).\n' +\
                       'Add os.environ["DISABLE_V2_BEHAVIOR"] = "1" to the top of script if you really want to use it.')
        preproc.p._use_char=False

    if verbose:
        emb_names = []
        if wv_path_or_url is not None: 
            emb_names.append('word embeddings initialized with fasttext word vectors (%s)' % (os.path.basename(wv_path_or_url)))
        else:
            emb_names.append('word embeddings initialized randomly')
        if name in TRANSFORMER_MODELS: emb_names.append('BERT embeddings with ' + bert_model)
        if name in ELMO_MODELS: emb_names.append('Elmo embeddings for English')
        if preproc.p._use_char:  emb_names.append('character embeddings')
        if len(emb_names) > 1:
            print('Embedding schemes employed (combined with concatenation):')
        else:
            print('embedding schemes employed:')
        for emb_name in emb_names:
            print('\t%s' % (emb_name))
        print()

    # setup embedding
    if wv_path_or_url is not None:
        wv_model, word_embedding_dim = preproc.get_wv_model(wv_path_or_url, verbose=verbose)
    else:
        wv_model = None
    if name == BILSTM_CRF:
        use_crf = False if not DISABLE_V2_BEHAVIOR else True # fallback to bilstm 
    elif name == BILSTM_CRF_ELMO:
        use_crf = False if not DISABLE_V2_BEHAVIOR else True # fallback to bilstm
        preproc.p.activate_elmo()
    elif name == BILSTM:
        use_crf = False
    elif name == BILSTM_ELMO:
        use_crf = False
        preproc.p.activate_elmo()
    elif name == BILSTM_TRANSFORMER:
        use_crf = False
        preproc.p.activate_transformer(bert_model, layers=bert_layers_to_use)
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
                      use_char=preproc.p._use_char,
                      embeddings=wv_model,
                      use_elmo=preproc.p.elmo_is_activated(),
                      use_transformer_with_dim=preproc.p.get_transformer_dim())
    model, loss = model.build()
    model.compile(loss=loss, optimizer=U.DEFAULT_OPT)
    return model

