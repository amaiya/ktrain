
from ...imports import *
from ... import utils as U
from .. import preprocessor as tpp
from ...keras_contrib.layers import CRF
from ...keras_contrib.metrics import crf_accuracy
from ...keras_contrib.losses import crf_loss


BILSTM_CRF = 'bilstm-crf'
SEQUENCE_TAGGERS = {
                     BILSTM_CRF: 'Bidirectional LSTM-CRF  (https://arxiv.org/abs/1508.01991)',
                     }
EMBEDDING = 40


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
    
    if name == BILSTM_CRF:
        #inp = Input(shape=(preproc.maxlen,))
        #model = Embedding(input_dim=preproc.max_features, output_dim=EMBEDDING,
                          #input_length=preproc.maxlen, mask_zero=True)(inp)
        #model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   #recurrent_dropout=0.1))(model)
        #model = TimeDistributed(Dense(50, activation="relu"))(model)
        #crf = CRF(len(preproc.get_classes()))
        #out = crf(model)
        #model = Model(inp, out)
        #model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_accuracy])
        #return model

        # mask_zero causes problems for CRF in Keras v2.2.5
        # https://github.com/keras-team/keras-contrib/issues/498


        #mask_zero = True
        #if keras.__version__ == '2.2.5': 
            #warnings.warn('Due to bug in keras_contrib and Keras 2.2.5, '+
                          #'we are setting mask_zero=False.')
            #mask_zero = False 
        mask_zero=False
        inp = Input(shape=(preproc.maxlen,))
        model = Embedding(input_dim=preproc.max_features, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                          input_length=preproc.maxlen, mask_zero=mask_zero)(inp)  # default: 20-dim embedding
        model = Bidirectional(LSTM(units=512, return_sequences=True,
                                   recurrent_dropout=0.2, dropout=0.2))(model)  # variational biLSTM
        model = TimeDistributed(Dense(512, activation="relu", 
                                      kernel_initializer='he_normal'))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(len(preproc.get_classes()))  # CRF layer, n_tags+1(PAD)
        out = crf(model)
        model = Model(inp, out)
        model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_accuracy])
        return model
    else:
        raise ValueError('Invalid value for name:  %s' % (name))





