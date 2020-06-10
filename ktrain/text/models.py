from ..imports import *
from .. import utils as U
from . import preprocessor as tpp


NBSVM = 'nbsvm'
FASTTEXT = 'fasttext'
LOGREG = 'logreg'
BIGRU = 'bigru'
STANDARD_GRU = 'standard_gru'
BERT = 'bert'
DISTILBERT = tpp.DISTILBERT
HUGGINGFACE_MODELS = [DISTILBERT]
LINREG = 'linreg'
TEXT_CLASSIFIERS = {
                    FASTTEXT: "a fastText-like model [http://arxiv.org/pdf/1607.01759.pdf]",
                    LOGREG:  "logistic regression using a trainable Embedding layer",
                    NBSVM:  "NBSVM model [http://www.aclweb.org/anthology/P12-2018]",
                    BIGRU:  'Bidirectional GRU with pretrained fasttext word vectors [https://fasttext.cc/docs/en/crawl-vectors.html]',
                    STANDARD_GRU: 'simple 2-layer GRU with randomly initialized embeddings',
                    BERT:  'Bidirectional Encoder Representations from Transformers (BERT) [https://arxiv.org/abs/1810.04805]',
                    DISTILBERT:  'distilled, smaller, and faster BERT from Hugging Face [https://arxiv.org/abs/1910.01108]',
                    } 

TEXT_REGRESSION_MODELS = {
                    FASTTEXT: "a fastText-like model [http://arxiv.org/pdf/1607.01759.pdf]",
                    LINREG:  "linear text regression using a trainable Embedding layer",
                    BIGRU:  'Bidirectional GRU with pretrained English word vectors [https://arxiv.org/abs/1712.09405]',
                    STANDARD_GRU: 'simple 2-layer GRU with randomly initialized embeddings',
                    BERT:  'Bidirectional Encoder Representations from Transformers (BERT) [https://arxiv.org/abs/1810.04805]',
                    DISTILBERT:  'distilled, smaller, and faster BERT from Hugging Face [https://arxiv.org/abs/1910.01108]',
                    }



def print_text_classifiers():
    for k,v in TEXT_CLASSIFIERS.items():
        print("%s: %s" % (k,v))

def print_text_regression_models():
    for k,v in TEXT_REGRESSION_MODELS.items():
        print("%s: %s" % (k,v))


def calc_pr(y_i, x, y, b):
    idx = np.argwhere((y==y_i)==b)
    ct = x[idx[:,0]].sum(0)+1
    tot = ((y==y_i)==b).sum()+1
    return ct/tot

def calc_r(y_i, x, y):
    return np.log(calc_pr(y_i, x, y, True) / calc_pr(y_i, x, y, False))


def _text_model(name, train_data, preproc=None, multilabel=None, classification=True, metrics=['accuracy'], verbose=1):
    """
    Build and return a text classification or text regression model.

    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'logreg' for logistic regression
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model
        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train) or ktrain.Dataset instance
                            returned from one of the texts_from_* functions
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        classification(bool): If True, will build a text classificaton model.
                              Otherwise, a text regression model will be returned.
        metrics(list): list of metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    # check arguments
    if not isinstance(train_data, tuple) and not U.is_huggingface_from_data(train_data):
        err ="""
            Please pass training data in the form of a tuple of numpy.ndarrays
            or data returned from a ktrain texts_from* function.
            """
        raise Exception(err)

    if not isinstance(preproc, tpp.TextPreprocessor):
        msg = 'The preproc argument is required.'
        msg += ' The preproc arg should be an instance of TextPreprocessor, which is '
        msg += ' the third return value from texts_from_folder, texts_from_csv, etc.'
        #warnings.warn(msg, FutureWarning)
        raise ValueError(msg)
    if name == BIGRU and preproc.ngram_count() != 1:
        raise ValueError('Data should be processed with ngram_range=1 for bigru model.')
    is_bert = U.bert_data_tuple(train_data)
    if (is_bert and name != BERT) or (not is_bert and name == BERT):
        raise ValueError("if '%s' is selected model, then preprocess_mode='%s' should be used and vice versa" % (BERT, BERT))
    is_huggingface = U.is_huggingface(data=train_data)
    if (is_huggingface and name not in HUGGINGFACE_MODELS) or (not is_huggingface and name in HUGGINGFACE_MODELS):
        raise ValueError('you are using a Hugging Face transformer model but did not preprocess as such (or vice versa)')
    if is_huggingface and preproc.name != name:
        raise ValueError('you preprocessed for %s but want to build a %s model' % (preproc.name, name))
 
    if not classification: # regression
        if metrics is None or metrics==['accuracy']: metrics=['mae']
        num_classes = 1
        multilabel = False
        loss_func = 'mse'
        activation = None
        max_features = preproc.max_features
        features = None
        maxlen = U.shape_from_data(train_data)[1]
        U.vprint('maxlen is %s' % (maxlen), verbose=verbose)
    else:                 # classification
        if metrics is None: metrics = ['accuracy']
        # set number of classes and multilabel flag
        num_classes = U.nclasses_from_data(train_data)

        # determine multilabel
        if multilabel is None:
            multilabel = U.is_multilabel(train_data)
        if multilabel and name in [NBSVM, LOGREG]:
            warnings.warn('switching to fasttext model, as data suggests '
                          'multilabel classification from data.')
            name = FASTTEXT
        U.vprint("Is Multi-Label? %s" % (multilabel), verbose=verbose)

        # set loss and activations
        loss_func = 'categorical_crossentropy'
        activation = 'softmax'
        if multilabel:
            loss_func = 'binary_crossentropy'
            activation = 'sigmoid'

        # determine number of classes, maxlen, and max_features
        max_features = preproc.max_features if preproc is not None else None
        features = set()
        if not is_bert and not is_huggingface:
            U.vprint('compiling word ID features...', verbose=verbose)
            x_train = train_data[0]
            y_train = train_data[1]
            if isinstance(y_train[0], int): raise ValueError('train labels should not be in sparse format')

            for x in x_train:
                features.update(x)
            #max_features = len(features)
            if max_features is None: 
                max_features = max(features)+1
                U.vprint('max_features is %s' % (max_features), verbose=verbose)
        maxlen = U.shape_from_data(train_data)[1]
        U.vprint('maxlen is %s' % (maxlen), verbose=verbose)


    # return appropriate model
    if name in [LOGREG, LINREG]:
        model =  _build_logreg(num_classes, 
                            maxlen,
                            max_features,
                            features,
                            loss_func=loss_func,
                            activation=activation, metrics=metrics, verbose=verbose)

    elif name==FASTTEXT:
        model = _build_fasttext(num_classes, 
                            maxlen,
                            max_features,
                            features,
                            loss_func=loss_func,
                            activation=activation, metrics=metrics, verbose=verbose)
    elif name==STANDARD_GRU:
        model = _build_standard_gru(num_classes, 
                                    maxlen,
                                    max_features,
                                    features,
                                    loss_func=loss_func,
                                    activation=activation, metrics=metrics, verbose=verbose)
    elif name==NBSVM:
        model = _build_nbsvm(num_classes, 
                            maxlen,
                            max_features,
                            features,
                            loss_func=loss_func,
                            activation=activation, metrics=metrics, verbose=verbose,
                            train_data=train_data)

    elif name==BIGRU:
        (tokenizer, tok_dct) = preproc.get_preprocessor()
        model = _build_bigru(num_classes, 
                            maxlen,
                            max_features,
                            features,
                            loss_func=loss_func,
                            activation=activation, metrics=metrics, verbose=verbose,
                            tokenizer=tokenizer,
                            preproc=preproc)
    elif name == BERT:
        model =  _build_bert(num_classes, 
                            maxlen,
                            max_features,
                            features,
                            loss_func=loss_func,
                            activation=activation, metrics=metrics, verbose=verbose,
                            preproc=preproc)
    elif name in HUGGINGFACE_MODELS:
        model =  _build_transformer(num_classes, 
                                   maxlen,
                                   max_features,
                                   features,
                                   loss_func=loss_func,
                                   activation=activation, metrics=metrics, verbose=verbose,
                                   preproc=preproc)

    else:
        raise ValueError('name for textclassifier is invalid')
    U.vprint('done.', verbose=verbose)
    return model



def _build_logreg(num_classes,
                  maxlen, max_features, features,
                 loss_func='categorical_crossentropy',
                 activation = 'softmax', metrics=['accuracy'], verbose=1):

    embedding_matrix = np.ones((max_features, 1))
    embedding_matrix[0] = 0

    # set up the model
    inp = Input(shape=(maxlen,))
    r = Embedding(max_features, 1, input_length=maxlen, 
                  weights=[embedding_matrix], trainable=False)(inp)
    x = Embedding(max_features, num_classes, input_length=maxlen, 
                  embeddings_initializer='glorot_normal')(inp)
    x = dot([x,r], axes=1)
    x = Flatten()(x)
    if activation: x = Activation(activation)(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss=loss_func,
                  optimizer=U.DEFAULT_OPT,
                  metrics=metrics)
    return model


def _build_bert(num_classes,
                maxlen, max_features, features,
               loss_func='categorical_crossentropy',
               activation = 'softmax', metrics=['accuracy'],  verbose=1,
               preproc=None):
    if preproc is None: raise ValueError('preproc is missing')
    lang = preproc.lang
    if lang is None: raise ValueError('lang is missing')
    config_path = os.path.join(tpp.get_bert_path(lang=lang), 'bert_config.json')
    checkpoint_path = os.path.join(tpp.get_bert_path(lang=lang), 'bert_model.ckpt')

    model = keras_bert.load_trained_model_from_checkpoint(
                                    config_path,
                                    checkpoint_path,
                                    training=True,
                                    trainable=True,
                                    seq_len=maxlen)
    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs = Dense(units=num_classes, activation=activation)(dense)
    model = Model(inputs, outputs)
    model.compile(loss=loss_func,
                  optimizer=U.DEFAULT_OPT,
                  metrics=metrics)
    return model


def _build_transformer(num_classes,
                      maxlen, max_features, features,
                      loss_func='categorical_crossentropy',
                      activation = 'softmax', metrics=['accuracy'],  verbose=1,
                      preproc=None):
    if not isinstance(preproc, tpp.TransformersPreprocessor): 
        raise ValueError('preproc must be instance of %s' % (str(tpp.TransformersPreprocessor)))

    #model = preproc.model_type.from_pretrained(preproc.model_name, num_labels=num_classes)
    #loss_map =  {'categorical_crossentropy': keras.losses.CategoricalCrossentropy(from_logits=True),
                 #'binary_crossentropy': keras.losses.BinaryCrossentropy(from_logits=True), 
                 #'mse': 'mse'}
    #model.compile(loss=loss_map[loss_func],
                  #optimizer=keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08),
                  #metrics=metrics)
    if loss_func == 'mse':
        if preproc.get_classes(): 
            raise Exception('This is supposed to be regression problem, but preproc.get_classes() is not empty. ' +\
                            'Something went wrong.  Please open a GitHub issue.')
            if len(preproc.get_classes()) != num_classes:
                raise Exception('Number of labels from preproc.get_classes() is not equal to num_classes. ' +\
                                'Something went wrong. Please open GitHub issue.')
    else:
        if not preproc.get_classes():
            raise Exception('This is supposed to be a classification problem, but preproc.get_classes() is empty. ' +\
                            'Something went wrong.  Please open a GitHub issue.')
    return preproc.get_model()



def _build_nbsvm(num_classes,
                 maxlen, max_features, features,
                 loss_func='categorical_crossentropy',
                 activation = 'softmax', metrics=['accuracy'], verbose=1, train_data=None):
    if train_data is None: raise ValueError('train_data is required')
    x_train = train_data[0]
    y_train = train_data[1]
    Y = np.array([np.argmax(row) for row in y_train])
    num_columns = max(features) + 1
    num_rows = len(x_train)

    # set up document-term matrix
    X = csr_matrix((num_rows, num_columns), dtype=np.int8)
    #X = lil_matrix((num_rows, num_columns), dtype=np.int8)
    U.vprint('building document-term matrix... this may take a few moments...',
            verbose=verbose)
    r_ids = []
    c_ids = []
    data = []
    for row_id, row in enumerate(x_train):
        trigger = 10000
        trigger_end =  min(row_id+trigger, num_rows)
        if row_id % trigger == 0: 
            U.vprint('rows: %s-%s' % (row_id+1, trigger_end), 
                     verbose=verbose)
        tmp_c_ids = [column_id for column_id in row if column_id >0 ]
        num = len(tmp_c_ids)
        c_ids.extend(tmp_c_ids)
        r_ids.extend([row_id]* num)
        data.extend([1] * num)
    X = csr_matrix( (data,(r_ids,c_ids)), shape=(num_rows, num_columns) )

    # compute Naive Bayes log-count ratios
    U.vprint('computing log-count ratios...', verbose=verbose)
    nbratios = np.stack([calc_r(i, X, Y).A1 for i in range(num_classes)])
    nbratios = nbratios.T
    embedding_matrix = np.zeros((num_columns, num_classes))
    for i in range(1, num_columns): 
        for j in range(num_classes):
            embedding_matrix[i,j] = nbratios[i,j]

    # set up the model
    inp = Input(shape=(maxlen,))
    r = Embedding(num_columns, num_classes, input_length=maxlen, 
                  weights=[embedding_matrix], trainable=False)(inp)
    x = Embedding(num_columns, 1, input_length=maxlen, 
                  embeddings_initializer='glorot_normal')(inp)
    x = dot([r,x], axes=1)
    x = Flatten()(x)
    x = Activation(activation)(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss=loss_func,
                  optimizer=U.DEFAULT_OPT,
                  metrics=metrics)
    return model


def _build_fasttext(num_classes,
                 maxlen, max_features, features,
                 loss_func='categorical_crossentropy',
                 activation = 'softmax', metrics=['accuracy'],  verbose=1):

    model = Sequential()
    model.add(Embedding(max_features, 64, input_length=maxlen))
    model.add(SpatialDropout1D(0.25))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation))
    model.compile(loss=loss_func, optimizer=U.DEFAULT_OPT, metrics=metrics)

    return model


def _build_standard_gru(num_classes,
                 maxlen, max_features, features,
                 loss_func='categorical_crossentropy',
                 activation = 'softmax', metrics=['accuracy'], verbose=1):
    model = Sequential()
    model.add(Embedding(max_features, 256, input_length = maxlen))
    model.add(GRU(256, dropout=0.9, return_sequences=True))
    model.add(GRU(256, dropout=0.9))
    model.add(Dense(num_classes, activation=activation))
    model.compile(loss=loss_func, optimizer=U.DEFAULT_OPT, metrics=metrics)
    return model



def _build_bigru(num_classes,
                  maxlen, max_features, features,
                 loss_func='categorical_crossentropy',
                 activation = 'softmax', metrics=['accuracy'], verbose=1,
                 tokenizer=None, preproc=None):


    if tokenizer is None: raise ValueError('bigru requires valid Tokenizer object')
    if preproc is None: raise ValueError('bigru requires valid preproc')
    if not hasattr(preproc, 'lang') or preproc.lang is None: 
        lang = 'en'
    else:
        lang = preproc.lang
    wv_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.%s.300.vec.gz" % (lang.split('-')[0])
    if verbose: print('word vectors will be loaded from: %s' % (wv_url))



    # setup pre-trained word embeddings
    embed_size = 300
    U.vprint('processing pretrained word vectors...', verbose=verbose)
    embeddings_index = tpp.load_wv(wv_path_or_url=wv_url, verbose=verbose)
    word_index = tokenizer.word_index 
    #nb_words = min(max_features, len(word_index))
    nb_words = max_features
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    # define model
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(num_classes, activation=activation)(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=loss_func,
                  optimizer=U.DEFAULT_OPT,
                  metrics=metrics)
    return model



def text_classifier(name, train_data, preproc=None, multilabel=None, metrics=['accuracy'], verbose=1):
    """
    Build and return a text classification model.

    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'logreg' for logistic regression using embedding layers
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model

        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train) or ktrain.Dataset instance
                            returned from one of the texts_from_* functions
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        metrics(list): metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    if name not in TEXT_CLASSIFIERS:
        raise ValueError('invalid name for text classification: %s' % (name)) 
    if preproc is not None and not preproc.get_classes():
        raise ValueError('preproc.get_classes() is empty, but required for text classification')
    return _text_model(name, train_data, preproc=preproc,
                       multilabel=multilabel, classification=True, metrics=metrics, verbose=verbose)


def text_regression_model(name, train_data, preproc=None, metrics=['mae'],  verbose=1):
    """
    Build and return a text regression model.

    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'linreg' for linear regression using embedding layers
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model

        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train)
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        metrics(list): metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    if name not in TEXT_REGRESSION_MODELS:
        raise ValueError('invalid name for text classification: %s' % (name) )
    if preproc is not None and preproc.get_classes():
        raise ValueError('preproc.get_classes() is supposed to be empty for text regression tasks')
    return _text_model(name, train_data, preproc=preproc,
                      multilabel=False, classification=False, metrics=metrics, verbose=verbose)
