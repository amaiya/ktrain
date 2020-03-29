from .imports import *
from .data import Dataset


#------------------------------------------------------------------------------
# KTRAIN DEFAULTS
#------------------------------------------------------------------------------
DEFAULT_BS = 32
DEFAULT_ES = 5 
DEFAULT_ROP = 2 
DEFAULT_OPT = 'adam'
DEFAULT_TRANSFORMER_LAYERS = [-2] # second-to-last hidden state
DEFAULT_TRANSFORMER_MAXLEN = 512
DEFAULT_TRANSFORMER_NUM_SPECIAL = 2



#------------------------------------------------------------------------------
# DATA/MODEL INSPECTORS
#------------------------------------------------------------------------------

def is_classifier(model):
    """
    checks for classification and mutlilabel from model
    """
    is_classifier = False
    is_multilabel = False

    # get loss name
    loss = model.loss
    if callable(loss): 
        if hasattr(loss, '__name__'):
            loss = loss.__name__
        elif hasattr(loss, 'name'):
            loss = loss.name
        else:
            raise Exception('could not get loss name')

    # check for classification
    if loss in ['categorical_crossentropy',
                 'sparse_categorical_crossentropy',
                 'binary_crossentropy']:
        is_classifier = True

    # check for multilabel
    if loss == 'binary_crossentropy':
        if is_huggingface(model=model):
            is_multilabel = True
        else:
            last = model.layers[-1]
            output_shape = last.output_shape
            mult_output = True if len(output_shape) ==2 and output_shape[1] >  1 else False
            if ( (hasattr(last, 'activation') and isinstance(last.activation, type(sigmoid))) or\
               isinstance(last, type(sigmoid)) ) and mult_output:
                is_multilabel = True
    return (is_classifier, is_multilabel)


def is_huggingface(model=None, data=None):
    """
    check for hugging face transformer model
    from  model and/or data
    """
    huggingface = False
    if model is not None and is_huggingface_from_model(model):
        huggingface = True
    elif data is not None and is_huggingface_from_data(data):
        huggingface = True
    return huggingface


def is_huggingface_from_model(model):
    return 'transformers.modeling_tf' in str(type(model))


def is_huggingface_from_data(data):
    return type(data).__name__ in ['TransformerDataset']



def is_ner(model=None, data=None):
    ner = False
    if data is None:
        warnings.warn('is_ner only detects CRF-based NER models when data is None')
    if model is not None and is_crf(model):
        ner = True
    elif data is not None and is_ner_from_data(data):
        ner = True
    return ner 


def is_crf(model):
    """
    checks for CRF sequence tagger.
    """
    #loss = model.loss
    #if callable(loss): 
        #if hasattr(loss, '__name__'):
            #loss = loss.__name__
        #elif hasattr(loss, 'name'):
            #loss = loss.name
        #else:
            #raise Exception('could not get loss name')
    #return loss == 'crf_loss' or 'CRF.loss_function' in str(model.loss)
    return type(model.layers[-1]).__name__ == 'CRF'


#def is_ner_from_model(model):
    #"""
    #checks for sequence tagger.
    #Curently, only checks for a CRF-based sequence tagger
    #"""
    #loss = model.loss
    #if callable(loss): 
        #if hasattr(loss, '__name__'):
            #loss = loss.__name__
        #elif hasattr(loss, 'name'):
            #loss = loss.name
        #else:
            #raise Exception('could not get loss name')

    #return loss == 'crf_loss' or 'CRF.loss_function' in str(model.loss)


def is_ner_from_data(data):
    return type(data).__name__ == 'NERSequence'


def is_nodeclass(model=None, data=None):
    result = False
    if data is not None and type(data).__name__ == 'NodeSequenceWrapper':
        result = True
    return result


def is_imageclass_from_data(data):
    return type(data).__name__ in ['DirectoryIterator', 'DataFrameIterator', 'NumpyArrayIterator']


def is_multilabel(data):
    """
    checks for multilabel from data
    """
    data_arg_check(val_data=data, val_required=True)
    if is_iter(data): 
        if is_ner(data=data): return False   # NERSequence
        elif is_nodeclass(data=data): return False  # NodeSequenceWrapper
        multilabel = False
        for idx, v in enumerate(data):
            if idx >= 16: break
            y = v[1]
            if len(y.shape) == 1 or y.shape[1] == 1: return False
            total_per_batch = np.sum(y, axis=1)
            if any(i>1 for i in total_per_batch):
                multilabel=True
                break
        return multilabel
    else:
        if len(data[1].shape) == 1: return False
        multilabel = False
        for idx, y in enumerate(data[1]):
            if idx >= 128: break
            total = sum(y)
            if total > 1:
                multilabel=True
                break
        return multilabel



def shape_from_data(data):
    err_msg = 'could not determine shape from %s' % (type(data))
    if is_iter(data):
        if isinstance(data, Dataset): return data.xshape()
        elif hasattr(data, 'image_shape'): return data.image_shape          # DirectoryIterator/DataFrameIterator
        elif hasattr(data, 'x'):                                            # NumpyIterator
            return data.x.shape[1:]
        else:
            try:
                return data[0][0].shape[1:]
            except:
                raise Exception(err_msg)
    else:
        try:
            if type(data[0]) == list: # BERT-style tuple
                return data[0][0].shape
            else:
                return data[0].shape  # standard tuple
        except:
            raise Exception(err_msg)


def ondisk(data):
    if hasattr(data, 'ondisk'): return data.ondisk()

    ondisk = is_iter(data) and \
             (type(data).__name__ not in  ['NumpyArrayIterator'])
    return ondisk


def nsamples_from_data(data):
    err_msg = 'could not determine number of samples from %s' % (type(data))
    if is_iter(data):
        if isinstance(data, Dataset): return data.nsamples()
        elif hasattr(data, 'samples'):  # DirectoryIterator/DataFrameIterator
            return data.samples
        elif hasattr(data, 'n'):     # DirectoryIterator/DataFrameIterator/NumpyIterator
            return data.n
        else:
            raise Exception(err_msg)
    else:
        try:
            if type(data[0]) == list: # BERT-style tuple
                return len(data[0][0])
            else:
                return len(data[0])   # standard tuple
        except:
            raise Exception(err_msg)


def nclasses_from_data(data):
    if is_iter(data):
        if isinstance(data, Dataset): return data.nclasses()
        elif hasattr(data, 'classes'):   # DirectoryIterator
            return len(set(data.classes))
        else:
            try:
                return data[0][1].shape[1]  # DataFrameIterator/NumpyIterator
            except:
                raise Exception('could not determine number of classes from %s' % (type(data)))
    else:
        try:
            return data[1].shape[1]
        except:
                raise Exception('could not determine number of classes from %s' % (type(data)))


def y_from_data(data):
    if is_iter(data):
        if isinstance(data, Dataset): return data.get_y()
        elif hasattr(data, 'classes'): # DirectoryIterator
            return to_categorical(data.classes)
        elif hasattr(data, 'labels'):  # DataFrameIterator
            return data.labels
        elif hasattr(data, 'y'): # NumpyArrayIterator
            return to_categorical(data.y)
        else:
            raise Exception('could not determine number of classes from %s' % (type(data)))
    else:
        try:
            return data[1]
        except:
            raise Exception('could not determine number of classes from %s' % (type(data)))


def is_iter(data, ignore=False):
    if ignore: return True
    iter_classes = ["NumpyArrayIterator", "DirectoryIterator", "DataFrameIterator"]
    return data.__class__.__name__ in iter_classes or isinstance(data, Dataset)



def data_arg_check(train_data=None, val_data=None, train_required=False, val_required=False,
                   ndarray_only=False):
    if train_required and train_data is None:
        raise ValueError('train_data is required')
    if val_required and val_data is None:
        raise ValueError('val_data is required')
    if train_data is not None and not is_iter(train_data, ndarray_only):
        if bad_data_tuple(train_data):
            err_msg = 'data must be tuple of numpy.ndarrays'
            if not ndarray_only: err_msg += ' or an instance of ktrain.Dataset'
            raise ValueError(err_msg)
    if val_data is not None and not is_iter(val_data, ndarray_only):
        if bad_data_tuple(val_data):
            err_msg = 'data must be tuple of numpy.ndarrays or BERT-style tuple'
            if not ndarray_only: err_msg += ' or an instance of Iterator'
            raise ValueError(err_msg)
    return


def bert_data_tuple(data):
    """
    checks if data tuple is BERT-style format
    """
    if is_iter(data): return False
    if type(data[0]) == list and len(data[0]) == 2 and \
       type(data[0][0]) is np.ndarray and type(data[0][1]) is np.ndarray and \
       type(data[1]) is np.ndarray and np.count_nonzero(data[0][1]) == 0:
           return True
    else:
        return False


def bad_data_tuple(data):
    """
    Checks for standard tuple or BERT-style tuple
    """
    if not isinstance(data, tuple) or len(data) != 2 or \
       type(data[0]) not in [np.ndarray, list] or \
       (type(data[0]) in [list] and type(data[0][0]) is not np.ndarray) or \
       type(data[1]) is not np.ndarray: 
        return True
    else:
        return False



#------------------------------------------------------------------------------
# PLOTTING UTILITIES
#------------------------------------------------------------------------------


# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    #if type(ims[0]) is np.ndarray:
        #ims = np.array(ims).astype(np.uint8)
        #if (ims.shape[-1] != 3):
            #ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#------------------------------------------------------------------------------
# DOWNLOAD UTILITIES
#------------------------------------------------------------------------------


def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True,  verify=False)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            #print(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()


def get_ktrain_data():
    home = os.path.expanduser('~')
    ktrain_data = os.path.join(home, 'ktrain_data')
    if not os.path.isdir(ktrain_data):
        os.mkdir(ktrain_data)
    return ktrain_data



#------------------------------------------------------------------------------
# MISC UTILITIES
#------------------------------------------------------------------------------

def is_tf_keras():
    if keras.__name__ == 'keras':
        is_tf_keras = False
    elif keras.__name__ in ['tensorflow.keras', 'tensorflow.python.keras', 'tensorflow_core.keras'] or\
            keras.__version__[-3:] == '-tf':
        is_tf_keras = True
    else:
        raise KeyError('Cannot detect if using keras or tf.keras.')
    return is_tf_keras


def vprint(s=None, verbose=1):
    if not s: s = '\n'
    if verbose:
        print(s)


def add_headers_to_df(fname_in, header_dict, fname_out=None):

    df = pd.read_csv(fname_in, header=None)
    df.rename(columns=header_dict, inplace=True)
    if fname_out is None:
        name, ext = os.path.splitext(fname_in)
        name += '-headers'
        fname_out = name + '.' + ext
    df.to_csv(fname_out, index=False) # save to new csv file
    return


def get_random_colors(n, name='hsv', hex_format=True):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    cmap = plt.cm.get_cmap(name, n)
    result = []
    for i in range(n):
        color = cmap(i)
        if hex_format: color = rgb2hex(color)
        result.append(color)
    return np.array(result)


