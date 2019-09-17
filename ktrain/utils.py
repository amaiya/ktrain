from .imports import *

DEFAULT_BS = 32
DEFAULT_ES = 5 
DEFAULT_ROP = 2 
DEFAULT_OPT = 'adam'


def is_classifier(model):
    """
    checks for classification and mutlilabel from model
    """
    is_classifier = False
    is_multilabel = False

    # get loss name
    loss = model.loss
    if callable(loss): loss = loss.__name__

    # check for classification
    if loss in ['categorical_crossentropy',
                 'sparse_categorical_crossentropy',
                 'binary_crossentropy']:
        is_classifier = True

    # check for multilabel
    if loss == 'binary_crossentropy':
        last = model.layers[-1]
        output_shape = last.output_shape
        mult_output = True if len(output_shape) ==2 and output_shape[1] >  1 else False
        if ( (hasattr(last, 'activation') and isinstance(last.activation, type(sigmoid))) or\
           isinstance(last, type(sigmoid)) ) and mult_output:
            is_multilabel = True
    return (is_classifier, is_multilabel)


def is_ner(model=None, data=None):
    ner = False
    if model is not None and is_ner_from_model(model):
        ner = True
    elif data is not None and is_ner_from_data(data):
        ner = True
    return ner 


def is_ner_from_model(model):
    """
    checks for sequence tagger.
    Curently, only checks for a CRF-based sequence tagger
    """
    loss = model.loss
    if callable(loss): loss = loss.__name__
    return loss == 'crf_loss' or 'CRF.loss_function' in str(model.loss)


def is_crf(model):
    """
    This is currently simpley an alias for is_ner_from_model
    """
    return is_ner_from_model(model)


def is_ner_from_data(data):
    return type(data).__name__ == 'NERSequence'
        


def is_multilabel(data):
    """
    checks for multilabel from data
    """
    data_arg_check(val_data=data, val_required=True)
    if is_iter(data):
        if is_ner(data=data): return False   # NERSequence
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
        if is_ner(data=data): return (len(data.x), data[0][0][0].shape[1])  # NERSequence
        elif hasattr(data, 'image_shape'): return data.image_shape
        elif hasattr(data, 'x'):
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
    ondisk = is_iter(data) and \
             (type(data).__name__ not in  ['NumpyArrayIterator', 'NERSequence'])
    return ondisk


def nsamples_from_data(data):
    err_msg = 'could not determine number of samples from %s' % (type(data))
    if is_iter(data):
        if is_ner(data=data): return len(data.x)
        elif hasattr(data, 'samples'):
            return data.samples
        elif hasattr(data, 'n'):
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
        if is_ner(data=data): return len(data.p._label_vocab._id2token)
        elif hasattr(data, 'classes'):
            return len(set(data.classes))
        else:
            try:
                return data[0][1].shape[1]
            except:
                raise Exception('could not determine number of classes from %s' % (type(data)))
    else:
        try:
            return data[1].shape[1]
        except:
                raise Exception('could not determine number of classes from %s' % (type(data)))


def y_from_data(data):
    if is_iter(data):
        if is_ner(data=data): return data.y
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



def vprint(s=None, verbose=1):
    if not s: s = '\n'
    if verbose:
        print(s)


def is_iter(data, ignore=False):
    if ignore: return True
    iter_classes = ["NumpyArrayIterator", "DirectoryIterator",
                    "DataFrameIterator", "Iterator", "Sequence", 
                    "NERSequence"]
    return data.__class__.__name__ in iter_classes


def data_arg_check(train_data=None, val_data=None, train_required=False, val_required=False,
                   ndarray_only=False):
    if train_required and train_data is None:
        raise ValueError('train_data is required')
    if val_required and val_data is None:
        raise ValueError('val_data is required')
    if train_data is not None and not is_iter(train_data, ndarray_only):
        if bad_data_tuple(train_data):
            err_msg = 'data must be tuple of numpy.ndarrays'
            if not ndarray_only: err_msg += ' or an instance of Iterator'
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
    if type(data[0]) == list and type(data[0][0]) is np.ndarray and\
       type(data[0] is np.ndarray):
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



def set_row_csr(A, row_idx, new_row):
    '''
    Replace a row in a CSR sparse matrix A.

    Parameters
    ----------
    A: csr_matrix
        Matrix to change
    row_idx: int
        index of the row to be changed
    new_row: np.array
        list of new values for the row of A

    Returns
    -------
    None (the matrix A is changed in place)

    Prerequisites
    -------------
    The row index shall be smaller than the number of rows in A
    The number of elements in new row must be equal to the number of columns in matrix A
    '''
    assert sparse.isspmatrix_csr(A), 'A shall be a csr_matrix'
    assert row_idx < A.shape[0], \
            'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
            .format(row_idx, A.shape[0])
    try:
        N_elements_new_row = len(new_row)
    except TypeError:
        msg = 'Argument new_row shall be a list or numpy array, is now a {0}'\
        .format(type(new_row))
        raise AssertionError(msg)
    N_cols = A.shape[1]
    assert N_cols == N_elements_new_row, \
            'The number of elements in new row ({0}) must be equal to ' \
            'the number of columns in matrix A ({1})' \
            .format(N_elements_new_row, N_cols)

    idx_start_row = A.indptr[row_idx]
    idx_end_row = A.indptr[row_idx + 1]
    additional_nnz = N_cols - (idx_end_row - idx_start_row)

    A.data = np.r_[A.data[:idx_start_row], new_row, A.data[idx_end_row:]]
    A.indices = np.r_[A.indices[:idx_start_row], np.arange(N_cols), A.indices[idx_end_row:]]
    A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):] + additional_nnz]





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


def get_img_fit_flow(image_config, fit_smpl_size, directory, target_size, batch_size, shuffle):   
    '''                                                                            
    Sample the generators to get fit data    
    image_config  dict   holds the vars for data augmentation & 
    fit_smpl_size float  subunit multiplier to get the sample size for normalization
    
    directory     str    folder of the images
    target_size   tuple  images processed size
    batch_size    str    
    shuffle       bool
    '''                                                                            
    if 'featurewise_std_normalization' in image_config and image_config['image_config']:                                      
       img_gen = ImageDataGenerator()                                              
       batches = img_gen.flow_from_directory(                                      
          directory=directory,                                                     
          target_size=target_size,                                                 
          batch_size=batch_size,                                                   
          shuffle=shuffle,                                                         
        )                                                                          
       fit_samples = np.array([])                                                  
       fit_samples.resize((0, target_size[0], target_size[1], 3))                  
       for i in range(batches.samples/batch_size):                                 
           imgs, labels = next(batches)                                            
           idx = np.random.choice(imgs.shape[0], batch_size*fit_smpl_size, replace=False)     
           np.vstack((fit_samples, imgs[idx]))                                     
    new_img_gen = ImageDataGenerator(**image_config)                               
    if 'featurewise_std_normalization' in image_config and image_config['image_config']:                                      
        new_img_gen.fit(fit_samples)                                               
    return new_img_gen.flow_from_directory(                                        
       directory=directory,                                                        
       target_size=target_size,                                                    
       batch_size=batch_size,                                                      
       shuffle=shuffle,                                                            
    )

def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True,  verify=False)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
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


def add_headers_to_df(fname_in, header_dict, fname_out=None):

    df = pd.read_csv(fname_in, header=None)
    df.rename(columns=header_dict, inplace=True)
    if fname_out is None:
        name, ext = os.path.splitext(fname_in)
        name += '-headers'
        fname_out = name + '.' + ext
    df.to_csv(fname_out, index=False) # save to new csv file
    return
