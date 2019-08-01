from scipy import sparse
import numpy as np
from keras.preprocessing.image import NumpyArrayIterator
from keras.preprocessing.image import Iterator
from keras.utils import Sequence, to_categorical

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras.preprocessing.image import ImageDataGenerator


DEFAULT_BS = 32
DEFAULT_ES = 5 
DEFAULT_ROP = 2 


def is_multilabel(data):
    data_arg_check(val_data=data, val_required=True)
    if is_iter(data):
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
        if hasattr(data, 'image_shape'): return data.image_shape
        elif hasattr(data, 'x'):
            return data.x.shape[1:]
        else:
            try:
                return data[0][0].shape[1:]
            except:
                raise Exception(err_msg)
    else:
        try:
            return data[0].shape
        except:
            raise Exception(err_msg)


def ondisk(data):
    ondisk = is_iter(data) and (type(data).__name__ != 'NumpyArrayIterator')
    return ondisk


def nsamples_from_data(data):
    err_msg = 'could not determine number of samples from %s' % (type(data))
    if is_iter(data):
        if hasattr(data, 'samples'):
            return data.samples
        elif hasattr(data, 'n'):
            return data.n
        else:
            raise Exception(err_msg)
    else:
        try:
            return len(data[0])
        except:
            raise Exception(err_msg)


def nclasses_from_data(data):
    if is_iter(data):
        if hasattr(data, 'classes'):
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
        if hasattr(data, 'classes'):
            return to_categorical(data.classes)
        elif hasattr(data, 'data'):
            return data.data
        elif hasattr(data, 'y'):
            return data.y
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
    #return isinstance(data, DirectoryIterator)
    return isinstance(data, Sequence)
    #return isinstance(data, Iterator)


def data_arg_check(train_data=None, val_data=None, train_required=False, val_required=False,
                   ndarray_only=False):

    if train_required and train_data is None:
        raise ValueError('train_data is required')
    if val_required and val_data is None:
        raise ValueError('val_data is required')
    if train_data is not None and not is_iter(train_data, ndarray_only):
        if not isinstance(train_data, tuple) or len(train_data) != 2 or \
           type(train_data[0]) is not np.ndarray or type(train_data[1]) is not np.ndarray:
            err_msg = 'train_data must be tuple of numpy.ndarrays'
            if not ndarray_only: err_msg += ' or an instance of Iterator'
            raise ValueError(err_msg)
    if val_data is not None and not is_iter(val_data, ndarray_only):
        if not isinstance(val_data, tuple) or len(val_data) != 2 or \
           type(val_data[0]) is not np.ndarray or type(val_data[1]) is not np.ndarray:
            err_msg = 'val_data must be tuple of numpy.ndarrays'
            if not ndarray_only: err_msg += ' or an instance of Iterator'
            raise ValueError(err_msg)



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
