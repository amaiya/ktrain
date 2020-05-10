from .imports import *


class Dataset:
    """
    Base class for custom datasets in ktrain.

    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.

    The signature of to_tfdataset is as follows:

    def to_tfdataset(self, train=True)

    See ktrain.text.preprocess.TransformerDataset as an example.
    """

    # required: used by ktrain.core.Learner instances
    def nsamples(self):
        raise NotImplemented

    # required: used by ktrain.core.Learner instances
    def get_y(self):
        raise NotImplemented

    # optional: to modify dataset between epochs (e.g., shuffle)
    def on_epoch_end(self):
        pass

    # optional
    def ondisk(self):
        """
        Is data being read from disk like with DirectoryIterators?
        """
        return False

    # optional: used only if invoking *_classifier functions
    def xshape(self):
        """
        shape of X
        Examples:
            for images: input_shape
            for text: (n_example, sequence_length)
        """
        raise NotImplemented
    
    # optional: used only if invoking *_classifier functions
    def nclasses(self):
        """
        Number of classes
        For classification problems: this is the number of labels
        Not used for regression problems
        """
        raise NotImplemented


class TFDataset(Dataset):
    """
    Wrapper for tf.data.Datasets
    """
    def __init__(self, tfdataset, n, y):
        """
        Args:
          tfdataset(tf.data.Dataset):  a tf.Dataset instance
          n(int): number of examples in dataset (cardinality, which can't reliably be extracted from tf.data.Datasets)
          y(np.ndarray): y values for each example - should be in the format expected by your moddel (e.g., 1-hot-encoded)
        """
        if not isinstance(tfdataset, tf.data.Dataset):
            raise ValueError('tfdataset must be a fully-configured tf.data.Dataset with batch_size, etc. set appropriately')
        self.tfdataset = tfdataset
        self.bs = next(tfdataset.as_numpy_iterator())[-1].shape[0] # extract batch_size from tfdataset
        self.n = n
        self.y = y

    @property
    def batch_size(self):
        return self.bs

    @batch_size.setter
    def batch_size(self, value):
        if value != self.bs:
            warnings.warn('batch_size parameter is ignored, as pre-configured batch_size of tf.data.Dataset is used')


    def nsamples(self):
        return self.n

    def get_y(self):
        return self.y

    def to_tfdataset(self, train=True):
        return self.tfdataset


class SequenceDataset(Dataset, Sequence):
    """
    Base class for custom datasets in ktrain.

    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.

    The signature of to_tfdataset is as follows:

    def to_tfdataset(self, training=True)

    See ktrain.text.preprocess.TransformerDataset as an example.
    """
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    # required by keras.utils.Sequence instances
    def __len__(self):
        raise NotImplemented

    # required by keras.utils.Sequence instances
    def __getitem__(self, idx):
        raise NotImplemented

        return False




class MultiArrayDataset(SequenceDataset):
    def __init__(self, x, y, batch_size=32, shuffle=True):
        # error checks
        err = False
        if type(x) == np.ndarray and len(x.shape) != 2: err = True
        elif type(x) == list:
            for d in x:
                if type(d) != np.ndarray or len(d.shape) != 2:
                    err = True
                    break
        else: err = True
        if err:
            raise ValueError('x must be a 2d numpy array or a list of 2d numpy arrays')
        if type(y) != np.ndarray:
            raise ValueError('y must be a numpy array')
        if type(x) == np.ndarray:
            x = [x]

        # set variables
        super().__init__(batch_size=batch_size)
        self.x, self.y = x, y
        self.indices = np.arange(self.x[0].shape[0])
        self.n_inputs = len(x)
        self.shuffle = shuffle


    def __len__(self):
        return math.ceil(self.x[0].shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for i in range(self.n_inputs):
            batch_x.append(self.x[i][inds])
        batch_y = self.y[inds]
        return tuple(batch_x), batch_y

    def nsamples(self):
        return self.x[0].shape[0]

    def get_y(self):
        return self.y

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

    def xshape(self):
        return self.x[0].shape

    def nclasses(self):
        return self.y.shape[1]

    def ondisk(self):
        return False

