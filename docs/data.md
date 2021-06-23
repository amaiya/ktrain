Module ktrain.data
==================

Classes
-------

`Dataset()`
:   Base class for custom datasets in ktrain.
    
    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.
    
    The signature of to_tfdataset is as follows:
    
    def to_tfdataset(self, train=True)
    
    See ktrain.text.preprocess.TransformerDataset as an example.

    ### Descendants

    * ktrain.data.SequenceDataset
    * ktrain.data.TFDataset

    ### Methods

    `get_y(self)`
    :

    `nclasses(self)`
    :   Number of classes
        For classification problems: this is the number of labels
        Not used for regression problems

    `nsamples(self)`
    :

    `on_epoch_end(self)`
    :

    `ondisk(self)`
    :   Is data being read from disk like with DirectoryIterators?

    `xshape(self)`
    :   shape of X
        Examples:
            for images: input_shape
            for text: (n_example, sequence_length)

`MultiArrayDataset(x, y, batch_size=32, shuffle=True)`
:   Base class for custom datasets in ktrain.
    
    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.
    
    The signature of to_tfdataset is as follows:
    
    def to_tfdataset(self, training=True)
    
    See ktrain.text.preprocess.TransformerDataset as an example.

    ### Ancestors (in MRO)

    * ktrain.data.SequenceDataset
    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Methods

    `get_y(self)`
    :

    `nsamples(self)`
    :

    `on_epoch_end(self)`
    :   Method called at the end of every epoch.

`SequenceDataset(batch_size=32)`
:   Base class for custom datasets in ktrain.
    
    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.
    
    The signature of to_tfdataset is as follows:
    
    def to_tfdataset(self, training=True)
    
    See ktrain.text.preprocess.TransformerDataset as an example.

    ### Ancestors (in MRO)

    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Descendants

    * ktrain.data.MultiArrayDataset
    * ktrain.graph.sg_wrappers.LinkSequenceWrapper
    * ktrain.graph.sg_wrappers.NodeSequenceWrapper
    * ktrain.tabular.preprocessor.TabularDataset
    * ktrain.text.ner.preprocessor.NERSequence
    * ktrain.text.preprocessor.TransformerDataset

`TFDataset(tfdataset, n, y)`
:   Wrapper for tf.data.Datasets
    
    Args:
      tfdataset(tf.data.Dataset):  a tf.Dataset instance
      n(int): number of examples in dataset (cardinality, which can't reliably be extracted from tf.data.Datasets)
      y(np.ndarray): y values for each example - should be in the format expected by your moddel (e.g., 1-hot-encoded)

    ### Ancestors (in MRO)

    * ktrain.data.Dataset

    ### Instance variables

    `batch_size`
    :

    ### Methods

    `get_y(self)`
    :

    `nsamples(self)`
    :

    `to_tfdataset(self, train=True)`
    :