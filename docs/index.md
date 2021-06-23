Module ktrain
=============

Sub-modules
-----------
* ktrain.core
* ktrain.data
* ktrain.graph
* ktrain.imports
* ktrain.lroptimize
* ktrain.models
* ktrain.predictor
* ktrain.preprocessor
* ktrain.tabular
* ktrain.text
* ktrain.utils
* ktrain.version
* ktrain.vision

Functions
---------

    
`get_learner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Returns a Learner instance that can be used to tune and train Keras models.
    
    model (Model):        A compiled instance of keras.engine.training.Model
    train_data (tuple or generator): Either a: 
                                   1) tuple of (x_train, y_train), where x_train and 
                                      y_train are numpy.ndarrays or 
                                   2) Iterator
    val_data (tuple or generator): Either a: 
                                   1) tuple of (x_test, y_test), where x_testand 
                                      y_test are numpy.ndarrays or 
                                   2) Iterator
                                   Note: Should be same type as train_data.
    batch_size (int):              Batch size to use in training. default:32
    eval_batch_size(int):  batch size used by learner.predict
                           only applies to validaton data during training if
                           val_data is instance of utils.Sequence.
                           default:32
    workers (int): number of cpu processes used to load data.
                   This is ignored unless train_data/val_data is an instance of 
                   tf.keras.preprocessing.image.DirectoryIterator or tf.keras.preprocessing.image.DataFrameIterator. 
    use_multiprocessing(bool):  whether or not to use multiprocessing for workers
                               This is ignored unless train_data/val_data is an instance of 
                               tf.keras.preprocessing.image.DirectoryIterator or tf.keras.preprocessing.image.DataFrameIterator.

    
`get_predictor(model, preproc, batch_size=32)`
:   Returns a Predictor instance that can be used to make predictions on
    unlabeled examples.  Can be saved to disk and reloaded as part of a 
    larger application.
    
    Args
        model (Model):        A compiled instance of keras.engine.training.Model
        preproc(Preprocessor):   An instance of TextPreprocessor,ImagePreprocessor,
                                 or NERPreprocessor.
                                 These instances are returned from the data loading
                                 functions in the ktrain vision and text modules:
    
                                 ktrain.vision.images_from_folder
                                 ktrain.vision.images_from_csv
                                 ktrain.vision.images_from_array
                                 ktrain.text.texts_from_folder
                                 ktrain.text.texts_from_csv
                                 ktrain.text.ner.entities_from_csv
        batch_size(int):    batch size to use.  default:32

    
`load_predictor(fpath, batch_size=32, custom_objects=None)`
:   Loads a previously saved Predictor instance
    Args
      fpath(str): predictor path name (value supplied to predictor.save)
                  From v0.16.x, this is always the path to a folder.
                  Pre-v0.16.x, this is the base name used to save model and .preproc instance.
      batch_size(int): batch size to use for predictions. default:32
      custom_objects(dict): custom objects required to load model.
                            This is useful if you compiled the model with a custom loss function, for example.
                            For models included with ktrain as is, this is populated automatically
                            and can be disregarded.

    
`release_gpu_memory(device=0)`
:   Relase GPU memory allocated by Tensorflow
    Source: 
    https://stackoverflow.com/questions/51005147/keras-release-memory-after-finish-training-process

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