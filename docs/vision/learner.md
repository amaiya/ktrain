Module ktrain.vision.learner
============================

Classes
-------

`ImageClassLearner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Main class used to tune and train Keras models for image classification.
    Main parameters are:
    
    model (Model): A compiled instance of keras.engine.training.Model
    train_data (Iterator): a Iterator instance for training set
    val_data (Iterator):   A Iterator instance for validation set

    ### Ancestors (in MRO)

    * ktrain.core.GenLearner
    * ktrain.core.Learner
    * abc.ABC

    ### Methods

    `view_top_losses(self, n=4, preproc=None, val_data=None)`
    :   Views observations with top losses in validation set.
        Args:
         n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
         preproc (Preprocessor): A TextPreprocessor or ImagePreprocessor.
                                 For some data like text data, a preprocessor
                                 is required to undo the pre-processing
                                 to correctly view raw data.
          val_data:  optional val_data to use instead of self.val_data
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.