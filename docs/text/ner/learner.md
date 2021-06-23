Module ktrain.text.ner.learner
==============================

Classes
-------

`NERLearner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Learner for Sequence Taggers.

    ### Ancestors (in MRO)

    * ktrain.core.GenLearner
    * ktrain.core.Learner
    * abc.ABC

    ### Methods

    `save_model(self, fpath)`
    :   a wrapper to model.save

    `top_losses(self, n=4, val_data=None, preproc=None)`
    :   Computes losses on validation set sorted by examples with top losses
        Args:
          n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
          val_data:  optional val_data to use instead of self.val_data
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

    `validate(self, val_data=None, print_report=True, class_names=[])`
    :   Validate text sequence taggers

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