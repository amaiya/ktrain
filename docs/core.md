Module ktrain.core
==================

Functions
---------

    
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

`ArrayLearner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Main class used to tune and train Keras models
    using Array data.  An objects of this class should be instantiated
    via the ktrain.get_learner method instead of directly.
    Main parameters are:
    
    
    model (Model):        A compiled instance of keras.engine.training.Model
    train_data (ndarray): A tuple of (x_train, y_train), where x_train and 
                          y_train are numpy.ndarrays.
    val_data (ndarray):   A tuple of (x_test, y_test), where x_test and 
                          y_test are numpy.ndarrays.

    ### Ancestors (in MRO)

    * ktrain.core.Learner
    * abc.ABC

    ### Descendants

    * ktrain.text.learner.BERTTextClassLearner

    ### Methods

    `fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, lr_decay=1, checkpoint_folder=None, early_stopping=None, verbose=1, class_weight=None, callbacks=[], steps_per_epoch=None)`
    :   Trains the model. By default, fit is simply a wrapper for model.fit.
        When cycle_len parameter is supplied, an SGDR learning rate schedule is used.
        Trains the model.
        
        lr (float): learning rate 
        n_cycles (int):  n_cycles
        cycle_len (int): If not None, decay learning rate over <cycle_len>
                         epochs until restarting/resetting learning rate to <lr>.
                         If None, lr remains constant
        cycle_mult (int): Increase cycle_len by factor of cycle_mult.
                          This will gradually elongate the cycle.
                          Has no effect if cycle_len is None.
        lr_decay(float): rate of decay of learning rate each cycle
        checkpoint_folder (string): Folder path in which to save the model weights 
                                   for each epoch.
                                   File name will be of the form: 
                                   weights-{epoch:02d}-{val_loss:.2f}.hdf5
        early_stopping (int):     If not None, training will automatically stop after this many 
                                  epochs of no improvement in validation loss.
                                  Upon completion, model will be loaded with weights from epoch
                                  with lowest validation loss.
        callbacks (list):         list of Callback instances to employ during training
        class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
        steps_per_epoch(int):    Steps per epoch. If None, then, math.ceil(num_samples/batch_size) is used.
                                 Ignored unless training dataset is generator (and in ArrayLearner instances).
        verbose (bool):           whether or not to show progress bar

    `layer_output(self, layer_id, example_id=0, use_val=False)`
    :   Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from training set, by default.

    `view_top_losses(self, n=4, preproc=None, val_data=None)`
    :   Views observations with top losses in validation set.
        Typically over-ridden by Learner subclasses.
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

`GenLearner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Main class used to tune and train Keras models
    using a Keras generator (e.g., DirectoryIterator).
    Objects of this class should be instantiated using the
    ktrain.get_learner function, rather than directly.
    
    Main parameters are:
    
    model (Model): A compiled instance of keras.engine.training.Model
    train_data (Iterator): a Iterator instance for training set
    val_data (Iterator):   A Iterator instance for validation set

    ### Ancestors (in MRO)

    * ktrain.core.Learner
    * abc.ABC

    ### Descendants

    * ktrain.graph.learner.LinkPredLearner
    * ktrain.graph.learner.NodeClassLearner
    * ktrain.text.learner.TransformerTextClassLearner
    * ktrain.text.ner.learner.NERLearner
    * ktrain.vision.learner.ImageClassLearner

    ### Methods

    `fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, lr_decay=1.0, checkpoint_folder=None, early_stopping=None, class_weight=None, callbacks=[], steps_per_epoch=None, verbose=1)`
    :   Trains the model. By default, fit is simply a wrapper for model.fit (for generators/sequences).
        When cycle_len parameter is supplied, an SGDR learning rate schedule is used.
        
        lr (float): learning rate 
        n_cycles (int):  n_cycles
        cycle_len (int): If not None, decay learning rate over <cycle_len>
                         epochs until restarting/resetting learning rate to <lr>.
                         If None, lr remains constant
        cycle_mult (int): Increase cycle_len by factor of cycle_mult.
                          This will gradually elongate the cycle.
                          Has no effect if cycle_len is None.
        lr_decay (float): rate of decay of learning reach each cycle.
                          Has no effect if cycle_len is None
        checkpoint_folder (string): Folder path in which to save the model weights 
                                   for each epoch.
                                   File name will be of the form: 
                                   weights-{epoch:02d}-{val_loss:.2f}.hdf5
        early_stopping (int):     If not None, training will automatically stop after this many 
                                  epochs of no improvement in validation loss.
                                  Upon completion, model will be loaded with weights from epoch
                                  with lowest validation loss.
        class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
        callbacks (list):         list of Callback instances to employ during training
        steps_per_epoch(int):    Steps per epoch. If None, then, math.ceil(num_samples/batch_size) is used.
        verbose (boolean):       whether or not to print progress bar

    `layer_output(self, layer_id, example_id=0, batch_id=0, use_val=False)`
    :   Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from first batch from training set, by default.

    `view_top_losses(self, n=4, preproc=None, val_data=None)`
    :   Views observations with top losses in validation set.
        Typically over-ridden by Learner subclasses.
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

`Learner(model, workers=1, use_multiprocessing=False)`
:   Abstract class used to tune and train Keras models. The fit method is
    an abstract method and must be implemented by subclasses.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * ktrain.core.ArrayLearner
    * ktrain.core.GenLearner

    ### Methods

    `autofit(self, lr, epochs=None, early_stopping=None, reduce_on_plateau=None, reduce_factor=2, cycle_momentum=True, max_momentum=0.95, min_momentum=0.85, monitor='val_loss', checkpoint_folder=None, class_weight=None, callbacks=[], steps_per_epoch=None, verbose=1)`
    :   Automatically train model using a default learning rate schedule shown to work well
        in practice.  By default, this method currently employs a triangular learning 
        rate policy (https://arxiv.org/abs/1506.01186).  
        During each epoch, this learning rate policy varies the learning rate from lr/10 to lr
        and then back to a low learning rate that is near-zero. 
        If epochs is None, then early_stopping and reduce_on_plateau are atomatically
        set to 6 and 3, respectively.
        
        Args:
            lr (float): optional initial learning rate.  If missing,
                       lr will be estimated automatically.
                       It is recommended that you estimate lr yourself by 
                       running lr_finder (and lr_plot) and visually inspect plot
                       for dramatic loss drop.
            epochs (int): Number of epochs.  If None, training will continue until
                          validation loss no longer improves after 5 epochs.
            early_stopping (int):     If not None, training will automatically stop after this many 
                                      epochs of no improvement in validation loss.
                                      Upon completion, model will be loaded with weights from epoch
                                      with lowest validation loss.
                                      NOTE: If reduce_on_plateau is also enabled, then
                                      early_stopping must be greater than reduce_on_plateau.
                                      Example: early_stopping=6, reduce_on_plateau=3.
            reduce_on_plateau (int):  If not None, will lower learning rate when
                                      when validation loss fails to improve after
                                      the specified number of epochs.
                                      NOTE: If early_stopping is enabled, then
                                      reduce_on_plateu must be less than early_stopping.
                                      Example: early_stopping=6, reduce_on_plateau=3.
            reduce_factor (int):      Learning reate is reduced by this factor on plateau.
                                      Only takes effect if reduce_on_plateau > 0.
            cycle_momentum (bool):    If True and optimizer is Adam, Nadam, or Adamax, momentum of 
                                      optimzer will be cycled between 0.95 and 0.85 as described in 
                                      https://arxiv.org/abs/1803.09820.
                                      Only takes effect if Adam, Nadam, or Adamax optimizer is used.
            max_momentum(float):  maximum momentum to use when cycle_momentum=True
            min_momentum(float): minimum momentum to use when cycle_momentum=True
            checkpoint_folder (string): Folder path in which to save the model weights 
                                        for each epoch.
                                        File name will be of the form: 
                                        weights-{epoch:02d}-{val_loss:.2f}.hdf5
            monitor (str):              what metric to monitor for early_stopping
                                        and reduce_on_plateau. Defaults to 'val_loss'.
                                        Only used if early_stopping or reduce_on_plateau
                                        is enabled.
            class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
            callbacks (list): list of Callback instances to employ during training
            steps_per_epoch(int):    Steps per epoch. If None, then, math.ceil(num_samples/batch_size) is used.
                                     Ignored unless training dataset is generator.
            verbose (bool):  verbose mode

    `evaluate(self, test_data=None, print_report=True, save_path='ktrain_classification_report.csv', class_names=[])`
    :   alias for self.validate().
        Returns confusion matrix and optionally prints
        a classification report.
        This is currently only supported for binary and multiclass
        classification, not multilabel classification.
        
        By default, this uses val_data, as supplied to ktrain.get_learner().
        Other validation or test data can be optionally be supplied as argument via <test_data> argument.
        Supply class_names to include labels instead of intenger class integer values in classification report.
        Args:
          test_data(Dataset|np.ndarray): test or validation data.  If None, self.val_data is used.
          print_report(bool): If True, classification report will be printed. If False, report will be saved to CSV 
                              at save_path. Not applicable to regression models.
                              Not applicable to regression models.
          save_path(str): Classification report will be saved to this file path/name if print_report=False
                          Not applicable to regression models.
          class_names(list): list of class names to be used in classification report instead of 
                             class integer IDs.

    `fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, batch_size=32)`
    :

    `fit_onecycle(self, lr, epochs, checkpoint_folder=None, cycle_momentum=True, max_momentum=0.95, min_momentum=0.85, class_weight=None, callbacks=[], steps_per_epoch=None, verbose=1)`
    :   Train model using a version of Leslie Smith's 1cycle policy.
        This method can be used with any optimizer. Thus,
        cyclical momentum is not currently implemented.
        
        Args:
            lr (float): (maximum) learning rate.  
                       It is recommended that you estimate lr yourself by 
                       running lr_finder (and lr_plot) and visually inspect plot
                       for dramatic loss drop.
            epochs (int): Number of epochs.  Number of epochs
            checkpoint_folder (string): Folder path in which to save the model weights 
                                        for each epoch.
                                        File name will be of the form: 
                                        weights-{epoch:02d}-{val_loss:.2f}.hdf5
            cycle_momentum (bool):    If True and optimizer is Adam, Nadam, or Adamax, momentum of 
                                      optimzer will be cycled between 0.95 and 0.85 as described in 
                                      https://arxiv.org/abs/1803.09820.
                                      Only takes effect if Adam, Nadam, or Adamax optimizer is used.
            max_momentum(float): Maximum momentum to use if cycle_momentum=True
            min_momentum(float): minimum momentum to use if cycle_momentum=True
            class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
            callbacks (list): list of Callback instances to employ during training
            steps_per_epoch(int):    Steps per epoch. If None, then, math.ceil(num_samples/batch_size) is used.
                                     Ignored unless training dataset is generator.
            verbose (bool):  verbose mode

    `freeze(self, freeze_range=None)`
    :   If freeze_range is None, makes all layers trainable=False except last Dense layer.
        If freeze_range is given, freezes the first <freeze_range> layers and
        unfrezes all remaining layers.
        NOTE:      Freeze method does not currently work with 
                   multi-GPU models.  If you are using the load_imagemodel method,
                   please use the freeze_layers argument of load_imagemodel
                   to freeze layers.
        Args:
            freeze_range(int): number of layers to freeze
        Returns:
            None

    `get_weight_decay(self)`
    :   Get current weight decay rate

    `ground_truth(self, val_data=None)`
    :

    `layer_output(self, layer_id, example_id=0, use_val=False)`
    :

    `load_model(self, fpath, custom_objects=None, **kwargs)`
    :   loads model from folder.
        Note: **kwargs included for backwards compatibility only, as TransformerTextClassLearner.load_model was removed in v0.18.0.
        Args:
          fpath(str): path to folder containing model
          custom_objects(dict): custom objects required to load model.
                                For models included with ktrain, this is populated automatically
                                and can be disregarded.

    `lr_estimate(self)`
    :   Return numerical estimates of lr using two different methods:
            1. learning rate associated with minimum numerical gradient
            2. learning rate associated with minimum loss divided by 10
        Since neither of these methods are fool-proof and can 
        potentially return bad estimates, it is recommended that you 
        examine the plot generated by lr_plot to estimate the learning rate.
        Returns:
          tuple: tuple of the form (float, float), where 
            First element is lr associated with minimum numerical gradient (None if gradient computation fails).
            Second element is lr associated with minimum loss divided by 10.

    `lr_find(self, start_lr=1e-07, lr_mult=1.01, max_epochs=None, class_weight=None, stop_factor=4, show_plot=False, suggest=False, restore_weights_only=False, verbose=1)`
    :   Plots loss as learning rate is increased.  Highest learning rate 
        corresponding to a still falling loss should be chosen.
        
        If you find the LR finder is running for more epochs than you'd prefer,
        you can set max_epochs (e.g., max_epochs=5) to estimate LR with a 
        smaller sample size.
        
        If lr_mult is supplied and max_epochs is None, LR will increase until loss diverges.
        Reasonable values of lr_mult are between 1.01 and 1.05.
        
        If max_epochs is supplied, lr_mult argument is ignored and computed automatically.
        
        Reference: https://arxiv.org/abs/1506.01186
        
        Args:
            start_lr (float): smallest lr to start simulation
            lr_mult (float): multiplication factor to increase LR.
                             Ignored if max_epochs is supplied.
            max_epochs (int):  maximum number of epochs to simulate.
                               lr_mult is ignored if max_epoch is supplied.
                               Default is None. Set max_epochs to an integer
                               (e.g., 5) if lr_find is taking too long
                               and running for more epochs than desired.
            class_weight(dict): class_weight parameter passed to model.fit
                                for imbalanced datasets.
            stop_factor(int): factor used to determine threhsold that loss 
                              must exceed to stop training simulation.
                              Increase this if loss is erratic and lr_find
                              exits too early.
            show_plot (bool):  If True, automatically invoke lr_plot
            restore_weights_only(bool): If True, when training simulation is complete,
                                        the model weights only are restored, but not
                                        the original optimizer weights.  
                                        In at least a few cases, this seems to improve performance
                                        when actual training begins. Further investigation is needed,
                                        so it is False by default.
            verbose (bool): specifies how much output to print
        Returns:
            None

    `lr_plot(self, n_skip_beginning=10, n_skip_end=5, suggest=False, return_fig=False)`
    :   Plots the loss vs. learning rate to help identify
        The maximal learning rate associated with a falling loss.
        The nskip_beginning and n_skip_end arguments can be used
        to "zoom in" on the plot.
        Args:
            n_skip_beginning(int): number of batches to skip on the left.
            n_skip_end(int):  number of batches to skip on the right.
            suggest(bool): will highlight numerical estimate
                           of best lr if True - methods adapted from fastai
            return_fig(bool): If True, return matplotlib.figure.Figure
        Returns:
          matplotlib.figure.Figure if return_fig else None

    `plot(self, plot_type='loss', return_fig=False)`
    :   plots training history
        Args:
          plot_type (str):  one of {'loss', 'lr', 'momentum'}
          return_fig(bool):  If True, return matplotlib.figure.Figure
        Return:
          matplotlib.figure.Figure if return_fig else None

    `predict(self, val_data=None)`
    :   Makes predictions on validation set

    `print_layers(self, show_wd=False)`
    :   prints the layers of the model along with indices

    `reset_weights(self, verbose=1)`
    :   Re-initializes network with original weights

    `save_model(self, fpath)`
    :   a wrapper to model.save
        Args:
          fpath(str): path to folder in which to save model
        Returns:
          None

    `set_lr(self, lr)`
    :

    `set_model(self, model)`
    :   replace model in this Learner instance

    `set_weight_decay(self, wd=0.01)`
    :   Sets global weight decay via AdamWeightDecay optimizer
        Args:
          wd(float): weight decay
        Returns:
          None

    `top_losses(self, n=4, val_data=None, preproc=None)`
    :   Computes losses on validation set sorted by examples with top losses
        Args:
          n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
          val_data:  optional val_data to use instead of self.val_data
          preproc (Preprocessor): A TextPreprocessor or ImagePreprocessor.
                                  For some data like text data, a preprocessor
                                  is required to undo the pre-processing
                                   to correctly view raw data.
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

    `unfreeze(self, exclude_range=None)`
    :   Make every layer trainable except those in exclude_range.
        unfreeze is simply a proxy method to freeze.
        NOTE:      Unfreeze method does not currently work with 
                   multi-GPU models.  If you are using the load_imagemodel method,
                   please use the freeze_layers argument of load_imagemodel
                   to freeze layers.

    `validate(self, val_data=None, print_report=True, save_path='ktrain_classification_report.csv', class_names=[])`
    :   Returns confusion matrix and optionally prints
        a classification report.
        This is currently only supported for binary and multiclass
        classification, not multilabel classification.
        
        By default, this uses val_data, as supplied to ktrain.get_learner().
        Other validation or test data can be optionally be supplied as argument.
        Supply class_names to include labels instead of intenger class integer values in classification report.
        Args:
          val_data(Dataset|np.ndarray): validation data.  If None, self.val_data is used.
          print_report(bool): If True, classification report will be printed. If False, report will be saved to CSV 
                              at save path. Not applicable to regression models.
          save_path(str): Classification report will be saved to this file path/name if print_report=False
          class_names(list): list of class names to be used in classification report instead of 
                             class integer IDs.

    `view_top_losses(self, n=4, preproc=None, val_data=None)`
    :   View observations with top losses in validation set.
        Musta be overridden by Learner subclasses.