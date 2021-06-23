Module ktrain.text.ner.anago.callbacks
======================================
Custom callbacks.

Classes
-------

`F1score(seq, preprocessor=None)`
:   Abstract base class used to build new callbacks.
    
    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `keras.models.Model`.
            Reference of the model being trained.
    
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).

    ### Ancestors (in MRO)

    * tensorflow.python.keras.callbacks.Callback

    ### Methods

    `get_lengths(self, y_true)`
    :

    `on_epoch_end(self, epoch, logs={})`
    :   Called at the end of an epoch.
        
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        
        Arguments:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`.