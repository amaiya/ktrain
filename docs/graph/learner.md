Module ktrain.graph.learner
===========================

Classes
-------

`LinkPredLearner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Main class used to tune and train Keras models for link prediction
    Main parameters are:
    
    model (Model): A compiled instance of keras.engine.training.Model
    train_data (Iterator): a Iterator instance for training set
    val_data (Iterator):   A Iterator instance for validation set

    ### Ancestors (in MRO)

    * ktrain.core.GenLearner
    * ktrain.core.Learner
    * abc.ABC

    ### Methods

    `layer_output(self, layer_id, example_id=0, batch_id=0, use_val=False)`
    :   Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from training set, by default.

`NodeClassLearner(model, train_data=None, val_data=None, batch_size=32, eval_batch_size=32, workers=1, use_multiprocessing=False)`
:   Main class used to tune and train Keras models for node classification
    Main parameters are:
    
    model (Model): A compiled instance of keras.engine.training.Model
    train_data (Iterator): a Iterator instance for training set
    val_data (Iterator):   A Iterator instance for validation set

    ### Ancestors (in MRO)

    * ktrain.core.GenLearner
    * ktrain.core.Learner
    * abc.ABC

    ### Methods

    `layer_output(self, layer_id, example_id=0, batch_id=0, use_val=False)`
    :   Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from training set, by default.