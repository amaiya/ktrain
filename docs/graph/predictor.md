Module ktrain.graph.predictor
=============================

Classes
-------

`LinkPredictor(model, preproc, batch_size=32)`
:   predicts graph node's classes

    ### Ancestors (in MRO)

    * ktrain.predictor.Predictor
    * abc.ABC

    ### Methods

    `get_classes(self)`
    :

    `predict(self, G, edge_ids, return_proba=False)`
    :   Performs link prediction
        If return_proba is True, returns probabilities of each class.

`NodePredictor(model, preproc, batch_size=32)`
:   predicts graph node's classes

    ### Ancestors (in MRO)

    * ktrain.predictor.Predictor
    * abc.ABC

    ### Methods

    `get_classes(self)`
    :

    `predict(self, node_ids, return_proba=False)`
    :

    `predict_inductive(self, df, G, return_proba=False)`
    :   Performs inductive inference.
        If return_proba is True, returns probabilities of each class.

    `predict_transductive(self, node_ids, return_proba=False)`
    :   Performs transductive inference.
        If return_proba is True, returns probabilities of each class.