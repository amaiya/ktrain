Module ktrain.graph.models
==========================

Functions
---------

    
`graph_link_predictor(name, train_data, preproc, layer_sizes=[20, 20], verbose=1)`
:   Build and return a neural link prediction model.
    
    Args:
        name (string): one of:
                      - 'graphsage' for GraphSAGE model 
                      (only GraphSAGE currently supported)
    
        train_data (LinkSequenceWrapper): a ktrain.graph.sg_wrappers.LinkSequenceWrapper object
        preproc(LinkPreprocessor): a LinkPreprocessor instance
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`graph_node_classifier(name, train_data, layer_sizes=[32, 32], verbose=1)`
:   Build and return a neural node classification model.
    Notes: Only mutually-exclusive class labels are supported.
    
    Args:
        name (string): one of:
                      - 'graphsage' for GraphSAGE model 
                      (only GraphSAGE currently supported)
    
        train_data (NodeSequenceWrapper): a ktrain.graph.sg_wrappers.NodeSequenceWrapper object
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`print_link_predictors()`
:   

    
`print_node_classifiers()`
: