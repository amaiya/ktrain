Module ktrain.text.models
=========================

Functions
---------

    
`calc_pr(y_i, x, y, b)`
:   

    
`calc_r(y_i, x, y)`
:   

    
`print_text_classifiers()`
:   

    
`print_text_regression_models()`
:   

    
`text_classifier(name, train_data, preproc=None, multilabel=None, metrics=['accuracy'], verbose=1)`
:   Build and return a text classification model.
    
    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'logreg' for logistic regression using embedding layers
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model
    
        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train) or ktrain.Dataset instance
                            returned from one of the texts_from_* functions
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        metrics(list): metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`text_regression_model(name, train_data, preproc=None, metrics=['mae'], verbose=1)`
:   Build and return a text regression model.
    
    Args:
        name (string): one of:
                      - 'fasttext' for FastText model
                      - 'nbsvm' for NBSVM model  
                      - 'linreg' for linear regression using embedding layers
                      - 'bigru' for Bidirectional GRU with pretrained word vectors
                      - 'bert' for BERT Text Classification
                      - 'distilbert' for Hugging Face DistilBert model
    
        train_data (tuple): a tuple of numpy.ndarrays: (x_train, y_train)
        preproc: a ktrain.text.TextPreprocessor instance.
                 As of v0.8.0, this is required.
        metrics(list): metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance