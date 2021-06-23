Module ktrain.tabular.models
============================

Functions
---------

    
`print_tabular_classifiers()`
:   

    
`print_tabular_regression_models()`
:   

    
`tabular_classifier(name, train_data, multilabel=None, metrics=['accuracy'], hidden_layers=[1000, 500], hidden_dropouts=[0.0, 0.5], bn=False, verbose=1)`
:   Build and return a classification model for tabular data
    
    Args:
        name (string): currently accepts 'mlp' for multilayer perceptron
        train_data (TabularDataset): TabularDataset instance returned from one of the tabular_from_* functions
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        metrics(list): list of metrics to use
        hidden_layers(list): number of units in each hidden layer of NN
        hidden_dropouts(list): Dropout values after each hidden layer of NN
        bn(bool): If True, BatchNormalization will be used before each fully-connected layer in NN
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance

    
`tabular_regression_model(name, train_data, metrics=['mae'], hidden_layers=[1000, 500], hidden_dropouts=[0.0, 0.5], bn=False, verbose=1)`
:   Build and return a regression model for tabular data
    
    Args:
        name (string): currently accepts 'mlp' for multilayer perceptron
        train_data (TabularDataset): TabularDataset instance returned from one of the tabular_from_* functions
        metrics(list): list of metrics to use
        hidden_layers(list): number of units in each hidden layer of NN
        hidden_dropouts(list): Dropout values after each hidden layer of NN
        bn(bool): If True, BatchNormalization will be before used each fully-connected layer in NN
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance