from ..imports import *
from .. import utils as U


MLP = 'mlp'
TABULAR_MODELS = {
                    MLP: "a configurable multilayer perceptron with categorical variable embeddings [https://arxiv.org/abs/1604.06737]",
                    } 



def print_tabular_classifiers():
    for k,v in TABULAR_MODELS.items():
        print("%s: %s" % (k,v))

def print_tabular_regression_models():
    for k,v in TABULAR_MODELS.items():
        print("%s: %s" % (k,v))



def _tabular_model(name, train_data, multilabel=None, is_regression=False, metrics=['accuracy'], verbose=1):
    """
    Build and return a classification or regression model for tabular data

    Args:
        name (string): currently accepts 'mlp' for multilayer perceptron
        train_data (TabularDataset): TabularDataset instance returned from one of the tabular_from_* functions
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        is_regression(bool): If True, will build a regression model, else classification model.
        metrics(list): list of metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    # check arguments
    if not U.is_tabular_from_data(train_data):
        err ="""
            Please pass training data in the form of data returned from a ktrain table_from* function.
            """
        raise Exception(err)

    if is_regression: # regression
        if metrics is None or metrics==['accuracy']: metrics=['mae']
        num_classes = 1
        multilabel = False
        loss_func = 'mse'
        activation = 'linear'
    else:             # classification
        if metrics is None: metrics = ['accuracy']
        # set number of classes and multilabel flag
        num_classes = U.nclasses_from_data(train_data)

        # determine multilabel
        if multilabel is None:
            multilabel = U.is_multilabel(train_data)
        U.vprint("Is Multi-Label? %s" % (multilabel), verbose=verbose)

        # set loss and activations
        loss_func = 'categorical_crossentropy'
        activation = 'softmax'
        if multilabel:
            loss_func = 'binary_crossentropy'
            activation = 'sigmoid'

    # return dummy model temporarily as placeholder
    model = Sequential()
    model.add(Dense(num_classes, input_shape=(42,), activation=activation))
    model.compile(optimizer=U.DEFAULT_OPT, loss=loss_func, metrics=metrics])

    U.vprint('done.', verbose=verbose)
    return model


def tabular_classifier(name, train_data, multilabel=None, metrics=['accuracy'], verbose=1):
    """
    Build and return a classification model for tabular data

    Args:
        name (string): currently accepts 'mlp' for multilayer perceptron
        train_data (TabularDataset): TabularDataset instance returned from one of the tabular_from_* functions
        multilabel (bool):  If True, multilabel model will be returned.
                            If false, binary/multiclass model will be returned.
                            If None, multilabel will be inferred from data.
        metrics(list): list of metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """


    self._tabular_model(name, train_data, multilabel=multilabel, metrics=metrics, verbose=verbose, is_regression=False)


def tabular_regression_model(name, train_data,  metrics=['mae'], verbose=1):
    """
    Build and return a regression model for tabular data

    Args:
        name (string): currently accepts 'mlp' for multilayer perceptron
        train_data (TabularDataset): TabularDataset instance returned from one of the tabular_from_* functions
        metrics(list): list of metrics to use
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """


    self._tabular_model(name, train_data, multilabel=None, metrics=metrics, verbose=verbose, is_regression=True)
