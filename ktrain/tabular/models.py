from ..imports import *
from .. import utils as U
from ..models import bn_drop_lin

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



def _tabular_model(name, train_data, multilabel=None, is_regression=False, metrics=['accuracy'], 
                   hidden_layers=[1000, 500], hidden_dropouts=[0., 0.5], bn=False, verbose=1):
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
        hidden_layers(list): number of units in each hidden layer of NN
        hidden_dropouts(list): Dropout values after each hidden layer of NN
        bn(bool): If True, BatchNormalization will be used before each fully-connected layer in NN
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """

    # check arguments
    if not U.is_tabular_from_data(train_data):
        err ="""
            Please pass training data in the form of data returned from a ktrain tabular_from* function.
            """
        raise Exception(err)
    if len(hidden_layers) != len(hidden_dropouts): raise ValueError('len(hidden_layers) must equal len(hidden_dropouts)')

    # reformat dropouts for each of construction
    output_dropout = hidden_dropouts[1]
    hidden_dropouts[1] = hidden_dropouts[0]
    hidden_dropouts[0] = 0.

    # set model configuration values
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


    # construct model

    ilayers = []
    n_cat = len(train_data.cat_columns)
    n_cont = len(train_data.cont_columns)
    if n_cat ==0 and n_cont == 0: raise ValueError('There are zero continuous and cateorical variables.')

    # categorical inputs and embeddings
    if n_cat > 0:
        emblayers = []
        num_uniques = [max(c.cat.codes.values+1)+1 for n, c in train_data.df[train_data.cat_columns].items()]
        for i in range(n_cat):
            inp = keras.layers.Input(shape=(1,))
            ilayers.append(inp)
            emb_size = min(50, (num_uniques[i]//2)+1)
            #emb_size = min(600, round(1.6 * num_uniques[i]**0.56))
            emb = keras.layers.Embedding(num_uniques[i], emb_size, input_length=1)(inp)
            emblayers.append(emb)
        x = keras.layers.concatenate(emblayers)if len(emblayers) > 1 else emblayers[0]
        x = keras.layers.Flatten()(x)

    # continuous inputs
    if n_cont > 0:
        x_cont = keras.layers.Input(shape=(n_cont,))
        ilayers.append(x_cont)
        x = keras.layers.concatenate([x, x_cont]) if n_cat > 0 else x_cont

    # hidden layers
    output = x
    for i, n_out in enumerate(hidden_layers):
        output = bn_drop_lin(output, n_out, bn=bn, p=hidden_dropouts[i], actn='relu')

    # output layer
    output = bn_drop_lin(output, num_classes , bn=bn, p=output_dropout, actn=activation)

    # construct and compile model
    model = Model(inputs=ilayers, outputs=output)
    model.compile(optimizer=U.DEFAULT_OPT, loss=loss_func, metrics=metrics)
    U.vprint('done.', verbose=verbose)
    return model



def tabular_classifier(name, train_data, multilabel=None, metrics=['accuracy'], 
                       hidden_layers=[1000, 500], hidden_dropouts=[0., 0.5], bn=False, verbose=1):
    """
    Build and return a classification model for tabular data

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
    """


    return _tabular_model(name, train_data, multilabel=multilabel, metrics=metrics,
                          hidden_layers=hidden_layers, hidden_dropouts=hidden_dropouts, bn=bn,
                          verbose=verbose, is_regression=False)


def tabular_regression_model(name, train_data,  metrics=['mae'], 
                             hidden_layers=[1000, 500], hidden_dropouts=[0., 0.5], bn=False, verbose=1):
    """
    Build and return a regression model for tabular data

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
    """


    return _tabular_model(name, train_data, multilabel=None, metrics=metrics, 
                          hidden_layers=hidden_layers, hidden_dropouts=hidden_dropouts, bn=bn,
                          verbose=verbose, is_regression=True)
