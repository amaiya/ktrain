from ..imports import *
from .. import utils as U
from .node_generator import NodeSequenceWrapper
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator, GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE



GRAPHSAGE = 'graphsage'
NODE_CLASSIFIERS = {
        GRAPHSAGE: 'GraphSAGE:  http://arxiv.org/pdf/1607.01759.pdf'}

def print_node_classifiers():
    for k,v in NODE_CLASSIFIERS.items():
        print("%s: %s" % (k,v))


def graph_node_classifier(name, train_data, layer_sizes=[32,32], verbose=1):
    """
    Build and return a neural node classification model.
    Notes: Only mutually-exclusive class labels are supported.

    Args:
        name (string): one of:
                      - 'graphsage' for GraphSAGE model 
                      (only GraphSAGE currently supported)

        train_data (NodeSequenceWrapper): a ktrain.graph.node_generator.NodeSequenceWrapper object
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    """
    sg_version_error = False
    try:
        from packaging import version
        if version.parse(sg.__version__) < version.parse('0.8.0'):
            sg_version_error = True
    except:
       if not sg.__version__.startswith('0.8'): sg_version_error = True

    if sg_version_error:
        warnings.warn('Please install stellargraph 0.8 or above.')
        return


    # check argument
    if not isinstance(train_data, NodeSequenceWrapper):
        err ="""
            train_data must be a ktrain.graph.node_generator.NodeSequenceWrapper object
            """
        raise Exception(err)
    if len(layer_sizes) != 2:
        raise ValueError('layer_sizes must be of length 2')

    num_classes = U.nclasses_from_data(train_data)

    # determine multilabel
    multilabel = U.is_multilabel(train_data)
    if multilabel:
        raise ValueError('Multi-label classification not currently supported for graphs.')
    U.vprint("Is Multi-Label? %s" % (multilabel), verbose=verbose)

    # set loss and activations
    loss_func = 'categorical_crossentropy'
    activation = 'softmax'

    # build a GraphSAGE node classification model
    graphsage_model = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=train_data,
        bias=True,
        dropout=0.5,
	)
    #x_inp, x_out = graphsage_model.default_model(flatten_output=True)
    x_inp, x_out = graphsage_model.build()
    prediction = Dense(units=num_classes, activation=activation)(x_out)
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer='adam',
                  loss=loss_func,
                  metrics=["acc"])
    U.vprint('done', verbose=verbose)
    return model

