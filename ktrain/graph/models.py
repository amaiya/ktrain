from .. import utils as U
from ..imports import *

GRAPHSAGE = "graphsage"
NODE_CLASSIFIERS = {GRAPHSAGE: "GraphSAGE:  https://arxiv.org/pdf/1706.02216.pdf"}

LINK_PREDICTORS = {GRAPHSAGE: "GraphSAGE:  https://arxiv.org/pdf/1706.02216.pdf"}


def print_node_classifiers():
    for k, v in NODE_CLASSIFIERS.items():
        print("%s: %s" % (k, v))


def print_link_predictors():
    for k, v in LINK_PREDICTORS.items():
        print("%s: %s" % (k, v))


def graph_node_classifier(name, train_data, layer_sizes=[32, 32], verbose=1):
    """
    ```
    Build and return a neural node classification model.
    Notes: Only mutually-exclusive class labels are supported.

    Args:
        name (string): one of:
                      - 'graphsage' for GraphSAGE model
                      (only GraphSAGE currently supported)

        train_data (NodeSequenceWrapper): a ktrain.graph.sg_wrappers.NodeSequenceWrapper object
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    ```
    """
    from .sg_wrappers import NodeSequenceWrapper

    # check argument
    if not isinstance(train_data, NodeSequenceWrapper):
        err = """
            train_data must be a ktrain.graph.sg_wrappers.NodeSequenceWrapper object
            """
        raise Exception(err)
    if len(layer_sizes) != 2:
        raise ValueError("layer_sizes must be of length 2")

    num_classes = U.nclasses_from_data(train_data)

    # determine multilabel
    multilabel = U.is_multilabel(train_data)
    if multilabel:
        raise ValueError(
            "Multi-label classification not currently supported for graphs."
        )
    U.vprint("Is Multi-Label? %s" % (multilabel), verbose=verbose)

    # set loss and activations
    loss_func = "categorical_crossentropy"
    activation = "softmax"

    # import stellargraph
    try:
        import stellargraph as sg
        from stellargraph.layer import GraphSAGE
    except:
        raise Exception(SG_ERRMSG)
    if version.parse(sg.__version__) < version.parse("0.8"):
        raise Exception(SG_ERRMSG)

    # build a GraphSAGE node classification model
    graphsage_model = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=train_data,
        bias=True,
        dropout=0.5,
    )
    # x_inp, x_out = graphsage_model.default_model(flatten_output=True)
    x_inp, x_out = graphsage_model.build()
    prediction = keras.layers.Dense(units=num_classes, activation=activation)(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=U.DEFAULT_OPT, loss=loss_func, metrics=["accuracy"])
    U.vprint("done", verbose=verbose)
    return model


def graph_link_predictor(name, train_data, preproc, layer_sizes=[20, 20], verbose=1):
    """
    ```
    Build and return a neural link prediction model.

    Args:
        name (string): one of:
                      - 'graphsage' for GraphSAGE model
                      (only GraphSAGE currently supported)

        train_data (LinkSequenceWrapper): a ktrain.graph.sg_wrappers.LinkSequenceWrapper object
        preproc(LinkPreprocessor): a LinkPreprocessor instance
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance
    ```
    """
    from .sg_wrappers import LinkSequenceWrapper

    # check argument
    if not isinstance(train_data, LinkSequenceWrapper):
        err = """
            train_data must be a ktrain.graph.sg_wrappers.LinkSequenceWrapper object
            """
        raise Exception(err)
    if len(layer_sizes) != len(preproc.sample_sizes):
        raise ValueError("number of layer_sizes must match len(preproc.sample_sizes)")

    num_classes = U.nclasses_from_data(train_data)

    # set loss and activations
    loss_func = "categorical_crossentropy"
    activation = "softmax"

    # import stellargraph
    try:
        import stellargraph as sg
        from stellargraph.layer import GraphSAGE, link_classification
    except:
        raise Exception(SG_ERRMSG)
    if version.parse(sg.__version__) < version.parse("0.8"):
        raise Exception(SG_ERRMSG)

    # build a GraphSAGE link prediction model
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_data, bias=True, dropout=0.3
    )
    x_inp, x_out = graphsage.build()
    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=U.DEFAULT_OPT, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model
