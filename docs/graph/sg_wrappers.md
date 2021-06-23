Module ktrain.graph.sg_wrappers
===============================

Classes
-------

`LinkSequenceWrapper(link_seq)`
:   Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit_generator`,
    :meth:`keras.Model.evaluate_generator`, and :meth:`keras.Model.predict_generator`
    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator` or :class:`Attri2VecLinkGenerator`.
    Args:
        generator: An instance of :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator` or 
        :class:`Attri2VecLinkGenerator`.
        ids (list or iterable): Link IDs to batch, each link id being a tuple of (src, dst) node ids.
            (The graph nodes must have a "feature" attribute that is used as input to the GraphSAGE/Attri2Vec model.)
            These are the links that are to be used to train or inference, and the embeddings
            calculated for these links via a binary operator applied to their source and destination nodes,
            are passed to the downstream task of link prediction or link attribute inference.
            The source and target nodes of the links are used as head nodes for which subgraphs are sampled.
            The subgraphs are sampled from all nodes.
        targets (list or iterable): Labels corresponding to the above links, e.g., 0 or 1 for the link prediction problem.
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.

    ### Ancestors (in MRO)

    * stellargraph.mapper.link_mappers.LinkSequence
    * ktrain.data.SequenceDataset
    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Methods

    `get_y(self)`
    :

    `nsamples(self)`
    :

`NodeSequenceWrapper(node_seq)`
:   Keras-compatible data generator to use with the Keras
    methods :meth:`keras.Model.fit_generator`, :meth:`keras.Model.evaluate_generator`,
    and :meth:`keras.Model.predict_generator`.
    
    This class generated data samples for node inference models
    and should be created using the `.flow(...)` method of
    :class:`GraphSAGENodeGenerator` or :class:`DirectedGraphSAGENodeGenerator` 
    or :class:`HinSAGENodeGenerator` or :class:`Attri2VecNodeGenerator`.
    
    GraphSAGENodeGenerator, DirectedGraphSAGENodeGenerator,and HinSAGENodeGenerator 
    are classes that capture the graph structure and the feature vectors of each node. 
    These generator classes are used within the NodeSequence to generate
    samples of k-hop neighbourhoods in the graph and to return to this 
    class the features from the sampled neighbourhoods.
    
    Attri2VecNodeGenerator is the class that captures node feature vectors
    of each node.
    
    Args:
        generator: GraphSAGENodeGenerator, DirectedGraphSAGENodeGenerator or 
            HinSAGENodeGenerator or Attri2VecNodeGenerator. The generator object 
            containing the graph information.
        ids: list
            A list of the node_ids to be used as head-nodes in the
            downstream task.
        targets: list, optional (default=None)
            A list of targets or labels to be used in the downstream
            class.
    
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.

    ### Ancestors (in MRO)

    * stellargraph.mapper.node_mappers.NodeSequence
    * ktrain.data.SequenceDataset
    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Methods

    `get_y(self)`
    :

    `nsamples(self)`
    :