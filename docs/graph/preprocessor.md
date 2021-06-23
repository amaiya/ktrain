Module ktrain.graph.preprocessor
================================

Classes
-------

`LinkPreprocessor(G, sample_sizes=[10, 20])`
:   Link preprocessing base class

    ### Ancestors (in MRO)

    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `preprocess(self, G, edge_ids)`
    :

    `preprocess_train(self, G, edge_ids, edge_labels, mode='train')`
    :   preprocess training set
        Args:
          G (networkx graph): networkx graph
          edge_ids(list): list of tuples representing edge ids
          edge_labels(list): edge labels (1 or 0 to indicated whether it is a true edge in original graph or not)

    `preprocess_valid(self, G, edge_ids, edge_labels)`
    :   preprocess training set
        Args:
          G (networkx graph): networkx graph
          edge_ids(list): list of tuples representing edge ids
          edge_labels(list): edge labels (1 or 0 to indicated whether it is a true edge in original graph or not)

`NodePreprocessor(G_nx, df, sample_size=10, missing_label_value=None)`
:   Node preprocessing base class

    ### Ancestors (in MRO)

    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Instance variables

    `feature_names`
    :

    ### Methods

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `ids_exist(self, node_ids)`
    :   check validity of node IDs

    `preprocess(self, df, G)`
    :

    `preprocess_test(self, df_te, G_te)`
    :   preprocess for inductive inference
        df_te (DataFrame): pandas dataframe containing new node attributes
        G_te (Graph):  a networkx Graph containing new nodes

    `preprocess_train(self, node_ids)`
    :   preprocess training set

    `preprocess_valid(self, node_ids)`
    :   preprocess validation nodes (transductive inference)
        node_ids (list):  list of node IDs that generator will yield