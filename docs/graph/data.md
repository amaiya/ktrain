Module ktrain.graph.data
========================

Functions
---------

    
`graph_links_from_csv(nodes_filepath, links_filepath, sample_sizes=[10, 20], train_pct=0.1, val_pct=0.1, sep=',', holdout_pct=None, holdout_for_inductive=False, missing_label_value=None, random_state=None, verbose=1)`
:   Loads graph data from CSV files. 
    Returns generators for links in graph for use with GraphSAGE model.
    Args:
        nodes_filepath(str): file path to training CSV containing node attributes
        links_filepath(str): file path to training CSV describing links among nodes
        sample_sizes(int): Number of nodes to sample at each neighborhood level.
        train_pct(float): Proportion of edges to use for training.
                          Default is 0.1.
                          Note that train_pct is applied after val_pct is applied.
        val_pct(float): Proportion of edges to use for validation
        sep (str):  delimiter for CSVs. Default is comma.
        random_state (int):  random seed for train/test split
        verbose (boolean): verbosity
    Return:
        tuple of EdgeSequenceWrapper objects for train and validation sets and LinkPreprocessor

    
`graph_nodes_from_csv(nodes_filepath, links_filepath, use_lcc=True, sample_size=10, train_pct=0.1, sep=',', holdout_pct=None, holdout_for_inductive=False, missing_label_value=None, random_state=None, verbose=1)`
:   Loads graph data from CSV files. 
    Returns generators for nodes in graph for use with GraphSAGE model.
    Args:
        nodes_filepath(str): file path to training CSV containing node attributes
        links_filepath(str): file path to training CSV describing links among nodes
        use_lcc(bool):  If True, consider the largest connected component only.
        sample_size(int): Number of nodes to sample at each neighborhood level
        train_pct(float): Proportion of nodes to use for training.
                          Default is 0.1.
        sep (str):  delimiter for CSVs. Default is comma.
        holdout_pct(float): Percentage of nodes to remove and return separately
                        for later transductive/inductive inference.
                        Example -->  train_pct=0.1 and holdout_pct=0.2:
    
                        Out of 1000 nodes, 200 (holdout_pct*1000) will be held out.
                        Of the remaining 800, 80 (train_pct*800) will be used for training
                        and 720 ((1-train_pct)*800) will be used for validation.
                        200 nodes will be used for transductive or inductive inference.
    
                        Note that holdout_pct is ignored if at least one node has
                        a missing label in nodes_filepath, in which case
                        these nodes are assumed to be the holdout set.
        holdout_for_inductive(bool):  If True, the holdout nodes will be removed from 
                                      training graph and their features will not be visible
                                      during training.  Only features of training and
                                      validation nodes will be visible.
                                      If False, holdout nodes will be included in graph
                                      and their features (but not labels) are accessible
                                      during training.
        random_state (int):  random seed for train/test split
        verbose (boolean): verbosity
    Return:
        tuple of NodeSequenceWrapper objects for train and validation sets and NodePreprocessor
        If holdout_pct is not None or number of nodes with missing labels is non-zero, 
        fourth and fifth return values are pd.DataFrame and nx.Graph
        comprising the held out nodes.