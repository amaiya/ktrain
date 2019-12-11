from ..imports import *
from .. import utils as U
from .preprocessor import NodePreprocessor


def graph_nodes_from_csv(nodes_filepath, 
                         links_filepath,
                         use_lcc=True,
                         sample_size=10,
                         train_pct=0.1, sep=',', 
                         holdout_pct=None, 
                         holdout_for_inductive=False,
                         missing_label_value=None,
                         random_state=None,
                         verbose=1):
    """
    Loads graph data from CSV files. 
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
    """

    #----------------------------------------------------------------
    # read graph structure
    #----------------------------------------------------------------
    nx_sep = None if sep in [' ', '\t'] else sep
    g_nx = nx.read_edgelist(path=links_filepath, delimiter=nx_sep)

    # read node attributes
    #node_attr = pd.read_csv(nodes_filepath, sep=sep, header=None)

    # store class labels within graph nodes
    #values = { str(row.tolist()[0]): row.tolist()[-1] for _, row in node_attr.iterrows()}
    #nx.set_node_attributes(g_nx, values, 'target')

    # select largest connected component
    if use_lcc:
        g_nx_ccs = (g_nx.subgraph(c).copy() for c in nx.connected_components(g_nx))
        g_nx = max(g_nx_ccs, key=len)
        if verbose:
            print("Largest subgraph statistics: {} nodes, {} edges".format(
            g_nx.number_of_nodes(), g_nx.number_of_edges()))


    #----------------------------------------------------------------
    # read node attributes and split into train/validation
    #----------------------------------------------------------------
    node_attr = pd.read_csv(nodes_filepath, sep=sep, header=None)
    num_features = len(node_attr.columns.values) - 2 # subract ID and target
    feature_names = ["w_{}".format(ii) for ii in range(num_features)]
    column_names =  feature_names + ["target"]
    node_data = pd.read_csv(nodes_filepath, header=None, names=column_names, sep=sep)
    node_data.index = node_data.index.map(str)
    node_data = node_data[node_data.index.isin(list(g_nx.nodes()))]


    #----------------------------------------------------------------
    # check for holdout nodes
    #----------------------------------------------------------------
    num_null = node_data[node_data.target.isnull()].shape[0]
    num_missing = 0
    if missing_label_value is not None:
        num_missing = node_data[node_data.target == missing_label_value].shape[0]

    if num_missing > 0 and num_null >0:
        raise ValueError('Param missing_label_value is not None but there are ' +\
                         'NULLs in last column. Replace these with missing_label_value.')

    if (num_null > 0 or num_missing > 0) and holdout_pct is not None:
        warnings.warn('Number of nodes in having NULL  or missing_label_value in target '+\
                      'column is non-zero. Using these as holdout nodes and ignoring holdout_pct.')



    #----------------------------------------------------------------
    # set df and G and optionally holdout nodes
    #----------------------------------------------------------------
    if num_null > 0:
        df_annotated = node_data[~node_data.target.isnull()]
        df_holdout = node_data[~node_data.target.isnull()]
        G_holdout = g_nx
        df_G = df_annotated if holdout_for_inductive else node_data
        G = g_nx.subgraph(df_annotated.index).copy() if holdout_for_inductive else g_nx
        U.vprint('using %s nodes with target=NULL as holdout set' % (num_null), verbose=verbose)
    elif num_missing > 0:
        df_annotated = node_data[node_data.target != missing_label_value]
        df_holdout = node_data[node_data.target == missing_label_value]
        G_holdout = g_nx
        df_G = df_annotated if holdout_for_inductive else node_data
        G = g_nx.subgraph(df_annotated.index).copy() if holdout_for_inductive else g_nx
        U.vprint('using %s nodes with missing target as holdout set' % (num_missing), verbose=verbose)
    elif holdout_pct is not None:
        df_annotated = node_data.sample(frac=1-holdout_pct, replace=False, random_state=None)
        df_holdout = node_data[~node_data.index.isin(df_annotated.index)]
        G_holdout = g_nx
        df_G = df_annotated if holdout_for_inductive else node_data
        G = g_nx.subgraph(df_annotated.index).copy() if holdout_for_inductive else g_nx
    else:
        if holdout_for_inductive:
            warnings.warn('holdout_for_inductive is True but no nodes were heldout '
                          'because holdout_pct is None and no missing targets')
        df_annotated = node_data
        df_holdout = None
        G_holdout = None
        df_G = node_data
        G = g_nx


    #----------------------------------------------------------------
    # split into train and validation
    #----------------------------------------------------------------
    tr_data, te_data = sklearn.model_selection.train_test_split(df_annotated, 
                                                        train_size=train_pct,
                                                        test_size=None,
                                                        stratify=df_annotated['target'], 
                                                        random_state=random_state)
    #te_data, test_data = sklearn.model_selection.train_test_split(test_data,
                                                                #train_size=0.2,
                                                                #test_size=None,
                                                                 #stratify=test_data["target"],
                                                                 #random_state=100)

    #----------------------------------------------------------------
    # print summary
    #----------------------------------------------------------------
    if verbose:
        print("Size of training graph: %s nodes" % (G.number_of_nodes()))
        print("Training nodes: %s" % (tr_data.shape[0]))
        print("Validation nodes: %s" % (te_data.shape[0]))
        if df_holdout is not None and G_holdout is not None:
            print("Nodes treated as unlabeled for testing/inference: %s" % (df_holdout.shape[0]))
            if holdout_for_inductive:
                print("Size of graph with added holdout nodes: %s" % (G_holdout.number_of_nodes()))
                print("Holdout node features are not visible during training (inductive inference)")
            else:
                print("Holdout node features are visible during training (transductive inference)")
        print()



    #----------------------------------------------------------------
    # Preprocess training and validation datasets using NodePreprocessor
    #----------------------------------------------------------------
    preproc = NodePreprocessor(G, df_G, sample_size=sample_size, missing_label_value=missing_label_value)
    trn = preproc.preprocess_train(list(tr_data.index))
    val = preproc.preprocess_valid(list(te_data.index))
    from .node_generator import NodeSequenceWrapper
    if df_holdout is not None and G_holdout is not None: 
        return (NodeSequenceWrapper(trn), NodeSequenceWrapper(val), preproc, df_holdout, G_holdout)
    else:
        return (NodeSequenceWrapper(trn), NodeSequenceWrapper(val), preproc)

