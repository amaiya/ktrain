#import networkx as nx
#import pandas as pd
#import stellargraph as sg
#from stellargraph.mapper import GraphSAGENodeGenerator, GraphSAGELinkGenerator
#from sklearn import preprocessing, feature_extraction, model_selection
#from stellargraph.data import EdgeSplitter


from ..imports import *
from .. import utils as U
from .node_generator import NodeSequenceWrapper
from .preprocessor import NodePreprocessor


def graph_nodes_from_csv(nodes_filepath, 
                         links_filepath,
                         use_lcc=True,
                         sample_size=10,
                         train_pct=0.1, sep=',', 
                         holdout_pct=None, verbose=1):
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
                        for later inductive inference.
                        Example -->  train_pct=0.1 and holdout_pct=0.2:

                        Out of 1000 nodes, 200 (holdout_pct*1000) will be held out.
                        Of the remaining 800, 80 (train_pct*800) will be used for training
                        and 720 ((1-train_pct)*800) will be used for validation.
                        That is, 720 nodes will be used for transductive inference
                        and 200 nodes will be used for inductive inference.
        verbose (boolean): verbosity
    Return:
        tuple of NodeSequenceWrapper objects for train and validation sets and NodePreprocessor
        If holdout_pct is not None, fourth and fifth return values are pd.DataFrame and nx.Graph
        comprising the held out nodes.
    """

    #----------------------------------------------------------------
    # read graph structure
    #----------------------------------------------------------------
    g_nx = nx.read_edgelist(path=links_filepath, delimiter=sep)

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
    # set df and G and optionally holdout nodes
    #----------------------------------------------------------------
    if holdout_pct is not None:
        df = node_data.sample(frac=1-holdout_pct, replace=False, random_state=101)
        G = g_nx.subgraph(df.index).copy()
        df_holdout = node_data[~node_data.index.isin(df.index)]
        #G_holdout = g_nx.subgraph(df_holdout.index).copy()
        G_holdout = g_nx

    else:
        df = node_data
        G = g_nx
        df_holdout = None
        G_holdout = None




    # split into train and validation
    tr_data, test_data = sklearn.model_selection.train_test_split(df, 
                                                        train_size=train_pct,
                                                        test_size=None,
                                                        stratify=df['target'], random_state=42)
    te_data, test_data = sklearn.model_selection.train_test_split(test_data,
                                                                train_size=0.2,
                                                                test_size=None,
                                                                 stratify=test_data["target"],
                                                                 random_state=100)

    #----------------------------------------------------------------
    # Preprocess training and validation datasets using NodePreprocessor
    #----------------------------------------------------------------
    preproc = NodePreprocessor(G, df, sample_size=sample_size)
    trn = preproc.preprocess_train(list(tr_data.index))
    val = preproc.preprocess_valid(list(te_data.index))
    if holdout_pct is not None:
        return (NodeSequenceWrapper(trn), NodeSequenceWrapper(val), preproc, df_holdout, G_holdout)
    else:
        return (NodeSequenceWrapper(trn), NodeSequenceWrapper(val), preproc)



def graph_edges_from_csv(nodes_filepath, 
                         links_filepath,
                         use_lcc=True,
                         sample_size=10,
                         train_pct=0.1, sep=',', verbose=1):
    """
    Loads graph data from CSV files. 
    Returns generators for edges in graph
    for use with GraphSAGE model.
    Args:
        nodes_filepath(str): file path to training CSV containing node attributes
        links_filepath(str): file path to training CSV describing links among nodes
        use_lcc(bool):  If True, consider the largest connected component only.
        sample_size(int): Number of nodes to sample at each neighborhood level
        train_pct(float): Proportion of edges to sample for train (and test)
                        0.1 is recommended.
        sep (str):  delimiter for CSVs. Default is comma.
        verbose (boolean): verbosity
    Return:
        tuple of NodeSequenceWrapper objects for train and validation sets
    """

    # read edge list
    g_nx = nx.read_edgelist(path=links_filepath, delimiter=sep)

    # read node attributes
    node_attr = pd.read_csv(nodes_filepath, sep=sep, header=None)

    # store class labels within graph nodes
    values = { str(row.tolist()[0]): row.tolist()[-1] for _, row in node_attr.iterrows()}
    nx.set_node_attributes(g_nx, values, 'target')

    # select largest connected component
    if use_lcc:
        g_nx_ccs = (g_nx.subgraph(c).copy() for c in nx.connected_components(g_nx))
        g_nx = max(g_nx_ccs, key=len)
        if verbose:
            print("Largest subgraph statistics: {} nodes, {} edges".format(
            g_nx.number_of_nodes(), g_nx.number_of_edges()))


    # set up the features
    num_features = len(node_attr.columns.values) - 2 # subract ID and target
    feature_names = ["w_{}".format(ii) for ii in range(num_features)]
    column_names =  feature_names + ["target"]
    node_data = pd.read_csv(nodes_filepath, header=None, names=column_names, sep=sep)
    node_data.drop(['target'], axis=1, inplace=True)
    node_data.index = node_data.index.map(str)
    node_data = node_data[node_data.index.isin(list(g_nx.nodes()))]
    node_features = node_data[feature_names].values

    # add features to graph
    for nid, f in zip(node_data.index, node_features):
        g_nx.node[nid][sg.globalvar.TYPE_ATTR_NAME] = "nodetype"  # specify node type
        g_nx.node[nid]["feature"] = f


    # Build train and validation set for edges.
    edge_splitter_test = EdgeSplitter(g_nx)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=train_pct, method="global", keep_connected=True
    )

    edge_splitter_train = EdgeSplitter(G_test)

    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=train_pct, method="global", keep_connected=True
    )

    G_train = sg.StellarGraph(G_train, node_features="feature")
    G_test = sg.StellarGraph(G_test, node_features="feature")
    if verbose:
        print('\nTRAIN GRAPH:\n')
        print(G_train.info())
        print('\nTEST GRAPH:\n')
        print(G_test.info())



    return (None, None, None)

