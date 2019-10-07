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
                         train_pct=0.1, sep=',', verbose=1):
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
        verbose (boolean): verbosity
    Return:
        tuple of NodeSequenceWrapper objects for train and validation sets
    """

    #----------------------------------------------------------------
    # process graph structure
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
    # process node attributes
    #----------------------------------------------------------------
    node_attr = pd.read_csv(nodes_filepath, sep=sep, header=None)
    num_features = len(node_attr.columns.values) - 2 # subract ID and target
    feature_names = ["w_{}".format(ii) for ii in range(num_features)]
    column_names =  feature_names + ["target"]
    node_data = pd.read_csv(nodes_filepath, header=None, names=column_names, sep=sep)
    node_data.index = node_data.index.map(str)
    node_data = node_data[node_data.index.isin(list(g_nx.nodes()))]

    # split into train and validation
    tr_data, te_data = sklearn.model_selection.train_test_split(node_data, 
                                                        train_size=None, 
                                                        test_size=1-train_pct, 
                                                        stratify=node_data['target'], 
                                                        random_state=42)

    # one-hot-encode target
    target_encoding = sklearn.feature_extraction.DictVectorizer(sparse=False)
    train_targets = target_encoding.fit_transform(tr_data[["target"]].to_dict('records'))
    test_targets = target_encoding.transform(te_data[["target"]].to_dict('records'))
    class_names = list(set([c[0] for c in node_data[['target']].values]))
    class_names.sort()

    #----------------------------------------------------------------
    # setup generators
    #----------------------------------------------------------------

    # create generators for training and validation
    G = sg.StellarGraph(g_nx, node_features=node_data[feature_names])
    generator = GraphSAGENodeGenerator(G, U.DEFAULT_BS, [sample_size, sample_size])
    train_gen = generator.flow(tr_data.index, train_targets, shuffle=True)
    test_gen = generator.flow(te_data.index, test_targets, shuffle=False)

    preproc = NodePreprocessor(class_names)
    return (NodeSequenceWrapper(train_gen), NodeSequenceWrapper(test_gen), preproc)



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

