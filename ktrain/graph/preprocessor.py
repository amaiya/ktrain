from ..imports import *
from .. import utils as U
from ..preprocessor import Preprocessor


class NodePreprocessor(Preprocessor):
    """
    Node preprocessing base class
    """

    def __init__(self, G_nx, df, sample_size=10, missing_label_value=None):

        self.sampsize = sample_size       # neighbor sample size
        self.df = df                      # node attributes and targets
        # TODO: eliminate storage redundancy
        self.G = G_nx                     # networkx graph
        self.G_sg = None                  # StellarGraph 

        # clean df
        df.index = df.index.map(str)
        df= df[df.index.isin(list(self.G.nodes()))]

        # class names
        self.c = list(set([c[0] for c in df[['target']].values]))
        if missing_label_value is not None: self.c.remove(missing_label_value)
        self.c.sort()

        # feature names + target
        self.colnames = list(df.columns.values)
        if self.colnames[-1] != 'target':
            raise ValueError('last column of df must be "target"')

        # set by preprocess_train
        self.y_encoding = None


    def get_preprocessor(self):
        return (self.G, self.df)


    def get_classes(self):
        return self.c

    @property
    def feature_names(self):
        return self.colnames[:-1]


    def preprocess(self, df, G):
        return self.preprocess_test(df, G)


    def ids_exist(self, node_ids):
        """
        check validity of node IDs
        """
        df = self.df[self.df.index.isin(node_ids)]
        return df.shape[0] > 0



    def preprocess_train(self, node_ids):
        """
        preprocess training set
        """
        if not self.ids_exist(node_ids): raise ValueError('node_ids must exist in self.df')

        # subset df for training nodes
        df_tr = self.df[self.df.index.isin(node_ids)]

        # one-hot-encode target
        self.y_encoding = sklearn.feature_extraction.DictVectorizer(sparse=False)
        train_targets = self.y_encoding.fit_transform(df_tr[["target"]].to_dict('records'))



        # import stellargraph
        try:
            import stellargraph as sg
            from stellargraph.mapper import GraphSAGENodeGenerator
        except:
            raise Exception(SG_ERRMSG)
        if version.parse(sg.__version__) < version.parse('0.8'):
            raise Exception(SG_ERRMSG)



        # return generator
        G_sg = sg.StellarGraph(self.G, node_features=self.df[self.feature_names])
        self.G_sg = G_sg
        generator = GraphSAGENodeGenerator(G_sg, U.DEFAULT_BS, [self.sampsize, self.sampsize])
        train_gen = generator.flow(df_tr.index, train_targets, shuffle=True)
        from .sg_wrappers import NodeSequenceWrapper
        return NodeSequenceWrapper(train_gen)


    def preprocess_valid(self, node_ids):
        """
        preprocess validation nodes (transductive inference)
        node_ids (list):  list of node IDs that generator will yield
        """
        if not self.ids_exist(node_ids): raise ValueError('node_ids must exist in self.df')
        if self.y_encoding is None:
            raise Exception('Unset parameters. Are you sure you called preprocess_train first?')

        # subset df for validation nodes
        df_val = self.df[self.df.index.isin(node_ids)]


        # one-hot-encode target
        val_targets = self.y_encoding.transform(df_val[["target"]].to_dict('records'))


        # import stellargraph
        try:
            import stellargraph as sg
            from stellargraph.mapper import GraphSAGENodeGenerator
        except:
            raise Exception(SG_ERRMSG)
        if version.parse(sg.__version__) < version.parse('0.8'):
            raise Exception(SG_ERRMSG)


        # return generator
        if self.G_sg is None:
            self.G_sg = sg.StellarGraph(self.G, node_features=self.df[self.feature_names])
        generator = GraphSAGENodeGenerator(self.G_sg, U.DEFAULT_BS, [self.sampsize,self.sampsize])
        val_gen = generator.flow(df_val.index, val_targets, shuffle=False)
        from .sg_wrappers import NodeSequenceWrapper
        return NodeSequenceWrapper(val_gen)



    def preprocess_test(self, df_te, G_te):
        """
        preprocess for inductive inference
        df_te (DataFrame): pandas dataframe containing new node attributes
        G_te (Graph):  a networkx Graph containing new nodes
        """
        if self.y_encoding is None:
            raise Exception('Unset parameters. Are you sure you called preprocess_train first?')

        # get aggregrated df
        #df_agg = pd.concat([df_te, self.df]).drop_duplicates(keep='last')
        df_agg = pd.concat([df_te, self.df])
        #df_te = pd.concat([self.df, df_agg]).drop_duplicates(keep=False)


        # get aggregrated graph
        is_subset = set(self.G.nodes()) <= set(G_te.nodes())
        if not is_subset:
            raise ValueError('Nodes in self.G must be subset of G_te')
        G_agg = nx.compose(self.G, G_te)    

        
        # one-hot-encode target
        if 'target' in df_te.columns:
            test_targets = self.y_encoding.transform(df_te[["target"]].to_dict('records'))
        else:
            test_targets = [-1] * len(df_te.shape[0])


        # import stellargraph
        try:
            import stellargraph as sg
            from stellargraph.mapper import GraphSAGENodeGenerator
        except:
            raise Exception(SG_ERRMSG)
        if version.parse(sg.__version__) < version.parse('0.8'):
            raise Exception(SG_ERRMSG)


        # return generator
        G_sg = sg.StellarGraph(G_agg, node_features=df_agg[self.feature_names])
        generator = GraphSAGENodeGenerator(G_sg, U.DEFAULT_BS, [self.sampsize,self.sampsize])
        test_gen = generator.flow(df_te.index, test_targets, shuffle=False)
        from .sg_wrappers import NodeSequenceWrapper
        return NodeSequenceWrapper(test_gen)




class LinkPreprocessor(Preprocessor):
    """
    Link preprocessing base class
    """

    def __init__(self, G,  sample_sizes=[10, 20]):
        self.sample_sizes = sample_sizes
        self.G = G # original graph under consideration with all original links


        # class names
        self.c = ['negative', 'positive']


    def get_preprocessor(self):
        return self


    def get_classes(self):
        return self.c


    def preprocess(self, G, edge_ids):
        edge_labels = [1] * len(edge_ids)
        return self.preprocess_valid(G, edge_ids, edge_labels)


    def preprocess_train(self, G, edge_ids, edge_labels, mode='train'):
        """
        preprocess training set
        Args:
          G (networkx graph): networkx graph
          edge_ids(list): list of tuples representing edge ids
          edge_labels(list): edge labels (1 or 0 to indicated whether it is a true edge in original graph or not)
        """
        # import stellargraph
        try:
            import stellargraph as sg
            from stellargraph.mapper import GraphSAGELinkGenerator
        except:
            raise Exception(SG_ERRMSG)
        if version.parse(sg.__version__) < version.parse('0.8'):
            raise Exception(SG_ERRMSG)

        #edge_labels = to_categorical(edge_labels)
        G_sg = sg.StellarGraph(G, node_features="feature")
        #print(G_sg.info())
        shuffle = True if mode == 'train' else False
        link_seq = GraphSAGELinkGenerator(G_sg, U.DEFAULT_BS, self.sample_sizes).flow(edge_ids, edge_labels, shuffle=shuffle)
        from .sg_wrappers import LinkSequenceWrapper
        return LinkSequenceWrapper(link_seq)


    def preprocess_valid(self, G, edge_ids, edge_labels):
        """
        preprocess training set
        Args:
          G (networkx graph): networkx graph
          edge_ids(list): list of tuples representing edge ids
          edge_labels(list): edge labels (1 or 0 to indicated whether it is a true edge in original graph or not)
        """
        return self.preprocess_train(G, edge_ids, edge_labels, mode='valid')


