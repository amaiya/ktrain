from ..dataset import SequenceDataset
from ..imports import *

# import stellargraph
try:
    import stellargraph as sg
    from stellargraph.mapper import link_mappers, node_mappers
except:
    raise Exception(SG_ERRMSG)
if version.parse(sg.__version__) < version.parse("0.8"):
    raise Exception(SG_ERRMSG)


class NodeSequenceWrapper(node_mappers.NodeSequence, SequenceDataset):
    def __init__(self, node_seq):
        if not isinstance(node_seq, node_mappers.NodeSequence):
            raise ValueError("node_seq must by a stellargraph NodeSequence object")
        self.node_seq = node_seq
        self.targets = node_seq.targets
        self.generator = node_seq.generator
        self.ids = node_seq.ids
        self.__len__ = node_seq.__len__
        self.__getitem__ = node_seq.__getitem__
        self.on_epoch_end = node_seq.on_epoch_end
        self.indices = node_seq.indices

    def __setattr__(self, name, value):
        if name == "batch_size":
            self.generator.batch_size = value
        elif name == "data_size":
            self.node_seq.data_size = value
        elif name == "shuffle":
            self.node_seq.shuffle = value
        elif name == "head_node_types":
            self.node_seq.head_node_types = value
        elif name == "_sampling_schema":
            self.node_seq._sample_schema = value
        else:
            self.__dict__[name] = value
        return

    def __getattr__(self, name):
        if name == "batch_size":
            return self.generator.batch_size
        elif name == "data_size":
            return self.node_seq.data_size
        elif name == "shuffle":
            return self.node_seq.shuffle
        elif name == "head_node_types":
            return self.node_seq.head_node_types
        elif name == "_sampling_schema":
            return self.node_seq._sampling_schema
        elif name == "reset":
            # stellargraph did not implement reset for its generators
            # return a zero-argument lambda that returns None
            return lambda: None
        elif name == "graph":
            return self.generator.graph
        else:
            try:
                return self.__dict__[name]
            except:
                raise AttributeError
        return

    def nsamples(self):
        return self.targets.shape[0]

    def get_y(self):
        return self.targets

    def xshape(self):
        return self[0][0][0].shape[1:]  # returns 1st neighborhood only

    def nclasses(self):
        return self[0][1].shape[1]


class LinkSequenceWrapper(link_mappers.LinkSequence, SequenceDataset):
    def __init__(self, link_seq):
        if not isinstance(link_seq, link_mappers.LinkSequence):
            raise ValueError("link_seq must by a stellargraph LinkSequence object")
        self.link_seq = link_seq
        self.targets = link_seq.targets
        self.generator = link_seq.generator
        self.ids = link_seq.ids
        self.__len__ = link_seq.__len__
        self.__getitem__ = link_seq.__getitem__
        self.on_epoch_end = link_seq.on_epoch_end
        self.indices = link_seq.indices

    def __setattr__(self, name, value):
        if name == "batch_size":
            self.generator.batch_size = value
        elif name == "data_size":
            self.link_seq.data_size = value
        elif name == "shuffle":
            self.link_seq.shuffle = value
        elif name == "head_node_types":
            self.link_seq.head_node_types = value
        elif name == "_sampling_schema":
            self.link_seq._sample_schema = value
        else:
            self.__dict__[name] = value
        return

    def __getattr__(self, name):
        if name == "batch_size":
            return self.generator.batch_size
        elif name == "data_size":
            return self.link_seq.data_size
        elif name == "shuffle":
            return self.link_seq.shuffle
        elif name == "head_node_types":
            return self.link_seq.head_node_types
        elif name == "_sampling_schema":
            return self.link_seq._sampling_schema
        elif name == "reset":
            # stellargraph did not implement reset for its generators
            # return a zero-argument lambda that returns None
            return lambda: None
        elif name == "graph":
            return self.generator.graph
        else:
            try:
                return self.__dict__[name]
            except:
                raise AttributeError
        return

    def nsamples(self):
        return self.targets.shape[0]

    def get_y(self):
        return self.targets

    def xshape(self):
        return self[0][0][0].shape[1:]  # returns 1st neighborhood only

    def nclasses(self):
        return 2
        return self[0][1].shape[1]
