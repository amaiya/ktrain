from stellargraph.mapper import node_mappers

class NodeSequenceWrapper(node_mappers.NodeSequence):
    def __init__(self, node_seq):
        if not isinstance(node_seq, node_mappers.NodeSequence):
            raise ValueError('node_seq must by a stellargraph NodeSequene object')
        self.node_seq = node_seq
        self.targets = node_seq.targets
        self.generator = node_seq.generator
        self.ids = node_seq.ids
        self.__len__ = node_seq.__len__
        self.__getitem__ = node_seq.__getitem__
        self.on_epoch_end = node_seq.on_epoch_end
        self.indices = node_seq.indices




    def __setattr__(self, name, value):
        if name == 'batch_size':
            self.generator.batch_size = value
        elif name == 'data_size':
            self.node_seq.data_size = value
        elif name == 'shuffle':
            self.node_seq.shuffle = value
        elif name == 'head_node_types':
            self.node_seq.head_node_types = value
        elif name == '_sampling_schema':
            self.node_seq._sample_schema = value
        else:
            self.__dict__[name] = value
        return



    def __getattr__(self, name):
        if name == 'batch_size':
            return self.generator.batch_size
        elif name == 'data_size':
            return self.node_seq.data_size
        elif name == 'shuffle':
            return self.node_seq.shuffle
        elif name == 'head_node_types':
            return self.node_seq.head_node_types
        elif name == '_sampling_schema':
            return self.node_seq._sampling_schema
        elif name == 'reset':
            # stellargraph did not implement reset for its generators
            # return a zero-argument lambda that returns None
            return lambda:None 
        else:
            try:
                return self.__dict__[name] 
            except:
                raise AttributeError
        return

