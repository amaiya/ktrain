from ..imports import *
from .. import utils as U
from ..preprocessor import Preprocessor


class NodePreprocessor(Preprocessor):
    """
    Text preprocessing base class
    """

    def __init__(self, classes):

        self.c = classes


    def get_preprocessor(self):
        raise NotImplementedError


    def get_classes(self):
        return self.c


    def preprocess(self, texts):
        raise NotImplementedError

