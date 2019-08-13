from .imports import *
class Preprocessor(ABC):
    """
    Abstract class to preprocess data
    """
    @abstractmethod
    def get_preprocessor(self):
        pass
    @abstractmethod
    def get_classes(self):
        pass
    @abstractmethod
    def preprocess(self):
        pass

    def undo(self, data_instance):
        return data_instance




