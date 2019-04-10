from abc import ABC, abstractmethod
import pickle

class Predictor(ABC):
    """
    Abstract class to preprocess data
    """
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def get_classes(self, filename):
        pass

    def save(self, fname):
        self.model.save(fname)
        fname_preproc = fname+'.preproc'
        with open(fname_preproc, 'wb') as f:
            pickle.dump(self.preproc, f)
        return



