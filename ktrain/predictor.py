from .imports import *
from . import utils as U
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

    def explain(self, x):
        raise NotImplementedError('explain is not currently supported for this model')

    def save(self, fname):

        if U.is_crf(self.model):
            from .text.ner import crf_loss
            self.model.compile(loss=crf_loss, optimizer=U.DEFAULT_OPT)

        self.model.save(fname, save_format='h5')
        fname_preproc = fname+'.preproc'
        with open(fname_preproc, 'wb') as f:
            pickle.dump(self.preproc, f)
        return



