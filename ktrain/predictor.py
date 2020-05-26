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


    def _make_predictor_folder(self, fpath):
        if os.path.isfile(fpath):
            raise ValueError(f'There is an existing file named {fpath}. ' +\
                              'Please use dfferent value for fpath.')
        elif os.path.exists(fpath):
            #warnings.warn('predictor files are being saved to folder that already exists: %s' % (fpath))
            pass
        elif not os.path.exists(fpath):
            os.makedirs(fpath)
        return


    def _save_preproc(self, fpath):
        with open(os.path.join(fpath, U.PREPROC_NAME), 'wb') as f:
            pickle.dump(self.preproc, f)
        return


    def _save_model(self, fpath):
        if U.is_crf(self.model): # TODO: fix/refactor this
            from .text.ner import crf_loss
            self.model.compile(loss=crf_loss, optimizer=U.DEFAULT_OPT)
        model_path = os.path.join(fpath, U.MODEL_NAME)
        self.model.save(model_path, save_format='h5')
        return



    def save(self, fpath):
        """
        saves both model and Preprocessor instance associated with Predictor 
        Args:
          fpath(str): path to folder to store model and Preprocessor instance (.preproc file)
        Returns:
          None
        """
        self._make_predictor_folder(fpath)
        self._save_model(fpath)
        self._save_preproc(fpath)
        return

