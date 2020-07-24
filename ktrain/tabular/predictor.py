from ..imports import *
from ..predictor import Predictor
from .. import utils as U
from .preprocessor import TabularPreprocessor

class TabularPredictor(Predictor):
    """
    predictions for tabular data
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):

        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        if not isinstance(preproc, TabularPreprocessor):
            raise ValueError('preproc must be a NERPreprocessor object')
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size 


    def get_classes(self):
        return self.c


    def predict(self, df, return_proba=False):
        """
        Makes predictions for a test dataframe
        Args:
          df(pd.DataFrame):  a pandas DataFrame in same format as DataFrame used for training model
          return_proba(bool): If True, return probabilities instead of predicted class labels
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pd.DataFrame')

        classification, multilabel = U.is_classifier(self.model)

        # get predictions
        tseq = self.preproc.preprocess_test(df, verbose=0)
        tseq.batch_size = self.batch_size
        preds = self.model.predict(tseq)
        result =  preds if return_proba or multilabel or not self.c else [self.c[np.argmax(pred)] for pred in preds] 
        if multilabel and not return_proba:
            result =  [list(zip(self.c, r)) for r in result]
        return result

