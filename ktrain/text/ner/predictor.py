from ...imports import *
from ...predictor import Predictor
from .preprocessor import NERPreprocessor, tokenize
from ... import utils as U

class NERPredictor(Predictor):
    """
    predicts  classes for string-representation of sentence
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):

        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        if not isinstance(preproc, NERPreprocessor):
        #if type(preproc).__name__ != 'NERPreprocessor':
            raise ValueError('preproc must be a NERPreprocessor object')
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size 


    def get_classes(self):
        return self.c


    def predict(self, sentence):
        """
        Makes predictions for a string-representation of a sentence
        If return_proba is True, returns probabilities of each class.
        """
        if not isinstance(sentence, str):
            raise ValueError('Param sentence must be a string-representation of a sentence')
        nerseq = self.preproc.preprocess([sentence])
        nerseq.batch_size = self.batch_size 
        x_true, _ = nerseq[0]
        lengths = nerseq.get_lengths(0)
        y_pred = self.model.predict_on_batch(x_true)
        y_pred = self.preproc.p.inverse_transform(y_pred, lengths)
        y_pred = y_pred[0]
        return list(zip(nerseq.x[0], y_pred))
