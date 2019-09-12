from ...imports import *
from ...predictor import Predictor
from .preprocessor import NERPreprocessor, tokenize
from ... import utils as U

class NERPredictor(Predictor):
    """
    predicts  classes for string-representation of sentence
    """

    def __init__(self, model, preproc):

        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        if not isinstance(preproc, NERPreprocessor):
        #if type(preproc).__name__ != 'NERPreprocessor':
            raise ValueError('preproc must be a NERPreprocessor object')
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()


    def get_classes(self):
        return self.c


    def predict(self, sentence, return_proba=False, include_tokens=True):
        """
        Makes predictions for a string-representation of a sentence
        If return_proba is True, returns probabilities of each class.
        """
        if not isinstance(sentence, str):
            raise ValueError('Param sentence must be a string-representation of a sentence')
        tokens = tokenize(sentence)
        x = self.preproc.preprocess(sentence)
        preds = self.model.predict(x)
        preds = np.squeeze(preds)
        preds = preds[:len(tokens)]
        result =  preds if return_proba else [self.c[np.argmax(pred)] for pred in preds] 
        #result = result if return_proba else np.array([r for r in result if r != PAD] )
        if include_tokens: result = list(zip(tokens, result))
        return result



    def predict_proba(self, texts):
        """
        Makes predictions for a string-represneta
        or text snippet.
        Returns probabilities of each class.
        """
        return self.predict(texts, return_proba=True)



