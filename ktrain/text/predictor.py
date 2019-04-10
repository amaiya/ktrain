from keras.engine.training import Model
import numpy as np
import math
from ..predictor import Predictor
from .preprocessor import TextPreprocessor
from sklearn.metrics import classification_report, confusion_matrix
from .. import utils as U
import warnings

class TextPredictor(Predictor):
    """
    predicts text classes
    """

    def __init__(self, model, preproc):

        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        if not isinstance(preproc, TextPreprocessor):
        #if type(preproc).__name__ != 'TextPreprocessor':
            raise ValueError('preproc must be a TextPreprocessor object')
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()

    def get_classes(self):
        return self.c

    def predict(self, texts, return_proba=False):
        """
        Makes predictions for a list of strings where each string is a document
        or text snippet.
        If return_proba is True, returns probabilities of each class.
        """
        if not isinstance(texts, np.ndarray) and not isinstance(texts, list):
            raise ValueError('data must be numpy.ndarray or list (of texts)')
        loss = self.model.loss
        treat_multilabel = False
        if loss != 'categorical_crossentropy' and not return_proba:
            return_proba=True
            treat_multilabel = True
        texts = self.preproc.preprocess(texts)
        preds = self.model.predict(texts)
        result =  preds if return_proba else [self.c[np.argmax(pred)] for pred in preds] 
        if treat_multilabel:
            return [list(zip(self.c, r)) for r in result]
        else:
            return result



    def analyze_valid(self, val_tup, print_report=True, multilabel=None):
        """
        Makes predictions on validation set and returns the confusion matrix.
        Accepts as input the validation set in the standard form of a tuple of
        two arrays: (X_test, y_test), wehre X_test is a Numpy array of strings
        where each string is a document or text snippet in the validation set.

        Optionally prints a classification report.
        Currently, this method is only supported for binary and multiclass 
        problems, not multilabel classification problems.
        """
        U.data_arg_check(val_data=val_tup, val_required=True, ndarray_only=True)
        if multilabel is None:
            multilabel = U.is_multilabel(val_tup)
        if multilabel:
            warnings.warn('multilabel_confusion_matrix not yet supported')
            return

        y_true = val_tup[1]
        y_true = np.argmax(y_true, axis=1)
        y_pred = self.model.predict(val_tup[0])
        y_pred = np.argmax(y_pred, axis=1)
        
        if print_report:
            print(classification_report(y_true, y_pred, target_names=self.c))
            cm_func = confusion_matrix
        cm =  confusion_matrix(y_true,  y_pred)
        return cm
