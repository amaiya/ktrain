from .. import utils as U
from ..imports import *
from ..predictor import Predictor
from .preprocessor import LinkPreprocessor, NodePreprocessor


class NodePredictor(Predictor):
    """
    ```
    predicts graph node's classes
    ```
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):
        if not isinstance(model, keras.Model):
            raise ValueError("model must be of instance keras.Model")
        if not isinstance(preproc, NodePreprocessor):
            raise ValueError("preproc must be a NodePreprocessor object")
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size

    def get_classes(self):
        return self.c

    def predict(self, node_ids, return_proba=False, verbose=0):
        return self.predict_transductive(
            node_ids, return_proba=return_proba, verbose=verbose
        )

    def predict_transductive(self, node_ids, return_proba=False, verbose=0):
        """
        ```
        Performs transductive inference.
        If return_proba is True, returns probabilities of each class.
        ```
        """
        gen = self.preproc.preprocess_valid(node_ids)
        gen.batch_size = self.batch_size
        # *_generator methods are deprecated from TF 2.1.0
        # preds = self.model.predict_generator(gen)
        preds = self.model.predict(gen, verbose=verbose)
        result = preds if return_proba else [self.c[np.argmax(pred)] for pred in preds]
        return result

    def predict_inductive(self, df, G, return_proba=False, verbose=0):
        """
        ```
        Performs inductive inference.
        If return_proba is True, returns probabilities of each class.
        ```
        """

        gen = self.preproc.preprocess(df, G)
        gen.batch_size = self.batch_size
        # *_generator methods are deprecated from TF 2.1.0
        # preds = self.model.predict_generator(gen)
        preds = self.model.predict(gen, verbose=verbose)
        result = preds if return_proba else [self.c[np.argmax(pred)] for pred in preds]
        return result


class LinkPredictor(Predictor):
    """
    ```
    predicts graph node's classes
    ```
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):
        if not isinstance(model, keras.Model):
            raise ValueError("model must be of instance keras.Model")
        if not isinstance(preproc, LinkPreprocessor):
            raise ValueError("preproc must be a LinkPreprocessor object")
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size

    def get_classes(self):
        return self.c

    def predict(self, G, edge_ids, return_proba=False, verbose=0):
        """
        ```
        Performs link prediction
        If return_proba is True, returns probabilities of each class.
        ```
        """
        gen = self.preproc.preprocess(G, edge_ids)
        gen.batch_size = self.batch_size
        # *_generator methods are deprecated from TF 2.1.0
        # preds = self.model.predict_generator(gen)
        preds = self.model.predict(gen, verbose=verbose)
        preds = np.squeeze(preds)
        if return_proba:
            return [[1 - pred, pred] for pred in preds]
        result = np.where(preds > 0.5, self.c[1], self.c[0])
        return result
