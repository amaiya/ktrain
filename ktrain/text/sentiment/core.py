from typing import Union
from transformers import pipeline

from ... import utils as U
from ...torch_base import TorchBase


class SentimentAnalyzer(TorchBase):
    """
    interface to Sentiment Analyzer
    """

    def __init__(self, device=None, **kwargs):
        """
        ```
        ImageCaptioner constructor

        Args:
          device(str): device to use (e.g., 'cuda', 'cpu')
        ```
        """

        super().__init__(
            device=device, quantize=False, min_transformers_version="4.12.3"
        )
        self.pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=self.device_to_id(),
            **kwargs
        )
        self.mapping = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE",
        }

    def predict(
        self,
        texts: Union[str, list],
        return_all_scores=False,
        batch_size=U.DEFAULT_BS,
        **kwargs
    ):
        """
        ```
        Performs sentiment analysis

        This method accepts a list of texts and predicts their sentiment as either 'NEGATIVE', 'NEUTRAL', 'POSITIVE'.
        Args:
            texts: str|list
            return_all_scores(bool): If True, return all labels/scores
            batch_size: size of batches sent to model
        Returns:
            A dictionary of labels and scores

        ```
        """
        str_input = isinstance(texts, str)
        if str_input:
            texts = [texts]
        chunks = U.batchify(texts, batch_size)
        results = []
        for chunk in chunks:
            preds = self.pipeline(
                chunk, top_k=len(self.mapping) if return_all_scores else 1, **kwargs
            )
            results.extend(preds)
        results = [self._flatten_prediction(pred) for pred in results]
        return results[0] if str_input else results

    def _flatten_prediction(self, prediction: list):
        """
        ```
        flatten prediction to the form {'label':score}
        ```
        """
        return_dict = {}
        for d in prediction:
            return_dict[self.mapping[d["label"]]] = d["score"]
        return return_dict
