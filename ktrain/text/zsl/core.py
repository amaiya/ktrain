import math
import warnings

import numpy as np

from ... import utils as U
from ...torch_base import TorchBase

list2chunks = U.list2chunks


class ZeroShotClassifier(TorchBase):
    """
    interface to Zero Shot Topic Classifier
    """

    def __init__(
        self, model_name="facebook/bart-large-mnli", device=None, quantize=False
    ):
        """
        ```
        ZeroShotClassifier constructor

        Args:
          model_name(str): name of a BART NLI model
          device(str): device to use (e.g., 'cuda', 'cpu')
          quantize(bool): If True, faster quantization will be used
        ```
        """
        if "mnli" not in model_name and "xnli" not in model_name:
            raise ValueError("ZeroShotClasifier requires an MNLI or XNLI model")

        super().__init__(device=device, quantize=quantize)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.torch_device
        )
        if quantize:
            self.model = self.quantize_model(self.model)

    def predict(
        self,
        docs,
        labels=[],
        include_labels=False,
        multilabel=True,
        max_length=512,
        batch_size=8,
        nli_template="This text is about {}.",
        topic_strings=[],
    ):
        """
        ```
        This method performs zero-shot text classification using Natural Language Inference (NLI).
        Args:
          docs(list|str): text of document or list of texts
          labels(list): a list of strings representing topics of your choice
                        Example:
                          labels=['political science', 'sports', 'science']
          include_labels(bool): If True, will return topic labels along with topic probabilities
          multilabel(bool): If True, labels are considered independent and multiple labels can predicted true for document and be close to 1.
                            If False, scores are normalized such that probabilities sum to 1.
          max_length(int): truncate long documents to this many tokens
          batch_size(int): batch_size to use. default:8
                           Increase this value to speed up predictions - especially
                           if len(topic_strings) is large.
          nli_template(str): labels are inserted into this template for use as hypotheses in natural language inference
          topic_strings(list): alias for labels parameter for backwards compatibility
        Returns:
          inferred probabilities or list of inferred probabilities if doc is list
        ```
        """

        # error checks
        is_str_input = False
        if not isinstance(docs, (list, np.ndarray)):
            docs = [docs]
            is_str_input = True
        if not isinstance(docs[0], str):
            raise ValueError(
                "docs must be string or a list of strings representing document(s)"
            )
        if len(labels) > 0 and len(topic_strings) > 0:
            raise ValueError("labels and topic_strings are mutually exclusive")
        if not labels and not topic_strings:
            raise ValueError("labels must be a list of strings")
        if topic_strings:
            labels = topic_strings

        # convert to sequences
        sequence_pairs = []
        for premise in docs:
            sequence_pairs.extend(
                [[premise, nli_template.format(label)] for label in labels]
            )
        if batch_size > len(sequence_pairs):
            batch_size = len(sequence_pairs)
        if len(sequence_pairs) >= 100 and batch_size == 8:
            warnings.warn(
                "TIP: Try increasing batch_size to speedup ZeroShotClassifier predictions"
            )
        num_chunks = math.ceil(len(sequence_pairs) / batch_size)
        sequence_chunks = list2chunks(sequence_pairs, n=num_chunks)

        # inference
        import torch

        with torch.no_grad():
            outputs = []
            for sequences in sequence_chunks:
                batch = self.tokenizer.batch_encode_plus(
                    sequences,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation="only_first",
                    padding=True,
                ).to(self.torch_device)
                logits = self.model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=False,
                )[0]
                outputs.extend(logits.cpu().detach().numpy())
                # entail_contradiction_logits = logits[:,[0,2]]

                # probs = entail_contradiction_logits.softmax(dim=1)
                # true_probs = list(probs[:,1].cpu().detach().numpy())
                # result.extend(true_probs)
        outputs = np.array(outputs)
        outputs = outputs.reshape((len(docs), len(labels), -1))

        # process outputs
        # 2020-08-24: modified based on transformers pipeline implementation
        if multilabel:
            # softmax over the entailment vs. contradiction dim for each label independently
            entail_contr_logits = outputs[..., [0, -1]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(
                -1, keepdims=True
            )
            scores = scores[..., 1]
        else:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = outputs[..., -1]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(
                -1, keepdims=True
            )
        scores = scores.tolist()
        if include_labels:
            scores = [list(zip(labels, s)) for s in scores]
        if is_str_input:
            scores = scores[0]
        return scores
