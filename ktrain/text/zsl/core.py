from ...imports import *
from ... import utils as U

class ZeroShotClassifier():
    """
    interface to Zero Shot Topic Classifier
    """

    def __init__(self, model_name='facebook/bart-large-mnli', device=None):
        """
        interface to BART-based text summarization using transformers library

        Args:
          model_name(str): name of BART model
          device(str): device to use (e.g., 'cuda', 'cpu')
        """
        if 'mnli' not in model_name:
            raise ValueError('ZeroShotClasifier requires an MNLI model')
        try:
            import torch
        except ImportError:
            raise Exception('ZeroShotClassifier requires PyTorch to be installed.')
        self.torch_device = device
        if self.torch_device is None: self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from transformers import BartForSequenceClassification, BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForSequenceClassification.from_pretrained(model_name).to(self.torch_device)


    def predict(self, doc, topic_strings=[], include_labels=False):
        """
        zero-shot topic classification
        Args:
          doc(str): text of document
          topic_strings(list): a list of strings representing topics of your choice
                               Example:
                               topic_strings=['political science', 'sports', 'science']
        Returns:
          inferred probabilities
        """
        if topic_strings is None or len(topic_strings) == 0:
            raise ValueError('topic_strings must be a list of strings')
        true_probs = []
        for topic_string in topic_strings:
            premise = doc
            hypothesis = 'This text is about %s.' % (topic_string)
            input_ids = self.tokenizer.encode(premise, hypothesis, return_tensors='pt').to(self.torch_device)
            logits = self.model(input_ids)[0]

            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true 
            # reference: https://joeddav.github.io/blog/2020/05/29/ZSL.html
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            true_prob = probs[:,1].item() 
            true_probs.append(true_prob)
        if include_labels:
            true_probs = list(zip(topic_strings, true_probs))
        return true_probs

