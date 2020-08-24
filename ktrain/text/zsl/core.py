# 2020-08-10: unnecessary imports removed for ZSL to address #225
#from ...imports import *
#from ... import utils as U

import math
import warnings
import numpy as np

# duplicated from ktrain.utils
def list2chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

class ZeroShotClassifier():
    """
    interface to Zero Shot Topic Classifier
    """

    def __init__(self, model_name='facebook/bart-large-mnli', device=None):
        """
        ZeroShotClassifier constructor

        Args:
          model_name(str): name of a BART NLI model
          device(str): device to use (e.g., 'cuda', 'cpu')
        """
        if 'mnli' not in model_name and 'xnli' not in model_name:
            raise ValueError('ZeroShotClasifier requires an MNLI or XNLI model')
        try:
            import torch
        except ImportError:
            raise Exception('ZeroShotClassifier requires PyTorch to be installed.')
        self.torch_device = device
        if self.torch_device is None: self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.torch_device)


    def predict(self, docs, labels=[], include_labels=False, 
               max_length=512, batch_size=8, nli_template='This text is about {}.',  topic_strings=[]):
        """
        This method performs zero-shot text classification using Natural Language Inference (NLI).
        Args:
          docs(list|str): text of document or list of texts
          labels(list): a list of strings representing topics of your choice
                        Example:
                          labels=['political science', 'sports', 'science']
          include_labels(bool): If True, will return topic labels along with topic probabilities
          max_length(int): truncate long documents to this many tokens
          batch_size(int): batch_size to use. default:8
                           Increase this value to speed up predictions - especially
                           if len(topic_strings) is large.
          nli_template(str): labels are inserted into this template for use as hypotheses in natural language inference
          topic_strings(list): alias for label_strings for backwards compatibility
        Returns:
          inferred probabilities or list of inferred probabilities if doc is list
        """

        # error checks
        is_str_input = False
        if not isinstance(docs, (list, np.ndarray)): 
            docs = [docs]
            is_str_input = True
        if not isinstance(docs[0], str): raise ValueError('docs must be string or a list of strings representing document(s)')
        if len(labels) > 0 and len(topic_strings) > 0: raise ValueError('labels and topic_strings are mutually exclusive')
        if not labels and not topic_strings: raise ValueError('labels must be a list of strings')
        if topic_strings: 
            labels = topic_strings




        # convert to sequences
        sequence_pairs = []
        for premise in docs:
            sequence_pairs.extend([[premise, nli_template.format(label)] for label in labels])
        if batch_size  > len(sequence_pairs): batch_size = len(sequence_pairs)
        if len(sequence_pairs) >= 100 and batch_size==8:
            warnings.warn('TIP: Try increasing batch_size to speedup ZeroShotClassifier predictions')
        num_chunks = math.ceil(len(sequence_pairs)/batch_size)
        sequence_chunks = list2chunks(sequence_pairs, n=num_chunks)

        # inference
        import torch
        with torch.no_grad():
            result = []
            for sequences in sequence_chunks:
                batch = self.tokenizer.batch_encode_plus(sequences, return_tensors='pt', max_length=max_length, truncation='only_first', padding=True).to(self.torch_device)
                logits = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])[0]
                entail_contradiction_logits = logits[:,[0,2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                true_probs = list(probs[:,1].cpu().detach().numpy())
                result.extend(true_probs)

        # format results
        num_labels = len(labels)
        result_per_doc = []
        output = []
        for i, r in enumerate(result):
            l = labels[i%num_labels]
            element = (labels[i%num_labels], r) if include_labels else r
            result_per_doc.append(element)
            if i%num_labels+1 == num_labels:
                output.append(result_per_doc)
                result_per_doc = []
        if is_str_input: output = output[0]
        return output

