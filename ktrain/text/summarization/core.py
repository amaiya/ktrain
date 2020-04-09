from ...imports import *
from ... import utils as U

class TransformerSummarizer():
    """
    interface to Transformer-based text summarization
    """

    def __init__(self, model_name='bart-large-cnn'):
        """
        interface to BART-based text summarization using transformers library

        Args:
          model_name(str): name of BART model
        """
        if model_name.split('-')[0] != 'bart': 
            raise ValueError('TransformerSummarizer currently only accepts BART models')
        try:
            import torch
        except ImportError:
            raise Exception('TransformerSummarizer requires PyTorch to be installed.')
        from transformers import BartTokenizer, BartForConditionalGeneration
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)


    def summarize(self, doc):
        """
        summarize document text
        Args:
          doc(str): text of document
        Returns:
          str: summary text
        """

        answers_input_ids = self.tokenizer.batch_encode_plus([doc], 
                                                             return_tensors='pt', 
                                                             max_length=1024)['input_ids'].to(self.torch_device)
        summary_ids = self.model.generate(answers_input_ids,
                                          num_beams=4,
                                          length_penalty=2.0,
                                          max_length=142,
                                          min_length=56,
                                          no_repeat_ngram_size=3)

        exec_sum = self.tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return exec_sum
