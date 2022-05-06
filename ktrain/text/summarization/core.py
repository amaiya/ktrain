from ...torch_base import TorchBase


class TransformerSummarizer(TorchBase):
    """
    interface to Transformer-based text summarization
    """

    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        """
        ```
        interface to BART-based text summarization using transformers library

        Args:
          model_name(str): name of BART model for summarization
          device(str): device to use (e.g., 'cuda', 'cpu')
        ```
        """
        if "bart" not in model_name:
            raise ValueError("TransformerSummarizer currently only accepts BART models")
        super().__init__(device=device)
        from transformers import BartForConditionalGeneration, BartTokenizer

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(
            self.torch_device
        )

    def summarize(self, doc):
        """
        ```
        summarize document text
        Args:
          doc(str): text of document
        Returns:
          str: summary text
        ```
        """
        import torch

        with torch.no_grad():
            answers_input_ids = self.tokenizer.batch_encode_plus(
                [doc], return_tensors="pt", truncation=True, max_length=1024
            )["input_ids"].to(self.torch_device)
            summary_ids = self.model.generate(
                answers_input_ids,
                num_beams=4,
                length_penalty=2.0,
                max_length=142,
                min_length=56,
                no_repeat_ngram_size=3,
            )

            exec_sum = self.tokenizer.decode(
                summary_ids.squeeze(), skip_special_tokens=True
            )
        return exec_sum
