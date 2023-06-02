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

    def summarize(
        self,
        doc,
        max_length=150,
        min_length=56,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        num_beams=4,
        **kwargs,
    ):
        """
        ```
        Summarize document text.  Extra arguments are fed to generate method
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
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs,
            )

            exec_sum = self.tokenizer.decode(
                summary_ids.squeeze(), skip_special_tokens=True
            )
        return exec_sum


class LexRankSummarizer:
    """
    interface to Lexrank-based text summarization
    """

    def __init__(self, language="english"):
        """
        ```
        interface to Lexrank-based text summarization using sumy library

        Args:
          language(str): default is "english"
        ```
        """

        try:
            from sumy.nlp.stemmers import Stemmer
            from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
            from sumy.utils import get_stop_words
        except ImportError:
            raise ImportError("Please install the sumy package: pip install sumy")

        self.language = language
        stemmer = Stemmer(self.language)
        self.summarizer = Summarizer(stemmer)
        self.summarizer.stop_words = get_stop_words(self.language)

    def summarize(
        self,
        doc,
        num_sentences=3,
        maximum_length=2000,
        minimum_length=40,
        join_sentences=True,
        num_candidate_sentences=100,
        **kwargs,
    ):
        """
        ```
        summarize document text
        Args:
          doc(str): text of document
          num_sentences(int): Number of sentences for summary
          maximum_length(int): Maximum length of sentence in summary
          minimumlength(int): Minimum length of sentence in summary
          join_sentences(bool): If True, summary is a single string instead of a list of sentences.
          num_candidate_sentences(int): Number of candidate sentences from which to select final summary.
        Returns:
          str: summary text
        ```
        """
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.html import HtmlParser
        from sumy.parsers.plaintext import PlaintextParser

        parser = PlaintextParser.from_string(doc, Tokenizer(self.language))
        results = []
        for sentence in self.summarizer(parser.document, num_candidate_sentences):
            if (
                len(sentence._text) > maximum_length
                or len(sentence._text) < minimum_length
                or sentence._text[0].isdigit()
            ):
                continue
            results.append(
                sentence._text + "."
                if sentence._text[-1] not in [".", "?", "!", ";"]
                else sentence._text
            )
        return (
            " ".join(results[:num_sentences])
            if join_sentences
            else results[:num_sentences]
        )
