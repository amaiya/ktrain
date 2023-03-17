from ... import utils as U
from ...imports import *
from ...torch_base import TorchBase
from .. import textutils as TU

SUPPORTED_SRC_LANGS = ["zh", "ar", "ru", "de", "af", "es", "fr", "it", "pt"]


class Translator(TorchBase):
    """
    Translator: basic wrapper around MarianMT model for language translation
    """

    def __init__(self, model_name=None, device=None, quantize=False):
        """
        ```
        basic wrapper around MarianMT model for language translation

        Args:
          model_name(str): Helsinki-NLP model
          device(str): device to use (e.g., 'cuda', 'cpu')
          quantize(bool): If True, use quantization.
        ```
        """
        if "Helsinki-NLP" not in model_name:
            warnings.warn(
                "Translator requires a Helsinki-NLP model: https://huggingface.co/Helsinki-NLP"
            )
        super().__init__(device=device, quantize=quantize)
        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.torch_device)
        if quantize:
            self.model = self.quantize_model(self.model)

    def translate(self, src_text, join_with="\n", num_beams=1, early_stopping=False):
        """
        ```
        Translate document (src_text).
        To speed up translations, you can set num_beams and early_stopping (e.g., num_beams=4, early_stopping=True).
        Args:
          src_text(str): source text.
                         The source text can either be a single sentence or an entire document with multiple sentences
                         and paragraphs.
                         IMPORTANT NOTE: Sentences are joined together and fed to model as single batch.
                                         If the input text is very large (e.g., an entire book), you should
                                         break it up into reasonbly-sized chunks (e.g., pages, paragraphs, or sentences) and
                                         feed each chunk separately into translate to avoid out-of-memory issues.
          join_with(str):  list of translated sentences will be delimited with this character.
                           default: each sentence on separate line
          num_beams(int): Number of beams for beam search. Defaults to None.  If None, the transformers library defaults this to 1,
                          whicn means no beam search.
          early_stopping(bool):  Whether to stop the beam search when at least ``num_beams`` sentences
                                 are finished per batch or not. Defaults to None.  If None, the transformers library
                                 sets this to False.
        Returns:
          str: translated text
        ```
        """
        sentences = TU.sent_tokenize(src_text)
        tgt_sentences = self.translate_sentences(
            sentences, num_beams=num_beams, early_stopping=early_stopping
        )
        return join_with.join(tgt_sentences)

    def translate_sentences(self, sentences, num_beams=1, early_stopping=False):
        """
        ```
        Translate sentences using model_name as model.
        To speed up translations, you can set num_beams and early_stopping (e.g., num_beams=4, early_stopping=True).
        Args:
          sentences(list): list of strings representing sentences that need to be translated
                         IMPORTANT NOTE: Sentences are joined together and fed to model as single batch.
                                         If the input text is very large (e.g., an entire book), you should
                                         break it up into reasonbly-sized chunks (e.g., pages, paragraphs, or sentences) and
                                         feed each chunk separately into translate to avoid out-of-memory issues.
          num_beams(int): Number of beams for beam search. Defaults to None.  If None, the transformers library defaults this to 1,
                          whicn means no beam search.
          early_stopping(bool):  Whether to stop the beam search when at least ``num_beams`` sentences
                                 are finished per batch or not. Defaults to None.  If None, the transformers library
                                 sets this to False.
        Returns:
          str: translated sentences
        ```
        """
        import torch

        with torch.no_grad():
            translated = self.model.generate(
                **self.tokenizer.prepare_seq2seq_batch(
                    sentences, return_tensors="pt"
                ).to(self.torch_device),
                num_beams=num_beams,
                early_stopping=early_stopping
            )
            tgt_sentences = [
                self.tokenizer.decode(t, skip_special_tokens=True) for t in translated
            ]
        return tgt_sentences


class EnglishTranslator:
    """
    Class to translate text in various languages to English.
    """

    def __init__(self, src_lang=None, device=None, quantize=False):
        """
        ```
        Constructor for English translator

        Args:
          src_lang(str): language code of source language.
                         Must be one of SUPPORTED_SRC_LANGS:
                           'zh': Chinese (either tradtional or simplified)
                           'ar': Arabic
                           'ru' : Russian
                           'de': German
                           'af': Afrikaans
                           'es': Spanish
                           'fr': French
                           'it': Italian
                           'pt': Portuguese
          device(str): device to use (e.g., 'cuda', 'cpu')
          quantize(bool): If True, use quantization.
        ```
        """

        if src_lang is None or src_lang not in SUPPORTED_SRC_LANGS:
            raise ValueError(
                "A src_lang must be supplied and be one of: %s" % (SUPPORTED_SRC_LANGS)
            )
        self.src_lang = src_lang
        self.translators = []
        if src_lang == "ar":
            self.translators.append(
                Translator(
                    model_name="Helsinki-NLP/opus-mt-ar-en",
                    device=device,
                    quantize=quantize,
                )
            )
        elif src_lang == "ru":
            self.translators.append(
                Translator(
                    model_name="Helsinki-NLP/opus-mt-ru-en",
                    device=device,
                    quantize=quantize,
                )
            )
        elif src_lang == "de":
            self.translators.append(
                Translator(
                    model_name="Helsinki-NLP/opus-mt-de-en",
                    device=device,
                    quantize=quantize,
                )
            )
        elif src_lang == "af":
            self.translators.append(
                Translator(
                    model_name="Helsinki-NLP/opus-mt-af-en",
                    device=device,
                    quantize=quantize,
                )
            )
        elif src_lang in ["es", "fr", "it", "pt"]:
            self.translators.append(
                Translator(
                    model_name="Helsinki-NLP/opus-mt-ROMANCE-en",
                    device=device,
                    quantize=quantize,
                )
            )
        # elif src_lang == 'zh': # could not find zh->en model, so currently doing two-step translation to English via German
        # self.translators.append(Translator(model_name='Helsinki-NLP/opus-mt-ZH-de', device=device))
        # self.translators.append(Translator(model_name='Helsinki-NLP/opus-mt-de-en', device=device))
        elif src_lang == "zh":
            self.translators.append(
                Translator(
                    model_name="Helsinki-NLP/opus-mt-zh-en",
                    device=device,
                    quantize=quantize,
                )
            )
        else:
            raise ValueError("lang:%s is currently not supported." % (src_lang))

    def translate(self, src_text, join_with="\n", num_beams=1, early_stopping=False):
        """
        ```
        Translate source document to English.
        To speed up translations, you can set num_beams and early_stopping (e.g., num_beams=4, early_stopping=True).

        Args:
          src_text(str): source text. Must be in language specified by src_lang (language code) supplied to constructor
                         The source text can either be a single sentence or an entire document with multiple sentences
                         and paragraphs.
                         IMPORTANT NOTE: Sentences are joined together and fed to model as single batch.
                                         If the input text is very large (e.g., an entire book), you should
                                         break it up into reasonbly-sized chunks (e.g., pages, paragraphs, or sentences) and
                                         feed each chunk separately into translate to avoid out-of-memory issues.
          join_with(str):  list of translated sentences will be delimited with this character.
                           default: each sentence on separate line
          num_beams(int): Number of beams for beam search. Defaults to None.  If None, the transformers library defaults this to 1,
                          whicn means no beam search.
          early_stopping(bool):  Whether to stop the beam search when at least ``num_beams`` sentences
                                 are finished per batch or not. Defaults to None.  If None, the transformers library
                                 sets this to False.
        Returns:
          str: translated text
        ```
        """
        text = src_text
        for t in self.translators:
            text = t.translate(
                text,
                join_with=join_with,
                num_beams=num_beams,
                early_stopping=early_stopping,
            )
        return text
