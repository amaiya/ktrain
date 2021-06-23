Module ktrain.text.translation.core
===================================

Classes
-------

`EnglishTranslator(src_lang=None, device=None)`
:   Class to translate text in various languages to English.
    
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

    ### Methods

    `translate(self, src_text, join_with='\n', num_beams=None, early_stopping=None)`
    :   Translate source document to English.
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

`Translator(model_name=None, device=None, half=False)`
:   Translator: basic wrapper around MarianMT model for language translation
    
    basic wrapper around MarianMT model for language translation
    
    Args:
      model_name(str): Helsinki-NLP model
      device(str): device to use (e.g., 'cuda', 'cpu')
      half(bool): If True, use half precision.

    ### Methods

    `translate(self, src_text, join_with='\n', num_beams=None, early_stopping=None)`
    :   Translate document (src_text).
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

    `translate_sentences(self, sentences, num_beams=None, early_stopping=None)`
    :   Translate sentences using model_name as model.
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