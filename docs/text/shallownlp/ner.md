Module ktrain.text.shallownlp.ner
=================================

Classes
-------

`NER(lang='en', predictor_path=None)`
:   pretrained NER.
    Only English and Chinese are currenty supported.
    
    Args:
      lang(str): Currently, one of {'en', 'zh', 'ru'}: en=English , zh=Chinese, or ru=Russian

    ### Methods

    `predict(self, texts, merge_tokens=True, batch_size=32)`
    :   Extract named entities from supplied text
        
        Args:
          texts (list of str or str): list of texts to annotate
          merge_tokens(bool):  If True, tokens will be merged together by the entity
                               to which they are associated:
                               ('Paul', 'B-PER'), ('Newman', 'I-PER') becomes ('Paul Newman', 'PER')
          batch_size(int):    Batch size to use for predictions (default:32)