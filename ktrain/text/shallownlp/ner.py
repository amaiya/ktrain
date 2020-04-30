from .imports import *


class NER:
    def __init__(self, lang='en', predictor_path=None):
        """
        pretrained NER.
        Only English and Chinese are currenty supported.

        Args:
          lang(str): Currently, one of {'en', 'zh', 'ru'}: en=English , zh=Chinese, or ru=Russian
        """
        if lang is None:
            raise ValueError('lang is required (e.g., "en" for English, "zh" for Chinese, "ru" for Russian, etc.')
        if predictor_path is None and lang not in ['en', 'zh', 'ru']:
            raise ValueError("Unsupported language: if predictor_path is None,  then lang must be " +\
                             "'en' for English, 'zh' for Chinese, or 'ru' for Chinese")
        self.lang = lang
        if os.environ.get('DISABLE_V2_BEHAVIOR', None) != '1':
            warnings.warn("Please add os.environ['DISABLE_V2_BEHAVIOR'] = '1' at top of your script or notebook")
            msg = "\nNER in ktrain uses the CRF module from keras_contrib, which is not yet\n" +\
                    "fully compatible with TensorFlow 2. To use NER, you must add the following to the top of your\n" +\
                    "script or notebook BEFORE you import ktrain (after restarting runtime):\n\n" +\
                  "import os\n" +\
                  "os.environ['DISABLE_V2_BEHAVIOR'] = '1'\n"
            print(msg)
            return
        else:
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()

        if predictor_path is None and self.lang == 'zh':
            dirpath = os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(dirpath, 'ner_models/ner_chinese')
        elif predictor_path is None and self.lang == 'ru':
            dirpath = os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(dirpath, 'ner_models/ner_russian')
        elif predictor_path is None and self.lang=='en':
            dirpath = os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(dirpath, 'ner_models/ner_english')
        elif predictor_path is None:
            raise ValueError("Unsupported language: if predictor_path is None,  then lang must be " +\
                             "'en' for English, 'zh' for Chinese, or 'ru' for Chinese")
        else:
            if not os.path.isfile(predictor_path) or not os.path.isfile(predictor_path +'.preproc'):
                raise ValueError('could not find a valid predictor model '+\
                                 '%s or valid Preprocessor %s at specified path' % (predictor_path, predictor_path+'.preproc'))
            fpath = predictor_path
        try:
           import io
           from contextlib import redirect_stdout
           f = io.StringIO()
           with redirect_stdout(f):
               import ktrain
        except:
           raise ValueError('ktrain could not be imported. Install with: pip3 install ktrain')
        self.predictor = ktrain.load_predictor(fpath)


    def predict(self, texts, merge_tokens=True):
        """
        Extract named entities from supplied text

        Args:
          texts (list of str or str): list of texts to annotate
          merge_tokens(bool):  If True, tokens will be merged together by the entity
                               to which they are associated:
                               ('Paul', 'B-PER'), ('Newman', 'I-PER') becomes ('Paul Newman', 'PER')
        """
        if isinstance(texts, str): texts = [texts]
        results = []
        for text in texts:
            text = text.strip()
            result = self.predictor.predict(text, merge_tokens=merge_tokens)
            #if merge_tokens:
                #result = self.merge_tokens(result)
            results.append(result)
        if len(result) == 1: result = result[0]
        return result


    # 2020-04-30: moved to text.ner.predictor
    #def merge_tokens(self, annotated_sentence):
    #    if self.lang.startswith('zh'):
    #        sep = ''
    #    else:
    #        sep = ' '
    #    current_token = ""
    #    current_tag = ""
    #    entities = []

    #    for tup in annotated_sentence:
    #        token = tup[0]
    #        entity = tup[1]
    #        tag = entity.split('-')[1] if '-' in entity else None
    #        prefix = entity.split('-')[0] if '-' in entity else None
    #        # not within entity
    #        if tag is None and not current_token:
    #            continue
    #        # beginning of entity
    #        #elif tag and prefix=='B':
    #        elif tag and (prefix=='B' or prefix=='I' and not current_token):
    #            if current_token: # consecutive entities
    #                entities.append((current_token, current_tag))
    #                current_token = ""
    #                current_tag = None
    #            current_token = token
    #            current_tag = tag
    #        # end of entity
    #        elif tag is None and current_token:
    #            entities.append((current_token, current_tag))
    #            current_token = ""
    #            current_tag = None
    #            continue
    #        # within entity
    #        elif tag and current_token:  #  prefix I
    #            current_token = current_token + sep + token
    #            current_tag = tag
    #    return entities  

