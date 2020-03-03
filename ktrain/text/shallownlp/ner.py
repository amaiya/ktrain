from .imports import *


class NER:
    def __init__(self, lang='en'):
        """
        pretrained NER.
        Only English and Chinese are currenty supported.

        Args:
          lang(str): Currently, one of {'en', 'zh', 'ru'}: en=English , zh=Chinese, or ru=Russian
        """
        if lang not in ['en', 'zh', 'ru']:
            raise ValueError("Unsupported langauge:  choose either 'en' for English, 'zh' for Chinese, or 'ru' for Chinese")
        self.lang = lang


    def predict(self, texts, merge_tokens=True):
        """
        Extract named entities from supplied text
        """
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

        #old_do = os.environ.get('CUDA_DEVICE_ORDER', None)
        #old_vd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if isinstance(texts, str): texts = [texts]
        if self.lang == 'zh':
            dirpath = os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(dirpath, 'ner_models/ner_chinese')
        elif self.lang == 'ru':
            dirpath = os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(dirpath, 'ner_models/ner_russian')
        elif self.lang=='en':
            dirpath = os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(dirpath, 'ner_models/ner_english')
        else:
            raise ValueError('lang %s is not supported by NER'  % (self.lang))
        try:
           import io
           from contextlib import redirect_stdout
           f = io.StringIO()
           with redirect_stdout(f):
               import ktrain
        except:
           raise ValueError('ktrain could not be imported. Install with: pip3 install ktrain')
        predictor = ktrain.load_predictor(fpath)
        results = []
        for text in texts:
            text = text.strip()
            result = predictor.predict(text)
            if merge_tokens:
                result = self.merge_entities(result)
            results.append(result)
        if len(result) == 1: result = result[0]

        #if old_do is not None:
            #os.environ["CUDA_DEVICE_ORDER"] = old_do
        #else:
            #del os.environ['CUDA_DEVICE_ORDER']
        #if old_vd is not None:
            #os.environ['CUDA_VISIBLE_DEVICES'] = old_vd
        #else:
            #del os.environ['CUDA_VISIBLE_DEVICES']
        return result

    def merge_entities(self, annotated_sentence):
        if self.lang == 'zh':
            sep = ''
        else:
            sep = ' '
        current_token = ""
        current_tag = ""
        entities = []

        for tup in annotated_sentence:
            token = tup[0]
            entity = tup[1]
            tag = entity.split('-')[1] if '-' in entity else None
            prefix = entity.split('-')[0] if '-' in entity else None
            # not within entity
            if tag is None and not current_token:
                continue
            # beginning of entity
            elif tag and prefix=='B':
                if current_token: # consecutive entities
                    entities.append((current_token, current_tag))
                    current_token = ""
                    current_tag = None
                current_token = token
                current_tag = tag
            # end of entity
            elif tag is None and current_token:
                entities.append((current_token, current_tag))
                current_token = ""
                current_tag = None
                continue
            # within entity
            elif tag and current_token:  #  prefix I
                current_token = current_token + sep + token
                current_tag = tag
        return entities  

