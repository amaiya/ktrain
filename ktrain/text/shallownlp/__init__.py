import os, logging, warnings

#os.environ['DISABLE_V2_BEHAVIOR'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import tensorflow as tf
    TF_INSTALLED = True
except ImportError:
    TF_INSTALLED = False
if TF_INSTALLED:
    tf.autograph.set_verbosity(1)



from .classifier import Classifier
from .searcher import *
from .ner import NER
from .utils import sent_tokenize, extract_filenames, read_text


__all__ = ['Classifier', 
           'Searcher', 'search', 'find_chinese', 'find_arabic', 'find_russian', 'read_text',
           'NER',
           'sent_tokenize', 'extract_filenames', 'read_text']
