from .models import *
from .data import *
from .ner.data import entities_from_gmb, entities_from_conll2003, entities_from_txt, entities_from_df, entities_from_array
from .ner.models import sequence_tagger, print_sequence_taggers
from .eda import get_topic_model
from .textutils import extract_filenames, load_text_files, filter_by_id
from .preprocessor import Transformer, TransformerEmbedding
from . import shallownlp

__all__ = [
           'text_classifier', 'text_regression_model',
           'print_text_classifiers', 'print_text_regression_models',
           'texts_from_folder', 'texts_from_csv',
           'entities_from_gmb',
           'entities_from_conll2003',
           'entities_from_txt',
           'entities_from_array',
           'entities_from_df',
           'sequence_tagger',
           'print_sequence_taggers',
           'get_topic_model',
           'extract_filenames', 
           'load_text_files',
           'Transformer',
           'TranformerEmbedding',
           'shallownlp'
           ]


def load_topic_model(fname):
    """
    Load saved TopicModel object
    Args:
        fname(str): base filename for all saved files
    """
    with open(fname+'.tm_vect', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(fname+'.tm_model', 'rb') as f:
        model = pickle.load(f)
    with open(fname+'.tm_params', 'rb') as f:
        params = pickle.load(f)
    tm = get_topic_model(n_topics=params['n_topics'],
                         n_features = params['n_features'],
                         verbose = params['verbose'])
    tm.model = model
    tm.vectorizer = vectorizer
    return tm



seqlen_stats = Transformer.seqlen_stats
