import os, logging, warnings
#os.environ['DISABLE_V2_BEHAVIOR'] = '1'

from ...imports import SUPPRESS_TF_WARNINGS
if SUPPRESS_TF_WARNINGS:
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



import re
import string
import os.path
import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression, SGDClassifier
from joblib import dump, load
import syntok.segmenter as segmenter

# ktrain imported locally in ner.py
#import ktrain 

# pandas imported locally in classifier.py
#import pandas as pd

try:
    import langdetect
    LANGDETECT=True
except:
    LANGDETECT=False
try:
    import cchardet as chardet
    CHARDET=True
except:
    CHARDET=False
try:
    import jieba
    JIEBA=True
except:
    JIEBA=False
