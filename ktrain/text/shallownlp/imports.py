import logging
import os
import warnings

from ...imports import SUPPRESS_DEP_WARNINGS

# os.environ['DISABLE_V2_BEHAVIOR'] = '1'


if SUPPRESS_DEP_WARNINGS:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow_hub").setLevel(logging.ERROR)
    warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    import tensorflow as tf

    TF_INSTALLED = True
except ImportError:
    TF_INSTALLED = False
if TF_INSTALLED:
    tf.autograph.set_verbosity(1)


import os.path
import re
import string

import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from sklearn.base import BaseEstimator

try:  # sklearn<0.24.x
    from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
except ImportError:  # sklearn>=0.24.x
    from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin

import syntok.segmenter as segmenter
from joblib import dump, load
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ktrain imported locally in ner.py
# import ktrain

# pandas imported locally in classifier.py
# import pandas as pd

try:
    import langdetect

    LANGDETECT = True
except:
    LANGDETECT = False
try:
    import charset_normalizer as chardet

    CHARDET = True
except:
    CHARDET = False
try:
    import jieba

    JIEBA = True
except:
    JIEBA = False
