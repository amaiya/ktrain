# --------------------------
# Tensorflow Keras imports
# --------------------------

import logging
import os
import re
import warnings
from distutils.util import strtobool

from packaging import version

os.environ["NUMEXPR_MAX_THREADS"] = (
    "8"  # suppress warning from NumExpr on machines with many CPUs
)

# TensorFlow
SUPPRESS_DEP_WARNINGS = strtobool(os.environ.get("SUPPRESS_DEP_WARNINGS", "1"))
if (
    SUPPRESS_DEP_WARNINGS
):  # 2021-11-12:  copied this here to properly suppress TF/CUDA warnings in Kaggle notebooks, etc.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
TF_WARNING = (
    "TensorFlow is not installed and will be needed if training neural networks, "
    + "but non-TensorFlow features in ktrain can still be used. See https://github.com/amaiya/ktrain/blob/master/README.md"
)
DISABLE_V2_BEHAVIOR = strtobool(os.environ.get("DISABLE_V2_BEHAVIOR", "0"))
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # for contiued use of legacy optimizers
try:
    if DISABLE_V2_BEHAVIOR:
        # TF2-transition
        ACC_NAME = "acc"
        VAL_ACC_NAME = "val_acc"
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        from tensorflow.compat.v1 import keras

        print("Using DISABLE_V2_BEHAVIOR with TensorFlow")
    else:
        # TF2
        ACC_NAME = "accuracy"
        VAL_ACC_NAME = "val_accuracy"
        import tensorflow as tf
        from tensorflow import keras
    K = keras.backend
    # suppress autograph warnings
    tf.autograph.set_verbosity(1)
    if version.parse(tf.__version__) < version.parse("2.0"):
        raise Exception(
            "As of v0.8.x, ktrain needs TensorFlow 2. Please upgrade TensorFlow."
        )
    os.environ["TF_KERAS"] = "1"  # for eli5-tf (and previously keras_bert)
    TF_INSTALLED = True
except ImportError:
    keras = None
    K = None
    tf = None
    TF_INSTALLED = False
    warnings.warn(TF_WARNING)


# for TF backwards compatibility (e.g., support for TF 2.3.x):
try:
    MobileNetV3Small = keras.applications.MobileNetV3Small
    pre_mobilenetv3small = keras.applications.mobilenet_v3.preprocess_input
    HAS_MOBILENETV3 = True
except:
    HAS_MOBILENETV3 = False


# ----------------------------------------------------------
# standards
# ----------------------------------------------------------

import codecs
import copy
import csv
import glob
import gzip
import itertools
import json
import math
import mimetypes
import operator
import os
import os.path
import pickle
import random
import re
import shutil
import string

# import warnings # imported above
import sys
import tempfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from distutils.version import StrictVersion

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex

# ----------------------------------------------------------
# external dependencies
# ----------------------------------------------------------


plt.ion()  # interactive mode
# from sklearn.externals import joblib
import joblib
import pandas as pd
import sklearn
from scipy import sparse  # utils
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_files
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    # fastprogress >= v0.2.0
    from fastprogress.fastprogress import master_bar, progress_bar
except:
    # fastprogress < v0.2.0
    from fastprogress import master_bar, progress_bar

import requests

# verify=False added to avoid headaches from some corporate networks
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import charset_normalizer as chardet
import jieba

# multilingual text processing
import langdetect

# text processing
import syntok.segmenter as segmenter

# 'bert' text classification model
if TF_INSTALLED:
    try:
        import keras_bert
        from keras_bert import Tokenizer as BERT_Tokenizer

        KERAS_BERT_INSTALLED = True
    except ImportError:
        warnings.warn(
            "keras_bert is not installed. keras_bert is only needed only for 'bert' text classification model"
        )
        KERAS_BERT_INSTALLED = False


def check_keras_bert():
    if not KERAS_BERT_INSTALLED:
        raise Exception("Please install keras_bert: pip install keras_bert")


# transformers for models in 'text' module
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    import transformers
except ImportError:
    warnings.warn(
        "transformers not installed - needed by various models in 'text' module"
    )

if sys.version.startswith("3.6") and (
    version.parse(transformers.__version__) >= version.parse("4.11.0")
):
    raise Exception(
        "You are using Python 3.6.  Please downgrade transformers "
        + "(and ignore the resultant pip ERROR): pip install transformers==4.10.3"
    )

try:
    from PIL import Image

    PIL_INSTALLED = True
except:
    PIL_INSTALLED = False

SG_ERRMSG = (
    "ktrain currently uses a forked version of stellargraph v0.8.2. "
    + "Please install with: "
    + "pip install https://github.com/amaiya/stellargraph/archive/refs/heads/no_tf_dep_082.zip"
)

ALLENNLP_ERRMSG = (
    "To use ELMo embedings, please install allenlp:\n" + "pip install allennlp"
)


# ELI5
KTRAIN_ELI5_TAG = "0.10.1-1"


# Suppress Warnings
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


if SUPPRESS_DEP_WARNINGS:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    # elevate warnings to errors for debugging dependencies
    # warnings.simplefilter('error', FutureWarning)
    set_global_logging_level(
        logging.ERROR,
        [
            "transformers",
            "nlp",
            "torch",
            "tensorflow",
            "tensorboard",
            "wandb",
            "mosestokenizer",
            "shap",
        ],
    )
