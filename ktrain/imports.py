
#--------------------------
# Tensorflow Keras imports
#--------------------------

import os
import warnings
import logging
from distutils.util import strtobool
from packaging import version


# suppress warnings
SUPPRESS_TF_WARNINGS = strtobool(os.environ.get('SUPPRESS_TF_WARNINGS', '1'))
if SUPPRESS_TF_WARNINGS:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # elevate warnings to errors for debugging dependencies
    #warnings.simplefilter('error', FutureWarning)



# TF1
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#from tensorflow import keras

DISABLE_V2_BEHAVIOR = strtobool(os.environ.get('DISABLE_V2_BEHAVIOR', '0'))
if DISABLE_V2_BEHAVIOR:
    # TF2-transition
    ACC_NAME = 'acc'
    VAL_ACC_NAME = 'val_acc'
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.compat.v1 import keras
    print('Using DISABLE_V2_BEHAVIOR with TensorFlow')
else:
    # TF2
    ACC_NAME = 'accuracy'
    VAL_ACC_NAME = 'val_accuracy'
    import tensorflow as tf
    from tensorflow import keras

# suppress autograph warnings
tf.autograph.set_verbosity(1)
#if SUPPRESS_TF_WARNINGS:
    #tf.autograph.set_verbosity(1)

if version.parse(tf.__version__) < version.parse('2.0'):
    raise Exception('As of v0.8.x, ktrain needs TensorFlow 2. Please upgrade TensorFlow.')

os.environ['TF_KERAS'] = '1' # to use keras_bert package below with tf.Keras





# output Keras version
#print("using Keras version: %s" % (keras.__version__))

K = keras.backend
Layer = keras.layers.Layer
InputSpec = keras.layers.InputSpec
Model = keras.Model
model_from_json = keras.models.model_from_json
load_model = keras.models.load_model
Sequential = keras.models.Sequential
ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping
LambdaCallback = keras.callbacks.LambdaCallback
Callback = keras.callbacks.Callback
Dense = keras.layers.Dense
Embedding = keras.layers.Embedding
Input = keras.layers.Input
Flatten = keras.layers.Flatten
GRU = keras.layers.GRU
Bidirectional = keras.layers.Bidirectional
LSTM = keras.layers.LSTM
LeakyReLU = keras.layers.LeakyReLU # SG
Multiply = keras.layers.Multiply   # SG
Average = keras.layers.Average     # SG
Reshape = keras.layers.Reshape     #SG
SpatialDropout1D = keras.layers.SpatialDropout1D
GlobalMaxPool1D = keras.layers.GlobalMaxPool1D
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
concatenate = keras.layers.concatenate
dot = keras.layers.dot
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
Add = keras.layers.Add
Convolution2D = keras.layers.Convolution2D
MaxPooling2D = keras.layers.MaxPooling2D
AveragePooling2D = keras.layers.AveragePooling2D
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
TimeDistributed = keras.layers.TimeDistributed
Lambda = keras.layers.Lambda
Activation = keras.layers.Activation
add = keras.layers.add
Concatenate = keras.layers.Concatenate
initializers = keras.initializers
glorot_uniform = keras.initializers.glorot_uniform
regularizers = keras.regularizers
l2 = keras.regularizers.l2
constraints = keras.constraints
sequence = keras.preprocessing.sequence
image = keras.preprocessing.image
NumpyArrayIterator = keras.preprocessing.image.NumpyArrayIterator
Iterator = keras.preprocessing.image.Iterator
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
Tokenizer = keras.preprocessing.text.Tokenizer
Sequence = keras.utils.Sequence
get_file = keras.utils.get_file
plot_model = keras.utils.plot_model
to_categorical = keras.utils.to_categorical
multi_gpu_model = keras.utils.multi_gpu_model
activations = keras.activations
sigmoid = keras.activations.sigmoid
categorical_crossentropy = keras.losses.categorical_crossentropy
sparse_categorical_crossentropy = keras.losses.sparse_categorical_crossentropy
ResNet50 = keras.applications.ResNet50
MobileNet = keras.applications.mobilenet.MobileNet
InceptionV3 = keras.applications.inception_v3.InceptionV3
pre_resnet50 = keras.applications.resnet50.preprocess_input
pre_mobilenet = keras.applications.mobilenet.preprocess_input
pre_inception = keras.applications.inception_v3.preprocess_input


#----------------------------------------------------------
# standards
#----------------------------------------------------------

#import warnings # imported above
import sys
import os
import os.path
import re
import operator
from collections import Counter
from distutils.version import StrictVersion
import tempfile
import pickle
from abc import ABC, abstractmethod
import math
import itertools
import csv
import copy
import glob
import codecs
import urllib.request
import zipfile
import gzip
import shutil
import string
import random
import json
import mimetypes









#----------------------------------------------------------
# external dependencies
#----------------------------------------------------------


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
plt.ion() # interactive mode
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder



#from sklearn.externals import joblib
import joblib
from scipy import sparse # utils
from scipy.sparse import csr_matrix
import pandas as pd
try:
    # fastprogress >= v0.2.0
    from fastprogress.fastprogress import master_bar, progress_bar 
except:
    # fastprogress < v0.2.0
    from fastprogress import master_bar, progress_bar 
import keras_bert
from keras_bert import Tokenizer as BERT_Tokenizer
import requests
# verify=False added to avoid headaches from some corporate networks
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# multilingual
import langdetect
import jieba
import cchardet as chardet

# graphs
import networkx as nx
#from sklearn import preprocessing, feature_extraction, model_selection

# ner
from seqeval.metrics import classification_report as ner_classification_report
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics import accuracy_score as ner_accuracy_score
from seqeval.metrics.sequence_labeling import get_entities
import syntok.segmenter as segmenter


# transformers
try:
    logging.getLogger('transformers').setLevel(logging.CRITICAL)
except: pass
import transformers


try:
    from PIL import Image
    PIL_INSTALLED = True
except:
    PIL_INSTALLED = False

SG_ERRMSG = 'ktrain currently uses a forked version of stellargraph v0.8.2. '+\
            'Please install with: '+\
            'pip3 install git+https://github.com/amaiya/stellargraph@no_tf_dep_082'

ALLENNLP_ERRMSG  = 'To use ELMo embedings, please install allenlp:\n' +\
                   'pip3 install allennlp'



