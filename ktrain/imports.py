
#------------------------
# Keras imports
#------------------------

import os
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TF_KERAS = False
EAGER_MODE = False

if os.environ.get('TF_KERAS', '0') != '0':
    import tensorflow as tf
    from tensorflow.python import keras
    TF_KERAS = True
    if os.environ.get('TF_EAGER', '0') != '0':
        try:
            tf.enable_eager_execution()
            raise AttributeError()
        except AttributeError as e:
            pass
    EAGER_MODE = tf.executing_eagerly()
else:
    import keras
print("using Keras version: %s" % (keras.__version__))

K = keras.backend
Layer = keras.engine.Layer
InputSpec = keras.engine.InputSpec
Model = keras.engine.training.Model
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
SpatialDropout1D = keras.layers.SpatialDropout1D
GlobalMaxPool1D = keras.layers.GlobalMaxPool1D
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
concatenate = keras.layers.concatenate
dot = keras.layers.dot
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
Add = keras.layers.Add
Convolution2D = keras.layers.convolutional.Convolution2D
MaxPooling2D = keras.layers.convolutional.MaxPooling2D
AveragePooling2D = keras.layers.convolutional.AveragePooling2D
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
TimeDistributed = keras.layers.TimeDistributed
Lambda = keras.layers.Lambda
Activation = keras.layers.core.Activation
add = keras.layers.merge.add
Concatenate = keras.layers.merge.Concatenate
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
ResNet50 = keras.applications.resnet50.ResNet50
MobilNet = keras.applications.mobilenet.MobileNet
InceptionV3 = keras.applications.inception_v3.InceptionV3
pre_resnet50 = keras.applications.resnet50.preprocess_input
pre_mobilenet = keras.applications.mobilenet.preprocess_input
pre_inception = keras.applications.inception_v3.preprocess_input


#----------------------------------------------------------
# standards
#----------------------------------------------------------

import sys
import os
import os.path
import re
import warnings
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
import string
import random
import json



#----------------------------------------------------------
# external dependencies
#----------------------------------------------------------


import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.externals import joblib
import joblib
from scipy import sparse # utils
from scipy.sparse import csr_matrix
import pandas as pd
from fastprogress import master_bar, progress_bar 
import keras_bert
from keras_bert import Tokenizer as BERT_Tokenizer
import requests
# verify=False added to avoid headaches from some corporate networks
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import eli5
from eli5.lime import TextExplainer

# ner
from seqeval.metrics import classification_report as ner_classification_report
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics.sequence_labeling import get_entities

# multilingual
import langdetect
import jieba
import cchardet as chardet

# graphs
#import networkx as nx
#import stellargraph as sg
#from stellargraph.mapper import GraphSAGENodeGenerator, GraphSAGELinkGenerator
#from stellargraph.layer import GraphSAGE
#from stellargraph.data import EdgeSplitter
#from sklearn import preprocessing, feature_extraction, model_selection





try:
    from PIL import Image
    PIL_INSTALLED = True
except:
    PIL_INSTALLED = False


