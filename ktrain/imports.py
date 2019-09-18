import sys
import os
import os.path
import re
import numpy as np
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



from matplotlib import pyplot as plt
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
from seqeval.metrics import classification_report as ner_classification_report
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics.sequence_labeling import get_entities



try:
    from PIL import Image
    PIL_INSTALLED = True
except:
    PIL_INSTALLED = False



#------------------------
# Keras imports
#------------------------
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras

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


#from keras import backend as K
#from keras.engine.training import Model
#from keras.models import load_model, Model, Sequential
#from keras.callbacks import ModelCheckpoint, EarlyStopping,LambdaCallback, Callback
#from keras.layers import Dense, Embedding, Input, Flatten, GRU, Bidirectional, LSTM
#from keras.layers import SpatialDropout1D, GlobalMaxPool1D, GlobalAveragePooling1D
#from keras.layers import concatenate, dot, Dropout, BatchNormalization, Add
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
#from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Lambda
#from keras.layers.core import Activation
#from keras.layers.merge import add
#from keras.initializers import glorot_uniform  
#from keras import regularizers
#from keras.regularizers import l2
#from keras.models import load_model
#from keras.preprocessing import sequence
#from keras.preprocessing import image
#from keras.preprocessing import sequence
#from keras.preprocessing.image import NumpyArrayIterator
#from keras.preprocessing.image import Iterator
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.text import Tokenizer
#from keras.utils import Sequence, to_categorical
#from keras.utils import multi_gpu_model
#from keras.activations import sigmoid
#from keras.losses import categorical_crossentropy
#from keras.losses import sparse_categorical_crossentropy

#from keras.applications.resnet50 import ResNet50
#from keras.applications.mobilenet import MobileNet
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import preprocess_input as pre_resnet50
#from keras.applications.mobilenet import preprocess_input as pre_mobilenet
#from keras.applications.inception_v3 import preprocess_input as pre_inception

