import sys
import os
import os.path
import re
import numpy as np
import warnings
import operator
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



from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
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





try:
    from PIL import Image
    PIL_INSTALLED = True
except:
    PIL_INSTALLED = False

import keras
from keras import backend as K
from keras.engine.training import Model
from keras.models import load_model, Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping,LambdaCallback, Callback
from keras.layers import Dense, Embedding, Input, Flatten, GRU, Bidirectional
from keras.layers import SpatialDropout1D, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.layers import concatenate, dot, Dropout, BatchNormalization, Add
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.initializers import glorot_uniform  
from keras import regularizers
from keras.regularizers import l2
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.preprocessing.image import NumpyArrayIterator
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence, to_categorical
from keras.utils import multi_gpu_model

from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import preprocess_input as pre_resnet50
from keras.applications.mobilenet import preprocess_input as pre_mobilenet
from keras.applications.inception_v3 import preprocess_input as pre_inception

