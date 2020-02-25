### [Overview](#overview) | [Tutorials](#tutorials) | [Examples](#examples) |  [Installation](#installation)

# ktrain



### News and Announcements
- **2020-01-31:**  
  - ***ktrain*** **v0.9.x is released** and now includes out-of-the-box support for text regression in addition to support for custom data formats. See [this tutorial notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/tutorials/tutorial-A4-customdata-text_regression_with_extra_regressors.ipynb) for more information on both these topics.
- **2020-01-14:**  
  - ***ktrain*** **v0.8.x is released** and now includes a thin and easy-to-use wrapper to [HuggingFace Transformers](https://github.com/huggingface/transformers) for text classification.  See [this tutorial notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb) for more details. 
  - As of v0.8.x, *ktrain* now uses **TensorFlow 2**. TensorFlow 1.x is no longer supported.  If you're using Google Colab and `import tensorflow as tf;  print(tf.__version__)` shows v1.15 is installed, you must install TensorFlow 2: `!pip3 install -q tensorflow_gpu>=2.0`.  Remember to import Keras modules like this:  `from tensorflow.keras.layers import Dense`.  (That is, don't do this:  `from keras.layers import Dense`.)


----

### Overview

*ktrain* is a lightweight wrapper for the deep learning library [TensorFlow Keras](https://www.tensorflow.org/guide/keras/overview) (and other libraries) to help build, train, and deploy neural networks.  With only a few lines of code, ktrain allows you to easily and quickly:

- estimate an optimal learning rate for your model given your data using a Learning Rate Finder
- utilize learning rate schedules such as the [triangular policy](https://arxiv.org/abs/1506.01186), the [1cycle policy](https://arxiv.org/abs/1803.09820), and [SGDR](https://arxiv.org/abs/1608.03983) to effectively minimize loss and improve generalization
- employ fast and easy-to-use pre-canned models for  `text`, `vision`, and `graph` data:
  - `text` data:
     - **Text Classification**: [BERT](https://arxiv.org/abs/1810.04805), [DistilBERT](https://arxiv.org/abs/1910.01108), [NBSVM](https://www.aclweb.org/anthology/P12-2018), [fastText](https://arxiv.org/abs/1607.01759), and other models <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/IMDb-BERT.ipynb)]</sup></sub>
     - **Text Regression**: [BERT](https://arxiv.org/abs/1810.04805), [DistilBERT](https://arxiv.org/abs/1910.01108), Embedding-based linear text regression, [fastText](https://arxiv.org/abs/1607.01759), and other models <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/examples/text/text_regression_example.ipynb)]</sup></sub>
     - **Sequence Labeling**:  [Bidirectional LSTM-CRF](https://arxiv.org/abs/1603.01360) with optional pretrained word embeddings <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorial-06-sequence-tagging.ipynb)]</sup></sub>
     - **Unsupervised Topic Modeling** with [LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/20newsgroups-topic_modeling.ipynb)]</sup></sub>
     - **Document Similarity with One-Class Learning**:  given some documents of interest, find and score new documents that are semantically similar to them using [One-Class Text Classification](https://en.wikipedia.org/wiki/One-class_classification) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/20newsgroups-document_similarity_scorer.ipynb)]</sup></sub>
     - **Document Recommendation Engine**:  given text from a sample document, recommend documents that are semantically-related to it from a larger corpus  <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/20newsgroups-recommendation_engine.ipynb)]</sup></sub>
  - `vision` data:
    - **image classification** (e.g., [ResNet](https://arxiv.org/abs/1512.03385), [Wide ResNet](https://arxiv.org/abs/1605.07146), [Inception](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/vision/dogs_vs_cats-ResNet50.ipynb)]</sup></sub>
  - `graph` data:
    - **graph node classification** with graph neural networks (e.g., [GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/graphs/pubmed-GraphSAGE.ipynb)]</sup></sub>
- perform multilingual text classification (e.g., [Chinese Sentiment Analysis with BERT](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/ChineseHotelReviews-BERT.ipynb), [Arabic Sentiment Analysis with NBSVM](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/ArabicHotelReviews-nbsvm.ipynb))
- load and preprocess text and image data from a variety of formats 
- inspect data points that were misclassified and [provide explanations](https://eli5.readthedocs.io/en/latest/) to help improve your model
- leverage a simple prediction API for saving and deploying both models and data-preprocessing steps to make predictions on new raw data


### Tutorials
Please see the following tutorial notebooks for a guide on how to use *ktrain* on your projects:
* Tutorial 1:  [Introduction](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-01-introduction.ipynb)
* Tutorial 2:  [Tuning Learning Rates](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-02-tuning-learning-rates.ipynb)
* Tutorial 3: [Image Classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-03-image-classification.ipynb)
* Tutorial 4: [Text Classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-04-text-classification.ipynb)
* Tutorial 5: [Learning from Unlabeled Text Data](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-05-learning_from_unlabeled_text_data.ipynb)
* Tutorial 6: [Text Sequence Tagging](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-06-sequence-tagging.ipynb) for Named Entity Recognition
* Tutorial 7: [Graph Node Classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-07-graph-node_classification.ipynb) with Graph Neural Networks
* Tutorial A1: [Additional tricks](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A1-additional-tricks.ipynb), which covers topics such as previewing data augmentation schemes, inspecting intermediate output of Keras models for debugging, setting global weight decay, and use of built-in and custom callbacks.
* Tutorial A2: [Explaining Predictions and Misclassifications](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A2-explaining-predictions.ipynb)
* Tutorial A3: [Text Classification with Hugging Face Transformers](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb)
* Tutorial A4: [Using Custom Data Formats and Models: Text Regression with Extra Regressors](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/tutorials/tutorial-A4-customdata-text_regression_with_extra_regressors.ipynb)


Some blog tutorials about *ktrain* are shown below:

> [**ktrain: A Lightweight Wrapper for Keras to Help Train Neural Networks**](https://towardsdatascience.com/ktrain-a-lightweight-wrapper-for-keras-to-help-train-neural-networks-82851ba889c) 


> [**BERT Text Classification in 3 Lines of Code**](https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358)  

> [**Text Classification with Hugging Face Transformers in  TensorFlow 2 (Without Tears)**](https://medium.com/@asmaiya/text-classification-with-hugging-face-transformers-in-tensorflow-2-without-tears-ee50e4f3e7ed)









### Examples

Tasks such as text classification and image classification can be accomplished easily with 
only a few lines of code.

#### Example: Text Classification of [IMDb Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) Using [BERT](https://arxiv.org/pdf/1810.04805.pdf)
```
import ktrain
from ktrain import text as txt

# load data
(x_train, y_train), (x_test, y_test), preproc = txt.texts_from_folder('data/aclImdb', maxlen=500, 
                                                                     preprocess_mode='bert',
                                                                     train_test_names=['train', 'test'],
                                                                     classes=['pos', 'neg'])

# load model
model = txt.text_classifier('bert', (x_train, y_train), preproc=preproc)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test), 
                             batch_size=6)

# find good learning rate
learner.lr_find()             # briefly simulate training to find good learning rate
learner.lr_plot()             # visually identify best learning rate

# train using 1cycle learning rate schedule for 3 epochs
learner.fit_onecycle(2e-5, 3) 
```


#### Example: Classifying Images of [Dogs and Cats](https://www.kaggle.com/c/dogs-vs-cats) Using a Pretrained [ResNet50](https://arxiv.org/abs/1512.03385) model
```
import ktrain
from ktrain import vision as vis

# load data
(train_data, val_data, preproc) = vis.images_from_folder(
                                              datadir='data/dogscats',
                                              data_aug = vis.get_data_aug(horizontal_flip=True),
                                              train_test_names=['train', 'valid'], 
                                              target_size=(224,224), color_mode='rgb')

# load model
model = vis.image_classifier('pretrained_resnet50', train_data, val_data, freeze_layers=80)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, 
                             workers=8, use_multiprocessing=False, batch_size=64)

# find good learning rate
learner.lr_find()             # briefly simulate training to find good learning rate
learner.lr_plot()             # visually identify best learning rate

# train using triangular policy with ModelCheckpoint and implicit ReduceLROnPlateau and EarlyStopping
learner.autofit(1e-4, checkpoint_folder='/tmp/saved_weights') 
```

#### Example: Sequence Labeling for [Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/2) using a randomly initialized [Bidirectional LSTM CRF](https://arxiv.org/abs/1603.01360) model
```
import ktrain
from ktrain import text as txt

# load data
(trn, val, preproc) = txt.entities_from_txt('data/ner_dataset.csv',
                                            sentence_column='Sentence #',
                                            word_column='Word',
                                            tag_column='Tag', 
                                            data_format='gmb')

# load model
model = txt.sequence_tagger('bilstm-crf', preproc)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model, train_data=trn, val_data=val)


# conventional training for 1 epoch using a learning rate of 0.001 (Keras default for Adam optmizer)
learner.fit(1e-3, 1) 
```


#### Example: Node Classification on [Cora Citation Graph](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) using a [GraphSAGE](https://arxiv.org/abs/1706.02216) model
```
import ktrain
from ktrain import graph as gr

# load data with supervision ratio of 10%
(trn, val, preproc)  = gr.graph_nodes_from_csv(
                                               'cora.content', # node attributes/labels
                                               'cora.cites',   # edge list
                                               sample_size=20, 
                                               holdout_pct=None, 
                                               holdout_for_inductive=False,
                                              train_pct=0.1, sep='\t')

# load model
model=gr.graph_node_classifier('graphsage', trn)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=64)


# find good learning rate
learner.lr_find(max_epochs=100) # briefly simulate training to find good learning rate
learner.lr_plot()               # visually identify best learning rate

# train using triangular policy with ModelCheckpoint and implicit ReduceLROnPlateau and EarlyStopping
learner.autofit(0.01, checkpoint_folder='/tmp/saved_weights')
```


Using *ktrain* on **Google Colab**?  See [this simple demo of Multiclass Text Classification with BERT](https://colab.research.google.com/drive/1AH3fkKiEqBpVpO5ua00scp7zcHs5IDLK).

**Additional examples can be found [here](https://github.com/amaiya/ktrain/tree/master/examples).**



### Installation

Make sure pip is up-to-date with: `pip3 install -U pip`.

1. Ensure TensorFlow 2 [is installed](https://www.tensorflow.org/install/pip?lang=python3) if it is not already

> For GPU: `pip3 install "tensorflow_gpu>=2.0.0"`

> For CPU: `pip3 install "tensorflow>=2.0.0"`


2. Install *ktrain*: `pip3 install ktrain`

**Some things to note:**
- As of v0.8.x, *ktrain* requires TensorFlow 2.  TensorFlow 1.x (1.14, 1.15) is no longer suppoted. 
- Since some *ktrain* dependencies have not yet been migrated to `tf.keras` in TensorFlow 2 (or may have other issues), 
  *ktrain* is temporarily using forked versions of some libraries. Specifically, *ktrain* uses forked versions of the `eli5` and `stellargraph` libraries.  If not installed, *ktrain* will complain  when a method or function needing 
  either of these libraries is invoked.
  To install these forked versions, you can do the following:
```
pip3 install git+https://github.com/amaiya/eli5@tfkeras_0_10_1
pip3 install git+https://github.com/amaiya/stellargraph@no_tf_dep_082
```




<!--
### Requirements

The following software/libraries should be installed:

- [Python 3.6+](https://www.python.org/) (tested on 3.6.7)
- [Keras](https://keras.io/)  (tested on 2.2.4)
- [TensorFlow](https://www.tensorflow.org/)  (tested on 1.10.1)
- [scikit-learn](https://scikit-learn.org/stable/) (tested on 0.20.0)
- [matplotlib](https://matplotlib.org/) (tested on 3.0.0)
- [pandas](https://pandas.pydata.org/) (tested on 0.24.2)
- [keras_bert](https://github.com/CyberZHG/keras-bert/tree/master/keras_bert) 
- [fastprogress](https://github.com/fastai/fastprogress) 
-->


This code was tested on Ubuntu 18.04 LTS using TensorFlow 2.0 (Keras version 2.2.4-tf).

----
**Creator:  [Arun S. Maiya](http://arun.maiya.net)**

**Email:** arun [at] maiya [dot] net
