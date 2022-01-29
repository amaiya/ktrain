### [Overview](#overview) | [Tutorials](#tutorials) | [Examples](#examples) |  [Installation](#installation) | [FAQ](https://github.com/amaiya/ktrain/blob/master/FAQ.md) | [API Docs](https://amaiya.github.io/ktrain/index.html) |  [How to Cite](#how-to-cite)
[![PyPI Status](https://badge.fury.io/py/ktrain.svg)](https://badge.fury.io/py/ktrain) [![ktrain python compatibility](https://img.shields.io/pypi/pyversions/ktrain.svg)](https://pypi.python.org/pypi/ktrain) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/amaiya/ktrain/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/ktrain)](https://pepy.tech/project/ktrain)

<p align="center">
<img src="https://github.com/amaiya/ktrain/raw/master/ktrain_logo_200x100.png" width="200"/>
</p>

# Welcome to ktrain



### News and Announcements
- **2022-01-28**
  - **ktrain v0.29.x** is released and includes miscellaneous enhancements contributed by [Sandy Khosasi](https://github.com/ilos-vigil) such as [support for MobileNetV3 and EfficientNet](https://colab.research.google.com/drive/1EJHpMVG6fBCg33UPla_Ly_6LQdswU2Ur?usp=sharing), [plotting improvements](https://colab.research.google.com/drive/1_WaRQ0J4g0VTn6HWS3kszdFZbBBWoa7R?usp=sharing), and [raw confidence scores in QA](https://colab.research.google.com/drive/1ParprLN9hFX6cxJ1w7bv91PYx4o0J1zm?usp=sharing).
- **2021-10-13**
  - **ktrain v0.28.x** is released and now includes the `AnswerExtractor`, which allows you to extract any information of interest from documents by simply phrasing it in the form of a question. A short example is shown here, but see the [example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/qa_information_extraction.ipynb) for more information.
```python
# QA-Based Information Extraction

# DataFrame BEFORE
df.head()
#     Text
#0    Three major risk factors for COVID-19 were sex (male), age (≥60), and severe pneumonia.
#1    His speciality is medical risk assessments, and he is 30 years old.
#2    A total of nine studies including 356 patients were included in this study.

# AnswerExtractor will create two new columns:  'Risk Factors' and 'Sample Size'
from ktrain.text.qa import AnswerExtractor
ae = AnswerExtractor()
df = ae.extract(df.Text.values, df, [('What are the risk factors?', 'Risk Factors'),
                                     ('How many individuals in sample?', 'Sample Size')])

# DataFrame AFTER
df[['Risk Fctors', 'Sample Size']].head()
#     Risk Factors                                       Sample Size
#0    sex (male), age (≥60), and severe pneumonia        None
#1    None                                               None
#2    None                                               356
```
- **2021-07-20**
  - **ktrain v0.27.x** is released and now supports causal inference using [meta-learners](https://arxiv.org/abs/1706.03461). See the [example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/tabular/causal_inference_example.ipynb) for more information.
- **2021-07-15**
  - **ktrain** was used to train machine learning models for [CoronaCentral.ai](https://coronacentral.ai/), a machine-learning-enhanced search engine for COVID publications at Stanford University. The CoronaCentral document classifier, **CoronaBERT**, is [available on the Hugging Face model hub](https://huggingface.co/jakelever/coronabert).  CoronaCentral.ai was developed by Jake Lever and Russ Altman and funded by the Chan Zuckerberg Biohub. Check out [their paper](https://www.biorxiv.org/content/10.1101/2020.12.21.423860v1).
----

### Overview

**ktrain** is a lightweight wrapper for the deep learning library [TensorFlow Keras](https://www.tensorflow.org/guide/keras/overview) (and other libraries) to help build, train, and deploy neural networks and other machine learning models.  Inspired by ML framework extensions like *fastai* and *ludwig*, **ktrain** is designed to make deep learning and AI more accessible and easier to apply for both newcomers and experienced practitioners. With only a few lines of code, **ktrain** allows you to easily and quickly:

- employ fast, accurate, and easy-to-use pre-canned models for  `text`, `vision`, `graph`, and `tabular` data:
  - `text` data:
     - **Text Classification**: [BERT](https://arxiv.org/abs/1810.04805), [DistilBERT](https://arxiv.org/abs/1910.01108), [NBSVM](https://www.aclweb.org/anthology/P12-2018), [fastText](https://arxiv.org/abs/1607.01759), and other models <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/IMDb-BERT.ipynb)]</sup></sub>
     - **Text Regression**: [BERT](https://arxiv.org/abs/1810.04805), [DistilBERT](https://arxiv.org/abs/1910.01108), Embedding-based linear text regression, [fastText](https://arxiv.org/abs/1607.01759), and other models <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/text_regression_example.ipynb)]</sup></sub>
     - **Sequence Labeling (NER)**:  Bidirectional LSTM with optional [CRF layer](https://arxiv.org/abs/1603.01360) and various embedding schemes such as pretrained [BERT](https://huggingface.co/transformers/pretrained_models.html) and [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html) word embeddings and character embeddings <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/CoNLL2002_Dutch-BiLSTM.ipynb)]</sup></sub>
     - **Ready-to-Use NER models for English, Chinese, and Russian** with no training required <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/shallownlp-examples.ipynb)]</sup></sub>
     - **Sentence Pair Classification**  for tasks like paraphrase detection <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/MRPC-BERT.ipynb)]</sup></sub>
     - **Unsupervised Topic Modeling** with [LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/20newsgroups-topic_modeling.ipynb)]</sup></sub>
     - **Document Similarity with One-Class Learning**:  given some documents of interest, find and score new documents that are thematically similar to them using [One-Class Text Classification](https://en.wikipedia.org/wiki/One-class_classification) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/20newsgroups-document_similarity_scorer.ipynb)]</sup></sub>
     - **Document Recommendation Engines and Semantic Searches**:  given a text snippet from a sample document, recommend documents that are semantically-related from a larger corpus  <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/20newsgroups-recommendation_engine.ipynb)]</sup></sub>
     - **Text Summarization**:  summarize long documents with a pretrained BART model - no training required <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/text_summarization_with_bart.ipynb)]</sup></sub>
     - **End-to-End Question-Answering**:  ask a large text corpus questions and receive exact answers <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb)]</sup></sub>
     - **Easy-to-Use Built-In Search Engine**:  perform keyword searches on large collections of documents <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb)]</sup></sub>
     - **Zero-Shot Learning**:  classify documents into user-provided topics **without** training examples <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/zero_shot_learning_with_nli.ipynb)]</sup></sub>
     - **Language Translation**:  translate text from one language to another <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/language_translation_example.ipynb)]</sup></sub>
     - **Text Extraction**: Extract text from PDFs, Word documents, etc. <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/text_extraction_example.ipynb)]</sup></sub>
     - **Speech Transcription**: Extract text from audio files <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/examples/text/speech_transcription_example.ipynb)]</sup></sub>
     - **Universal Information Extraction**:  extract any kind of information from documents by simply phrasing it in the form of a question <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/qa_information_extraction.ipynb)]</sup></sub>
  - `vision` data:
    - **image classification** (e.g., [ResNet](https://arxiv.org/abs/1512.03385), [Wide ResNet](https://arxiv.org/abs/1605.07146), [Inception](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)) <sub><sup>[[example notebook](https://colab.research.google.com/drive/1WipQJUPL7zqyvLT10yekxf_HNMXDDtyR)]</sup></sub>
    - **image regression** for predicting numerical targets from photos (e.g., age prediction) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/vision/utk_faces_age_prediction-resnet50.ipynb)]</sup></sub>
  - `graph` data:
    - **node classification** with graph neural networks ([GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/graphs/pubmed_node_classification-GraphSAGE.ipynb)]</sup></sub>
    - **link prediction** with graph neural networks ([GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/graphs/cora_link_prediction-GraphSAGE.ipynb)]</sup></sub>
  - `tabular` data:
    - **tabular classification** (e.g., Titanic survival prediction) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-08-tabular_classification_and_regression.ipynb)]</sup></sub>
    - **tabular regression** (e.g., predicting house prices) <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/tabular/HousePricePrediction-MLP.ipynb)]</sup></sub>
    - **causal inference** using meta-learners <sub><sup>[[example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/examples/tabular/causal_inference_example.ipynb)]</sup></sub>

- estimate an optimal learning rate for your model given your data using a Learning Rate Finder
- utilize learning rate schedules such as the [triangular policy](https://arxiv.org/abs/1506.01186), the [1cycle policy](https://arxiv.org/abs/1803.09820), and [SGDR](https://arxiv.org/abs/1608.03983) to effectively minimize loss and improve generalization
- build text classifiers for any language (e.g., [Arabic Sentiment Analysis with BERT](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/ArabicHotelReviews-AraBERT.ipynb), [Chinese Sentiment Analysis with NBSVM](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/ChineseHotelReviews-nbsvm.ipynb))
- easily train NER models for any language (e.g., [Dutch NER](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/examples/text/CoNLL2002_Dutch-BiLSTM.ipynb) )
- load and preprocess text and image data from a variety of formats 
- inspect data points that were misclassified and [provide explanations](https://eli5.readthedocs.io/en/latest/) to help improve your model
- leverage a simple prediction API for saving and deploying both models and data-preprocessing steps to make predictions on new raw data
- built-in support for exporting models to [ONNX](https://onnx.ai/) and  [TensorFlow Lite](https://www.tensorflow.org/lite) (see [example notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/examples/text/ktrain-ONNX-TFLite-examples.ipynb) for more information)



### Tutorials
Please see the following tutorial notebooks for a guide on how to use **ktrain** on your projects:
* Tutorial 1:  [Introduction](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-01-introduction.ipynb)
* Tutorial 2:  [Tuning Learning Rates](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-02-tuning-learning-rates.ipynb)
* Tutorial 3: [Image Classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-03-image-classification.ipynb)
* Tutorial 4: [Text Classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-04-text-classification.ipynb)
* Tutorial 5: [Learning from Unlabeled Text Data](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-05-learning_from_unlabeled_text_data.ipynb)
* Tutorial 6: [Text Sequence Tagging](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-06-sequence-tagging.ipynb) for Named Entity Recognition
* Tutorial 7: [Graph Node Classification](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-07-graph-node_classification.ipynb) with Graph Neural Networks
* Tutorial 8: [Tabular Classification and Regression](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-08-tabular_classification_and_regression.ipynb) 
* Tutorial A1: [Additional tricks](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A1-additional-tricks.ipynb), which covers topics such as previewing data augmentation schemes, inspecting intermediate output of Keras models for debugging, setting global weight decay, and use of built-in and custom callbacks.
* Tutorial A2: [Explaining Predictions and Misclassifications](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A2-explaining-predictions.ipynb)
* Tutorial A3: [Text Classification with Hugging Face Transformers](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/tutorials/tutorial-A3-hugging_face_transformers.ipynb)
* Tutorial A4: [Using Custom Data Formats and Models: Text Regression with Extra Regressors](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A4-customdata-text_regression_with_extra_regressors.ipynb)


Some blog tutorials and other guides about **ktrain** are shown below:

> [**ktrain: A Lightweight Wrapper for Keras to Help Train Neural Networks**](https://towardsdatascience.com/ktrain-a-lightweight-wrapper-for-keras-to-help-train-neural-networks-82851ba889c) 


> [**BERT Text Classification in 3 Lines of Code**](https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358)  

> [**Text Classification with Hugging Face Transformers in  TensorFlow 2 (Without Tears)**](https://medium.com/@asmaiya/text-classification-with-hugging-face-transformers-in-tensorflow-2-without-tears-ee50e4f3e7ed)

> [**Build an Open-Domain Question-Answering System With BERT in 3 Lines of Code**](https://towardsdatascience.com/build-an-open-domain-question-answering-system-with-bert-in-3-lines-of-code-da0131bc516b)

> [**Finetuning BERT using ktrain for Disaster Tweets Classification**](https://medium.com/analytics-vidhya/finetuning-bert-using-ktrain-for-disaster-tweets-classification-18f64a50910b) by Hamiz Ahmed

> [**Indonesian NLP Examples with ktrain**](https://github.com/ilos-vigil/ktrain-assessment-study) by Sandy Khosasi









### Examples

Tasks such as text classification and image classification can be accomplished easily with 
only a few lines of code.

#### Example: Text Classification of [IMDb Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) Using [BERT](https://arxiv.org/pdf/1810.04805.pdf) <sub><sup>[[see notebook](https://github.com/amaiya/ktrain/blob/master/examples/text/IMDb-BERT.ipynb)]</sup></sub>
```python
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


#### Example: Classifying Images of [Dogs and Cats](https://www.kaggle.com/c/dogs-vs-cats) Using a Pretrained [ResNet50](https://arxiv.org/abs/1512.03385) model <sub><sup>[[see notebook](https://colab.research.google.com/drive/1WipQJUPL7zqyvLT10yekxf_HNMXDDtyR)]</sup></sub>
```python
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

#### Example: Sequence Labeling for [Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/version/2) using a randomly initialized [Bidirectional LSTM CRF](https://arxiv.org/abs/1603.01360) model <sub><sup>[[see notebook](https://github.com/amaiya/ktrain/blob/master/examples/text/CoNLL2003-BiLSTM_CRF.ipynb)]</sup></sub>
```python
import ktrain
from ktrain import text as txt

# load data
(trn, val, preproc) = txt.entities_from_txt('data/ner_dataset.csv',
                                            sentence_column='Sentence #',
                                            word_column='Word',
                                            tag_column='Tag', 
                                            data_format='gmb',
                                            use_char=True) # enable character embeddings

# load model
model = txt.sequence_tagger('bilstm-crf', preproc)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model, train_data=trn, val_data=val)


# conventional training for 1 epoch using a learning rate of 0.001 (Keras default for Adam optmizer)
learner.fit(1e-3, 1) 
```


#### Example: Node Classification on [Cora Citation Graph](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) using a [GraphSAGE](https://arxiv.org/abs/1706.02216) model <sub><sup>[[see notbook](https://github.com/amaiya/ktrain/blob/master/examples/graphs/cora_node_classification-GraphSAGE.ipynb)]</sup></sub>
```python
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


#### Example: Text Classification with [Hugging Face Transformers](https://github.com/huggingface/transformers) on [20 Newsgroups Dataset](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) Using [DistilBERT](https://arxiv.org/abs/1910.01108) <sub><sup>[[see notebook](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb)]</sup></sub>
```python
# load text data
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
train_b = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_b = fetch_20newsgroups(subset='test',categories=categories, shuffle=True)
(x_train, y_train) = (train_b.data, train_b.target)
(x_test, y_test) = (test_b.data, test_b.target)

# build, train, and validate model (Transformer is wrapper around transformers library)
import ktrain
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=train_b.target_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(5e-5, 4)
learner.validate(class_names=t.get_classes()) # class_names must be string values

# Output from learner.validate()
#                        precision    recall  f1-score   support
#
#           alt.atheism       0.92      0.93      0.93       319
#         comp.graphics       0.97      0.97      0.97       389
#               sci.med       0.97      0.95      0.96       396
#soc.religion.christian       0.96      0.96      0.96       398
#
#              accuracy                           0.96      1502
#             macro avg       0.95      0.96      0.95      1502
#          weighted avg       0.96      0.96      0.96      1502
```

<!--
#### Example: NER With [BioBERT](https://arxiv.org/abs/1901.08746) Embeddings
```python
# NER with BioBERT embeddings
import ktrain
from ktrain import text as txt
x_train= [['IL-2', 'responsiveness', 'requires', 'three', 'distinct', 'elements', 'within', 'the', 'enhancer', '.'], ...]
y_train=[['B-protein', 'O', 'O', 'O', 'O', 'B-DNA', 'O', 'O', 'B-DNA', 'O'], ...]
(trn, val, preproc) = txt.entities_from_array(x_train, y_train)
model = txt.sequence_tagger('bilstm-bert', preproc, bert_model='monologg/biobert_v1.1_pubmed')
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=128)
learner.fit(0.01, 1, cycle_len=5)
```
-->

#### Example: Tabular Classification for [Titanic Survival Prediction](https://www.kaggle.com/c/titanic) Using an MLP  <sub><sup>[[see notebook](https://github.com/amaiya/ktrain/blob/master/examples/tabular/tabular_classification_and_regression_example.ipynb)]</sup></sub>
```python
import ktrain
from ktrain import tabular
import pandas as pd
train_df = pd.read_csv('train.csv', index_col=0)
train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], 1)
trn, val, preproc = tabular.tabular_from_df(train_df, label_columns=['Survived'], random_state=42)
learner = ktrain.get_learner(tabular.tabular_classifier('mlp', trn), train_data=trn, val_data=val)
learner.lr_find(show_plot=True, max_epochs=5) # estimate learning rate
learner.fit_onecycle(5e-3, 10)

# evaluate held-out labeled test set
tst = preproc.preprocess_test(pd.read_csv('heldout.csv', index_col=0))
learner.evaluate(tst, class_names=preproc.get_classes())
```


Using **ktrain** on **Google Colab**?  See these Colab examples:
-  [a simple demo of Multiclass Text Classification with BERT](https://colab.research.google.com/drive/1AH3fkKiEqBpVpO5ua00scp7zcHs5IDLK)
-  [a simple demo of Multiclass Text Classification with Hugging Face Transformers](https://colab.research.google.com/drive/1YxcceZxsNlvK35pRURgbwvkgejXwFxUt)
-  [image classification with Cats vs. Dogs](https://colab.research.google.com/drive/1WipQJUPL7zqyvLT10yekxf_HNMXDDtyR)

#### Additional examples can be found [here](https://github.com/amaiya/ktrain/tree/master/examples).



### Installation

1. Make sure pip is up-to-date with: `pip install -U pip`

2. [Install TensorFlow 2](https://www.tensorflow.org/install) if it is not already installed (e.g., `pip install tensorflow`)

3. Install *ktrain*: `pip install ktrain`


The above should be all you need on Linux systems and cloud computing environments like Google Colab and AWS EC2.  If you are using **ktrain** on a **Windows computer**, you can follow these 
[more detailed instructions](https://github.com/amaiya/ktrain/blob/master/FAQ.md#how-do-i-install-ktrain-on-a-windows-machine) that include some extra steps.

**ktrain** should currently support any version of TensorFlow at or above to v2.3: i.e., `pip install tensorflow>=2.3`.

**Some important things to note about installation:**

Some optional, extra libraries used for some operations can be installed as needed:
```python
# for graph module
pip install https://github.com/amaiya/stellargraph/archive/refs/heads/no_tf_dep_082.zip
# for text.TextPredictor.explain and vision.ImagePredictor.explain
pip install https://github.com/amaiya/eli5/archive/refs/heads/tfkeras_0_10_1.zip
# for tabular.TabularPredictor.explain
pip install shap
# for text.zsl (ZeroShotClassifier), text.summarization, text.translation, text.speech
pip install torch
# for text.speech
pip install librosa
# for tabular.causal_inference_model
pip install causalnlp
# for text.TextExtractor
pip install textract
# for text.qa.AnswerExtractor.finetune
pip install datasets
```
Notice that **ktrain** is using forked versions of the `eli5` and `stellargraph` libraries above in order to support TensorFlow2.

<!--
pip install pdoc3==0.9.2
pdoc3 --html -o docs ktrain
diff -qr docs/ktrain/ /path/to/repo/ktrain/docs
-->

### How to Cite

Please cite the [following paper](https://arxiv.org/abs/2004.10703) when using **ktrain**:
```
@article{maiya2020ktrain,
    title={ktrain: A Low-Code Library for Augmented Machine Learning},
    author={Arun S. Maiya},
    year={2020},
    eprint={2004.10703},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    journal={arXiv preprint arXiv:2004.10703},
}

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



----
**Creator:  [Arun S. Maiya](http://arun.maiya.net)**

**Email:** arun [at] maiya [dot] net
