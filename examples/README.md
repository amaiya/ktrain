# Example Notebooks

This directory contains various example notebooks using *ktrain*.  The directory currently has four folders:
- `text`:  
  - [text classification](#textclass): examples using various text classification models and datasets
  - [text regression](#textregression): example for predicting continuous value purely from text
  - [text sequence labeling](#seqlab):  sequence tagging models
  - [sentence pair classification](#sentpair):  sentence pair classification for tasks such as paraphrase or sarcasm detection
  - [topic modeling](#lda):  unsupervised learning from unlabeled text data
  - [document similarity with one-class learning](#docsim): given a sample of interesting documents, find and score new documents that are semantically similar to it using One-Class text classification
  - [document recommender system](#docrec):  given text from a sample document, recommend documents that are semantically similar to it from a larger corpus 
  - [Shallow NLP](#shallownlp):  a small collection of miscellaneous text utilities amenable to being used on machines with only a CPU available (no GPU required)
  - [Text Summarization](#bart):  an example of text summarization using a pretrained BART model
  - [Open-Domain Question-Answering](#textqa):  ask questions to a large text corpus and receive exact candidate answers
  - [Zero-Shot Learning](#zsl):  classify documents by user-supplied topics **without** any training examples
  - [Language Translation](#translation): an example of language translation using pretrained MarianMT models
  - [Text Extraction](#textextraction): extract text from PDFs, Word documents, etc.
  - [Speech Transcription](#transcription): extract text audio file
  - [Universal Information Extraction](#extraction): an example of using a Question-Answering model for information extraction
  - [Indonesian Text Examples](#indonesian):  examples such as zero-Shot text classification and question-answering on Indonesian text by [Sandy Khosasi](https://github.com/ilos-vigil)
- `vision`:  
  - [image classification](#imageclass):  models for image datasetsimage classification examples using various models and datasets
  - [image regression](#imageregression): example of predicting numerical values purely from images/photos
  - [image captioning](#imagecaptioning): example of captioning images with a pretrained model
- `graphs`: 
  - [node classification](#-graph-node-classification-datasets): node classification in graphs or networks
  - [link prediction](#-graph-link-prediction-datasets): link prediction in graphs or networks
- `tabular`: 
  - [classification](#-tabular-classification-datasets): classification for tabular data
  - [regression](#-tabular-regression-datasets): regression for tabular data
  - [causal inference](#-tabular-causal-inference): causal inference using meta-learners


## Text Data

### <a name="textclass"></a>Text Classification

#### [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/):  Binary Classification

IMDb is a dataset containing 50K movie reviews labeled as positive or negative.  The corpus is split evenly between training and validation.
The dataset is in the form of folders of images.

- [IMDb-fasttext.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  A simple and fast "custom" fasttext model.
- [IMDb-BERT.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  BERT text classification to predict sentiment of movie reviews.


#### [Chinese Sentiment Analysis](https://github.com/Tony607/Chinese_sentiment_analysis/tree/master/data/ChnSentiCorp_htl_ba_6000):  Binary Classification

This dataset consists of roughly 6000 hotel reviews in Chinese.  The objective is to predict the positive or negative sentiment of each review. This notebook shows an example of using *ktrain* with non-English text.

- [ChineseHotelReviews-nbsvm.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Training a simple and fast NBSVM model on this dataset with bigram/trigram features can achieve a validation accuracy of **92%** with only 7 seconds of training.  

- [ChineseHotelReviews-fasttext.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Using a fast and simple fasttext-like model to predict sentiment of Chinese-language hotel reviews. 

- [ChineseHotelReviews-BERT.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  BERT text classification to predict sentiment of Chinese-language hotel reviews.


#### [Arabic Sentiment Analysis](https://github.com/elnagara/HARD-Arabic-Dataset):  Binary Classification

This dataset consists contains hotel reviews in Arabic.  The objective is to predict the positive or negative sentiment of each review. This notebook shows an example of using *ktrain* with non-English text.

- [ArabicHotelReviews-nbsvm.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Training a simple and fast NBSVM model on this dataset with bigram/trigram features can achieve a validation accuracy of **94%** with only seconds of training.

- [ArabicHotelReviews-BERT.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  BERT text classification to predict sentiment of Arabic-language hotel reviews.


#### [20 News Groups](http://qwone.com/~jason/20Newsgroups/): Multiclass Classification
This is a small sample of the 20newsgroups dataset based on considering 4 newsgroups similar to what was done in the
[Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) scikit-learn tutorial. 
Data are in the form of arrays fetched via scikit-learn library.
These examples show the results of training on a relatively small training set.
- [20newsgroups-NBVSM.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  NBSVM model using unigram features only.
- [20newsgroups-BERT.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  BERT text classification in a multiclass setting.
- [20newsgroups-distilbert.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text): a faster, smaller version of BERT called DistilBert for multiclass text classification
- [ktrain-ONNX-TFLite-examples.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text): text classification using ONNX and TensorFlow Lite


#### [Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge):  Multi-Label Text Classification
In multi-label classification, a single document can belong to multiple classes.  The objective here is
to categorize each text comment into one or more categories of toxic online behavior.
Dataset is in the form of a CSV file.
- [toxic_comments-fasttext.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  A fasttext-like model applied in a multi-label setting.
- [toxic_comments-bigru.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  A bidirectional GRU using pretrained Glove vectors. This example shows how to use pretreained word vectors using *ktrain*.


### <a name="textregression"></a>Text Regression 

#### [Wine Prices](https://www.floydhub.com/floydhub/datasets/wine-reviews/1/wine_data.csv):  Text Regression

This dataset consists of prices of various wines along with textual descriptions of the wine.
We will attempt to predict the price of a wine based purely on its text description.
- [text_regression_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Using an Embedding-based implementation of linear regression to predict wine prices from text column.



### <a name="seqlab"></a> Sequence Labeling 
#### [CoNLL2003 NER Task](https://github.com/amaiya/ktrain/tree/master/ktrain/tests/conll2003):  Named Entity Recognition
The objective of the CoNLL2003 task is to classify sequences of words as belonging to one of several categories of concepts such as Persons or Locations. See the [original paper](https://www.aclweb.org/anthology/W03-0419) for more information on the format.

- [CoNLL2003-BiLSTM_CRF.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  A simple and fast Bidirectional LSTM-CRF model with randomly initialized word embeddings.
- [CoNLL2003-BiLSTM.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  A Bidirectional LSTM model with pretrained BERT embeddings


#### [CoNLL2002 NER Task (Dutch)](https://www.clips.uantwerpen.be/conll2002/ner/):  Named Entity Recognition for Dutch

- [CoNLL2002_Dutch-BiLSTM.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  A Bidirectional LSTM model that uses pretrained BERT embeddings along with pretrained fasttext word embeddings - both for Dutch.


### <a name="sentpair"></a> Sentence Pair Classification

#### [Microsoft Research Paraphrase Corpus (MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398):  Paraphrase Detection

- [MRPC-BERT.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Using BERT for sentence pair classification on MRPC dataset


### <a name="lda"></a> Topic Modeling

#### [20 News Groups](http://qwone.com/~jason/20Newsgroups/): unsupervised learning on 20newsgroups corpus
- [20newsgroups-topic_modeling.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Discover latent topics and themes in the 20 newsgroups corpus 


### <a name="docsim"></a> One-Class Text Classification

#### [20 News Groups](http://qwone.com/~jason/20Newsgroups/): select a set of positive examples from 20newsgroups dataset
- [20newsgroups-document_similarity_scorer.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  Given a selected seed set of documents from 20newsgroup corpus, find and score new documents that are semantically similar to it.


### <a name="docrec"></a> Text Recommender System

#### [20 News Groups](http://qwone.com/~jason/20Newsgroups/): recommend posts from 20newsgroups
- [20newsgroups-recommendation_engine.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  given text from a sample document, recommend documents that are semantically similar to it from a larger corpus


### <a name="shallownlp"></a> Shallow NLP

#### [shallownlp-examples.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
- Shallow NLP is a submodule in *ktrain* that provides a small collection of miscellaneous text utiilities to analyze text on machines with no GPU and minimal computational resources.  Includes:
    - *text classification*: a non-neural version of NBSVM that can be trained in a single line of code
	- *named-entity-recognition (NER)*:  out-of-the box, ready-to-use NER for English, Russian, and Chinese that just works with no training
	- *multilingual text searches*:  searching text in multiple languages


#### [20 News Groups](http://qwone.com/~jason/20Newsgroups/): recommend posts from 20newsgroups
- [20newsgroups-recommendation_engine.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text):  given text from a sample document, recommend documents that are semantically similar to it from a larger corpus

### <a name="bart"></a>Text Summarization with pretrained BART: [text_summarization_with_bart.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="textqa"></a>Open-Domain Question-Answering: [question_answering_with_bert.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="zsl"></a>Zero-Shot Learning: [zero_shot_learning_with_nli.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="translation"></a>Language Translation: [language_translation_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="textextraction"></a>Text Extraction: [text_extraction_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="transcription"></a>Speech Transcription: [speech_transcription_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="extraction"></a>Universal Information Extraction: [qa_information_extraction.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/text)
### <a name="indonesian"></a> [Indonesian NLP examples by Sandy Khosasi](https://github.com/ilos-vigil/ktrain-assessment-study) including Indonesian question-answering, emotion recognition, and document similarity


## Vision Data

### <a name="imageclass"></a> Image Classification 

#### [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats):  Binary Classification
- [dogs_vs_cats-ResNet50.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  ResNet50 pretrained on ImageNet.  

#### [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats):  Binary Classification
- [dogs_vs_cats-MobileNet.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  MobileNet pretrained on ImageNet on filtered version of Dogs vs. Cats dataset

#### [MNIST](http://yann.lecun.com/exdb/mnist/):  Multiclass Classification
- [mnist-WRN22.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  A randomly-initialized Wide Residual Network applied to MNIST

#### [MNIST](http://yann.lecun.com/exdb/mnist/):  Multiclass Classification
- [mnist-image_from_array_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  Build an MNIST model using `images_from_array`

#### [MNIST](http://yann.lecun.com/exdb/mnist/):  Multiclass Classification
- [mnist-tf_workflow.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  Illustrates how *ktrain* can be used in minimally-invasive way with normal TF workflow

#### [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html):  Multiclass Classification
- [cifar10-WRN22.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  A randomly-initialized Wide Residual Network applied to CIFAR10


#### [Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/): Multiclass Classification
- [pets-ResNet50.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  Categorizing dogs and cats by breed using a pretrained ResNet50. Uses the `images_from_fname` function, as class labels are embedded in the file names of images.


#### [Planet](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space): Multilabel Classification
The Kaggle Planet dataset consists of satellite images - each of which are categorized into multiple categories.
Image labels are in the form of a CSV containing paths to images.
- [planet-ResNet50.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  Using a pretrained ResNet50 model for multi-label classification.


### <a name="imageregression"></a> Image Regression

#### [Age Prediction](http://aicip.eecs.utk.edu/wiki/UTKFace):  Image Regression 
- [utk_faces_age_prediction-resnet50.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/vision):  ResNet50 pretrained on ImageNet for age prediction  using UTK Face dataset

### <a name="imagecaptioning"></a> [Image Captioning]((https://en.wikipedia.org/wiki/Automatic_image_annotation)
### [image_captioning_example.ipynb]((https://github.com/amaiya/ktrain/tree/master/examples/vision) is a notebook illustrating pretrained image-captioning in **ktrain**

## Graph Data

### <a name="#nodeclass"></a> Graph Node Classification Datasets

#### [PubMed-Diabetes](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz):  Node Classification

In the PubMed graph, each node represents a paper pertaining to one of three topics:  *Diabetes Mellitus - Experimental*, *Diabetes Mellitus - Type 1*, and *Diabetes Mellitus - Type 2*.  Links represent citations between papers.  The attributes or features assigned to each node are in the form of a vector of words in each paper and their corresponding TF-IDF scores.

- [pubmed_node_classification-GraphSAGE.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/graphs): GraphSAGE model for transductive and inductive inference.

#### [Cora Citation Graph](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz):  Node Classification

In the Cora citation graph, each node represents a paper pertaining to one of several topic categories.  Links represent citations between papers.  The attributes or features assigned to each node is in the form of a multi-hot-encoded vector of words in each paper.

- [cora_node_classification-GraphSAGE.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/graphs): GraphSAGE model for transductive inference on validation and test set of nodes in graph.


#### [Hateful Twitter Users](https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter):  Node Classification
Dataset of Twitter users and their attributes.  A small portion of the user accounts are annotated as `hateful` or `normal`.  The goal is to predict hateful accounts based on user features and graph structure.

- [hateful_twitter_users-GraphSAGE.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/graphs): GraphSAGE model to predict hateful Twitter users using transductive inference.


### <a name="#linkpred"></a> Graph Link Prediction Datasets


#### [Cora Citation Graph](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz):  Node Classification

In the Cora citation graph, each node represents a paper. Links represent citations between papers.  The attributes or features assigned to each node is in the form of a multi-hot-encoded vector of words in each paper.

- [cora_link_prediction-GraphSAGE.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/graphs): GraphSAGE model to predict missing links in the citation network.


## Tabular Data

### <a name="#tabularclass"></a> Tabular Classification Datasets

#### [Titanic Survival Prediction](https://www.kaggle.com/c/titanic):  Tabular Classification

This is the well-studied Titanic dataset from Kaggle.  The goal is to predict which passengers survived the Titanic disaster based on their attributes.

- [tabular_classification_and_regression_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/tabular): MLP for tabular classification


#### [Income Prediction from Census Data](http://archive.ics.uci.edu/ml/datasets/Adult):  Tabular Classification

This is the same dataset used in the [AutoGluon classification example](https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html).
The goal is to predict which individuals make over $50K per year.


- [IncomePrediction-MLP.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/tabular): MLP for tabular classification


### <a name="#tabularreg"></a> Tabular Regression Datasets


#### [Adults Census Dataset](http://archive.ics.uci.edu/ml/datasets/Adult):  Tabular Regression

The original goal of this dataset is to predict the individuals that make over $50K in this Census dataset.  We change the task to a regression problem
and predict the Age attribute for each individual.  This is the same dataset used in the [AutoGluon regression example](https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html).

- [tabular_classification_and_regression_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/tabular): MLP for tabular regression


#### [House Price Prediction](https://www.kaggle.com/c/house-prices-advanced-regression-techniques):  Tabular Regression

- [HousePricePrediction-MLP.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/tabular): MLP for tabular regression


### <a name="#tabularci"></a> Tabular Causal Inference


#### [Adults Census Dataset](http://archive.ics.uci.edu/ml/datasets/Adult):  Tabular Causal Inference

The original goal of this dataset is to predict the individuals that make over $50K in this Census dataset. Here, we will use causal inference to estimate
the causal impact of having a PhD on earning over $50K.  

- [causal_inference_example.ipynb](https://github.com/amaiya/ktrain/tree/master/examples/tabular): use meta-learners for causal inference and uplift modeling

