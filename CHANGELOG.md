# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour


## 0.14.0 (2020-04-15)

### New:
- support for building Question-Answering systems
- `textutils` now contains `paragraph_tokenize` function

### Changed
- N/A

### Fixed:
- resolved import issue with `textutils.sent_tokenize'



## 0.13.2 (2020-04-09)

### New:
- N/A

### Changed
- `TransformerSummarizer` accepts BART `model_name` as parameter


### Fixed:
- N/A



## 0.13.1 including 0.13.0 (2020-04-09)

### New:
- support for link prediction with graph neural networks
- text summarization with pretrained BART (included in 0.13.1 but not in 0.13.0)
- `bigru` method now selects pretrained word vectors based on detected language

### Changed
- instead of throwing error, default to English if `detect_lang` could not detect language from batch of texts
- `layers` argument moved to `TransformerEmbedding` constructor
- enforce specific version of TensorFlow due to undocumented breaking changes in newer TF versions
- `AdamWeightDecay` optimizer is now used to support global weight decay. Used when user
   excplictly sets a weight decay


### Fixed:
- force re-instantiation of `TransformerEmbedding` object with `sequence_tagger` function is re-invoked


## 0.12.3 (2020-04-02)

### New:
- Added `max_momentum` and `min_momentum` parameters to `autofit` and `fit_onecycle` to control cyclical momentum

### Changed
- Prevent loading errors of previously saved NERPreprocessor objects


### Fixed:
- N/A


## 0.12.2 (2020-04-01)

### New:
- N/A

### Changed
- Require at least TensorFlow 2.1.0 is installed in `setup.py` due to TF 2.0.0 bug with `lr_find`
- Added lower bounds to scikit-learn and networkx versions


### Fixed:
- N/A


## 0.12.1 (2020-04-01)

### New:
- N/A

### Changed
- N/A


### Fixed:
- check and ensure AllenNLP is installed when Elmo embeddings are selected for NER


## 0.12.0 (2020-03-31)

### New:
- BERT and Elmo embeddings for NER and other downstream tasks

### Changed
- `wv_path_or_url` parameter moved from `entities_from*` to `sequence_taggers`
- Added `use_char` parameter and ensure it is not used unless `DISABLE_V2_BEHAVIOR` is enabled:
- `batch_size` argument added to `get_predictor` and `load_predictor`
- `eval_batch_size` argument added to `get_learner`
- added `val_pct` argument to `entities_from_array`


### Fixed:
- properly set threshold in `text.eda` (PR #99)
- fixed error when no validation data is supplied to `entities_from_array`


## 0.11.3 (2020-03-18)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- prevent errors with reading word vector files on Windows by specifying `encoding='utf-8'`


## 0.11.2 (2020-03-18)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- `ktrain.text.eda.visualize_documents` now properly processes filepath argument


## 0.11.1 (2020-03-18)

### New:
- `entities_from_txt`, `entities_from_gmb`, and `entities_from_conll2003` functions now discover
   the encoding of the file automatically when `encoding=None` (which is the default now)

### Changed:
- N/A

### Fixed:
- N/A


## 0.11.0 (2020-03-18)

### New:
- sequence-taging (e.g., NER) now supports ELMo embeddings with `use_elmo=True` argument to data-loading
  functions like `entities_from_array`  and `entities_from_txt`A
- pretrained word embeddings (i.e., fasttext word2vec embeddings) can be specified by providing the URL to
  a `.vec.gz` file from [here](https://fasttext.cc/docs/en/crawl-vectors.html). The URL (or path) is
  supplied as `wv_path_or_url` argument to data-loading functions like `entities_from_array` and `entities_from_txt`
- `show_random_images`: show random images from folder in Jupyter notebook
- `NERPreprocessor` now includes a `preprocess_test` method for easier evaluation of test sets in datasets
   that contain a training, validation, and test set

### Changed:
- ensure `DISABLE_V2_BEHAVIOR=True` when `ImagePredictor.explain` is invoked
- added `SUPPRESS_TF_WARNINGS` environment variable.  Default is '1'. If set to '0', TF warnings will be displayed.
- `merge_entities` method of `ktrain.text.shallownlp.ner.NER` changed to `merge_tokens` 
- moved `load_predictor` to constructor in `krain.text.shallownlp.ner.NER`
- `ktrain.text.shallownlp.ner.NER` now supports `predictor_path` argument

### Fixed:
- convert `class_names` to strings in `core.validate` to prevent error from scikit-learn
- fixed error arising when no data augmentation scheme is provided to the `images_from*` functions
- fixed bug in `images_from_fname` to ensure supplied `pattern` is used
- added `val_folder` argument to `images_from_fname`
- raise Exception when `preproc` is not found in `load_predictor`
- check for existence of `preproc` in `text_classifier` and `text_regression_model`
- fixed `text.eda` so that `detect_lang` is called correctly after being moved to `textutils`


## 0.10.1 (2020-03-04)

### New:
- N/A

### Changed:
- `shallownlp.Classifier.texts_from_folder` changed to `shallownlp.Classifier.load_texts_from_folder`
- `shallownlp.Classifier.texts_from_csv` changed to `shallownlp.Classifier.load_texts_from_csv`
- In `text.preprocessor`, added warning that `class_names` is being ignored when `class_names` were supplied
  and `y_train` and `y_test` contain string labels

### Fixed:
- N/A




## 0.10.0 (2020-03-03)

### New:
- `Transformer` API in *ktrain*  now supports using community-uploaded transformer models
- added `shallownlp` module with out-of-the-box NER for English, Russian, and Chinese
- `text.eda` module now supports NMF in addition to LDA

### Changed:
- `texts_from_csv` and `texts_from_df` now accept a single column of labels in string format and will 
   1-hot-encode labels automatically for classification or multi-class classification problems.
- reorganized language-handling to `text.textutils`
- more suppression of warnings due to spurious warnings from TF2 causing confusion in output
- `classes` argument to `Transformer` constructor has been changed to `class_names` for consistency with `texts_from_array`

### Fixed:
- N/A


## 0.9.4 (2020-02-13)

### New:
- N/A

### Changed:
- changed pandas dependency to `>=1.0.1` due to bug in pandas 1.0

### Fixed:
- N/A


## 0.9.3 (2020-02-11)

### New:
- N/A

### Changed:
- Transformed data containers for transformers, NER, and graph -node classification to be
  instances of `ktrain.data.Dataset`.

### Fixed:
- fixed `images_from_array` so that y labels are correctly 1-hot-encoded when necessary
- correct tokenization for `bert-base-japanese` Transformer models from PR 57


## 0.9.2 (2020-02-04)

### New:
- N/A

### Changed:
- Removed Exception when `distilbert` is selected in `text_classifier` for non-English language after 
  [Hugging Face fixed the reported bug](https://github.com/huggingface/transformers/issues/2462). 

### Fixed:
- XLNet models like `xlnet-base-cased` now works after casting input arrays to `int32`
- modified `TextPredictor.explain` to propogate correct error message from `eli5` for multilabel text classification.


## 0.9.1 (2020-02-01)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- fixed `utils.nclasses_from_data` for `ktrain.Dataset` instances
- prevent `detect_lang` failing when Pandas Series is supplied


## 0.9.0 (2020-01-31)

### New:
- support for out-of-the-box text regression in both the `Transformer` API and conventional API (i.e., `text.text_regression_model`).

### Changed:
- `text.TextPreprocessor` prints sequence length statistics

### Fixed:
- auto-detect language when using `Transformer` class to prevent printing `en` as default


## 0.8.3 (2020-01-22)

### New:
- N/A

### Changed:
- `MultiArrayDataset` accepts list of Numpy arrays 

### Fixed:
- fixed incorrect activation in `TextPredictor` for multi-label Transformer models
- fixed `top_losses` for regression tasks


## 0.8.2 (2020-01-19)

### New:
- initial base `ktrain.Dataset` class for use as a Sequence wrapper to better support custom datasets/models

### Changed:
- N/A

### Fixed:
- N/A




## 0.8.1 (2020-01-15)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- fix to support multilabel text classification in `Transformers`
- `_prepare_dataset` no longer breaks when validation dataset has not been supplied


## 0.8.0 (2020-01-14)

### New:
- availability of a new, simplied interface to Hugging Face transformer models
- added 'distilbert' as an available model in `text.text_classifier` function

### Changed:
- `preproc` argument is required for `text.text_classifier`

### Fixed:
- `core._load_model` calls `_make_predict_function` before returning model
- added warning when non-adam optimizer is used with `cycle_momentum=True`



## 0.7.3 (2019-12-31)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- Fixed error when using *ktrain* with v0.2.x of `fastprogress`. *ktrain* can now be used with both v0.1.x and v0.2.x of `fastprogress`


## 0.7.2 (2019-12-11)

### New:
- All data-loading functions (e.g., `texts_from_csv`) accept a `random_state` argument
that will enable consistent reproduction of the train-test split.

### Changed:
- perform local checks for `stellargraph` where needed.  
- removed `stellargraph` as dependency due to issues with it overwriting `tensorflow-gpu`
- change `setup.py` to skip navigation links for pypi page

### Fixed:
- N/A


## 0.7.1 (2019-12-11)

### New:
- All data-loading functions (e.g., `texts_from_csv`) accept a `random_state` argument
that will enable consistent reproduction of the train-test split.

### Changed:
- perform local checks for `stellargraph` where needed.  
- removed `stellargraph` as dependency due to issues with it overwriting `tensorflow-gpu`

### Fixed:
- N/A


## 0.7.0 (2019-12-10)

### New:
- *ktrain* now uses tf.keras (`tensorflow>=1.14,<=2.0`) instead of stand-alone Keras.  

### Changed:
- N/A

### Fixed:
- N/A


## 0.6.2 (2019-12-02)

### New:
- N/A

### Changed:

### Fixed:
- added encoding argument when reading in word vectors to bypass error on Windows systems (PR #31)
- Change preprocessing defaults and apply special preprocessing in `text.eda.get_topic_model` 
  when non-English is detected.


## 0.6.1 (2019-11-16)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- `TextPredictor.explain` now correcty supports non-English languages.
- Parameter `activation` is no longer ignored in `_build_bert` function


## 0.6.0 (2019-11-12)

### New:
- support for learning from unlabeled or partially-labeled text data
  - unsupervised topic modeling with LDA
  - one-class text classification to score documents based on similarity to a set of positive examples
  - document recommendation engine

### Changed:
- N/A


### Fixed:
- Removed dangling reference to external 'stellargraph' dependency from `_load_model`, so that we rely solely on
  local version of stellargraph


## 0.5.2 (2019-10-20)

### New:
- N/A

### Changed:
- N/A


### Fixed:
- Removed dangling reference to external 'stellargraph' dependency so that we rely solely on
  local version of stellargraph



## 0.5.1 (2019-10-17)

### New:
- N/A

### Changed:
- N/A


### Fixed:
- store a local version of `stellargraph` to prevent it from installing `tensorflow-cpu`
  and overriding existing `tensorflow-gpu` installation




## 0.5.0 (2019-10-16)

### New:
- Support for node classification in graphs with `ktrain.graph` module

### Changed:
- N/A


### Fixed:
- N/A


## 0.4.3 (2019-10-14)

### New:
- N/A

### Changed:
- N/A


### Fixed:
- Call `reset` before `predict_generator` for consistent  ordering of `view_top_losses` results
- Fixed incorrect reference to `train_df` instead of `val_df` in `texts_from_df`


## 0.4.2 (2019-10-01)

### New:
- All `fit` methods in *ktrain* now accept `class_weight` parameter to handle imbalanced datasets.

### Changed:
- N/A


### Fixed:
- Resolved problem with `text_classifier` incorrectly using `uncased_L-12_H-768_A-12` to build BERT model
  instead of `multi_cased_L-12_H-768_A-12` when non-English language was detected.
- Fixed error messages releated to preproc requirement in `text_classifier`
- Fixed test script for multingual text classification
- Fixed rendering of Chinese in `view_top_losses`



## 0.4.1 (2019-10-01)

### New:
- N/A

### Changed:
- N/A


### Fixed:
- Fix problem with `text_classifier` incorrectly using `uncased_L-12_H-768_A-12` to build BERT model
  instead of `multi_cased_L-12_H-768_A-12` when non-English language was detected.




## 0.4.0 (2019-09-30)

### New:
- Added multilingual support for text classification.  
- Added experimental support for tf.keras. By default, *ktrain* will use standalone Keras.
  If `os.environ['TF_KERAS']` is set, *ktrian* will attempt to use tf.keras. 
  Some capabilities (e.g., `predictor.explain` for images) are not yet supported for tf.keras

### Changed:
- When BERT is selected, check to make sure dataset is correctly preprocessed for BERT


### Fixed:
- Fixed `utils.bert_data_type` and ensures it does more checks to validate BERT-style data


## 0.3.1 (2019-09-19)

### New:
- N/A

### Changed:
- globally import tensorflow
- suppress tensorflow deprecation warnings from TF 1.14.0


### Fixed:
- Resolved issue with `text_classifier` failing when BERT is selected and Preprocessor is supplied.


## 0.3.0 (2019-09-17)

### New:
- Support for sequence tagging with Bidirectional LSTM-CRF. Word embeddings can currently be either
  random or word2vec(cbow).  If latter chosen, word vectors will be downloaded automaticlaly from Facebook fasttext 
  site.
- Added `ktra.text.texts_from_df` function

### Changed:
- Added FutureWarning in ```text.text_classifier```, that ```preproc``` will be required argument in future.
- In ```text.text_classifier```, when ```preproc=None```, use the maximum feature ID to populate max_features.


### Fixed:
- Fixed construction of custom_objects dictionary for BERT to ensure load_model works for 
  custom BERT models
- Resolved issue with pretrained bigru models failing when max_features >= than total word count.



## 0.2.5 (2019-08-27)

### New:
- ```explain``` methods have been added to ```TextPredictor``` and ```ImagePredictor``` objects.
- ```TextPredictor.predict_proba``` and ```ImagePredictor.predict_proba_*``` convenience
methods have been added.
- Added ```utils.is_classifier``` utility function

### Changed:
- ```TextPredictor.predict``` method can now accept a single document as input instead of
always requiring a list.
- Output of ```core.view_top_losses``` now includes the ground truth label of examples

### Fixed:
- Fixed test of data loading


## 0.2.4 (2019-08-20)

### New:
- added additional tests of *ktrain*

### Changed:
- Added ```classes``` argument to ```vision.images_from_folder```.  Only classes/subfolders
  matching a name in the ```classes``` list will be considered.

### Fixed:
- Resolved issue with using ```learner.view_top_losses``` with BERT models.


## 0.2.3 (2019-08-18)

### New:
- N/A

### Changed:
- Added ```classes``` argument to ```vision.images_from_folder```.  Only classes/subfolders
  matching a name in the ```classes``` list will be considered.

### Fixed:
- Fixed issue with ```learner.validate``` and ```learner.predict``` failing when validation data is in 
  the form of an Iterator (e.g., DirectoryIterator).


## 0.2.2 (2019-08-16)

### New:
- N/A

### Changed:
- Added check in ```ktrain.lroptimize.lrfinder``` to stop training if learning rate exceeds a fixed maximum,
  which may happen when bad/dysfunctional model is supplied to learning rate finder.

### Fixed:
- In ```ktrain.text.data.texts_from_folder``` function, only subfolders specified in classes argument 
  are read in as training and validation data.

## 0.2.1 (2019-08-15)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- Fixed error related to validation_steps=None in call to fit_generator in ```ktrain.core``` on Google Colab.


## 0.2.0 (2019-08-12)

### New:
- Support for pretrained BERT Text Classification 

### Changed:
- For ```Learner.lr_find```, added optional ```max_epochs``` argument. 
- Changed ```Learner.confusion_matrix``` to ```Learner.validate``` and added optional ```val_data``` argument.
  The ```use_valid``` argument has been removed.
- Removed ```pretrained_fpath``` argument to ```text.text_classifier```.  Pretrained word vectors are
  now downloaded automatically when 'bigru' is selected as model.

### Fixed:
- Further cleanup of  ```utils.is_iter``` function to use type check.





## 0.1.10 (2019-08-02)

### New:
- N/A

### Changed:
- For ```Learner.lr_find```, removed epochs and max_lr arguments and added lr_mult argument
  Default lr_mult is 1.01, but can be changed to control size of sample being used
  to estimate learning rate.
- Changed structure of examples folder

### Fixed:
- Resolved issue with ```utils.y_from_data``` not working correctly with DataFrameIterator objects.


## 0.1.9 (2019-08-01)

### New:
- N/A

### Changed:
- Use class check in utils.is_iter as temporary fix
- revert to epochs=5 for ```Learner.lr_find```

### Fixed:
- N/A

## 0.1.8 (2019-06-04)

### New:
- N/A

### Changed:
- N/A

### Fixed:
- ```Learner.set_weight_decay``` now works correctly


## 0.1.7 (2019-05-24)

### New:
- BIGRU text classifier: Bidirectional GRU using pretrained word embeddings

### Changed:
- Epochs are calculated automatically in ```LRFinder```

### Fixed:
- Number of epochs that ```Learner.lr_find``` runs can be explicitly set again


## 0.1.6 (2019-05-03)

### New:

### Changed:
- relocated calls to tensorflow 
- installation instructions and reformatted examples

### Fixed:



## 0.1.5 (2019-05-01)

### New:
- **cycle\_momentum** argument for both```autofit``` and ```fit_onecycle``` method that will cycle momentum between 0.95 and 0.85 as described in [this paper](https://arxiv.org/abs/1803.09820)
- ```Learner.plot``` method that will plot training-validation loss, LR schedule, or momentum schedule
- added ```set_weight_decay``` and ```get_weight_decay``` methods to get/set "global" weight decay in Keras

### Changed:
- ```vision.data.preview_data_aug``` now displays images in rows by default
- added multigpu flag to ```core.get_learner``` with comment that it is only supported by```vision.model.image_classifier```
- added ```he_normal``` initialization to FastText model

### Fixed:

- Bug in ```vision.data.images_from_fname``` that prevented relative paths for directory argument
- Bug in ```utils.y_from_data``` that returned incorrect information for array-based training/validation data
- Bug in ```core.autofit``` with callback failure when validation data is not set
- Bug in ```core.autofit``` and ```core.fit_onecycle``` with learning rate setting at end of cycle


## 0.1.4 (2019-04-10)

- Last release without CHANGELOG updates



