# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

## 0.21.1 (2020-09-03)

### New:
- N/A

### Changed
- added `num_beams` and `early_stopping` arguments to `translate` methods in `translation` module that can be set to improve translation speed

### Fixed:
- N/A



## 0.21.0 (2020-09-03)

### New:
- Added `translate_sentences` method to `Translator` class that translates list of sentences, where list is fed to model as single batch

### Changed
- Removed TensorFlow dependency from `setup.py` to allow users to use *ktrain* with any version of TensorFlow 2 they choose.
- Added `truncation=True` to tokenization in `summarization` module
- Require `transformers>=3.1.0` due to breaking changes
- `SUPPRESS_TF_WARNINGS` environment variable changed to `SUPPRESS_KTRAIN_WARNINGS`

### Fixed:
- Use `prepare_seq2seq_batch` insteadd of `prepare_translation_batch` in `translation` module due to breaking change in `transformers==3.1.0`


## 0.20.2 (2020-08-27)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Always use `*Auto*` classes to load `transformers` models to prevent loading errors


## 0.20.1 (2020-08-25)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Added missing `torch.no_grad()` scope in `text.translation` and `text.summarization` modules



## 0.20.0 (2020-08-24)

### New:
- added `nli_template` parameter to `ZeroShotClassifier.predict` to allow versatility in the kinds of labels that
  can be predicted
- efficiency improvements to `ZeroShotClassifier.predict` that allow faster predictions on large sequences
  of documents and a large numer of labels to predict
- added 'multilabel` parameter to `ZeroShotClassifier.predict`
- added `labels` parameter to `ZeroShotClassifer.predict`, an alias to `topic_strings` parameter

### Changed
- N/A

### Fixed:
- Allow variations on `accuracy` metric such as `binary_accuracy` when inpecting model in `is_classifier`


## 0.19.9 (2020-08-17)

### New:
- N/A

### Changed
- N/A

### Fixed:
- In `texts_from_array`, check `class_names` only after preprocessing before printing classification vs. regression status.


## 0.19.8 (2020-08-17)

### New:
- N/A

### Changed
- N/A

### Fixed:
- In `TextPreprocessor` instances, correctly reset `class_names` when targets are in string format.


## 0.19.7 (2020-08-16)

### New:
- N/A

### Changed
- added `class_weight` parameter to `lr_find` for imbalanced datasets
- removed pins for `cchardet` and `scikitlearn` from `setup.py`
- added version check for `eli5` fork
- removed `scipy` pin from `setup.py`
- Allow TensorFlow 2.3 for Python 3.8
- Request  manual installation of `shap` in `TabularPredictor.explain` instead of inclusion in `setup.py`

### Fixed:
- N/A


## 0.19.6 (2020-08-12)

### New:
- N/A

### Changed
-N/A

### Fixed:
- include metrics check in `is_classifier` function to support with non-standard loss functions


## 0.19.5 (2020-08-11)

### New:
- N/A

### Changed
-N/A

### Fixed:
- Ensure transition to `YTransform` is backwards compatibility for `StandardTextPreprocessor` and `BertPreprocessor`


## 0.19.4 (2020-08-10)

### New:
- N/A

### Changed
- `TextPreprocessor` instances now use `YTransform` class to transform targets
- `texts_from_df`, `texts_from_csv`, and `texts_from_array` employ the use of either `YTransformDataFrame` or `YTransform`
- `images_from_df`, `images_from_fname`, `images_from_csv`, and `imagas_from_array` use `YTransformDataFrame` or `YTransform`
- Extra imports removed from PyTorch-based `zsl.core.ZeroShotClassifier` and `summarization.core.TransformerSummarizer`. If necessary, 
   both can now be used without having TensorFlow installed by installing ktrain using `--no-deps` and importing these modules using 
    a method like [this](https://stackoverflow.com/a/58423785).

### Fixed:
- N/A


## 0.19.3 (2020-08-05)

### New:
- N/A/

### Changed
- `NERPredictor.predict` was changed to accept an optional `custom_tokenizer` argument

### Fixed:
- N/A



## 0.19.2 (2020-08-03)

### New:
- N/A

### Changed
- N/A

### Fixed:
- added missing `num_classes` argument to `to_categorical`


## 0.19.1 (2020-07-29)

### New:
- N/A

### Changed
- Adjusted `no_grad` scope in `ZeroShotClassifier.predict`

### Fixed:
- N/A


## 0.19.0 (2020-07-29)

### New:
- support for `tabular` data including explainable AI for tabular predictions
- `learner.validate` and `learner.evaluate` now support regression models
- added `restore_weights_only` flag to `lr_find`.  When True, only the model weights will be restored after
  simulating training, not the optimizer weights. In at least a few observed cases, this "warm up" seems to improve performance
  when actual training begins. Further investigation is needed, so it is False by default.

### Changed
- N/A

### Fixed:
- added `save_path` argument to `Learner.validate` and `Learner.evaluate`.  If `print_report=False`, classification
  report will be saved as CSV to `save_path`.
- Use `torch.no_grad` with `ZeroShotClassifier.predict` to [prevent OOM](https://github.com/amaiya/ktrain/issues/215)
- Added `max_length` parameter to `ZeroShotClassifier.predict` to [prevent errors on long documnets](https://github.com/amaiya/ktrain/issues/215)
- Added type check to `TransformersPreprocessor.preprocess_train`


## 0.18.5 (2020-07-20)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Changed `qa` module to use use 'Auto' when loading `QuestionAnswering` models and tokenizer
- try `from_pt=True` for `qa` module if initial model-loading fails
- use `get_hf_model_name` in `qa` module


## 0.18.4 (2020-07-17)

### New:
- N/A

### Changed
- N/A

### Fixed:
- return gracefully if no documents match question in `qa` module
- tokenize question in `qa` module to ensure all candidate documents are returned
- Added error in `text.preprocessor` when training set has incomplete integer labels


## 0.18.3 (2020-07-12)

### New:
- added `batch_size` argument to `ZeroShotClassifier.predict` that can be increased to speed up predictions.
  This is especially useful if `len(topic_strings)` is large.

### Changed
- N/A

### Fixed:
- fixed typo in `load_predictor` error message


## 0.18.2 (2020-07-08)

### New:
- N/A

### Changed
- updated doc comments in core module
- removed unused `nosave` parameter from `reset_weights`
- added warning about obsolete `show_wd` parameter in `print_layers` method
- pin to `scipy==1.4.1` due to TensorFlow requirement

### Fixed:
- N/A


## 0.18.1 (2020-07-07)

### New:
- N/A

### Changed
- Use `tensorflow==2.1.0` if Python 3.6/3.7 and use `tensorflow==2.2.0` only if on Python 3.8 due to TensorFlow v2.2.0 issues

### Fixed:
- N/A


## 0.18.0 (2020-07-07)

### New:
- N/A

### Changed
- Fixes to address changes or issues in TensorFlow 2.2.0:
  - created `metrics_from_model` function due to changes in the way metrics are extracted from compiled model
  - use `loss_fn_from_model` function due to changes in they way loss functions are extracted from compiled model
  - addd `**kwargs` to `AdamWeightDecay based on [this issue](https://github.com/tensorflow/addons/issues/1645)
  - changed `TransformerTextClassLearner.predict` and `TextPredictor.predict` to deal with tuples being returned by `predict` in TensorFlow 2.2.0
  - changed multilabel test to use loss insead of accuracy due to [TF 2.2.0 issue](https://github.com/tensorflow/tensorflow/issues/41114)
  - changed `Learner.lr_find` to use `save_model` and `load_model` to restore weights due to [this TF issue](https://github.com/tensorflow/tensorflow/issues/41116) 
    and added `TransformersPreprocessor.load_model_and_configure_from_data` to support this

### Fixed:
- N/A




## 0.17.5 (2020-07-02)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Explicitly supply `'truncate='longest_first'` to prevent sentence pair classification from breaking in `transformers==3.0.0`
- Fixed typo in `encode_plus` invocation


## 0.17.4 (2020-07-02)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Explicitly supply `'truncate='longest_first'` to prevent sentence pair classification from breaking in `transformers==3.0.0`



## 0.17.3 (2020-06-26)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Changed `setup.py` to open README file using `encoding="utf-8"` to prevent installation problems on Windows machines with `cp1252` encoding


## 0.17.2 (2020-06-25)

### New:
- Added support for Russian in `text.EnglishTranslator`

### Changed
- N/A

### Fixed:
- N/A

## 0.17.1 (2020-06-24)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Properly set device in `text.Translator` and use cuda when available


## 0.17.0 (2020-06-24)

### New:
- support for language translation using pretraiend `MarianMT` models
- added `core.evaluate` as alias to `core.validate`
- `Learner.estimate_lr` method will return numerical estimates of learning rate using two different methods.
   Should only be called **after** running `Learner.lr_find`.

### Changed
- `text.zsl.ZeroShotClassifier` changed to use `AutoModel*` and `AutoTokenizer` in order to load any `mlni` model
- remove external modules from `ktrain.__init__.py` so that they do not appear when pressing TAB in notebook
- added `Transformer.save_tokenizer` and `Transformer.get_tokenizer` methods to facilitate training on machines
  with no internet

### Fixed:
- explicitly call `plt.show()` in `LRFinder.plot_loss` to resolved issues with plot not displaying in certain cases (PR #170)
- suppress warning about text regression when making text regression predictions
- allow `xnli` models for `zsl` module


## 0.16.3 (2020-06-10)

### New:
- added `metrics` parameter to `text.text_classifier` and `text.text_regression_model` functions
- added `metrics` parameter to `Transformer.get_classifier` and `Transformer.get_regrssion_model` methods

### Changed
- `metric` parameter in `vision.image_classifier` and `vision.image_regression_model` functions changed to `metrics`

### Fixed:
- N/A


## 0.16.2 (2020-06-07)

### New:
- N/A

### Changed
- default model for summarization changed to `facebook/bart-large-cnn` due to breaking change in v2.11
- added `device` argument to `TransformerSummarizer` constructor to control PyTorch device

### Fixed:
- require `transformers>=2.11.0` due to breaking changes in v2.11 related to `BART` models


## 0.16.1 (2020-06-05)

### New:
- N/A

### Changed
- N/A/

### Fixed:
- prevent `transformer` tokenizers from being pickled during `predictor.save`, as it causes problems for
  some community-uploaded models like `bert-base-japanese-whole-word-masking`.

## 0.16.0 (2020-06-03)

### New:
- support for Zero-Shot Topic Classification via the `text.ZeroShotClassifier`.  

### Changed
- N/A/

### Fixed:
- N/A


## 0.15.4 (2020-06-03)

### New:
- N/A

### Changed
- N/A/

### Fixed:
- Added the `procs`, `limitmb`, and `multisegment` argumetns to `index_from_list` and `index_from_folder` method in `text.SimpleQA`
  to speedup indexing when necessary.  Supplying `multisegment=True` speeds things up significantly, for example. Defaults, however, are
  the same as before. Users must explicitly change values if desiring a speedup.
- Load `xlm-roberta*` as `jplu/tf-xlm-roberta*` to bypass error from `transformers`


## 0.15.3 (2020-05-28)

### New:
- N/A

### Changed
- [**breaking change**] The `multilabel` argument in `text.Transformer` constructor was moved to `Transformer.get_classifier` and now correctly allows
  users to forcibly configure model for multilabel task regardless as to what data suggests. However, it is recommended to leave this value as `None`.
- The methods `predictor.save`, `ktrain.load_predictor`, `learner.save_model`, `learner.load_model` all now accept a path to folder where
  all files (e.g., model file, `.preproc` file) will be saved. If path does not exist, it will be created.
   This should not be a breaking change as the `load*` methods will still look for files in the old location if model or predictor was saved
  using an older version of *ktrain*.

### Fixed:
- N/A





## 0.15.2 (2020-05-15)

### New:
- N/A

### Changed
- Added `n_samples` argument to `TextPredictor.explain` to address slowness of `explain` on Google Colab
- Lock to version 0.21.3 of `scikit-learn` to ensure old-style explanations are generated from `TextPredictor.explain`

### Fixed:
- added missing `import pickle` to ensure saved topic models can be loaded


## 0.15.1 (2020-05-14)

### New:
- N/A

### Changed
- Changed `Transformer.preprocess*` methods to accept sentence pairs for sentence pair classification

### Fixed:
- N/A

## 0.15.0 (2020-05-13)

### New:
- Out-of-the-box support for image regression
- `vision.images_from_df` function to load image data from *pandas* DataFrames

### Changed
- references to `fit_generator` and `predict_generator` converted to `fit` and `predict`

### Fixed:
- Resolved issue with multilabel detection returning `False` for valid multilabel problems when data is in form of generator


## 0.14.7 (2020-05-10)

### New:
- Added `TFDataset` class for use as wrapper around arbitrary `tf.data.Dataset` objects for use in *ktrain*

### Changed
- Added `NERPreprocessor.preprocess_train_from_conll2003`
- Removed extraneous imports from `text.__init__.py` and `vision.__init__.py`
- `classes` argument in `images_from_array` changed to `class_names`

### Fixed:
- ensure NER data is properly prepared `text.ner.learner.validate`
- fixed typo with `df` reference in `images_from_fname`


## 0.14.6 (2020-05-06)

### New:
- If no validation data is supplied to `images_from_array`, training data is split to generate validation data

### Changed
- issue warning if Learner cannot save original weights
- `images_from_array` accepts labels in the form of integer class IDs

### Fixed:
- fix pandas `SettingwithCopyWarning` from `images_from_csv`
- fixed issue with `return_proba=True` including class labels for multilabel image classification
- resolved issue with class labels not being set correctly in `images_from_array`
- lock to `cchardet==2.1.5` due to [this issue](https://stackoverflow.com/questions/60784527/ktrain-importerror-dll-load-failed-the-specified-module-could-not-be-found)
- fixed `y_from_data` from NumpyArrayIterators in image classification


## 0.14.5 (2020-05-03)

### New:
- N/A

### Changed
- N/A

### Fixed:
- fixed issue with MobileNet model due to typo and added MobileNet example notebook

## 0.14.4 (2020-04-30)

### New:
- N/A

### Changed
- added `merge_tokens` and `return_proba` options to `NERPredictor.predict`

### Fixed:
- N/A


## 0.14.3 (2020-04-27)

### New:
- N/A

### Changed
- added `textutils` to `text` namespace and added note about `sent_tokenize` to sequence-tagging tutorial

### Fixed:
- cast dependent variable to `tf.float32` instead of `tf.int64` for text regression problems using `transformers` library



## 0.14.2 (2020-04-21)

### New:
- N/A

### Changed
- added `suggest` option to `core.Learner.lr_plot`

### Fixed:
- set interactive mode for matplotlib so plots show automatically from Python console and PyCharm
- run prepare for NER sequence predictor to avoid matrix mismatch


## 0.14.1 (2020-04-17)

### New:
- N/A

### Changed
- N/A

### Fixed:
- ensure `text.eda.TopicModel.visualize_documents` works with `bokeh` v2.0.x


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



