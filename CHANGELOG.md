# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour


## 0.37.5 (TBD)

### new:
- N/A

### changed
- N/A

### fixed:
- Removed pin on `scikit-learn`, as `eli5-tf` repo was updated to support `scikit-learn>=1.3` (#505)


## 0.37.4 (2023-07-22)

### new:
- N/A

### changed
- N/A

### fixed:
- Temporarily pin to `scikit-learn<1.3` to avoid `eli5` import error (#505) 
- Temporarily changed `generative_qa` imports to avoid `OPENAI_API_KEY error (#506)


## 0.37.3 (2023-07-22)

### new:
- N/A

### changed
- N/A

### fixed:
- fix `eda.py` topic visualization to work with `bokeh>=3.0.0` (#504)


## 0.37.2 (2023-06-14)

### new:
- N/A

### changed
- `text.models`, `vision.models`, and `tabular.models` now all automatically set metrics to use `binary_accuracy` for multilabel problems

### fixed:
- fix `validate` to support multilabel classification problems (#498)
- add a warning to `TransformerPreprocessor.get_classifier` to use `binary_accuracy` for multilabel problems (#498)


## 0.37.1 (2023-06-05)

### new:
- Supply arguments to `generate` in `TransformerSummarizer.summarize`

### changed
- N/A

### fixed:
- N/A


## 0.37.0 (2023-05-11)

### new:
- Support for **Generative Question-Answering** powered by OpenAI models, LangChan, and PaperQA.  Ask questions to any set of documents and get back answers with citations to where the answer was found in your documents.

### changed
- N/A

### fixed:
- N/A


## 0.36.1 (2023-05-09)

### new:
- N/A

### changed
- N/A

### fixed:
- resolved issue with using DeBERTa embedding models with NER (#492)


## 0.36.0 (2023-04-21)

### new:
- easy-to-use-wrapper for sentiment analysis

### changed
- N/A

### fixed:
- N/A


## 0.35.1 (2023-04-02)

### new:
- N/A

### changed
- N/A

### fixed:
- Ensure `do_sample=True` for `GenerativeAI`


## 0.35.0 (2023-04-01)

### new:
- Support for generative AI with few-shot and zero-shot prompting using a model that can run on your own machine.

### changed
- N/A

### fixed:
- N/A


## 0.34.0 (2023-03-30)

### new:
- Support for LexRank summarization

### changed
- N/A

### fixed:
- Bug fix in `dataset` module (#486)


## 0.33.4 (2023-03-22)

### new:
- N/A

### changed
- Added `verbose` parameter to `predict*` methods in all `Predictor` classes

### fixed:
- N/A


## 0.33.3 (2023-03-17)

### new:
- N/A

### changed
- Added `exclude_unigrams` argument to `text.kw` module and support unigram extraction when `noun_phrases` is selected

### fixed:
- explicitly set `num_beams` and `early_stopping` for `generate` in `ktrain.text.translation.core` to prevent errors in `transformers>=4.26.0`


## 0.33.2 (2023-02-06)

### new:
- N/A

### changed
- N/A

### fixed:
- fixed typo in `translation` module (#479)
- removed superfluous warning when inspecting `transformer` model signature


## 0.33.1 (2023-02-03)

### new:
- N/A

### changed
- N/A

### fixed:
- Resolved bug that causes problems when loading PyTorch models (#478)


## 0.33.0 (2023-01-14)

### new:
- Support for the latest version of `transformers`.  

### changed
- Removed pin to `transformers==4.17`

### fixed:
- Changed `numpy.float` and `numpy.int` to `numpy.float64` and `numpy.int_` respectively, in `ktrain.utils` (#474)
- Removed  `pandas` deprecation warnings from `ktrain.tabular.prepreprocessor` (#475)
- Ensure `use_token_type_ids` always exists in `TransformerPreprocessor` objects to ensure backwards compatibility
- Removed reference to `networkx.info`, as it was removed in `networkx>=3`


## 0.32.3 (2022-12-12)

### new:
- N/A

### changed
- N/A

### fixed:
- Changed NMF to accept optional parameters `nmf_alpha_W` and `nmf_alpha_H` based on changes in `scikit-learn==1.2.0`.
- Change `ktrain.utils` to check for TensorFlow before doing a version check, so that **ktrain** can be imported without TensorFlow being installed.


## 0.32.1 (2022-12-11)

### new:
- N/A

### changed
- N/A

### fixed:
- In TensorFlow 2.11, the `tf.optimizers.Optimizer` base class points the new keras optimizer that seems to have problems.  Users should use legacy optimizers in `tf.keras.optimizers.legacy` with **ktrain** (which evidently will never be deleted). This means that, in TF 2.11, supplying a string representation of an optimizer like `"adam"` to `model.compile` uses the new optimizer instead of the legacy optimizers. In these cases, **ktrain** will issue a warning and automatically recompile the model with the default `tf.keras.optimizers.legacy.Adam` optimizer. 


## 0.32.0 (2022-12-08)

### new:
- Support for TensorFlow 2.11. For now, as recommended in the [TF release notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0), **ktrain** has been changed to use the legacy optimizers in `tf.keras.optimizers.legacy`.  This means that, when compiling Keras models, you should supply `tf.keras.optimizers.legacy.Adam()` instead of the string `"adam"`. 
- Support for Python 3.10. Changed references from `CountVectorizer.get_field_names` to `CountVectorizer.get_field_names_out`.  Updated supported versions in `setup.py`.

### changed
- N/A

### fixed:
- fixed error in docs


## 0.31.10 (2022-10-01)

### new:
- N/A

### changed
- N/A

### fixed:
- Adjusted tika imports due to issue with `/tmp/tika.log` in multi-user scenario


## 0.31.9 (2022-09-24)

### new:
- N/A

### changed
- N/A

### fixed:
- Adjustment for kwe
- Fixed problem with importing `ktrain` without  TensorFlow installed


## 0.31.8 (2022-09-08)

### new:
- N/A

### changed
- N/A

### fixed:
- Fixed paragraph tokenization in `AnswerExtractor`


## 0.31.7 (2022-08-04)

### new:
- N/A

### changed
- re-arranged dep warnings for TF
- **ktrain** now pinned to `transformers==4.17.0`. Python 3.6 users can downgrade to `transformers==4.10.3` and still use **ktrain**.

### fixed:
- N/A


## 0.31.6 (2022-08-02)

### new:
- N/A

### changed
- updated dependencies to work with newer versions (but temporarily continue pinning to `transformers==4.10.1`)

### fixed:
- fixes for newer `networkx`


## 0.31.5 (2022-08-01)

### new:
- N/A

### changed
- N/A

### fixed:
- fix release



## 0.31.4 (2022-08-01)

### new:
- N/A

### changed
- `TextPredictor.explain` and `ImagePredictor.explain` now use a different fork of `eli5`: `pip install https://github.com/amaiya/eli5-tf/archive/refs/heads/master.zip`

### fixed:
- Fixed `loss_fn_from_model` function to work with `DISABLE_V2_BEHAVIOR` properly
- `TextPredictor.explain` and `ImagePredictor.explain` now work with `tensorflow>=2.9` and `scipy>=1.9` (due to new `eli5-tf` fork -- see above)


## 0.31.3 (2022-07-15)

### new:
- N/A

### changed
- added `alnum` check and period check to `KeywordExtractor`

### fixed:
- fixed bug in `text.qa.core` caused by previous refactoring of `paragraph_tokenize` and `tokenize`


## 0.31.2 (2022-05-20)

### new:
- N/A

### changed
- added `truncate_to` argument (default:5000) and `minchars` argument (default:3) argument to `KeywordExtractor.extract_keywords` method.
- added `score_by` argument to `KeywordExtractor.extract_keywords`.  Default is `freqpos`, which means keywords are now ranked by a combination of frequency and position in document.


### fixed:
- N/A

## 0.31.1 (2022-05-17)

### new:
- N/A

### changed
- Allow for returning prediction probabilities when merging tokens in sequence-tagging (PR #445)
- added basic ML pipeline test to workflow using latest TensorFlow

### fixed:
- N/A


## 0.31.0 (2022-05-07)

### new:
- The `text.ner.models.sequence_tagger` now supports word embeddings from non-BERT transformer models (e.g., `roberta-base`, `openai-gpt`). Thank to @Niekvdplas.
- Custom tokenization can now be used in sequence-tagging even when using transformer word embeddings.  See `custom_tokenizer` argument to `NERPredictor.predict`.

### changed
- [**breaking change**] In the `text.ner.models.sequence_tagger` function, the `bilstm-bert` model  is now called `bilstm-transformer` and the `bert_model` parameter has been renamed to `transformer_model`.
- [**breaking change**] The  `syntok` package is now used as the default tokenizer for `NERPredictor` (sequence-tagging prediction). To use the tokenization scheme from older versions of ktrain, you can import the `re` and  `string` packages and supply this function to the `custom_tokenizer` argument: `lambda s: re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])").sub(r" \1 ", s).split()`.
- Code base was reformatted using [black](https://github.com/psf/black)
- **ktrain** now supports TIKA for text extraction in the `text.textractor.TextExtractor` package with the `use_tika=True` argument as default.  To use the old-style text extraction based on the `textract` package, you can supply `use_tika=False` to `TextExtractor`.
- removed warning about sentence pair classification to avoid confusion

### fixed:
- N/A



## 0.30.0 (2022-03-28)

### new:
- **ktrain** now supports simple, fast, and robust keyphrase extraction with the `ktran.text.kw.KeywordExtractor` module
- **ktrain** now only issues a warning if TensorFlow is not installed, insteading of halting and preventing further use. This means that
  pre-trained PyTorch models (e.g., `text.zsl.ZeroShotClassifier`) and sklearn models (e.g., `text.eda.TopicModel`) in **ktrain** can now be used
  **without** having TensorFlow installed.
- `text.qa.SimpleQA` and `text.qa.AnswerExtractor` now both support PyTorch with optional quantization (use `framework='pt'` for PyTorch version)
- `text.qa.SimpleQA` and `text.qa.AnswerExtractor` now both support a `quantize` argument that can speed up
- `text.zsl.ZeroShotClassifier`, `text.translation.Translator`, and `text.translation.EnglishTranslator` all support a `quantize` argument.
- pretrained image-captioning and object-detection via `transformers` is now supported

### changed
- reorganized imports
- localized seqeval
- The `half` parameter to `text.translation.Translator`, and `text.translation.EnglishTranslator` was changed to `quantize` and now supports
  both CPU and GPU.

### fixed:
- N/A


## 0.29.3 (2022-03-09)

### new:
- `NERPredictor.predict` now includes a `return_offsets` parameter.  If True, the results will include character offsets of predicted entities.

### changed
- In `eda.TopicModel`, changed `lda_max_iter` to `max_iter` and `nmf_alpha` to `alpha`
- Added `show_counts` parameter to `TopicModel.get_topics` method
- Changed `qa.core._process_question` to `qa.core.process_question`
- In `qa.core`, added `remove_english_stopwords` and `and_np` parameters to `process_question`
-  The `valley` learning rate suggestion is now returned in `learner.lr_estimate` and `learner.lr_plot` (when `suggest=True` supplied to `learner.lr_plot`)

### fixed:
- save `TransformerEmbedding` model, tokenizer, and configuration when saving `NERPredictor` and reset `te_model` to
  facilitate loading NERPredictors with BERT embeddings offline (#423)
- switched from `keras2onnx` to `tf2onnx`, which supports newer versions of TensorFlow


## 0.29.2 (2022-02-09)

### new:
- N/A

### changed
- N/A

### fixed:
- added `get_tokenizer` call to `TransformersPreprocessor._load_pretrained` to address issue #416


## 0.29.1 (2022-02-08)

### new:
- N/A

### changed
- pin to `sklearn==0.24.2` due to breaking changes.  `eli5` fork for tf.keras updated for 0.24.2.
   To use `scikit-learn==0.24.2`, users must uninstall and re-install the `eli5` fork with: `pip install https://github.com/amaiya/eli5/archive/refs/heads/tfkeras_0_10_1.zip`

### fixed:
- N/A


## 0.29.0 (2022-01-28)

### new:
- New vision models: added MobileNetV3-Small and EfficientNet.  Thanks to @ilos-vigil.

### changed
- `core.Learner.plot` now supports plotting of any value that exists in the training `History` object (e.g., `mae` if previously specified as metric). Thanks to @ilos-vigil.
- added `raw_confidence` parameter to `QA.ask` method to return raw confidence scores. Thanks to @ilos-vigil.

### fixed:
- pin to `transformers==4.10.3` due to Issue #398
- pin to `syntok==1.3.3` due to bug with `syntok==1.4.1` causing paragraph tokenization in `qa` module to break
- properly suppress TF/CUDA warnings by default
- ensure document fed to `keras_bert`  tokenizer to avoid [this issue](https://stackoverflow.com/questions/67360987/bert-model-bug-encountered-during-training/67375675#67375675)


## 0.28.3 (2021-11-05)

### new:
- speech transcription support

### changed
- N/A

### fixed:
- N/A


## 0.28.2 (2021-10-17)

### new:
- N/A

### changed
- minor fix to installation due to pypi

### fixed:
- N/A


## 0.28.1 (2021-10-17)

### New:
- N/A

### Changed
- added `extra_requirements` to `setup.py`
- changed imports for summarization, translation, qa, and zsl in notebooks and tests

### Fixed:
- N/A


## 0.28.0 (2021-10-13)

### New:
- `text.AnswerExtractor` is a universal information extractor powered by a Question-Answering module and capable of extracting user-specfied information from texts.
- `text.TextExtractor` is a  text extraction pipeline (e.g., convert PDFs to plain text)

### Changed
- changed transformers pin to  `transformers>=4.0.0,<=4.10.3`

### Fixed:
- N/A


## 0.27.3 (2021-09-03)

### New:
- N/A

### Changed
-N/A

### Fixed:
- `SimpleQA` now can load PyTorch question-answering checkpoints
- change API call to support newest `causalnlp`


## 0.27.2 (2021-07-28)

### New:
- N/A

### Changed
- N/A

### Fixed:
- check for `logits` attribute when predicting using `transformers`
- change raised Exception to warning for longer sequence lengths for `transformers`


## 0.27.1 (2021-07-20)

### New:
- N/A

### Changed
- Added `method` parameter to `tabular.causal_inference_model`.

### Fixed:
- N/A


## 0.27.0 (2021-07-20)

### New:
- Added `tabular.causal_inference_model` function for causal inference support.

### Changed
- N/A

### Fixed:
- N/A


## 0.26.5 (2021-07-15)

### New:
- N/A

### Changed
- added `query` parameter to `SimpleQA.ask` so that an alternative query can be used to retrieve contexts from corpus
- added `chardet` as dependency for `stellargraph`

### Fixed:
- fixed issue with `TopicModel.build` when `threshold=None`


## 0.26.4 (2021-06-23)

### New:
- API documenation index

### Changed
- Added warning when a TensorFlow version of selected `transformers` model is not available and the PyTorch version is being downloaded and converted instead using `from_pt=True`.

### Fixed:
- Fixed `utils.metrics_from_model` to support alternative metrics
- Check for AUC `ktrain.utils` "inspect" function


## 0.26.3 (2021-05-19)

### New:
- N/A

### Changed
- `shallownlp.ner.NER.predict` processes lists of sentences in batches resulting in faster predictions
- `batch_size` argument added to `shallownlp.ner.NER.predict`
- added `verbose` parameter to `ktrain.text.textutils.extract_copy` to optionally see why each skipped document was skipped

### Fixed:
- Changed `TextPredictor.save` to save Hugging Face tokenizer files locally to ensure they can be easily reloaded when `text.Transformer` is supplied with local path.
- For `transformers` models, the `predictor.preproc.model_name` variable is automatically updated to be new `Predictor` folder to avoid having users manually update `model_name`.  Applies
  when a local path is supplied to `text.Transformer` and resultant `Predictor` is moved to new machine.


## 0.26.2 (2021-03-26)

### New:
- N/A

### Changed
- `NERPredictor.predict` now optionally accepts lists of sentences to make sequence-labeling predictions in batches (as all other `Predictor` instances already do).

### Fixed:
- N/A


## 0.26.1 (2021-03-11)

### New:
- N/A

### Changed
- expose errors from `transformers` in `_load_pretrained`
- Changed `TextPreprocessor.check_trained` to be a warning instead of Exception

### Fixed:
- N/A


## 0.26.0 (2021-03-10)

### New:
- Support for **transformers** 4.0 and above.

### Changed
- added `set_tokenizer to `TransformerPreprocessor`
- show error message when original weights cannot be saved (for `reset_weights` method)

### Fixed:
- cast filename to string before concatenating with suffix in `images_from_csv` and `images_from_df` (addresses issue #330)
-  resolved import error for `sklearn>=0.24.0`, but `eli5` still requires `sklearn<0.24.0`.


## 0.25.4 (2021-01-26)

### New:
- N/A

### Changed
- N/A

### Fixed:
- fixed problem with `LabelEncoder` not properly being stored when `texts_from_df` is invoked
- refrain from invoking  `max` on empty sequence (#307)
- corrected issue with `return_proba=True` in NER predictions (#316)


## 0.25.3 (2020-12-23)

### New:
- N/A

### Changed
- A `steps_per_epoch` argument has been added to all `*fit*` methods that operate on generators
- Added `get_tokenizer` methods to all instances of `TextPreprocessor`

### Fixed:
- propogate custom metrics to model when `distilbert` is chosen in `text_classifier` and `text_regression_model` functions
- pin `scikit-learn` to 0.24.0 sue to breaking change


## 0.25.2 (2020-12-05)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Added `custom_objects` argument to `load_predictor` to load models with custom loss functions, etc.
- Fixed bug #286 related to length computation when `use_dynamic_shape=True`


## 0.25.1 (2020-12-02)

### New:
- N/A

### Changed
- Added `use_dynamic_shape` parameter to `text.preprocessor.hf_convert_examples` which is set to `True` when running predictions.  This reduces the input length when making predictions, if possible..
- Added warnings to some imports in `imports.py` to allow for slightly lighter-weight deployments
- Temporarily pinning to `transformers>=3.1,<4.0` due to breaking changes in v4.0.

### Fixed:
- Suppress progress bar in `predictor.predict` for `keras_bert` models
- Fixed typo causing problems when loading predictor for Inception models
- Fixes to address documented/undocumented breaking changes in `transformers>=4.0`. But, temporarily pinning to `transformers>=3.1,<4.0` for backwards compatibility.



## 0.25.0 (2020-11-08)

### New:
- The `SimpleQA.index_from_folder` method now supports text extraction from many file types including PDFs, MS Word documents, and MS PowerPoint files.

### Changed
- The default in `SimpleQA.index_from_list` and `SimpleQA.index_from_folder` has been changed to `breakup_docs=True`.

### Fixed:
- N/A


## 0.24.2 (2020-11-07)

### New:
- N/A

### Changed
- `ktrain.text.textutils.extract_copy` now uses `textract` to extract text from many file types (e.g., PDF, DOC, PPT)
  instead of just PDFs,

### Fixed:
- N/A



## 0.24.1 (2020-11-06)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Change exception in model ID check in `Translator` to warning to better allow offline language translations


## 0.24.0 (2020-11-05)

### New:
- `Predictor` instances now provide built-in support for exporting to TensorFlow Lite and ONNX.

### Changed
- N/A


### Fixed:
- N/A


## 0.23.2 (2020-10-27)

### New:
- N/A

### Changed
- Use fast tokenizers for the following Hugging Face **transformers**  models: BERT, DistilBERT, and RoBERTa models. This change affects models created with either `text.Transformer(...` or `text.text_clasifier('distilbert',..')`.  BERT models created with `text_classifier('bert',..`, which uses `keras_bert` instead of `transformers`, are not affected by this change.


### Fixed:
- N/A


## 0.23.1 (2020-10-26)

### New:
- N/A

### Changed
- N/A


### Fixed:
- Resolved issue in `qa.ask` method occuring with embedding computations when full answer sentences exceed 512 tokens.


## 0.23.0 (2020-10-16)

### New:
- Support for upcoming release of TensorFlow 2.4 such as removal of references to obsolete `multi_gpu_model`

### Changed
- **[breaking change]** `TopicModel.get_docs` now returns a list of dicts instead of a list of tuples.  Each dict has keys: `text`, `doc_id`, `topic_proba`, `topic_id`.
- added `TopicModel.get_document_topic_distribution`
- added `TopicModel.get_sorted_docs` method to return all documents sorted by relevance to a given `topic_id`


### Fixed:
- Changed version check warning in `lr_find` to a raised Exception to avoid confusion when warnings from **ktrain** are suppressed
- Pass `verbose` parameter to `hf_convert_examples`


## 0.22.4 (2020-10-12)

### New:
- N/A

### Changed
- changed `qa.core.display_answers` to make URLs open in new tab


### Fixed:
- pin to `seqeval==0.0.19` due to `numpy` version incompatibility with latest TensorFlow and to suppress errors during installation


## 0.22.3 (2020-10-09)

### New:
- N/A

### Changed
- N/A


### Fixed:
- fixed issue with missing noun phrase at end of sentence in `extract_noun_phrases`
- fixed TensorFlow versioning issues with `utils.metrics_from_model`


## 0.22.2 (2020-10-09)

### New:
- added `extract_noun_phrases` to `textutils`

### Changed
- `SimpleQA.ask` now includes an `include_np` parameter.  When True, noun phrases will be used to retrieve documents
   containing candidate answers.


### Fixed:
- N/A



## 0.22.1 (2020-10-08)

### New:
- N/A

### Changed
- added optional `references` argument to `SimpleQA.index_from_list`
- added `min_words` argument to `SimpleQA.index_from_list` and `SimpleQA.index_from_folder` to prune small documents or paragraphs
  that are unlikely to include good answers
- `qa.display_answers` now supports hyperlinks for document references


### Fixed:
- N/A


## 0.22.0 (2020-10-06)

### New:
- added `breakup_docs` argument to `index_from_list` and `index_from_folder` that potentially speeds up `ask` method substantially
- added `batch_size` argument to `ask` and set default at 8 for faster answer-retrieval

### Changed
- refactored `QA` and `SimpleQA` for better extensibility


### Fixed:
- Ensure `save_path` is correctyl processed in `Learner.evaluate`



## 0.21.4 (2020-09-24)

### New:
- N/A

### Changed
- Changed installation instructions in `README.md` to reflect that using *ktrain* with TensorFlow 2.1 will require downgrading `transformers` to 3.1.0.
- updated requirements with `keras_bert>=0.86.0` due to TensorFlow 2.3 error with older versions of `keras_bert`
- In  `lr_find` and `lr_plot`, check for TF 2.2 or 2.3 and make necessary adjustments due to TF bug 41174.

### Fixed:
- fixed typos in `__all__` in `text` and graph` modules (PR #250)
- fixed Chinese language translation based on on name-changes of models with `zh` as source language


## 0.21.3 (2020-09-08)

### New:
- N/A

### Changed
- added `TopicModel.get_word_weights` method to retrieve the word weights for a given topic
- added `return_fig` option to `Learner.lr_plot` and `Learner.plot`, which allows the matplotlib `Figure` to be returned to user

### Fixed:
- N/A


## 0.21.2 (2020-09-03)

### New:
- N/A

### Changed
- `SUPPRESS_KTRAIN_WARNINGS` environment variable changed to `SUPPRESS_DEP_WARNINGS`

### Fixed:
- N/A


## 0.21.1 (2020-09-03)

### New:
- N/A

### Changed
- added `num_beams` and `early_stopping` arguments to `translate` methods in `translation` module that can be set to improve translation speed
- added `half` parameter to `Translator` construcor

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
