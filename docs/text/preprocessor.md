Module ktrain.text.preprocessor
===============================

Functions
---------

    
`bert_tokenize(docs, tokenizer, max_length, verbose=1)`
:   

    
`detect_text_format(texts)`
:   

    
`file_len(fname)`
:   

    
`fname_from_url(url)`
:   

    
`get_bert_path(lang='en')`
:   

    
`get_coefs(word, *arr)`
:   

    
`get_wv_path(wv_path_or_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')`
:   

    
`hf_convert_example(text_a, text_b=None, tokenizer=None, max_length=512, pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True)`
:   convert InputExample to InputFeature for Hugging Face transformer

    
`hf_convert_examples(texts, y=None, tokenizer=None, max_length=512, pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True, use_dynamic_shape=False, verbose=1)`
:   Loads a data file into a list of ``InputFeatures``
    Args:
        texts: texts of documents or sentence pairs
        y:  labels for documents
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
        use_dynamic_shape(bool):  If True, supplied max_length will be ignored and will be computed
                                  based on provided texts instead.
        verbose(bool): verbosity
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    
`hf_features_to_tfdataset(features_list, labels)`
:   

    
`is_nospace_lang(lang)`
:   

    
`load_wv(wv_path_or_url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip', verbose=1)`
:   

Classes
-------

`BERTPreprocessor(maxlen, max_features, class_names=[], classes=[], lang='en', ngram_range=1, multilabel=None)`
:   text preprocessing for BERT model

    ### Ancestors (in MRO)

    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `get_preprocessor(self)`
    :

    `get_tokenizer(self)`
    :

    `preprocess(self, texts)`
    :

    `preprocess_test(self, texts, y=None, mode='test', verbose=1)`
    :

    `preprocess_train(self, texts, y=None, mode='train', verbose=1)`
    :   preprocess training set

`DistilBertPreprocessor(maxlen, max_features, class_names=[], classes=[], lang='en', ngram_range=1)`
:   text preprocessing for Hugging Face DistlBert model

    ### Ancestors (in MRO)

    * ktrain.text.preprocessor.TransformersPreprocessor
    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.preprocessor.Preprocessor
    * abc.ABC

`StandardTextPreprocessor(maxlen, max_features, class_names=[], classes=[], lang='en', ngram_range=1, multilabel=None)`
:   Standard text preprocessing

    ### Ancestors (in MRO)

    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `get_preprocessor(self)`
    :

    `get_tokenizer(self)`
    :

    `ngram_count(self)`
    :

    `preprocess(self, texts)`
    :

    `preprocess_test(self, test_text, y_test=None, verbose=1)`
    :   preprocess validation or test dataset

    `preprocess_train(self, train_text, y_train, verbose=1)`
    :   preprocess training set

`TextPreprocessor(maxlen, class_names, lang='en', multilabel=None)`
:   Text preprocessing base class

    ### Ancestors (in MRO)

    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Descendants

    * ktrain.text.preprocessor.BERTPreprocessor
    * ktrain.text.preprocessor.StandardTextPreprocessor
    * ktrain.text.preprocessor.TransformersPreprocessor

    ### Static methods

    `seqlen_stats(list_of_texts)`
    :   compute sequence length stats from
        list of texts in any spaces-segmented language
        Args:
            list_of_texts: list of strings
        Returns:
            dict: dictionary with keys: mean, 95percentile, 99percentile

    ### Methods

    `check_trained(self)`
    :

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `get_tokenizer(self)`
    :

    `is_chinese(self)`
    :

    `is_nospace_lang(self)`
    :

    `migrate_classes(self, class_names, classes)`
    :

    `preprocess(self, texts)`
    :

    `print_seqlen_stats(self, texts, mode, verbose=1)`
    :   prints stats about sequence lengths

    `process_chinese(self, texts, lang=None)`
    :

    `set_classes(self, class_names)`
    :

    `set_multilabel(self, data, mode, verbose=1)`
    :

    `undo(self, doc)`
    :   undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words

`Transformer(model_name, maxlen=128, class_names=[], classes=[], batch_size=None, use_with_learner=True)`
:   convenience class for text classification Hugging Face transformers 
    Usage:
       t = Transformer('distilbert-base-uncased', maxlen=128, classes=['neg', 'pos'], batch_size=16)
       train_dataset = t.preprocess_train(train_texts, train_labels)
       model = t.get_classifier()
       model.fit(train_dataset)
    
    Args:
        model_name (str):  name of Hugging Face pretrained model
        maxlen (int):  sequence length
        class_names(list):  list of strings of class names (e.g., 'positive', 'negative').
                            The index position of string is the class ID.
                            Not required for:
                              - regression problems
                              - binary/multi classification problems where
                                labels in y_train/y_test are in string format.
                                In this case, classes will be populated automatically.
                                get_classes() can be called to view discovered class labels.
                            The class_names argument replaces the old classes argument.
        classes(list):  alias for class_names.  Included for backwards-compatiblity.
    
        use_with_learner(bool):  If False, preprocess_train and preprocess_test
                                 will return tf.Datasets for direct use with model.fit
                                 in tf.Keras.
                                 If True, preprocess_train and preprocess_test will
                                 return a ktrain TransformerDataset object for use with
                                 ktrain.get_learner.
        batch_size (int): batch_size - only required if use_with_learner=False

    ### Ancestors (in MRO)

    * ktrain.text.preprocessor.TransformersPreprocessor
    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `preprocess_test(self, texts, y=None, verbose=1)`
    :   Preprocess the validation or test set for a Transformer model
        Y values can be in one of the following forms:
        1) integers representing the class (index into array returned by get_classes)
           for binary and multiclass text classification.
           If labels are integers, class_names argument to Transformer constructor is required.
        2) strings representing the class (e.g., 'negative', 'positive').
           If labels are strings, class_names argument to Transformer constructor is ignored,
           as class labels will be extracted from y.
        3) multi-hot-encoded vector for multilabel text classification problems
           If labels are multi-hot-encoded, class_names argument to Transformer constructor is requird.
        4) Numerical values for regression problems.
           <class_names> argument to Transformer constructor should NOT be supplied
        
        Args:
            texts (list of strings): text of documents
            y: labels
            verbose(bool): verbosity
        Returns:
            TransformerDataset if self.use_with_learner = True else tf.Dataset

    `preprocess_train(self, texts, y=None, mode='train', verbose=1)`
    :   Preprocess training set for A Transformer model
        
        Y values can be in one of the following forms:
        1) integers representing the class (index into array returned by get_classes)
           for binary and multiclass text classification.
           If labels are integers, class_names argument to Transformer constructor is required.
        2) strings representing the class (e.g., 'negative', 'positive').
           If labels are strings, class_names argument to Transformer constructor is ignored,
           as class labels will be extracted from y.
        3) multi-hot-encoded vector for multilabel text classification problems
           If labels are multi-hot-encoded, class_names argument to Transformer constructor is requird.
        4) Numerical values for regression problems.
           <class_names> argument to Transformer constructor should NOT be supplied
        
        Args:
            texts (list of strings): text of documents
            y: labels
            mode (str):  If 'train' and prepare_for_learner=False,
                         a tf.Dataset will be returned with repeat enabled
                         for training with fit_generator
            verbose(bool): verbosity
        Returns:
          TransformerDataset if self.use_with_learner = True else tf.Dataset

`TransformerDataset(x, y, batch_size=1)`
:   Wrapper for Transformer datasets.

    ### Ancestors (in MRO)

    * ktrain.data.SequenceDataset
    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Methods

    `get_y(self)`
    :

    `nsamples(self)`
    :

    `to_tfdataset(self, train=True)`
    :   convert transformer features to tf.Dataset

`TransformerEmbedding(model_name, layers=[-2])`
:   Args:
        model_name (str):  name of Hugging Face pretrained model.
                           Choose from here: https://huggingface.co/transformers/pretrained_models.html
        layers(list): list of indexes indicating which hidden layers to use when
                      constructing the embedding (e.g., last=[-1])

    ### Methods

    `embed(self, texts, word_level=True, max_length=512)`
    :   get embedding for word, phrase, or sentence
        Args:
          text(str|list): word, phrase, or sentence or list of them representing a batch
          word_level(bool): If True, returns embedding for each token in supplied texts.
                            If False, returns embedding for each text in texts
          max_length(int): max length of tokens
        Returns:
            np.ndarray : embeddings

`TransformersPreprocessor(model_name, maxlen, max_features, class_names=[], classes=[], lang='en', ngram_range=1, multilabel=None)`
:   text preprocessing for Hugging Face Transformer models

    ### Ancestors (in MRO)

    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Descendants

    * ktrain.text.preprocessor.DistilBertPreprocessor
    * ktrain.text.preprocessor.Transformer

    ### Static methods

    `load_model_and_configure_from_data(fpath, transformer_ds)`
    :   loads model from file path and configures loss function and metrics automatically
        based on inspecting data
        Args:
          fpath(str): path to model folder
          transformer_ds(TransformerDataset): an instance of TransformerDataset

    ### Methods

    `get_classifier(self, fpath=None, multilabel=None, metrics=['accuracy'])`
    :   creates a model for text classification
        Args:
          fpath(str): optional path to saved pretrained model. Typically left as None.
          multilabel(bool): If None, multilabel status is discovered from data [recommended].
                            If True, model will be forcibly configured for multilabel task.
                            If False, model will be forcibly configured for non-multilabel task.
                            It is recommended to leave this as None.
          metrics(list): metrics to use

    `get_config(self)`
    :

    `get_model(self, fpath=None)`
    :

    `get_preprocessor(self)`
    :

    `get_regression_model(self, fpath=None, metrics=['mae'])`
    :   creates a model for text regression
        Args:
          fpath(str): optional path to saved pretrained model. Typically left as None.
          metrics(list): metrics to use

    `get_tokenizer(self, fpath=None)`
    :

    `preprocess(self, texts)`
    :

    `preprocess_test(self, texts, y=None, mode='test', verbose=1)`
    :

    `preprocess_train(self, texts, y=None, mode='train', verbose=1)`
    :   preprocess training set

    `save_tokenizer(self, fpath)`
    :

    `set_config(self, config)`
    :

    `set_tokenizer(self, tokenizer)`
    :