Module ktrain.text.ner.preprocessor
===================================

Functions
---------

    
`array_to_df(x_list, y_list)`
:   

    
`conll2003_to_df(filepath, encoding='latin1')`
:   

    
`gmb_to_df(filepath, encoding='latin1')`
:   

    
`process_df(df, sentence_column='SentenceID', word_column='Word', tag_column='Tag', verbose=1)`
:   Extract words, tags, and sentences from dataframe

Classes
-------

`NERPreprocessor(p)`
:   NER preprocessing base class

    ### Ancestors (in MRO)

    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `filter_embeddings(self, embeddings, vocab, dim)`
    :   Loads word vectors in numpy array.
        
        Args:
            embeddings (dict or TransformerEmbedding): a dictionary of numpy array or Transformer Embedding instance
            vocab (dict): word_index lookup table.
        
        Returns:
            numpy array: an array of word embeddings.

    `fit(self, X, y)`
    :   Learn vocabulary from training set

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `get_wv_model(self, wv_path_or_url, verbose=1)`
    :

    `preprocess(self, sentences, lang=None, custom_tokenizer=None)`
    :

    `preprocess_test(self, x_test, y_test, verbose=1)`
    :   Args:
          x_test(list of lists of str): lists of token lists
          x_test (list of lists of str):  lists of tag lists
          verbose(bool): verbosity
        Returns:
          NERSequence:  can be used as argument to NERLearner.validate() to evaluate test sets

    `preprocess_test_from_conll2003(self, filepath, verbose=1)`
    :

    `transform(self, X, y=None)`
    :   Transform documents to sequences of word IDs

    `undo(self, nerseq)`
    :   undoes preprocessing and returns raw data by:
        converting a list or array of Word IDs back to words

`NERSequence(x, y, batch_size=1, p=None)`
:   Base class for custom datasets in ktrain.
    
    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.
    
    The signature of to_tfdataset is as follows:
    
    def to_tfdataset(self, training=True)
    
    See ktrain.text.preprocess.TransformerDataset as an example.

    ### Ancestors (in MRO)

    * ktrain.data.SequenceDataset
    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Methods

    `get_lengths(self, idx)`
    :

    `get_y(self)`
    :

    `nsamples(self)`
    :

    `prepare(self)`
    :

`SentenceGetter(data, word_column, tag_column, sentence_column)`
:   Class to Get the sentence in this format:
    [(Token_1, Part_of_Speech_1, Tag_1), ..., (Token_n, Part_of_Speech_1, Tag_1)]
    
    Args:
    data is the pandas.DataFrame which contains the above dataset

    ### Methods

    `get_next(self)`
    :   Return one sentence