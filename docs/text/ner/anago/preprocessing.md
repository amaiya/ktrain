Module ktrain.text.ner.anago.preprocessing
==========================================
Preprocessors.

Functions
---------

    
`normalize_number(text)`
:   

    
`pad_nested_sequences(sequences, dtype='int32')`
:   Pads nested sequences to the same length.
    
    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.
    
    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.
    
    # Returns
        x: Numpy array.

Classes
-------

`IndexTransformer(lower=True, num_norm=True, use_char=True, initial_vocab=None, use_elmo=False)`
:   Convert a collection of raw documents to a document id matrix.
    
    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    
    Create a preprocessor object.
    
    Args:
        lower: boolean. Whether to convert the texts to lowercase.
        use_char: boolean. Whether to use char feature.
        num_norm: boolean. Whether to normalize text.
        initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
        use_elmo: If True, will generate contextual English Elmo embeddings

    ### Ancestors (in MRO)

    * sklearn.base.BaseEstimator
    * sklearn.base.TransformerMixin

    ### Static methods

    `load(file_path)`
    :

    ### Instance variables

    `char_vocab_size`
    :

    `label_size`
    :

    `word_vocab_size`
    :

    ### Methods

    `activate_elmo(self)`
    :

    `activate_transformer(self, model_name, layers=[-2], force=False)`
    :

    `elmo_is_activated(self)`
    :

    `fit(self, X, y)`
    :   Learn vocabulary from training set.
        
        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.
        
        Returns:
            self : IndexTransformer.

    `fit_transform(self, X, y=None, **params)`
    :   Learn vocabulary and return document id matrix.
        
        This is equivalent to fit followed by transform.
        
        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
        
        Returns:
            list : document id matrix.
            list: label id matrix.

    `fix_tokenization(self, X, Y, maxlen=512, num_special=2)`
    :   Should be called prior training

    `get_transformer_dim(self)`
    :

    `inverse_transform(self, y, lengths=None)`
    :   Return label strings.
        
        Args:
            y: label id matrix.
            lengths: sentences length.
        
        Returns:
            list: list of list of strings.

    `save(self, file_path)`
    :

    `transform(self, X, y=None)`
    :   Transform documents to document ids.
        
        Uses the vocabulary learned by fit.
        
        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.
        
        Returns:
            features: document id matrix.
            y: label id matrix.

    `transformer_is_activated(self)`
    :