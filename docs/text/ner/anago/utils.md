Module ktrain.text.ner.anago.utils
==================================
Utility functions.

Functions
---------

    
`download(url)`
:   Download a trained weights, config and preprocessor.
    
    Args:
        url (str): target url.

    
`filter_embeddings(embeddings, vocab, dim)`
:   Loads word vectors in numpy array.
    
    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.
    
    Returns:
        numpy array: an array of word embeddings.

    
`load_data_and_labels(filename, encoding='utf-8')`
:   Loads data and label from a file.
    
    Args:
        filename (str): path to the file.
        encoding (str): file encoding format.
    
        The file format is tab-separated values.
        A blank line is required at the end of a sentence.
    
        For example:
        ```
        EU      B-ORG
        rejects O
        German  B-MISC
        call    O
        to      O
        boycott O
        British B-MISC
        lamb    O
        .       O
    
        Peter   B-PER
        Blackburn       I-PER
        ...
        ```
    
    Returns:
        tuple(numpy array, numpy array): data and labels.
    
    Example:
        >>> filename = 'conll2003/en/ner/train.txt'
        >>> data, labels = load_data_and_labels(filename)

    
`load_glove(file)`
:   Loads GloVe vectors in numpy array.
    
    Args:
        file (str): a path to a glove file.
    
    Return:
        dict: a dict of numpy arrays.

Classes
-------

`AnagoNERSequence(x, y, batch_size=1, preprocess=None)`
:   Base object for fitting to a sequence of data, such as a dataset.
    
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`.
    The method `__getitem__` should return a complete batch.
    
    Notes:
    
    `Sequence` are a safer way to do multiprocessing. This structure guarantees
    that the network will only train once
     on each sample per epoch which is not the case with generators.
    
    Examples:
    
    ```python
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    import math
    
    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.
    
    class CIFAR10Sequence(Sequence):
    
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
    
        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)
    
        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) *
            self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) *
            self.batch_size]
    
            return np.array([
                resize(imread(file_name), (200, 200))
                   for file_name in batch_x]), np.array(batch_y)
    ```

    ### Ancestors (in MRO)

    * tensorflow.python.keras.utils.data_utils.Sequence

`Vocabulary(max_size=None, lower=True, unk_token=True, specials=('<pad>',))`
:   A vocabulary that maps tokens to ints (storing a vocabulary).
    
    Attributes:
        _token_count: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocabulary.
        _token2id: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        _id2token: A list of token strings indexed by their numerical identifiers.
    
    Create a Vocabulary object.
    
    Args:
        max_size: The maximum size of the vocabulary, or None for no
            maximum. Default: None.
        lower: boolean. Whether to convert the texts to lowercase.
        unk_token: boolean. Whether to add unknown token.
        specials: The list of special tokens (e.g., padding or eos) that
            will be prepended to the vocabulary. Default: ('<pad>',)

    ### Instance variables

    `reverse_vocab`
    :   Return the vocabulary as a reversed dict object.
        
        Returns:
            dict: reversed vocabulary object.

    `vocab`
    :   Return the vocabulary.
        
        Returns:
            dict: get the dict object of the vocabulary.

    ### Methods

    `add_documents(self, docs)`
    :   Update dictionary from a collection of documents. Each document is a list
        of tokens.
        
        Args:
            docs (list): documents to add.

    `add_token(self, token)`
    :   Add token to vocabulary.
        
        Args:
            token (str): token to add.

    `build(self)`
    :   Build vocabulary.

    `doc2id(self, doc)`
    :   Get the list of token_id given doc.
        
        Args:
            doc (list): document.
        
        Returns:
            list: int id of doc.

    `id2doc(self, ids)`
    :   Get the token list.
        
        Args:
            ids (list): token ids.
        
        Returns:
            list: token list.

    `id_to_token(self, idx)`
    :   token-id to token (string).
        
        Args:
            idx (int): token id.
        
        Returns:
            str: string of given token id.

    `process_token(self, token)`
    :   Process token before following methods:
        * add_token
        * add_documents
        * doc2id
        * token_to_id
        
        Args:
            token (str): token to process.
        
        Returns:
            str: processed token string.

    `token_to_id(self, token)`
    :   Get the token_id of given token.
        
        Args:
            token (str): token from vocabulary.
        
        Returns:
            int: int id of token.