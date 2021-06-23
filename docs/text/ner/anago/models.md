Module ktrain.text.ner.anago.models
===================================
Model definition.

Functions
---------

    
`load_model(weights_file, params_file)`
:   

    
`save_model(model, weights_file, params_file)`
:   

Classes
-------

`BiLSTMCRF(num_labels, word_vocab_size, char_vocab_size=None, word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, embeddings=None, use_char=True, use_crf=True, char_mask_zero=True, use_elmo=False, use_transformer_with_dim=None)`
:   A Keras implementation of BiLSTM-CRF for sequence labeling.
    
    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    
    Build a Bi-LSTM CRF model.
    
    Args:
        word_vocab_size (int): word vocabulary size.
        char_vocab_size (int): character vocabulary size.
        num_labels (int): number of entity labels.
        word_embedding_dim (int): word embedding dimensions.
        char_embedding_dim (int): character embedding dimensions.
        word_lstm_size (int): character LSTM feature extractor output dimensions.
        char_lstm_size (int): word tagger LSTM output dimensions.
        fc_dim (int): output fully-connected layer size.
        dropout (float): dropout rate.
        embeddings (numpy array): word embedding matrix.
        use_char (boolean): add char feature.
        use_crf (boolean): use crf as last layer.
        char_mask_zero(boolean): mask zero for character embedding (see TF2 isse #33148 and #33069)
        use_elmo(boolean): If True, model will be configured to accept Elmo embeddings
                           as an additional input to word and character embeddings
        use_transformer_with_dim(int): If not None, model will be configured to accept
                                       transformer embeddings of given dimension

    ### Methods

    `build(self)`
    :