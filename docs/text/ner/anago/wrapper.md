Module ktrain.text.ner.anago.wrapper
====================================
Wrapper class.

Classes
-------

`Sequence(word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, embeddings=None, use_char=True, use_crf=True, initial_vocab=None, optimizer='adam')`
:   

    ### Static methods

    `load(weights_file, params_file, preprocessor_file)`
    :

    ### Methods

    `analyze(self, text, tokenizer=<method 'split' of 'str' objects>)`
    :   Analyze text and return pretty format.
        
        Args:
            text: string, the input text.
            tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.
        
        Returns:
            res: dict.

    `fit(self, x_train, y_train, x_valid=None, y_valid=None, epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True)`
    :   Fit the model for a fixed number of epochs.
        
        Args:
            x_train: list of training data.
            y_train: list of training target (label) data.
            x_valid: list of validation data.
            y_valid: list of validation target (label) data.
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch). `shuffle` will default to True.

    `save(self, weights_file, params_file, preprocessor_file)`
    :

    `score(self, x_test, y_test)`
    :   Returns the f1-micro score on the given test data and labels.
        
        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.
        
            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.
        
        Returns:
            score : float, f1-micro score.