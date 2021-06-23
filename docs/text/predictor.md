Module ktrain.text.predictor
============================

Classes
-------

`TextPredictor(model, preproc, batch_size=32)`
:   predicts text classes

    ### Ancestors (in MRO)

    * ktrain.predictor.Predictor
    * abc.ABC

    ### Methods

    `analyze_valid(self, val_tup, print_report=True, multilabel=None)`
    :   Makes predictions on validation set and returns the confusion matrix.
        Accepts as input the validation set in the standard form of a tuple of
        two arrays: (X_test, y_test), wehre X_test is a Numpy array of strings
        where each string is a document or text snippet in the validation set.
        
        Optionally prints a classification report.
        Currently, this method is only supported for binary and multiclass 
        problems, not multilabel classification problems.

    `explain(self, doc, truncate_len=512, all_targets=False, n_samples=2500)`
    :   Highlights text to explain prediction
        Args:
            doc (str): text of documnet
            truncate_len(int): truncate document to this many words
            all_targets(bool):  If True, show visualization for
                                each target.
            n_samples(int): number of samples to generate and train on.
                            Larger values give better results, but will take more time.
                            Lower this value if explain is taking too long.

    `get_classes(self)`
    :

    `predict(self, texts, return_proba=False)`
    :   Makes predictions for a list of strings where each string is a document
        or text snippet.
        If return_proba is True, returns probabilities of each class.
        Args:
          texts(str|list): For text classification, texts should be either a str or
                           a list of str.
                           For sentence pair classification, texts should be either
                           a tuple of form (str, str) or list of tuples.
                           A single tuple of the form (str, str) is automatically treated as sentence pair classification, so
                           please refrain from using tuples for text classification tasks.
          return_proba(bool): If True, return probabilities instead of predicted class labels

    `predict_proba(self, texts)`
    :   Makes predictions for a list of strings where each string is a document
        or text snippet.
        Returns probabilities of each class.