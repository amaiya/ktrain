Module ktrain.vision.predictor
==============================

Classes
-------

`ImagePredictor(model, preproc, batch_size=32)`
:   predicts image classes

    ### Ancestors (in MRO)

    * ktrain.predictor.Predictor
    * abc.ABC

    ### Methods

    `analyze_valid(self, generator, print_report=True, multilabel=None)`
    :   Makes predictions on validation set and returns the confusion matrix.
        Accepts as input a genrator (e.g., DirectoryIterator, DataframeIterator)
        representing the validation set.
        
        
        Optionally prints a classification report.
        Currently, this method is only supported for binary and multiclass
        problems, not multilabel classification problems.

    `explain(self, img_fpath)`
    :   Highlights image to explain prediction

    `get_classes(self)`
    :

    `predict(self, data, return_proba=False)`
    :   Predicts class from image in array format.
        If return_proba is True, returns probabilities of each class.

    `predict_filename(self, img_path, return_proba=False)`
    :   Predicts class from filepath to single image file.
        If return_proba is True, returns probabilities of each class.

    `predict_folder(self, folder, return_proba=False)`
    :   Predicts the classes of all images in a folder.
        If return_proba is True, returns probabilities of each class.

    `predict_generator(self, generator, steps=None, return_proba=False)`
    :

    `predict_proba(self, data)`
    :

    `predict_proba_filename(self, img_path)`
    :

    `predict_proba_folder(self, folder)`
    :

    `predict_proba_generator(self, generator, steps=None)`
    :