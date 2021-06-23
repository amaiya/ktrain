Module ktrain.vision.preprocessor
=================================

Classes
-------

`ImagePreprocessor(datagen, classes, target_size=(224, 224), color_mode='rgb')`
:   Image preprocessing

    ### Ancestors (in MRO)

    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Methods

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `preprocess(self, data, batch_size=32)`
    :   Receives raw data and returns 
        tuple containing the generator and steps
        argument for model.predict.

    `preprocess_test(self, data, batch_size=32)`
    :   Alias for preprocess