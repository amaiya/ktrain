Module ktrain.preprocessor
==========================

Classes
-------

`Preprocessor()`
:   Abstract class to preprocess data

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * ktrain.graph.preprocessor.LinkPreprocessor
    * ktrain.graph.preprocessor.NodePreprocessor
    * ktrain.tabular.preprocessor.TabularPreprocessor
    * ktrain.text.ner.preprocessor.NERPreprocessor
    * ktrain.text.preprocessor.TextPreprocessor
    * ktrain.vision.preprocessor.ImagePreprocessor

    ### Methods

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `preprocess(self)`
    :

    `undo(self, data_instance)`
    :