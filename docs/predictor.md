Module ktrain.predictor
=======================

Classes
-------

`Predictor()`
:   Abstract class to preprocess data

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * ktrain.graph.predictor.LinkPredictor
    * ktrain.graph.predictor.NodePredictor
    * ktrain.tabular.predictor.TabularPredictor
    * ktrain.text.ner.predictor.NERPredictor
    * ktrain.text.predictor.TextPredictor
    * ktrain.vision.predictor.ImagePredictor

    ### Methods

    `create_onnx_session(self, onnx_model_path, provider='CPUExecutionProvider')`
    :   Creates ONNX inference session from provided onnx_model_path

    `explain(self, x)`
    :

    `export_model_to_onnx(self, fpath, quantize=False, target_opset=None, verbose=1)`
    :   Export model to onnx
        Args:
          fpath(str): String representing full path to model file where ONNX model will be saved.
                      Example: '/tmp/my_model.onnx'
          quantize(str): If True, will create a total of three model files will be created using transformers.convert_graph_to_onnx: 
                         1) ONNX model  (created directly using keras2onnx
                         2) an optimized ONNX model (created by transformers library)
                         3) a quantized version of optimized ONNX model (created by transformers library)
                         All files will be created in the parent folder of fpath:
                         Example: 
                           If fpath='/tmp/model.onnx', then both /tmp/model-optimized.onnx and
                           /tmp/model-optimized-quantized.onnx will also be created.
          verbose(bool): verbosity
        Returns:
          str: string representing fpath.  If quantize=True, returned fpath will be different than supplied fpath

    `export_model_to_tflite(self, fpath, verbose=1)`
    :   Export model to TFLite
        Args:
          fpath(str): String representing full path to model file where TFLite model will be saved.
                      Example: '/tmp/my_model.tflite'
          verbose(bool): verbosity
        Returns:
          str: fpath is returned back

    `get_classes(self, filename)`
    :

    `predict(self, data)`
    :

    `save(self, fpath)`
    :   saves both model and Preprocessor instance associated with Predictor 
        Args:
          fpath(str): path to folder to store model and Preprocessor instance (.preproc file)
        Returns:
          None