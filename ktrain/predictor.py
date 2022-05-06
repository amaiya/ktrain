from . import utils as U
from .imports import *


class Predictor(ABC):
    """
    ```
    Abstract class to preprocess data
    ```
    """

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def get_classes(self, filename):
        pass

    def explain(self, x):
        raise NotImplementedError("explain is not currently supported for this model")

    def _make_predictor_folder(self, fpath):
        if os.path.isfile(fpath):
            raise ValueError(
                f"There is an existing file named {fpath}. "
                + "Please use dfferent value for fpath."
            )
        elif os.path.exists(fpath):
            # warnings.warn('predictor files are being saved to folder that already exists: %s' % (fpath))
            pass
        elif not os.path.exists(fpath):
            os.makedirs(fpath)
        return

    def _save_preproc(self, fpath):
        with open(os.path.join(fpath, U.PREPROC_NAME), "wb") as f:
            pickle.dump(self.preproc, f)
        return

    def _save_model(self, fpath):
        if U.is_crf(self.model):  # TODO: fix/refactor this
            from .text.ner.anago.layers import crf_loss

            self.model.compile(loss=crf_loss, optimizer=U.DEFAULT_OPT)
        model_path = os.path.join(fpath, U.MODEL_NAME)
        self.model.save(model_path, save_format="h5")
        return

    def save(self, fpath):
        """
        ```
        saves both model and Preprocessor instance associated with Predictor
        Args:
          fpath(str): path to folder to store model and Preprocessor instance (.preproc file)
        Returns:
          None
        ```
        """
        self._make_predictor_folder(fpath)
        self._save_model(fpath)
        self._save_preproc(fpath)
        return

    def export_model_to_tflite(self, fpath, verbose=1):
        """
        ```
        Export model to TFLite
        Args:
          fpath(str): String representing full path to model file where TFLite model will be saved.
                      Example: '/tmp/my_model.tflite'
          verbose(bool): verbosity
        Returns:
          str: fpath is returned back
        ```
        """
        if verbose:
            print("converting to TFLite format ... this may take a few moments...")
        if U.is_huggingface(model=self.model):
            tokenizer = self.preproc.get_tokenizer()
            maxlen = self.preproc.maxlen
            input_dict = tokenizer(
                "Name", return_tensors="tf", padding="max_length", max_length=maxlen
            )

            if version.parse(tf.__version__) < version.parse("2.2"):
                raise Exception("export_model_to_tflite requires tensorflow>=2.2")
                # self.model._set_inputs(input_spec, training=False) # for tf < 2.2
            self.model._saved_model_inputs_spec = None  # for tf > 2.2
            self.model._set_save_spec(input_dict)  # for tf > 2.2
            self.model._get_save_spec()
            if verbose:
                print("----------------------------------------------")
                print(
                    'NOTE: For Hugging Face models, please ensure you use return_tensors="tf" and padding="max_length" when encoding your inputs.'
                )
                print("----------------------------------------------")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # normal conversion
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        open(fpath, "wb").write(tflite_model)
        if verbose:
            print("done.")
        return fpath

    def export_model_to_onnx(self, fpath, quantize=False, target_opset=None, verbose=1):
        """
        ```
        Export model to onnx
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
        ```
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise Exception(
                "This method requires ONNX libraries to be installed: "
                + "pip install -q --upgrade onnxruntime==1.10.0 onnx sympy tf2onnx"
            )
        from pathlib import Path

        if type(self.preproc).__name__ == "BERTPreprocessor":
            raise Exception(
                'currently_unsupported:  BERT models created with text_classifier("bert",...) are not supported (i.e., keras_bert models). '
                + "Only BERT models created with Transformer(...) are supported."
            )

        if verbose:
            print(
                "converting to ONNX format by way of TFLite ... this may take a few moments..."
            )
        if U.is_huggingface(model=self.model):
            tokenizer = self.preproc.get_tokenizer()
            maxlen = self.preproc.maxlen
            input_dict = tokenizer(
                "Name", return_tensors="tf", padding="max_length", max_length=maxlen
            )

            if version.parse(tf.__version__) < version.parse("2.2"):
                raise Exception("export_model_to_tflite requires tensorflow>=2.2")
                # self.model._set_inputs(input_spec, training=False) # for tf < 2.2
            self.model._saved_model_inputs_spec = None  # for tf > 2.2
            self.model._set_save_spec(input_dict)  # for tf > 2.2
            self.model._get_save_spec()

        # onnx_model = keras2onnx.convert_keras(self.model, self.model.name, target_opset=target_opset)
        # keras2onnx.save_model(onnx_model, fpath)
        tflite_model_path = self.export_model_to_tflite(
            fpath + "-TFLITE_TMP", verbose=verbose
        )

        import subprocess

        if verbose:
            print("converting to ONNX using tf2onnx...")
        proc = subprocess.run(
            f"python -m tf2onnx.convert --tflite {tflite_model_path} --output {fpath}".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if verbose:
            print(proc.returncode)
            print(proc.stdout.decode("ascii"))
            print(proc.stderr.decode("ascii"))
        return_fpath = fpath

        if quantize:
            from transformers.convert_graph_to_onnx import optimize, quantize

            # opt_path = optimize(Path(fpath))

            if U.is_huggingface(model=self.model) and type(self.model).__name__ in [
                "TFDistilBertForSequenceClassification",
                "TFBertForSequenceClassification",
            ]:
                try:
                    from onnxruntime.transformers import optimizer
                    from onnxruntime.transformers.onnx_model_bert import (
                        BertOptimizationOptions,
                    )

                    # disable embedding layer norm optimization for better model size reduction
                    opt_options = BertOptimizationOptions("bert")
                    opt_options.enable_embed_layer_norm = False
                    opt_model = optimizer.optimize_model(
                        fpath,
                        "bert",  # bert_keras causes error with transformers
                        num_heads=12,
                        hidden_size=768,
                        optimization_options=opt_options,
                    )
                    opt_model.save_model_to_file(fpath)
                except:
                    warnings.warn("Could not run BERT-specific optimizations")
                    pass
            quantize_path = quantize(Path(fpath))
            return_fpath = quantize_path.as_posix()
        if verbose:
            print("done.")
        return return_fpath

    def create_onnx_session(self, onnx_model_path, provider="CPUExecutionProvider"):
        """
        ```
        Creates ONNX inference session from provided onnx_model_path
        ```
        """

        from onnxruntime import (
            GraphOptimizationLevel,
            InferenceSession,
            SessionOptions,
            get_all_providers,
        )

        assert (
            provider in get_all_providers()
        ), f"provider {provider} not found, {get_all_providers()}"

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 0
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        session = InferenceSession(onnx_model_path, options, providers=[provider])
        session.disable_fallback()

        # if 'OMP_NUM_THREADS' not in os.environ or 'OMP_WAIT_POLICY' not in os.environ:
        # warnings.warn('''We recommend adding the following at top of script for CPU inference:

        # from psutil import cpu_count
        ##Constants from the performance optimization available in onnxruntime
        ##It needs to be done before importing onnxruntime
        # os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
        # os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'
        #''')
        return session
