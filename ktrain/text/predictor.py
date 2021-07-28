from ..imports import *
from ..predictor import Predictor
from .preprocessor import TextPreprocessor, TransformersPreprocessor, detect_text_format
from .. import utils as U

class TextPredictor(Predictor):
    """
    ```
    predicts text classes
    ```
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):

        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        if not isinstance(preproc, TextPreprocessor):
        #if type(preproc).__name__ != 'TextPreprocessor':
            raise ValueError('preproc must be a TextPreprocessor object')
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size


    def get_classes(self):
        return self.c


    def predict(self, texts, return_proba=False):
        """
        ```

        Makes predictions for a list of strings where each string is a document
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
        ```
        """

        is_array, is_pair = detect_text_format(texts)
        if not is_array: texts = [texts]

        classification, multilabel = U.is_classifier(self.model)

        # get predictions
        if U.is_huggingface(model=self.model):
            tseq = self.preproc.preprocess_test(texts, verbose=0)
            tseq.batch_size = self.batch_size
            tfd = tseq.to_tfdataset(train=False)
            preds = self.model.predict(tfd)
            if hasattr(preds, 'logits'): # dep_fix: breaking change - also needed for LongFormer
            #if type(preds).__name__ == 'TFSequenceClassifierOutput': # dep_fix: undocumented breaking change in transformers==4.0.0
                # REFERENCE: https://discuss.huggingface.co/t/new-model-output-types/195
                preds = preds.logits
            
            # dep_fix: transformers in TF 2.2.0 returns a tuple insead of NumPy array for some reason
            if isinstance(preds, tuple) and len(preds) == 1: preds = preds[0] 
        else:
            texts = self.preproc.preprocess(texts)
            preds = self.model.predict(texts, batch_size=self.batch_size)

        # process predictions
        if U.is_huggingface(model=self.model):
            # convert logits to probabilities for Hugging Face models
            if multilabel and self.c:
                preds = activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
            elif self.c:
                preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
            else:
                preds = np.squeeze(preds)
                if len(preds.shape) == 0: preds = np.expand_dims(preds, -1)
        result =  preds if return_proba or multilabel or not self.c else [self.c[np.argmax(pred)] for pred in preds] 
        if multilabel and not return_proba:
            result =  [list(zip(self.c, r)) for r in result]
        if not is_array: return result[0]
        else:      return result



    def predict_proba(self, texts):
        """
        ```
        Makes predictions for a list of strings where each string is a document
        or text snippet.
        Returns probabilities of each class.
        ```
        """
        return self.predict(texts, return_proba=True)


    def explain(self, doc, truncate_len=512, all_targets=False, n_samples=2500):
        """
        Highlights text to explain prediction
        Args:
            doc (str): text of documnet
            truncate_len(int): truncate document to this many words
            all_targets(bool):  If True, show visualization for
                                each target.
            n_samples(int): number of samples to generate and train on.
                            Larger values give better results, but will take more time.
                            Lower this value if explain is taking too long.
        """
        is_array, is_pair = detect_text_format(doc)
        if is_pair: 
            warnings.warn('currently_unsupported: explain does not currently support sentence pair classification')
            return
        if not self.c:
            warnings.warn('currently_unsupported: explain does not support text regression')
            return
        try:
            import eli5
            from eli5.lime import TextExplainer
        except:
            msg = 'ktrain requires a forked version of eli5 to support tf.keras. '+\
                  'Install with: pip install https://github.com/amaiya/eli5/archive/refs/heads/tfkeras_0_10_1.zip'
            warnings.warn(msg)
            return
        if not hasattr(eli5, 'KTRAIN_ELI5_TAG') or eli5.KTRAIN_ELI5_TAG != KTRAIN_ELI5_TAG:
            msg = 'ktrain requires a forked version of eli5 to support tf.keras. It is either missing or not up-to-date. '+\
                  'Uninstall the current version and install/re-install the fork with: pip install https://github.com/amaiya/eli5/archive/refs/heads/tfkeras_0_10_1.zip'
            warnings.warn(msg)
            return

        if not isinstance(doc, str): raise TypeError('text must of type str')
        prediction = [self.predict(doc)] if not all_targets else None

        if self.preproc.is_nospace_lang():
            doc = self.preproc.process_chinese([doc])
            doc = doc[0]
        doc = ' '.join(doc.split()[:truncate_len])
        te = TextExplainer(random_state=42, n_samples=n_samples)
        _ = te.fit(doc, self.predict_proba)
        return te.show_prediction(target_names=self.preproc.get_classes(), targets=prediction)


    def _save_model(self, fpath):
        if isinstance(self.preproc, TransformersPreprocessor):
            self.model.save_pretrained(fpath)
            # As of 0.26.3, make sure we save tokenizer in predictor folder
            tok = self.preproc.get_tokenizer()
            tok.save_pretrained(fpath)
        else:
            super()._save_model(fpath)
        return


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
            import onnxruntime, onnxruntime_tools, onnx, keras2onnx
        except ImportError:
            raise Exception('This method requires ONNX libraries to be installed: '+\
                            'pip install -q --upgrade onnxruntime==1.5.1 onnxruntime-tools onnx keras2onnx')
        from pathlib import Path
        if type(self.preproc).__name__ == 'BERTPreprocessor':
            raise Exception('currently_unsupported:  BERT models created with text_classifier("bert",...) are not supported (i.e., keras_bert models). ' +\
                            'Only BERT models created with Transformer(...) are supported.')

        if verbose: print('converting to ONNX format ... this may take a few moments...')
        if U.is_huggingface(model=self.model):
            tokenizer = self.preproc.get_tokenizer()
            maxlen = self.preproc.maxlen
            input_dict = tokenizer('Name', return_tensors='tf',
                                   padding='max_length', max_length=maxlen)

            if version.parse(tf.__version__) < version.parse('2.2'):
                raise Exception('export_model_to_tflite requires tensorflow>=2.2')
                #self.model._set_inputs(input_spec, training=False) # for tf < 2.2
            self.model._saved_model_inputs_spec = None # for tf > 2.2
            self.model._set_save_spec(input_dict) # for tf > 2.2
            self.model._get_save_spec()

        onnx_model = keras2onnx.convert_keras(self.model, self.model.name, target_opset=target_opset)
        keras2onnx.save_model(onnx_model, fpath)
        return_fpath = fpath

        if quantize:
            from transformers.convert_graph_to_onnx import optimize, quantize
            #opt_path = optimize(Path(fpath))

            if U.is_huggingface(model=self.model) and\
               type(self.model).__name__ in ['TFDistilBertForSequenceClassification', 'TFBertForSequenceClassification']:
                try:
                    from onnxruntime_tools import optimizer
                    from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
                    # disable embedding layer norm optimization for better model size reduction
                    opt_options = BertOptimizationOptions('bert')
                    opt_options.enable_embed_layer_norm = False
                    opt_model = optimizer.optimize_model(
                        fpath,
                       'bert',  # bert_keras causes error with transformers
                        num_heads=12,
                        hidden_size=768,
                        optimization_options=opt_options)
                    opt_model.save_model_to_file(fpath)
                except:
                    warnings.warn('Could not run BERT-specific optimizations')
                    pass
            quantize_path = quantize(Path(fpath))
            return_fpath = quantize_path.as_posix()
        if verbose: print('done.')
        return return_fpath
