from .. import utils as U
from ..imports import *
from ..predictor import Predictor
from .preprocessor import TextPreprocessor, TransformersPreprocessor, detect_text_format


class TextPredictor(Predictor):
    """
    ```
    predicts text classes
    ```
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):
        if not isinstance(model, keras.Model):
            raise ValueError("model must be of instance keras.Model")
        if not isinstance(preproc, TextPreprocessor):
            # if type(preproc).__name__ != 'TextPreprocessor':
            raise ValueError("preproc must be a TextPreprocessor object")
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size

    def get_classes(self):
        return self.c

    def predict(self, texts, return_proba=False, use_tf_dataset=False, verbose=0):
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
          use_tf_dataset(bool): If True, wraps dataset in a tf.Dataset when passing input to model.

          verbose(int): verbosity: 0 (silent), 1 (progress bar), 2 (single line)
        ```
        """

        is_array, is_pair = detect_text_format(texts)
        if not is_array:
            texts = [texts]

        classification, multilabel = U.is_classifier(self.model)

        # get predictions
        if U.is_huggingface(model=self.model):
            tseq = self.preproc.preprocess_test(texts, verbose=0)
            if use_tf_dataset:
                tseq.batch_size = self.batch_size
                tfd = tseq.to_tfdataset(train=False)
                preds = self.model.predict(tfd, verbose=verbose)
            else:
                data = tseq.to_array()
                preds = self.model.predict(
                    data, batch_size=self.batch_size, verbose=verbose
                )
            if hasattr(
                preds, "logits"
            ):  # dep_fix: breaking change - also needed for LongFormer
                # if type(preds).__name__ == 'TFSequenceClassifierOutput': # dep_fix: undocumented breaking change in transformers==4.0.0
                # REFERENCE: https://discuss.huggingface.co/t/new-model-output-types/195
                preds = preds.logits

            # dep_fix: transformers in TF 2.2.0 returns a tuple insead of NumPy array for some reason
            if isinstance(preds, tuple) and len(preds) == 1:
                preds = preds[0]
        else:
            texts = self.preproc.preprocess(texts)
            preds = self.model.predict(
                texts, batch_size=self.batch_size, verbose=verbose
            )

        # process predictions
        if U.is_huggingface(model=self.model):
            # convert logits to probabilities for Hugging Face models
            if multilabel and self.c:
                preds = keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
            elif self.c:
                preds = keras.activations.softmax(tf.convert_to_tensor(preds)).numpy()
            else:
                preds = np.squeeze(preds)
                if len(preds.shape) == 0:
                    preds = np.expand_dims(preds, -1)
        result = (
            preds
            if return_proba or multilabel or not self.c
            else [self.c[np.argmax(pred)] for pred in preds]
        )
        if multilabel and not return_proba:
            result = [list(zip(self.c, r)) for r in result]
        if not is_array:
            return result[0]
        else:
            return result

    def predict_proba(self, texts, verbose=0):
        """
        ```
        Makes predictions for a list of strings where each string is a document
        or text snippet.
        Returns probabilities of each class.
        ```
        """
        return self.predict(texts, return_proba=True, verbose=verbose)

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
            warnings.warn(
                "currently_unsupported: explain does not currently support sentence pair classification"
            )
            return
        if not self.c:
            warnings.warn(
                "currently_unsupported: explain does not support text regression"
            )
            return
        try:
            import eli5
            from eli5.lime import TextExplainer
        except:
            msg = (
                "ktrain requires a forked version of eli5 to support tf.keras. "
                + "Install with: pip install https://github.com/amaiya/eli5-tf/archive/refs/heads/master.zip"
            )
            warnings.warn(msg)
            return

        if not isinstance(doc, str):
            raise TypeError("text must of type str")
        prediction = [self.predict(doc)] if not all_targets else None

        if self.preproc.is_nospace_lang():
            doc = self.preproc.process_chinese([doc])
            doc = doc[0]
        doc = " ".join(doc.split()[:truncate_len])
        te = TextExplainer(random_state=42, n_samples=n_samples)
        _ = te.fit(doc, self.predict_proba)
        return te.show_prediction(
            target_names=self.preproc.get_classes(), targets=prediction
        )

    def _save_model(self, fpath):
        if isinstance(self.preproc, TransformersPreprocessor):
            self.model.save_pretrained(fpath)
            # As of 0.26.3, make sure we save tokenizer in predictor folder
            tok = self.preproc.get_tokenizer()
            tok.save_pretrained(fpath)
        else:
            super()._save_model(fpath)
        return
