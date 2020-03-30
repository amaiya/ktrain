from ..imports import *
from ..predictor import Predictor
from .preprocessor import TextPreprocessor, TransformersPreprocessor
from .. import utils as U

class TextPredictor(Predictor):
    """
    predicts text classes
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
        Makes predictions for a list of strings where each string is a document
        or text snippet.
        If return_proba is True, returns probabilities of each class.
        """

        is_str = False
        if isinstance(texts, str):
            is_str = True
            texts = [texts]
        elif not isinstance(texts, np.ndarray) and not isinstance(texts, list):
            raise ValueError('data must be numpy.ndarray or list (of texts)')
        classification, multilabel = U.is_classifier(self.model)

        # get predictions
        if U.is_huggingface(model=self.model):
            tseq = self.preproc.preprocess_test(texts, verbose=0)
            tseq.batch_size = self.batch_size
            texts = tseq.to_tfdataset(shuffle=False, repeat=False)
            preds = self.model.predict(texts)
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
        if is_str: return result[0]
        else:      return result



    def predict_proba(self, texts):
        """
        Makes predictions for a list of strings where each string is a document
        or text snippet.
        Returns probabilities of each class.
        """
        return self.predict(texts, return_proba=True)


    def explain(self, doc, truncate_len=512, all_targets=False):
        """
        Highlights text to explain prediction
        Args:
            doc (str): text of documnet
            truncate_len(int): truncate document to this many words
            all_targets(bool):  If True, show visualization for
                                each target.
        """
        if not self.c:
            warnings.warn('currently_unsupported:  explain does not support text regression')
            return
        try:
            import eli5
            from eli5.lime import TextExplainer
        except:
            msg = 'ktrain requires a forked version of eli5 to support tf.keras. '+\
                  'Install with: pip3 install git+https://github.com/amaiya/eli5@tfkeras_0_10_1'
            warnings.warn(msg)
            return

        prediction = [self.predict(doc)] if not all_targets else None

        if not isinstance(doc, str): raise Exception('text must of type str')
        if self.preproc.is_nospace_lang():
            doc = self.preproc.process_chinese([doc])
            doc = doc[0]
        doc = ' '.join(doc.split()[:truncate_len])
        te = TextExplainer(random_state=42)
        _ = te.fit(doc, self.predict_proba)
        return te.show_prediction(target_names=self.preproc.get_classes(), targets=prediction)


    def analyze_valid(self, val_tup, print_report=True, multilabel=None):
        """
        Makes predictions on validation set and returns the confusion matrix.
        Accepts as input the validation set in the standard form of a tuple of
        two arrays: (X_test, y_test), wehre X_test is a Numpy array of strings
        where each string is a document or text snippet in the validation set.

        Optionally prints a classification report.
        Currently, this method is only supported for binary and multiclass 
        problems, not multilabel classification problems.
        """
        U.data_arg_check(val_data=val_tup, val_required=True, ndarray_only=True)
        if multilabel is None:
            multilabel = U.is_multilabel(val_tup)
        if multilabel:
            warnings.warn('multilabel_confusion_matrix not yet supported')
            return

        y_true = val_tup[1]
        y_true = np.argmax(y_true, axis=1)
        y_pred = self.model.predict(val_tup[0])
        y_pred = np.argmax(y_pred, axis=1)
        
        if print_report:
            print(classification_report(y_true, y_pred, target_names=self.c))
            cm_func = confusion_matrix
        cm =  confusion_matrix(y_true,  y_pred)
        return cm


    def save(self, fpath):

        if isinstance(self.preproc, TransformersPreprocessor):
            if os.path.isfile(fpath):
                raise ValueError(f'There is an existing file named {fpath}. ' +\
                                  'Please use dfferent value for fpath.')
            elif not os.path.exists(fpath):
                os.mkdir(fpath)
            self.model.save_pretrained(fpath)
            fname_preproc = fpath+'.preproc'
            with open(fname_preproc, 'wb') as f:
                pickle.dump(self.preproc, f)
        else:
            super().save(fpath)
        return

