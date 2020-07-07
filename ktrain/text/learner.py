from ..imports import *
from .. import utils as U
from ..core import ArrayLearner, GenLearner, _load_model
from .preprocessor import TransformersPreprocessor




class BERTTextClassLearner(ArrayLearner):
    """
    Main class used to tune and train Keras models for text classification using Array data.
    """


    def __init__(self, model, train_data=None, val_data=None, 
                 batch_size=U.DEFAULT_BS, eval_batch_size=U.DEFAULT_BS,
                 workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, train_data=train_data, val_data=val_data,
                         batch_size=batch_size, eval_batch_size=eval_batch_size,
                         workers=workers, use_multiprocessing=use_multiprocessing,
                         multigpu=multigpu)
        return


    def view_top_losses(self, n=4, preproc=None, val_data=None):
        """
        Views observations with top losses in validation set.
        Args:
         n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
         preproc (Preprocessor): A TextPreprocessor or ImagePreprocessor.
                                 For some data like text data, a preprocessor
                                 is required to undo the pre-processing
                                 to correctly view raw data.
          val_data:  optional val_data to use instead of self.val_data
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

        """
        val = self._check_val(val_data)


        # get top losses and associated data
        tups = self.top_losses(n=n, val_data=val, preproc=preproc)

        # get multilabel status and class names
        classes = preproc.get_classes() if preproc is not None else None

        # iterate through losses
        for tup in tups:

            # get data
            idx = tup[0]
            loss = tup[1]
            truth = tup[2]
            pred = tup[3]

            # BERT-style tuple
            join_char = ' '
            obs = val[0][0][idx]
            if preproc is not None: 
                obs = preproc.undo(obs)
                if preproc.is_nospace_lang(): join_char = ''
            if type(obs) == str:
                obs = join_char.join(obs.split()[:512])
            print('----------')
            print("id:%s | loss:%s | true:%s | pred:%s)\n" % (idx, round(loss,2), truth, pred))
            print(obs)
        return


class TransformerTextClassLearner(GenLearner):
    """
    Main class used to tune and train Keras models for text classification using Array data.
    """


    def __init__(self, model, train_data=None, val_data=None, 
                 batch_size=U.DEFAULT_BS, eval_batch_size=U.DEFAULT_BS,
                 workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, train_data=train_data, val_data=val_data,
                         batch_size=batch_size, eval_batch_size=eval_batch_size,
                         workers=workers, use_multiprocessing=use_multiprocessing,
                         multigpu=multigpu)
        return


    def view_top_losses(self, n=4, preproc=None, val_data=None):
        """
        Views observations with top losses in validation set.
        Args:
         n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
         preproc (Preprocessor): A TextPreprocessor or ImagePreprocessor.
                                 For some data like text data, a preprocessor
                                 is required to undo the pre-processing
                                 to correctly view raw data.
          val_data:  optional val_data to use instead of self.val_data
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

        """
        val = self._check_val(val_data)


        # get top losses and associated data
        tups = self.top_losses(n=n, val_data=val, preproc=preproc)

        # get multilabel status and class names
        classes = preproc.get_classes() if preproc is not None else None

        # iterate through losses
        for tup in tups:

            # get data
            idx = tup[0]
            loss = tup[1]
            truth = tup[2]
            pred = tup[3]

            join_char = ' '
            #obs = val.x[idx][0]
            print('----------')
            print("id:%s | loss:%s | true:%s | pred:%s)\n" % (idx, round(loss,2), truth, pred))
        return


    def _prepare(self, data, train=True):
        """
        prepare data as tf.Dataset
        """
        # HF_EXCEPTION
        # convert arrays to TF dataset (iterator) on-the-fly
        # to work around issues with transformers and tf.Datasets
        if data is None: return None
        return data.to_tfdataset(train=train)


    def predict(self, val_data=None):
        """
        Makes predictions on validation set
        """
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if val is None: raise Exception('val_data must be supplied to get_learner or predict')
        if hasattr(val, 'reset'): val.reset()
        classification, multilabel = U.is_classifier(self.model)
        preds = self.model.predict(self._prepare(val, train=False))

        # dep_fix: transformers in TF 2.2.0 returns a tuple insead of NumPy array for some reason
        if isinstance(preds, tuple) and len(preds) == 1: preds = preds[0] 

        if classification:
            if multilabel:
                return activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
            else:
                return activations.softmax(tf.convert_to_tensor(preds)).numpy()
        else:
            return preds


    def save_model(self, fpath):
        """
        save Transformers model
        """
        self._make_model_folder(fpath)
        self.model.save_pretrained(fpath)
        return


    # 2020-07-07: removed, as core.Learner.load_model calls TransformerPreprocessor.load_model_and_configure
    #def load_model(self, fpath, preproc=None):
    #    """
    #    load Transformers model
    #    Args:
    #      fpath(str): path to folder containing model files
    #      preproc(TransformerPreprocessor): a TransformerPreprocessor instance.
    #    """
    #    if preproc is None or not isinstance(preproc, TransformersPreprocessor):
    #        raise ValueError('preproc arg is required to load Transformer models from disk. ' +\
    #                          'Supply a TransformersPreprocessor instance. This is ' +\
    #                          'either the third return value from texts_from* function or '+\
    #                          'the result of calling ktrain.text.Transformer')


    #    self.model = _load_model(fpath, preproc=preproc)
    #    return



