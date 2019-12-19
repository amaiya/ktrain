from ..imports import *
from .. import utils as U
from ..core import ArrayLearner, GenLearner




class BERTTextClassLearner(ArrayLearner):
    """
    Main class used to tune and train Keras models for text classification using Array data.
    """


    def __init__(self, model, train_data=None, val_data=None, 
                 batch_size=U.DEFAULT_BS, workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, train_data=train_data, val_data=val_data,
                         batch_size=batch_size, 
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
                 batch_size=U.DEFAULT_BS, workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, train_data=train_data, val_data=val_data,
                         batch_size=batch_size, 
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


    def _prepare(self, data, mode='train'):
        """
        prepare data as tf.Dataset
        """
        # HF_EXCEPTION
        # convert arrays to TF dataset (iterator) on-the-fly
        # to work around issues with transformers and tf.Datasets
        tfdataset = self.features_to_tfdataset(data)
        if mode == 'train':
            return tfdataset.shuffle(U.nsamples_from_data(data)).batch(self.batch_size).repeat(-1)
        else:
            return tfdataset.batch(self.batch_size)


    def features_to_tfdataset(self, data):
        """
        convert transformer features to tf.Dataset
        """

        features_list = data[0]['transformer_features']
        labels = data[1]
        if type(features_list) not in [list, np.ndarray] or\
                type(labels) not in [list, np.ndarray]:
            raise ValueError('features_list and labels must be numpy arrays')
        if type(features_list) == list: features_list = np.array(features_list)
        if type(labels) == list: labels = np.array(labels)
        tfdataset = tf.data.Dataset.from_tensor_slices((features_list, labels))
        tfdataset = tfdataset.map(lambda x,y: ({'input_ids': x[0], 
                                                'attention_mask': x[1], 
                                                 'token_type_ids': x[2]}, y))

        return tfdataset
