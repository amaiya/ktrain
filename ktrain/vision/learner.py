from ..imports import *
from .. import utils as U
from ..core import GenLearner
from .data import show_image




class ImageClassLearner(GenLearner):
    """
    Main class used to tune and train Keras models for image classification.
    Main parameters are:

    model (Model): A compiled instance of keras.engine.training.Model
    train_data (Iterator): a Iterator instance for training set
    val_data (Iterator):   A Iterator instance for validation set
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

        # check validation data and arguments
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if val is None: raise Exception('val_data must be supplied to get_learner or view_top_losses')

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

            # Image Classification
            if type(val).__name__ in ['DirectoryIterator', 'DataFrameIterator']:
                fpath = val.filepaths[tup[0]]
                fp = os.path.join(os.path.basename(os.path.dirname(fpath)), os.path.basename(fpath))
                plt.figure()
                plt.title("%s | loss:%s | true:%s | pred:%s)" % (fp, round(loss,2), truth, pred))
                show_image(fpath)
            elif type(val).__name__ in ['NumpyArrayIterator']:
                obs = val.x[idx]
                #if preproc is not None: obs = preproc.undo(obs)
                plt.figure()
                plt.title("id:%s | loss:%s | true:%s | pred:%s)" % (idx, round(loss,2), truth, pred))
                plt.imshow(np.squeeze(obs))
                # everything else including text classification
            else:
                raise Exception('ImageClassLearner.view_top_losses only supports ' +
                                'DirectoryIterators, DataFrameIterators, and NumpyArrayIterators')
        return


