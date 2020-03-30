from ..imports import *
from .. import utils as U
from ..core import GenLearner




class NodeClassLearner(GenLearner):
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
        Typically over-ridden by Learner subclasses.
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

            print('----------')
            print("id:%s | loss:%s | true:%s | pred:%s)\n" % (idx, round(loss,2), truth, pred))
            #print(obs)
        return



    def layer_output(self, layer_id, example_id=0, batch_id=0, use_val=False):
        """
        Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from training set, by default.
        """
        warnings.warn('currently_unsupported: layer_output method is not yet supported for ' +
                      'graph neural networks in ktrain')
        return


