from ...imports import *
from ... import utils as U
from ...core import GenLearner


class NERLearner(GenLearner):
    """
    Learner for Sequence Taggers.
    """


    def __init__(self, model, train_data=None, val_data=None, 
                 batch_size=U.DEFAULT_BS, eval_batch_size=U.DEFAULT_BS,
                 workers=1, use_multiprocessing=False,
                 multigpu=False):
        super().__init__(model, train_data=train_data, val_data=val_data, 
                         batch_size=batch_size, eval_batch_size=eval_batch_size,
                         workers=workers, use_multiprocessing=use_multiprocessing, 
                         multigpu=multigpu)
        return



    def validate(self, val_data=None, print_report=True, class_names=[]):
        """
        Validate text sequence taggers
        """
        val = self._check_val(val_data)

        if not U.is_ner(model=self.model, data=val):
            warnings.warn('learner.validate_ner is only for sequence taggers.')
            return

        label_true = []
        label_pred = []
        for i in range(len(val)):
            x_true, y_true = val[i]
            #lengths = self.ner_lengths(y_true)
            lengths = val.get_lengths(i)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = val.p.inverse_transform(y_true, lengths)
            y_pred = val.p.inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = ner_f1_score(label_true, label_pred)
        if print_report:
            print('   F1: {:04.2f}'.format(score * 100))
            print(ner_classification_report(label_true, label_pred))

        return score

    def top_losses(self, n=4, val_data=None, preproc=None):
        """
        Computes losses on validation set sorted by examples with top losses
        Args:
          n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
          val_data:  optional val_data to use instead of self.val_data
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

        """
        val = self._check_val(val_data)
        if type(n) == type(42):
            n = (0, n)

        # get predicictions and ground truth
        y_pred = self.predict(val_data=val)
        y_true = self.ground_truth(val_data=val)

        # compute losses and sort
        losses = []
        for idx, y_t in enumerate(y_true):
            y_p = y_pred[idx]
            #err = 1- sum(1 for x,y in zip(y_t,y_p) if x == y) / len(y_t)
            err = sum(1 for x,y in zip(y_t,y_p) if x != y) 
            losses.append(err)
        tups = [(i,x, y_true[i], y_pred[i]) for i, x in enumerate(losses) if x > 0]
        tups.sort(key=operator.itemgetter(1), reverse=True)

        # prune by given range
        tups = tups[n[0]:n[1]] if n is not None else tups
        return tups


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

        # check validation data and arguments
        val = self._check_val(val_data)

        tups = self.top_losses(n=n, val_data=val)

        # get multilabel status and class names
        classes = preproc.get_classes() if preproc is not None else None

        # iterate through losses
        for tup in tups:

            # get data
            idx = tup[0]
            loss = tup[1]
            truth = tup[2]
            pred = tup[3]

            seq = val.x[idx]
            print('total incorrect: %s' % (loss))
            print("{:15} {:5}: ({})".format("Word", "True", "Pred"))
            print("="*30)
            for w, true_tag, pred_tag in zip(seq, truth, pred):
                print("{:15}:{:5} ({})".format(w, true_tag, pred_tag))
            print('\n')
        return


    def save_model(self, fpath):
        """
        a wrapper to model.save
        """
        if U.is_crf(self.model):
            from .anago.layers import crf_loss
            self.model.compile(loss=crf_loss, optimizer=U.DEFAULT_OPT)
        self.model.save(fpath, save_format='h5')
        return


    def predict(self, val_data=None):
        """
        Makes predictions on validation set
        """
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if val is None: raise Exception('val_data must be supplied to get_learner or predict')
        steps = np.ceil(U.nsamples_from_data(val)/val.batch_size)
        results = []
        for idx, (X, y) in enumerate(val):
            y_pred = self.model.predict_on_batch(X)
            lengths = val.get_lengths(idx)
            y_pred = val.p.inverse_transform(y_pred, lengths)
            results.extend(y_pred)
        return results


    def _prepare(self, data, mode='train'):
        """
        prepare NERSequence for training
        """
        if data is None: return None
        if not data.prepare_called:
            print('preparing %s data ...' % (mode), end='')
            data.prepare()
            print('done.')
        return data

