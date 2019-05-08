import os
import os.path
import numpy as np
import warnings
import operator
from distutils.version import StrictVersion
import tempfile
import pickle
from abc import ABC, abstractmethod
import math

from matplotlib import pyplot as plt

import keras
from keras import backend as K
from keras.engine.training import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.initializers import glorot_uniform  
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from .lroptimize.sgdr import *
from .lroptimize.triangular import *
from .lroptimize.lrfinder import *
from . import utils as U

from .vision.preprocessor import ImagePreprocessor
from .vision.predictor import ImagePredictor
from .vision.data import show_image
from .text.preprocessor import TextPreprocessor
from .text.predictor import TextPredictor

from keras.applications.resnet50 import preprocess_input as pre_resnet50
from keras.applications.mobilenet import preprocess_input as pre_mobilenet
from keras.applications.inception_v3 import preprocess_input as pre_inception




def get_learner(model, train_data=None, val_data=None, 
                batch_size=U.DEFAULT_BS, workers=1, use_multiprocessing=False,
                multigpu=False):
    """
    Returns a Learner instance that can be used to tune and train Keras models.

    model (Model):        A compiled instance of keras.engine.training.Model
    train_data (tuple or generator): Either a: 
                                   1) tuple of (x_train, y_train), where x_train and 
                                      y_train are numpy.ndarrays or 
                                   2) Iterator
    val_data (tuple or generator): Either a: 
                                   1) tuple of (x_test, y_test), where x_testand 
                                      y_test are numpy.ndarrays or 
                                   2) Iterator
                                   Note: Should be same type as train_data.
    batch_size (int):              Batch size to use in training
    workers (int): number of cpu processes used to load data.
                   only applicable if train_data is is a generator.
    use_multiprocessing(bool):  whether or not to use multiprocessing for workers
    multigpu(bool):             Lets the Learner know that the model has been 
                                replicated on more than 1 GPU.
                                Only supported for models from vision.image_classifiers
                                at this time.
    """

    # check arguments
    if not isinstance(model, Model):
        raise ValueError('model must be of instance Model')
    U.data_arg_check(train_data=train_data, val_data=val_data)
    if type(workers) != type(1) or workers < 1:
        workers =1
    # check for NumpyArrayIterator 
    if train_data and not U.ondisk(train_data):
        if workers > 1 and not use_multiprocessing:
            use_multiprocessing = True
            wrn_msg = 'Changed use_multiprocessing to True because NumpyArrayIterator with workers>1'
            wrn_msg +=' is slow when use_multiprocessing=False.'
            wrn_msg += ' If you experience issues with this, please set workers=1 and use_multiprocessing=False.'
            warnings.warn(wrn_msg)

    # return the appropriate trainer
    if U.is_iter(train_data):
        learner = GenLearner
    else:
        learner = ArrayLearner
    return learner(model, train_data=train_data, val_data=val_data, 
                   batch_size=batch_size, workers=workers, use_multiprocessing=use_multiprocessing, multigpu=multigpu)


class Learner(ABC):
    """
    Abstract class used to tune and train Keras models. The fit method is
    an abstract method and must be implemented by subclasses.

    """
    def __init__(self, model, workers=1, use_multiprocessing=False, multigpu=False):
        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        self.model = model
        self.lr_finder = LRFinder(self.model)
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.multigpu=multigpu
        self.history = None

        # save original weights of model
        new_file, weightfile = tempfile.mkstemp()
        self.model.save_weights(weightfile)
        self._original_weights = weightfile



    def get_weight_decay(self):
        """
        Gets set of weight decays currently used in network.
        use print_layers(show_wd=True) to view weight decays per layer.
        """
        wds = []
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                reg = layer.kernel_regularizer
                if hasattr(reg, 'l2'):
                    wd = reg.l2
                elif hasattr(reg, 'l1'):
                    wd = reg.l1
                else:
                    wd = None
                wds.append(wd)
        return wds


    def set_weight_decay(self, wd=0.005):
        """
        Sets global weight decay layer-by-layer using L2 regularization.

        NOTE: Weight decay can be implemented in the form of
              L2 regularization, which is the case with Keras.
              Thus, he weight decay value must be divided by
              2 to obtain similar behavior.
              The default weight decay here is 0.01/2 = 0.005.
              See here for more information: 
              https://bbabenko.github.io/weight-decay/
        Args:
          wd(float): weight decay (see note above)
        Returns:
          None
              
        """
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= regularizers.l2(wd)
            if hasattr(layer, 'bias_regularizer'):
                layer.bias_regularizer= regularizers.l2(wd)
        return
        

    def confusion_matrix(self, print_report=False):
        """
        Returns confusion matrix and optionally prints
        a classification report.
        This is currently only supported for binary and multiclass
        classification, not multilabel classification.
        """
        if U.is_multilabel(self.val_data):
            warnings.warn('multilabel confusion matrices not yet supported')
            return
        y_pred = self.predict()
        y_true = self.ground_truth(use_valid=True)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        if print_report:
            print(classification_report(y_true, y_pred))
            cm_func = confusion_matrix
        cm =  confusion_matrix(y_true,  y_pred)
        return cm


    def top_losses(self, n=8):
        """
        Computes losses on validation set sorted by examples with top losses
        Args:
         n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

        """
        import tensorflow as tf

        # check validation data and arguments
        if not self.val_data:
            raise Exception('val_data was not supplied to get_learner')
        if type(n) == type(42):
            n = (0, n)

        multilabel = True if U.is_multilabel(self.val_data) else False

        # get predicictions and ground truth
        y_pred = self.predict()
        y_true = self.ground_truth(use_valid=True)
        y_true = y_true.astype('float32')

        # compute loss
        losses = self.model.loss_functions[0](tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
        losses = tf.Session().run(losses)

        # sort by loss and prune correct classifications, if necessary
        if 'acc' in self.model.metrics_names and not multilabel:
            y_p = np.argmax(y_pred, axis=1)
            y_t = np.argmax(y_true, axis=1)
            tups = [(i,x) for i, x in enumerate(losses) if y_p[i] != y_t[i]]
        else:
            tups = [(i,x) for i, x in enumerate(losses)]
        tups.sort(key=operator.itemgetter(1), reverse=True)

        # prune by given range
        tups = tups[n[0]:n[1]] if n is not None else tups
        return tups


    def view_top_losses(self, preproc=None, n=8):
        """
        Views observations with top losses in validation set.
        Args:
         preproc (Preprocessor): A TextPreprocessor or ImagePreprocessor.
                                 For some data like text data, a preprocessor
                                 is required to undo the pre-processing
                                 to correctly view raw data.
         n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

        """
        tups = self.top_losses(n=n)
        for tup in tups:
            idx = tup[0]
            loss = tup[1]
            if type(self.val_data).__name__ in ['DirectoryIterator', 'DataFrameIterator']:
                # idx is replaced with a file path
                # replace IDs with file paths, if possible
                fpath = self.val_data.filepaths[tup[0]]
                plt.figure()
                plt.title("%s (LOSS: %s)" % (fpath, round(loss,3)))
                show_image(fpath)
            else:
                if type(self.val_data).__name__ in ['NumpyArrayIterator']:
                    obs = self.val_data.x[idx]
                    if preproc is not None: obs = preproc.undo(obs)
                    plt.figure()
                    plt.title("id:%s (LOSS: %s)" % (idx, round(loss,3)))
                    plt.imshow(np.squeeze(obs))
                else:
                    obs = self.val_data[0][idx]
                    if preproc is not None: obs = preproc.undo(obs)
                    print('----------')
                    print("val_id:%s (LOSS: %s)\n" % (idx, round(loss,3)))
                    print(obs)
        return



            




    def save_model(self, fpath):
        """
        a wrapper to model.save
        """
        self.model.save(fpath)
        return


    def load_model(self, fpath):
        """
        a wrapper to load_model
        """
        self.model = load_model(fpath)
        return


    def _recompile(self):
        # ISSUE: recompile does not work correctly with multigpu models
        if self.multigpu:
            err_msg = """
                   IMPORTANT: freeze and unfreeze methods do not currently work with 
                   multi-GPU models.  If you are using the load_imagemodel method to
                   define your model, please reload your model and use the freeze_layers 
                   argument of load_imagemodel to selectively freeze layers.
                   """
            raise Exception(err_msg)

        if self.multigpu:
            import tensorflow as tf
            with tf.device("/cpu:0"):
                self.model.compile(optimizer=self.model.optimizer,
                                   loss=self.model.loss,
                                   metrics=self.model.metrics)
        else:
            self.model.compile(optimizer=self.model.optimizer,
                               loss=self.model.loss,
                               metrics=self.model.metrics)
        return


    def set_model(self, model):
        """
        replace model in this Learner instance
        """
        if not isinstance(model, Model):
            raise ValueError('model must be of instance Model')
        self.model = model
        self.history = None
        return


    def freeze(self, freeze_range=None):
        """
        If freeze_range is None, makes all layers trainable=False except last Dense layer.
        If freeze_range is given, freezes the first <freeze_range> layers and
        unfrezes all remaining layers.
        NOTE:      Freeze method does not currently work with 
                   multi-GPU models.  If you are using the load_imagemodel method,
                   please use the freeze_layers argument of load_imagemodel
                   to freeze layers.
        Args:
            freeze_range(int): number of layers to freeze
        Returns:
            None
        """

        if freeze_range is None:
            # freeze everything except last Dense layer
            # first find last dense layer
            dense_id = None
            for i, layer in reversed(list(enumerate(self.model.layers))):
                if isinstance(layer, Dense):
                    dense_id = i
                    break
            if dense_id is None: raise Exception('cannot find Dense layer in this model')
            for i, layer in enumerate(self.model.layers):
                if i < dense_id: 
                    layer.trainable=False
                else:
                    layer.trainable=True
        else:
            # freeze all layers up to and including layer_id
            if type(freeze_range) != type(1) or freeze_range <1: 
                raise ValueError('freeze_range must be integer > 0')
            for i, layer in enumerate(self.model.layers):
                if i < freeze_range: 
                    layer.trainable=False
                else:
                    layer.trainable=True
        self._recompile()
        return


    def unfreeze(self, exclude_range=None):
        """
        Make every layer trainable except those in exclude_range.
        unfreeze is simply a proxy method to freeze.
        NOTE:      Unfreeze method does not currently work with 
                   multi-GPU models.  If you are using the load_imagemodel method,
                   please use the freeze_layers argument of load_imagemodel
                   to freeze layers.
        """
        # make all layers trainable
        for i, layer in enumerate(self.model.layers):
            layer.trainable = True
        if exclude_range:
            for i, layer in enumerate(self.model.layers[:exclude_range]):
                layer.trainable = False
        self._recompile()
        return


    def reset_weights(self, nosave=False, verbose=1):
        """
        Re-initializes network - use with caution, as this may not be robust
        """
        #initial_weights = self.model.get_weights()
        #backend_name = K.backend()
        #if backend_name == 'tensorflow': 
            #k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
        #elif backend_name == 'theano': 
            #k_eval = lambda placeholder: placeholder.eval()
        #else: 
            #raise ValueError("Unsupported backend")
        #new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]
        #if nosave: return new_weights
        #self.model.set_weights(new_weights)
        #self.history = None
        #print('Weights of moedl have been reset.')

        if os.path.isfile(self._original_weights):
            self.model.load_weights(self._original_weights)
            self.history = None
            U.vprint('Model weights have been reset.', verbose=verbose)
        else:
            warnings.warn('Weights have not been reset because the original weights file '+\
                          '(%s) no longer exists.' % (self._original_weights))
        return



    def lr_find(self, start_lr=1e-7, epochs=None, verbose=1):
        """
        Plots loss as learning rate is increased.
        Highest learning rate corresponding to a still
        falling loss should be chosen.

        Reference: https://arxiv.org/abs/1506.01186

        Args:
            epochs (int): maximum number of epochs to simulate training
                          If None, chosen automatically.
            start_lr (float): smallest lr to start simulation
            verbose (bool): specifies how much output to print
        Returns:
            float:  Numerical estimate of best lr.  
                    The lr_plot method should be invoked to
                    identify the maximal loss associated with falling loss.
        """

        U.vprint('simulating training for different learning rates... this may take a few moments...',
                verbose=verbose)

        # save current weights and temporarily restore original weights
        new_file, weightfile = tempfile.mkstemp()
        self.model.save_weights(weightfile)
        #self.model.load_weights(self._original_weights)

        try:
            # track and plot learning rates
            self.lr_finder = LRFinder(self.model)
            self.lr_finder.find(self.train_data, start_lr=start_lr, end_lr=10, 
                                epochs=epochs,
                                workers=self.workers, 
                                use_multiprocessing=self.use_multiprocessing, 
                                verbose=verbose)
        except KeyboardInterrupt:
            # re-load current weights
            self.model.load_weights(weightfile)
            return

        # re-load current weights
        self.model.load_weights(weightfile)

        # instructions to invoker
        U.vprint('\n', verbose=verbose)
        U.vprint('done.', verbose=verbose)
        U.vprint('Please invoke the Learner.lr_plot() method to visually inspect '
              'the loss plot to help identify the maximal learning rate '
              'associated with falling loss.', verbose=verbose)

        return 


    def lr_plot(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss vs. learning rate to help identify
        The maximal learning rate associated with a falling loss.
        The nskip_beginning and n_skip_end arguments can be used
        to "zoom in" on the plot.
        """
        self.lr_finder.plot_loss(n_skip_beginning=n_skip_beginning,
                                 n_skip_end=n_skip_end)
        return


    def plot(self, plot_type='loss'):
        """
        plots training history
        Args:
          plot_type (str):  one of {'loss', 'lr', 'momentum'}
        Return:
          None
        """
        if self.history is None:
            raise Exception('No training history - did you train the model yet?')

        if plot_type == 'loss':
            plt.plot(self.history.history['loss'])
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'])
                legend_items = ['train', 'validation']
            else:
                legend_items = ['train']
            plt.title('Model Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(legend_items, loc='upper left')
            plt.show()
        elif plot_type == 'lr':
            if 'lr' not in self.history.history:
                raise ValueError('no lr in history: are you sure you used autofit or fit_onecycle to train?')
            plt.plot(self.history.history['lr'])
            plt.title('LR Schedule')
            plt.ylabel('lr')
            plt.xlabel('iterations')
            plt.show()
        elif plot_type == 'momentum':
            if 'momentum' not in self.history.history:
                raise ValueError('no momentum history: are you sure you used autofit or fit_onecycle to train?')
            plt.plot(self.history.history['momentum'])
            plt.title('Momentum Schedule')
            plt.ylabel('momentum')
            plt.xlabel('iterations')
            plt.show()
        else:
            raise ValueError('invalid type: choose loss, lr, or momentum')
        return


    def print_layers(self, show_wd=False):
        """
        prints the layers of the model along with indices
        """
        for i, layer in enumerate(self.model.layers):
            if show_wd and hasattr(layer, 'kernel_regularizer'):
                reg = layer.kernel_regularizer
                if hasattr(reg, 'l2'):
                    wd = reg.l2
                elif hasattr(reg, 'l1'):
                    wd = reg.l1
                else:
                    wd = None
                print("%s (trainable=%s, wd=%s) : %s" % (i, layer.trainable, wd, layer))
            else:
                print("%s (trainable=%s) : %s" % (i, layer.trainable, layer))
        return


    def layer_output(self, layer_id, example_id=0, use_val=False):
        # should implemented in subclass
        raise NotImplementedError


    def set_lr(self, lr):
        K.set_value(self.model.optimizer.lr, lr)
        return


    def _check_cycles(self, n_cycles, cycle_len, cycle_mult):
        if type(n_cycles) != type(1) or n_cycles <1:
            raise ValueError('n_cycles must be >= 1')
        if type(cycle_mult) != type(1) or cycle_mult < 1:
            raise ValueError('cycle_mult must by >= 1')
        if cycle_len is not None:
            if type(cycle_len) != type(1) or cycle_len < 1:
                raise ValueError('cycle_len must either be None or >= 1')

        # calculate number of epochs
        if cycle_len is None:
            epochs = n_cycles
        else:
            epochs = 0
            tmp_cycle_len = cycle_len
            for i in range(n_cycles):
                epochs += tmp_cycle_len
                tmp_cycle_len *= cycle_mult
        return epochs


    def _cb_sgdr(self, max_lr, steps_per_epoch, cycle_len, cycle_mult, lr_decay=1.0, callbacks=[]):
        # configuration
        min_lr = 1e-9
        if max_lr <= min_lr: min_lr = max_lr/10

        #  use learning_rate schedule
        if cycle_len is not None:
            if not isinstance(callbacks, list): callbacks = []
            schedule = SGDRScheduler(min_lr=min_lr,
                                     max_lr=max_lr,
                                     steps_per_epoch=steps_per_epoch,
                                     lr_decay=lr_decay,
                                     cycle_length=cycle_len,
                                     mult_factor=cycle_mult)
            callbacks.append(schedule)
        if not callbacks: callbacks=None
        return callbacks


    def _cb_checkpoint(self, folder, callbacks=[]):
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            if not isinstance(callbacks, list): callbacks = []
            #filepath=os.path.join(folder, "weights-{epoch:02d}-{val_loss:.2f}.hdf5")
            filepath=os.path.join(folder, "weights-{epoch:02d}.hdf5")
            callbacks.append(ModelCheckpoint(filepath, save_best_only=False, save_weights_only=True))
        if not callbacks: callbacks=None
        return callbacks


    def _cb_earlystopping(self, early_stopping, callbacks=[]):
        if early_stopping:
            if not isinstance(callbacks, list): callbacks = []
            #if StrictVersion(keras.__version__) >= StrictVersion('2.2.3'):
            try:
                callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping, 
                                               restore_best_weights=True, verbose=0, mode='auto'))
            except TypeError:
                warnings.warn("""
                              The early_stopping=True argument relies on EarlyStopping.restore_best_weights,
                              which is only supported on Keras 2.2.3 or greater. 
                              For now, we are falling back to EarlyStopping.restore_best_weights=False.
                              Please use checkpoint_folder option in fit() to restore best weights.""")
                callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping, 
                                               verbose=0, mode='auto'))

        if not callbacks: callbacks=None
        return callbacks


    @abstractmethod
    def fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, batch_size=U.DEFAULT_BS):
        pass


    def fit_onecycle(self, lr, epochs, checkpoint_folder=None, cycle_momentum=True,
                     verbose=1, callbacks=[]):
        """
        Train model using a version of Leslie Smith's 1cycle policy.
        This method can be used with any optimizer. Thus,
        cyclical momentum is not currently implemented.

        Args:
            lr (float): (maximum) learning rate.  
                       It is recommended that you estimate lr yourself by 
                       running lr_finder (and lr_plot) and visually inspect plot
                       for dramatic loss drop.
            epochs (int): Number of epochs.  Number of epochs
            checkpoint_folder (string): Folder path in which to save the model weights 
                                        for each epoch.
                                        File name will be of the form: 
                                        weights-{epoch:02d}-{val_loss:.2f}.hdf5
            cycle_momentum (bool):    If True and optimizer is Adam, Nadam, or Adamax, momentum of 
                                      optimzer will be cycled between 0.95 and 0.85 as described in 
                                      https://arxiv.org/abs/1803.09820.
                                      Only takes effect if Adam, Nadam, or Adamax optimizer is used.
            callbacks (list): list of Callback instances to employ during training
            verbose (bool):  verbose mode
        """

        num_samples = U.nsamples_from_data(self.train_data)
        steps_per_epoch = math.ceil(num_samples/self.batch_size)

        # setup callbacks for learning rates and early stopping
        if not callbacks: kcallbacks = []
        else:
            kcallbacks = callbacks[:] 
        if cycle_momentum:
            max_momentum = 0.95
            min_momentum = 0.85
        else:
            max_momentum = None
            min_momentum = None
        clr = CyclicLR(base_lr=lr/10, max_lr=lr,
                       step_size=math.ceil((steps_per_epoch*epochs)/2), 
                       reduce_on_plateau=0,
                       max_momentum=max_momentum,
                       min_momentum=min_momentum,
                       verbose=verbose)
        kcallbacks.append(clr)

        # start training
        policy='onecycle'
        U.vprint('\n', verbose=verbose)
        U.vprint('begin training using %s policy with max lr of %s...' % (policy, lr), 
                verbose=verbose)
        hist = self.fit(lr, epochs, early_stopping=None,
                        checkpoint_folder=checkpoint_folder,
                        verbose=verbose, callbacks=kcallbacks)
        hist.history['lr'] = clr.history['lr']
        hist.history['iterations'] = clr.history['iterations']
        if cycle_momentum:
            hist.history['momentum'] = clr.history['momentum']
        self.history = hist
        return hist



    def autofit(self, lr, epochs=None,  
                early_stopping=None, reduce_on_plateau=None, reduce_factor=2, 
                cycle_momentum=True,
                monitor='val_loss', checkpoint_folder=None, verbose=1, callbacks=[]):
        """
        Automatically train model using a default learning rate schedule shown to work well
        in practice.  This method currently employs a triangular learning 
        rate policy (https://arxiv.org/abs/1506.01186).
        During each epoch, this learning rate policy varies the learning rate from lr/10 to lr
        and then back to a low learning rate that is near-zero. 
        If epochs is None, then early_stopping and reduce_on_plateau are atomatically
        set to 6 and 3, respectively.

        Args:
            lr (float): optional initial learning rate.  If missing,
                       lr will be estimated automatically.
                       It is recommended that you estimate lr yourself by 
                       running lr_finder (and lr_plot) and visually inspect plot
                       for dramatic loss drop.
            epochs (int): Number of epochs.  If None, training will continue until
                          validation loss no longer improves after 5 epochs.
            early_stopping (int):     If not None, training will automatically stop after this many 
                                      epochs of no improvement in validation loss.
                                      Upon completion, model will be loaded with weights from epoch
                                      with lowest validation loss.
                                      NOTE: If reduce_on_plateau is also enabled, then
                                      early_stopping must be greater than reduce_on_plateau.
                                      Example: early_stopping=6, reduce_on_plateau=3.
            recuce_on_plateau (int):  If not None, will lower learning rate when
                                      when validation loss fails to improve after
                                      the specified number of epochs.
                                      NOTE: If early_stopping is enabled, then
                                      reduce_on_plateu must be less than early_stopping.
                                      Example: early_stopping=6, reduce_on_plateau=3.
            reduce_factor (int):      Learning reate is reduced by this factor on plateau.
                                      Only takes effect if reduce_on_plateau > 0.
            cycle_momentum (bool):    If True and optimizer is Adam, Nadam, or Adamax, momentum of 
                                      optimzer will be cycled between 0.95 and 0.85 as described in 
                                      https://arxiv.org/abs/1803.09820.
                                      Only takes effect if Adam, Nadam, or Adamax optimizer is used.
            checkpoint_folder (string): Folder path in which to save the model weights 
                                        for each epoch.
                                        File name will be of the form: 
                                        weights-{epoch:02d}-{val_loss:.2f}.hdf5
            monitor (str):              what metric to monitor for early_stopping
                                        and reduce_on_plateau (either val_loss or val_acc).
                                        Only used if early_stopping or reduce_on_plateau
                                        is enabled.
            callbacks (list): list of Callback instances to employ during training
            verbose (bool):  verbose mode
        """
        # check monitor
        if monitor not in ['val_acc', 'val_loss']:
            raise ValueError("monitor must be one of {'val_acc', val_loss'}")

        # setup learning rate policy 
        num_samples = U.nsamples_from_data(self.train_data)
        steps_per_epoch = num_samples//self.batch_size
        step_size = math.ceil(steps_per_epoch/2)

        # handle missing epochs
        if epochs is None:
            epochs = 1024
            if not early_stopping:
                early_stopping = U.DEFAULT_ES
                U.vprint('early_stopping automatically enabled at patience=%s' % (U.DEFAULT_ES),
                        verbose=verbose)
            if not reduce_on_plateau:
                reduce_on_plateau = U.DEFAULT_ROP
                U.vprint('reduce_on_plateau automatically enabled at patience=%s' % (U.DEFAULT_ROP),
                        verbose=verbose)
        if reduce_on_plateau and early_stopping and (reduce_on_plateau  > early_stopping):
            warnings.warn('reduce_on_plateau=%s and is greater than ' % (reduce_on_plateau) +\
                          'early_stopping=%s.  ' % (early_stopping)  +\
                          'Either reduce reduce_on_plateau or set early_stopping ' +\
                          'to be higher.')

        if self.val_data is None and monitor in ['val_loss', 'val_acc'] and\
           (reduce_on_plateau is not None or early_stopping is not None):
            raise Exception('cannot monitor %s ' % (monitor)  +\
                            'without validation data - please change monitor')



        # setup callbacks for learning rates and early stopping
        if not callbacks: kcallbacks = []
        else:
            kcallbacks = callbacks[:] 
        if cycle_momentum:
            max_momentum = 0.95
            min_momentum = 0.85
        else:
            max_momentum = None
            min_momentum = None
        clr = CyclicLR(base_lr=lr/10, max_lr=lr,
                       step_size=step_size, verbose=verbose,
                       monitor=monitor,
                       reduce_on_plateau=reduce_on_plateau,
                       reduce_factor=reduce_factor,
                       max_momentum=max_momentum,
                       min_momentum=min_momentum)
        kcallbacks.append(clr)
        if early_stopping:
            kcallbacks.append(EarlyStopping(monitor=monitor, min_delta=0, 
                                           patience=early_stopping,
                                           restore_best_weights=True, 
                                           verbose=1, mode='auto'))

        # start training
        U.vprint('\n', verbose=verbose)
        policy = 'triangular learning rate'
        U.vprint('begin training using %s policy with max lr of %s...' % (policy, lr), 
                verbose=verbose)
        hist = self.fit(lr, epochs, early_stopping=early_stopping,
                        checkpoint_folder=checkpoint_folder,
                        verbose=verbose, callbacks=kcallbacks)
        hist.history['lr'] = clr.history['lr']
        hist.history['iterations'] = clr.history['iterations']
        if cycle_momentum:
            hist.history['momentum'] = clr.history['momentum']
        self.history = hist
        return hist

    

class ArrayLearner(Learner):
    """
    Main class used to tune and train Keras models
    using Array data.  An objects of this class should be instantiated
    via the ktrain.get_learner method instead of directly.
    Main parameters are:

    
    model (Model):        A compiled instance of keras.engine.training.Model
    train_data (ndarray): A tuple of (x_train, y_train), where x_train and 
                          y_train are numpy.ndarrays.
    val_data (ndarray):   A tuple of (x_test, y_test), where x_test and 
                          y_test are numpy.ndarrays.
    """


    def __init__(self, model, train_data=None, val_data=None, 
                 batch_size=U.DEFAULT_BS, workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, workers=workers, use_multiprocessing=use_multiprocessing, multigpu=multigpu)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        return

    def layer_output(self, layer_id, example_id=0, use_val=False):
        """
        Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from training set, by default.
        """
                                                                                
        inp = self.model.layers[0].input
        outp = self.model.layers[layer_id].output
        f_out = K.function([inp], [outp])
        if not use_val:
            example = self.train_data[0][example_id]
        else:
            example = self.val_data[0][example_id]
        layer_out = f_out([np.array([example,])])[0]
        return layer_out

    
    def fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, 
            lr_decay=1, checkpoint_folder = None, early_stopping=None,
            verbose=1, callbacks=[]):
        """
        Trains the model. By default, fit is simply a wrapper for model.fit.
        When cycle_len parameter is supplied, an SGDR learning rate schedule is used.
        Trains the model.

        lr (float): learning rate 
        n_cycles (int):  n_cycles
        cycle_len (int): If not None, decay learning rate over <cycle_len>
                         epochs until restarting/resetting learning rate to <lr>.
                         If None, lr remains constant
        cycle_mult (int): Increase cycle_len by factor of cycle_mult.
                          This will gradually elongate the cycle.
                          Has no effect if cycle_len is None.
        lr_decay(float): rate of decay of learning rate each cycle
        checkpoint_folder (string): Folder path in which to save the model weights 
                                   for each epoch.
                                   File name will be of the form: 
                                   weights-{epoch:02d}-{val_loss:.2f}.hdf5
        early_stopping (int):     If not None, training will automatically stop after this many 
                                  epochs of no improvement in validation loss.
                                  Upon completion, model will be loaded with weights from epoch
                                  with lowest validation loss.
        callbacks (list):         list of Callback instances to employ during training
        verbose (bool):           whether or not to show progress bar
        """

        # check early_stopping
        if self.val_data is None and early_stopping is not None:
            raise ValueError('early_stopping monitors val_loss but validation data not set')


        # setup data
        x_train = self.train_data[0]
        y_train = self.train_data[1]
        validation = None
        if self.val_data:
            validation = (self.val_data[0], self.val_data[1])
        # setup learning rate schedule
        epochs = self._check_cycles(n_cycles, cycle_len, cycle_mult)
        self.set_lr(lr)
        kcallbacks = self._cb_sgdr(lr, 
                                  np.ceil(len(x_train)/self.batch_size), 
                                  cycle_len, cycle_mult, lr_decay=lr_decay, callbacks=None)
        sgdr = kcallbacks[0] if kcallbacks is not None else None
        kcallbacks = self._cb_checkpoint(checkpoint_folder, callbacks=kcallbacks)
        kcallbacks = self._cb_earlystopping(early_stopping, callbacks=kcallbacks)
        if callbacks:
            if kcallbacks is None: kcallbacks = []
            kcallbacks.extend(callbacks)

        # train model
        hist = self.model.fit(x_train, y_train,
                             batch_size=self.batch_size,
                             epochs=epochs,
                             validation_data=validation, verbose=verbose,
                             callbacks=kcallbacks)
        if sgdr is not None: hist.history['lr'] = sgdr.history['lr']
        self.history = hist

        if early_stopping:
            U.vprint('Weights from best epoch have been loaded into model.', verbose=verbose)
            #loss, acc = self.model.evaluate(self.val_data[0], self.val_data[1])
            #U.vprint('\n', verbose=verbose)
            #U.vprint('Early stopping due to no further improvement.', verbose=verbose)
            #U.vprint('final loss:%s, final score:%s' % (loss, acc), verbose=verbose)

        return hist


    def predict(self):
        """
        Makes predictions on validation set
        """
        if self.val_data is None:
            raise Exception('val_data is None')
        return self.model.predict(self.val_data[0])


    def ground_truth(self, use_valid=True):
        if use_valid and self.val_data is None:
            raise Exception('val_data is None')
        if use_valid:
            return self.val_data[1]
        else:
            return self.train_data[1]



class GenLearner(Learner):
    """
    Main class used to tune and train Keras models
    using a Keras generator (e.g., DirectoryIterator).
    Objects of this class should be instantiated using the
    ktrain.get_learner function, rather than directly.

    Main parameters are:

    model (Model): A compiled instance of keras.engine.training.Model
    train_data (Iterator): a Iterator instance for training set
    val_data (Iterator):   A Iterator instance for validation set
    """


    def __init__(self, model, train_data=None, val_data=None, 
                 batch_size=U.DEFAULT_BS, workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, workers=workers, use_multiprocessing=use_multiprocessing, multigpu=multigpu)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        if self.train_data:
            self.train_data.batch_size = batch_size
        if self.val_data:
            self.val_data.batch_size = batch_size
        return

    
    def fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1,
            lr_decay=1.0, checkpoint_folder=None, early_stopping=None, 
            callbacks=[], verbose=1):
        """
        Trains the model. By default, fit is simply a wrapper for model.fit_generator.
        When cycle_len parameter is supplied, an SGDR learning rate schedule is used.

        lr (float): learning rate 
        n_cycles (int):  n_cycles
        cycle_len (int): If not None, decay learning rate over <cycle_len>
                         epochs until restarting/resetting learning rate to <lr>.
                         If None, lr remains constant
        cycle_mult (int): Increase cycle_len by factor of cycle_mult.
                          This will gradually elongate the cycle.
                          Has no effect if cycle_len is None.
        lr_decay (float): rate of decay of learning reach each cycle.
                          Has no effect if cycle_len is None
        checkpoint_folder (string): Folder path in which to save the model weights 
                                   for each epoch.
                                   File name will be of the form: 
                                   weights-{epoch:02d}-{val_loss:.2f}.hdf5
        early_stopping (int):     If not None, training will automatically stop after this many 
                                  epochs of no improvement in validation loss.
                                  Upon completion, model will be loaded with weights from epoch
                                  with lowest validation loss.
        callbacks (list):         list of Callback instances to employ during training
        verbose (boolean):       whether or not to print progress bar
        """
        # check early_stopping
        if self.val_data is None and early_stopping is not None:
            raise ValueError('early_stopping monitors val_loss but validation data not set')

        
        # handle callbacks
        num_samples = U.nsamples_from_data(self.train_data)
        steps_per_epoch = num_samples // self.train_data.batch_size

        epochs = self._check_cycles(n_cycles, cycle_len, cycle_mult)
        self.set_lr(lr)
        kcallbacks = self._cb_sgdr(lr, 
                                  steps_per_epoch,
                                  cycle_len, cycle_mult, lr_decay, callbacks=None)
        sgdr = kcallbacks[0] if kcallbacks is not None else None
        kcallbacks = self._cb_checkpoint(checkpoint_folder, callbacks=kcallbacks)
        kcallbacks = self._cb_earlystopping(early_stopping, callbacks=kcallbacks)
        if callbacks:
            if kcallbacks is None: kcallbacks = []
            kcallbacks.extend(callbacks)
            
        # MNIST times per epoch on Titan V
        # workers=4, usemp=True 9 sec.
        # workers=1, usemp=True 12 sec.
        # workers=1, usemp=False 16 sec.
        # workers=4, usemp=False 30+ sec.
        #print(self.workers)
        #print(self.use_multiprocessing)

        # train model
        hist = self.model.fit_generator(self.train_data,
                                       steps_per_epoch = steps_per_epoch,
                                       epochs=epochs,
                                       validation_data=self.val_data,
                                       workers=self.workers,
                                       use_multiprocessing=self.use_multiprocessing, verbose=verbose,
                                       callbacks=kcallbacks)
        if sgdr is not None: hist.history['lr'] = sgdr.history['lr']
        self.history = hist

        if early_stopping:
            U.vprint('Weights from best epoch have been loaded into model.', verbose=verbose)
            #loss, acc = self.model.evaluate_generator(self.val_data)
            #U.vprint('\n', verbose=verbose)
            #U.vprint('Early stopping due to no further improvement.', verbose=verbose)
            #U.vprint('final loss:%s, final score:%s' % (loss, acc), verbose=verbose)
        return hist


    def layer_output(self, layer_id, example_id=0, batch_id=0, use_val=False):
        """
        Prints output of layer with index <layer_id> to help debug models.
        Uses first example (example_id=0) from training set, by default.
        """
                                                                                
        inp = self.model.layers[0].input
        outp = self.model.layers[layer_id].output
        f_out = K.function([inp], [outp])
        if not use_val:
            example = self.train_data[0][batch_id][example_id]
        else:
            example = self.val_data[0][batch_id][example_id]
        layer_out = f_out([np.array([example,])])[0]
        return layer_out


    def predict(self):
        """
        Makes predictions on validation set
        """
        if self.val_data is None:
            raise Exception('val_data is None')
        return self.model.predict_generator(self.val_data)


    def ground_truth(self, use_valid=False):
        if use_valid and self.val_data is None:
            raise Exception('val_data is None')
        if use_valid:
            return U.y_from_data(self.val_data)
        else:
            return U.y_from_data(self.train_data)


def get_predictor(model, preproc):
    """
    Returns a Predictor instance that can be used to make predictions on
    unlabeled examples.  Can be saved to disk and reloaded as part of a 
    larger application.

    Args
        model (Model):        A compiled instance of keras.engine.training.Model
        preproc(Preprocessor):   An instance of TextPreprocessor or ImagePreprocessor.
                                 These instances are returned from the data loading
                                 functions in the ktrain vision and text modules:

                                 ktrain.vision.images_from_folder
                                 ktrain.vision.images_from_csv
                                 ktrain.vision.images_from_array
                                 ktrain.text.texts_from_folder
                                 ktrain.text.texts_from_csv
    """

    # check arguments
    if not isinstance(model, Model):
        raise ValueError('model must be of instance Model')
    if not isinstance(preproc, ImagePreprocessor) and not isinstance(preproc, TextPreprocessor):
    #if not isinstance(preproc, ImagePreprocessor) and type(preproc).__name__ != 'TextPreprocessor':
        raise ValueError('preproc must be instance of ImagePreprocessor or TextPreprocessor')
    if isinstance(preproc, ImagePreprocessor):
        return ImagePredictor(model, preproc)
    elif isinstance(preproc, TextPreprocessor):
    #elif type(preproc).__name__ == 'TextPreprocessor':
        return TextPredictor(model, preproc)
    else:
        raise Exception('preproc of type %s not currently supported' % (type(preproc)))


def load_predictor(filename):
    """
    Loads a previously saved Predictor instance
    """

    model = load_model(filename)
    try:
        preproc = None
        with open(filename+'.preproc', 'rb') as f:
            preproc = pickle.load(f)
    except FileNotFoundError:
        print('load_predictor failed.\n'+\
              'Could not find the saved preprocessor (%s) for this model.' % (filename+'.preproc') +\
               ' Are you sure predictor.save method was called?')
        return
    # preprocessing functions in ImageDataGenerators are not pickable
    # so, we must reconstruct
    if hasattr(preproc, 'datagen') and hasattr(preproc.datagen, 'ktrain_preproc'):
        preproc_name = preproc.datagen.ktrain_preproc
        if preproc_name == 'resnet50':
            preproc.datagen.preprocessing_function = pre_resnet50
        elif preproc_name == 'mobilenet':
            preproc.datagen.preprocessing_function = pre_mobilenet
        elif preproc_name == 'inception':
            preproc.datagen.preprocessing_function = pre_incpeption
        else:
            raise Exception('Uknown preprocessing_function name: %s' % (preproc_name))
    
    # check arguments
    if not isinstance(model, Model):
        raise ValueError('model must be of instance Model')
    if not isinstance(preproc, ImagePreprocessor) and not isinstance(preproc, TextPreprocessor):
        raise ValueError('preproc must be instance of ImagePreprocessor or TextPreprocessor')
    if isinstance(preproc, ImagePreprocessor):
        return ImagePredictor(model, preproc)
    elif isinstance(preproc, TextPreprocessor):
        return TextPredictor(model, preproc)
    else:
        raise Exception('preprocessor not currently supported')

#----------------------------------------
# Utility Functions
#----------------------------------------




def release_gpu_memory(device=0):
    """
    Relase GPU memory allocated by Tensorflow
    Source: 
    https://stackoverflow.com/questions/51005147/keras-release-memory-after-finish-training-process
    """
    from numba import cuda
    K.clear_session()
    cuda.select_device(device)
    cuda.close()
    return
