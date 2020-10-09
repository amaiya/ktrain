from .imports import *

from .lroptimize.sgdr import *
from .lroptimize.triangular import *
from .lroptimize.lrfinder import *
from .lroptimize.optimization import AdamWeightDecay
from . import utils as U

from .vision.preprocessor import ImagePreprocessor
from .vision.predictor import ImagePredictor
from .text.preprocessor import TextPreprocessor, BERTPreprocessor, TransformersPreprocessor
from .text.predictor import TextPredictor
from .text.ner.predictor import NERPredictor
from .text.ner.preprocessor import NERPreprocessor
from .graph.predictor import NodePredictor, LinkPredictor
from .graph.preprocessor import NodePreprocessor, LinkPreprocessor
from .tabular.predictor import TabularPredictor
from .tabular.preprocessor import TabularPreprocessor


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
        try:
            new_file, weightfile = tempfile.mkstemp()
            self.model.save_weights(weightfile)
            self._original_weights = weightfile
        except:
            warnings.warn('Could not save original model weights')
            self._original_weights = None

    @property
    def _monitor_metrics(self):
        """
        monitor metrics
        """
        metrics = ['loss']
        try:
            m = U.metrics_from_model(self.model)
            if isinstance(m, list): metrics.extend(m)
        except:
            pass
        if self.val_data is not None:
            for m in metrics[:]:
                metrics.append('val_%s' % (m))
        return metrics


    def get_weight_decay(self):
        """
        Get current weight decay rate
        """
        if type(self.model.optimizer).__name__ == 'AdamWeightDecay':
            return self.model.optimizer.weight_decay_rate
        else:
            return None


    def set_weight_decay(self, wd=U.DEFAULT_WD):
        """
        Sets global weight decay via AdamWeightDecay optimizer
        Args:
          wd(float): weight decay
        Returns:
          None
              
        """
        self._recompile(wd=wd)
        return
        


    def evaluate(self, test_data=None, print_report=True, save_path='ktrain_classification_report.csv', class_names=[]):
        """
        alias for self.validate().
        Returns confusion matrix and optionally prints
        a classification report.
        This is currently only supported for binary and multiclass
        classification, not multilabel classification.

        By default, this uses val_data, as supplied to ktrain.get_learner().
        Other validation or test data can be optionally be supplied as argument via <test_data> argument.
        Supply class_names to include labels instead of intenger class integer values in classification report.
        Args:
          test_data(Dataset|np.ndarray): test or validation data.  If None, self.val_data is used.
          print_report(bool): If True, classification report will be printed. If False, report will be saved to CSV 
                              at save_path. Not applicable to regression models.
                              Not applicable to regression models.
          save_path(str): Classification report will be saved to this file path/name if print_report=False
                          Not applicable to regression models.
          class_names(list): list of class names to be used in classification report instead of 
                             class integer IDs.
        """
        return self.validate(val_data=test_data, print_report=print_report, save_path=save_path, class_names=class_names)



    def validate(self, val_data=None, 
                 print_report=True,
                 save_path='ktrain_classification_report.csv', 
                 class_names=[]):
        """
        Returns confusion matrix and optionally prints
        a classification report.
        This is currently only supported for binary and multiclass
        classification, not multilabel classification.

        By default, this uses val_data, as supplied to ktrain.get_learner().
        Other validation or test data can be optionally be supplied as argument.
        Supply class_names to include labels instead of intenger class integer values in classification report.
        Args:
          val_data(Dataset|np.ndarray): validation data.  If None, self.val_data is used.
          print_report(bool): If True, classification report will be printed. If False, report will be saved to CSV 
                              at save path. Not applicable to regression models.
          save_path(str): Classification report will be saved to this file path/name if print_report=False
          class_names(list): list of class names to be used in classification report instead of 
                             class integer IDs.
        """
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data

        classification, multilabel = U.is_classifier(self.model)
        if not classification:
            #warnings.warn('learner.validate is only for classification problems. ' 
                          #'For regression, etc., use learner.predict and learner.ground_truth '
                          #'to manually validate.')
            #return
            pass
            
        if U.is_multilabel(val) or multilabel:
            warnings.warn('multilabel confusion matrices not yet supported')
            return
        y_pred = self.predict(val_data=val)
        y_true = self.ground_truth(val_data=val)
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)


        # regression evaluation
        if not classification:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            regout = []
            metrics = U.metrics_from_model(self.model)
            for m in metrics:
                if m in ['mae', 'mean_absolute_error']:
                    regout.append( (m, mean_absolute_error(y_true,  y_pred)) )
                elif m in ['mse', 'mean_squared_error']:
                    regout.append( (m, mean_squared_error(y_true,  y_pred)) )
            if not regout:
                warnings.warn('%s is not supported by validate/evaluate - falling back to MAE')
                regout.append( ('mae', mean_absolute_error(y_true,  y_pred)) )
            return regout


        if len(y_pred.shape) == 1:
            y_pred = np.where(y_pred > 0.5, 1, 0)
            y_true = np.where(y_true > 0.5, 1, 0)
        else:
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
        if print_report or save_path is not None:
            if class_names:
                try:
                    class_names = [str(s) for s in class_names]
                except:
                    pass
                report = classification_report(y_true, y_pred, target_names=class_names, output_dict=not print_report)
            else:
                report = classification_report(y_true, y_pred, output_dict=not print_report)
            if print_report: 
                print(report)
            else:
                df = pd.DataFrame(report).transpose()
                df.to_csv(save_path)
                print('classification report saved to: %s' % (save_path))
            cm_func = confusion_matrix
        cm =  confusion_matrix(y_true,  y_pred)
        return cm


    def _check_val(self, val_data):
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if val is None: raise Exception('val_data must be supplied to get_learner or view_top_losses')
        return val


    def top_losses(self, n=4, val_data=None, preproc=None):
        """
        Computes losses on validation set sorted by examples with top losses
        Args:
          n(int or tuple): a range to select in form of int or tuple
                          e.g., n=8 is treated as n=(0,8)
          val_data:  optional val_data to use instead of self.val_data
          preproc (Preprocessor): A TextPreprocessor or ImagePreprocessor.
                                  For some data like text data, a preprocessor
                                  is required to undo the pre-processing
                                   to correctly view raw data.
        Returns:
            list of n tuples where first element is either 
            filepath or id of validation example and second element
            is loss.

        """


        # check validation data and arguments
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if val is None: raise Exception('val_data must be supplied to get_learner or top_losses')
        if type(n) == type(42):
            n = (0, n)


        #multilabel = True if U.is_multilabel(val) else False
        classification, multilabel = U.is_classifier(self.model)


        # get predicictions and ground truth
        y_pred = self.predict(val_data=val)
        y_true = self.ground_truth(val_data=val)
        y_true = y_true.astype('float32')

        # adjust y_true for regression problems
        if not classification and len(y_true.shape) == 1 and\
                (len(y_pred.shape) == 2 and y_pred.shape[1] == 1):
            y_true = np.expand_dims(y_true, -1)


        # compute loss
        # this doesn't work in tf.keras 1.14
        #losses = self.model.loss_functions[0](tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
        #if U.is_tf_keras():
            #L = self.model.loss_functions[0].fn
        #else:
            #L = self.model.loss_functions[0]
        L = U.loss_fn_from_model(self.model)
        losses = L(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
        if DISABLE_V2_BEHAVIOR:
            losses = tf.Session().run(losses)
        else:
            losses = losses.numpy()


        class_names = [] if preproc is None else preproc.get_classes()
        if preproc is None: 
            class_fcn = lambda x:"%s" % (x)
        else:
            class_fcn = lambda x:class_names[x]

        # regression output modifications
        if not classification:
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                y_pred = np.squeeze(y_pred)
                y_pred = np.around(y_pred, 2)
            if len(y_true.shape) == 2 and y_true.shape[1] == 1:
                y_true = np.squeeze(y_true)
                y_true = np.around(y_true, 2)

        # sort by loss and prune correct classifications, if necessary
        if classification and not multilabel:
            y_pred = np.squeeze(y_pred)
            y_true = np.squeeze(y_true)
            if len(y_pred.shape) == 1:
                y_p = np.where(y_pred > 0.5, 1, 0)
                y_t = np.where(y_true>0.5, 1, 0)
            else:
                y_p = np.argmax(y_pred, axis=1)
                y_t = np.argmax(y_true, axis=1)
            tups = [(i,x, class_fcn(y_t[i]), class_fcn(y_p[i])) for i, x in enumerate(losses) 
                     if y_p[i] != y_t[i]]
        else:
            tups = [(i,x, y_true[i], np.around(y_pred[i],2)) for i, x in enumerate(losses)]
        tups.sort(key=operator.itemgetter(1), reverse=True)

        # prune by given range
        tups = tups[n[0]:n[1]] if n is not None else tups
        return tups


    def view_top_losses(self, n=4, preproc=None, val_data=None):
        """
        View observations with top losses in validation set.
        Musta be overridden by Learner subclasses.
        """
        raise NotImplementedError('view_top_losses must be overriden by Learner subclass')


    def _make_model_folder(self, fpath):
        if os.path.isfile(fpath):
            raise ValueError(f'There is an existing file named {fpath}. ' +\
                              'Please use dfferent value for fpath.')
        elif os.path.exists(fpath):
            #warnings.warn('model is being saved to folder that already exists: %s' % (fpath))
            pass
        elif not os.path.exists(fpath):
            os.makedirs(fpath)


    def save_model(self, fpath):
        """
        a wrapper to model.save
        Args:
          fpath(str): path to folder in which to save model
        Returns:
          None
        """
        self._make_model_folder(fpath)
        self.model.save(os.path.join(fpath, U.MODEL_NAME), save_format='h5')
        return


    def load_model(self, fpath, custom_objects=None, **kwargs):
        """
        loads model from folder.
        Note: **kwargs included for backwards compatibility only, as TransformerTextClassLearner.load_model was removed in v0.18.0.
        Args:
          fpath(str): path to folder containing model
          custom_objects(dict): custom objects required to load model.
                                For models included with ktrain, this is populated automatically
                                and can be disregarded.
        
        """
        self.model = _load_model(fpath, train_data=self.train_data, custom_objects=custom_objects)
        return

    def _is_adamlike(self):
        """
        checks whether optimizer attached to model is an 
        "Adam-like" optimizer with beta_1 parameter.
        """
        return self.model is not None and hasattr(self.model.optimizer, 'beta_1')


    def _recompile(self, wd=None):
        # ISSUE: recompile does not work correctly with multigpu models
        if self.multigpu:
            err_msg = """
                   IMPORTANT: freeze and unfreeze methods do not currently work with 
                   multi-GPU models.  If you are using the load_imagemodel method to
                   define your model, please reload your model and use the freeze_layers 
                   argument of load_imagemodel to selectively freeze layers.
                   """
            raise Exception(err_msg)
        
        #if self.multigpu:
            #with tf.device("/cpu:0"):
                #metrics = [m.name for m in self.model.metrics] if U.is_tf_keras() else self.model.metrics
                #self.model.compile(optimizer=self.model.optimizer,
                                   #loss=self.model.loss,
                                   #metrics=metrics)
        metrics = U.metrics_from_model(self.model)
        if wd is not None and wd > 0 and type(self.model.optimizer).__name__ != 'AdamWeightDecay':
            warnings.warn('recompiling model to use AdamWeightDecay as opimizer with weight decay of %s' % (wd) )
            optimizer = U.get_default_optimizer(wd=wd)
        elif wd is not None and wd > 0:
            optimizer = U.get_default_optimizer(wd=wd)
        elif wd is not None and wd == 0:
            optimizer = U.DEFAULT_OPT
        else: # wd is None -> don't modify optimizer
            optimizer = self.model.optimizer
        self.model.compile(optimizer=optimizer,
                           loss=self.model.loss,
                           metrics=metrics)

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


    def reset_weights(self, verbose=1):
        """
        Re-initializes network with original weights
        """

        if os.path.isfile(self._original_weights):
            self.model.load_weights(self._original_weights)
            self.history = None
            U.vprint('Model weights have been reset.', verbose=verbose)
        else:
            warnings.warn('Weights have not been reset because the original weights file '+\
                          '(%s) no longer exists.' % (self._original_weights))
        return



    def lr_find(self, start_lr=1e-7, lr_mult=1.01, max_epochs=None, class_weight=None,
                stop_factor=4, show_plot=False, suggest=False, restore_weights_only=False, verbose=1):
        """
        Plots loss as learning rate is increased.  Highest learning rate 
        corresponding to a still falling loss should be chosen.

        If you find the LR finder is running for more epochs than you'd prefer,
        you can set max_epochs (e.g., max_epochs=5) to estimate LR with a 
        smaller sample size.

        If lr_mult is supplied and max_epochs is None, LR will increase until loss diverges.
        Reasonable values of lr_mult are between 1.01 and 1.05.

        If max_epochs is supplied, lr_mult argument is ignored and computed automatically.

        Reference: https://arxiv.org/abs/1506.01186

        Args:
            start_lr (float): smallest lr to start simulation
            lr_mult (float): multiplication factor to increase LR.
                             Ignored if max_epochs is supplied.
            max_epochs (int):  maximum number of epochs to simulate.
                               lr_mult is ignored if max_epoch is supplied.
                               Default is None. Set max_epochs to an integer
                               (e.g., 5) if lr_find is taking too long
                               and running for more epochs than desired.
            class_weight(dict): class_weight parameter passed to model.fit
                                for imbalanced datasets.
            stop_factor(int): factor used to determine threhsold that loss 
                              must exceed to stop training simulation.
                              Increase this if loss is erratic and lr_find
                              exits too early.
            show_plot (bool):  If True, automatically invoke lr_plot
            restore_weights_only(bool): If True, when training simulation is complete,
                                        the model weights only are restored, but not
                                        the original optimizer weights.  
                                        In at least a few cases, this seems to improve performance
                                        when actual training begins. Further investigation is needed,
                                        so it is False by default.
            verbose (bool): specifies how much output to print
        Returns:
            None
        """
        # dep_fix: bug in TF 2.2 and 2.3
        if version.parse(tf.__version__) > version.parse('2.1') and version.parse(tf.__version__) < version.parse('2.4'):
            if max_epochs is None:
                warnings.warn('Due to a bug in TensorFlow 2.2 and 2.3, the max_epochs argument is temporarily required. Please re-run with max_epochs. \n' +\
                              'More info: https://github.com/tensorflow/tensorflow/issues/41174#issuecomment-656330268')
                return


        U.vprint('simulating training for different learning rates... this may take a few moments...',
                verbose=verbose)
        # save current weights and temporarily restore original weights
        # dep_fix: temporarily use save_model instead of save_weights as default due to https://github.com/tensorflow/tensorflow/issues/41116
        _weights_only=True
        if restore_weights_only:
            new_file, weightfile = tempfile.mkstemp()
            self.model.save_weights(weightfile)
        else:
            temp_folder = tempfile.mkdtemp()
            self.save_model(temp_folder)


         # compute steps_per_epoch
        num_samples = U.nsamples_from_data(self.train_data)
        bs = self.train_data.batch_size if hasattr(self.train_data, 'batch_size') else self.batch_size
        if U.is_iter(self.train_data):
            use_gen = True
            steps_per_epoch = num_samples // bs
        else:
            use_gen = False
            steps_per_epoch = np.ceil(num_samples/bs)

        # check steps_per_epoch
        if steps_per_epoch <=64 and max_epochs is None:
            warnings.warn('max_epochs is being set to 5 since steps per epoch is small. ' +\
                          'If you wish to estimate LR using more epochs, set max_epochs manually.')
            max_epochs = 5


        try:
            # track and plot learning rates
            self.lr_finder = LRFinder(self.model, stop_factor=stop_factor)
            self.lr_finder.find(self._prepare(self.train_data), 
                                steps_per_epoch,
                                use_gen=use_gen,
                                start_lr=start_lr, lr_mult=lr_mult, 
                                max_epochs=max_epochs,
                                class_weight=class_weight,
                                workers=self.workers, 
                                use_multiprocessing=self.use_multiprocessing, 
                                batch_size=self.batch_size,
                                verbose=verbose)
        except KeyboardInterrupt:
            # re-load current weights
            #self.model.load_weights(weightfile)
            self.load_model(temp_folder)
            return

        # re-load current weights
        # dep_fix: temporarily use load_model instead of load_weights as default due to https://github.com/tensorflow/tensorflow/issues/41116
        if restore_weights_only:
            self.model.load_weights(weightfile)
        else:
            self.load_model(temp_folder)

        # instructions to invoker
        U.vprint('\n', verbose=verbose)
        U.vprint('done.', verbose=verbose)
        if show_plot:
            U.vprint('Visually inspect loss plot and select learning rate associated with falling loss', verbose=verbose)
            self.lr_plot()
        else:
            U.vprint('Please invoke the Learner.lr_plot() method to visually inspect '
                  'the loss plot to help identify the maximal learning rate '
                  'associated with falling loss.', verbose=verbose)
        return


    def lr_estimate(self):
        """
        Return numerical estimates of lr using two different methods:
            1. learning rate associated with minimum numerical gradient
            2. learning rate associated with minimum loss divided by 10
        Since neither of these methods are fool-proof and can 
        potentially return bad estimates, it is recommended that you 
        examine the plot generated by lr_plot to estimate the learning rate.
        Returns:
          tuple: tuple of the form (float, float), where 
            First element is lr associated with minimum numerical gradient (None if gradient computation fails).
            Second element is lr associated with minimum loss divided by 10.
        """
        if self.lr_finder is None or not self.lr_finder.find_called(): raise ValueError('Please call lr_find first.')
        return self.lr_finder.estimate_lr()
        


    def lr_plot(self, n_skip_beginning=10, n_skip_end=5, suggest=False, return_fig=False):
        """
        Plots the loss vs. learning rate to help identify
        The maximal learning rate associated with a falling loss.
        The nskip_beginning and n_skip_end arguments can be used
        to "zoom in" on the plot.
        Args:
            n_skip_beginning(int): number of batches to skip on the left.
            n_skip_end(int):  number of batches to skip on the right.
            suggest(bool): will highlight numerical estimate
                           of best lr if True - methods adapted from fastai
            return_fig(bool): If True, return matplotlib.figure.Figure
        Returns:
          matplotlib.figure.Figure if return_fig else None
          
        """
        # dep_fix: bug in TF 2.2 and 2.3
        if version.parse(tf.__version__) > version.parse('2.1') and version.parse(tf.__version__) < version.parse('2.4'):
            if n_skip_end == 5: n_skip_end=10

        if self.lr_finder is None or not self.lr_finder.find_called(): raise ValueError('Please call lr_find first.')
        return self.lr_finder.plot_loss(n_skip_beginning=n_skip_beginning,
                                        n_skip_end=n_skip_end, suggest=suggest, return_fig=return_fig)


    def plot(self, plot_type='loss', return_fig=False):
        """
        plots training history
        Args:
          plot_type (str):  one of {'loss', 'lr', 'momentum'}
          return_fig(bool):  If True, return matplotlib.figure.Figure
        Return:
          matplotlib.figure.Figure if return_fig else None
        """
        if self.history is None:
            raise Exception('No training history - did you train the model yet?')

        fig = None
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
        elif plot_type == 'lr':
            if 'lr' not in self.history.history:
                raise ValueError('no lr in history: are you sure you used autofit or fit_onecycle to train?')
            plt.plot(self.history.history['lr'])
            plt.title('LR Schedule')
            plt.ylabel('lr')
            plt.xlabel('iterations')
        elif plot_type == 'momentum':
            if 'momentum' not in self.history.history:
                raise ValueError('no momentum history: are you sure you used autofit or fit_onecycle to train?')
            plt.plot(self.history.history['momentum'])
            plt.title('Momentum Schedule')
            plt.ylabel('momentum')
            plt.xlabel('iterations')
        else:
            raise ValueError('invalid type: choose loss, lr, or momentum')
        fig = plt.gcf()
        plt.show()
        if return_fig: return fig
        return


    def print_layers(self, show_wd=False):
        """
        prints the layers of the model along with indices
        """
        if show_wd: warnings.warn('set_weight_decay now uses AdamWeightDecay instead of kernel_regularizers.')
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
        if callbacks and 'SGDRScheduler' in [type(cb).__name__ for cb in callbacks]: return callbacks
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
        if callbacks and 'ModelCheckpoint' in [type(cb).__name__ for cb in callbacks]: return callbacks
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            if not isinstance(callbacks, list): callbacks = []
            #filepath=os.path.join(folder, "weights-{epoch:02d}-{val_loss:.2f}.hdf5")
            filepath=os.path.join(folder, "weights-{epoch:02d}.hdf5")
            callbacks.append(ModelCheckpoint(filepath, save_best_only=False, save_weights_only=True))
        if not callbacks: callbacks=None
        return callbacks


    def _cb_earlystopping(self, early_stopping, callbacks=[]):
        if callbacks and 'EarlyStopping' in [type(cb).__name__ for cb in callbacks]: return callbacks
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


    def _prepare(self, data, train=True):
        """
        Subclasses can override this method if data
        needs to be specially-prepared prior to invoking fit methods
        Args:
          data:  dataset
          train(bool):  If True, prepare for training. Otherwise, prepare for evaluation.
        """
        if data is None: return None

        if hasattr(data, 'to_tfdataset'):
            return data.to_tfdataset(train=train)
        else:
            return data


    @abstractmethod
    def fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, batch_size=U.DEFAULT_BS):
        pass


    def fit_onecycle(self, lr, epochs, checkpoint_folder=None, 
                     cycle_momentum=True, max_momentum=0.95, min_momentum=0.85,
                     verbose=1, class_weight=None, callbacks=[]):
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
            max_momentum(float): Maximum momentum to use if cycle_momentum=True
            min_momentum(float): minimum momentum to use if cycle_momentum=True
            class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
            callbacks (list): list of Callback instances to employ during training
            verbose (bool):  verbose mode
        """
        if not self._is_adamlike() and cycle_momentum:
            warnings.warn('cyclical momentum has been disabled because '+\
                           'optimizer is not "Adam-like" with beta_1 param')
            cycle_momentum=False


        num_samples = U.nsamples_from_data(self.train_data)
        steps_per_epoch = math.ceil(num_samples/self.batch_size)

        # setup callbacks for learning rates and early stopping
        if not callbacks: kcallbacks = []
        else:
            kcallbacks = callbacks[:] 
        if cycle_momentum:
            max_momentum = max_momentum
            min_momentum = min_momentum
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
                        verbose=verbose, class_weight=class_weight, callbacks=kcallbacks)
        hist.history['lr'] = clr.history['lr']
        hist.history['iterations'] = clr.history['iterations']
        if cycle_momentum:
            hist.history['momentum'] = clr.history['momentum']
        self.history = hist
        return hist



    def autofit(self, lr, epochs=None, 
                early_stopping=None, reduce_on_plateau=None, reduce_factor=2, 
                cycle_momentum=True, max_momentum=0.95, min_momentum=0.85,
                monitor='val_loss', checkpoint_folder=None, verbose=1, 
                class_weight=None, callbacks=[]):
        """
        Automatically train model using a default learning rate schedule shown to work well
        in practice.  By default, this method currently employs a triangular learning 
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
            reduce_on_plateau (int):  If not None, will lower learning rate when
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
            max_momentum(float):  maximum momentum to use when cycle_momentum=True
            min_momentum(float): minimum momentum to use when cycle_momentum=True
            checkpoint_folder (string): Folder path in which to save the model weights 
                                        for each epoch.
                                        File name will be of the form: 
                                        weights-{epoch:02d}-{val_loss:.2f}.hdf5
            monitor (str):              what metric to monitor for early_stopping
                                        and reduce_on_plateau. Defaults to 'val_loss'.
                                        Only used if early_stopping or reduce_on_plateau
                                        is enabled.
            class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
            callbacks (list): list of Callback instances to employ during training
            verbose (bool):  verbose mode
        """
        # check optimizer
        if not self._is_adamlike() and cycle_momentum:
            warnings.warn('cyclical momentum has been disabled because '+\
                           'optimizer is not "Adam-like" with beta_1 param')
            cycle_momentum=False


        # setup learning rate policy 
        num_samples = U.nsamples_from_data(self.train_data)
        steps_per_epoch = math.ceil(num_samples/self.batch_size)
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

        # check monitor
        if reduce_on_plateau is not None or early_stopping is not None:
            if monitor.startswith('val_') and self.val_data is None:
                raise ValueError('monitor is %s but no val_data was supplied.\nChange monitor or supply val_data to get_learner function.' % monitor)
            if monitor != 'val_loss' and  monitor not in self._monitor_metrics:
                raise ValueError("monitor must be one of {%s}" % (self._monitor_metrics))


        # setup callbacks for learning rates and early stopping
        if not callbacks: kcallbacks = []
        else:
            kcallbacks = callbacks[:] 
        if cycle_momentum:
            max_momentum = max_momentum
            min_momentum = min_momentum
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
                        verbose=verbose, class_weight=class_weight, callbacks=kcallbacks)
        hist.history['lr'] = clr.history['lr']
        hist.history['iterations'] = clr.history['iterations']
        if cycle_momentum:
            hist.history['momentum'] = clr.history['momentum']
        self.history = hist
        return hist


    def ground_truth(self, val_data=None):
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if not val: raise Exception('val_data must be supplied to get_learner or ground_truth')
        return U.y_from_data(val)


    def predict(self, val_data=None):
        """
        Makes predictions on validation set
        """
        if val_data is not None:
            val = val_data
        else:
            val = self.val_data
        if val is None: raise Exception('val_data must be supplied to get_learner or predict')
        if U.is_iter(val):
            if hasattr(val, 'reset'): val.reset()
            steps = np.ceil(U.nsamples_from_data(val)/val.batch_size)
            # *_generator methods are deprecated from TF 2.1.0
            #result = self.model.predict_generator(self._prepare(val, train=False), 
                                                #steps=steps)
            result = self.model.predict(self._prepare(val, train=False), steps=steps)
            return result
        else:
            return self.model.predict(val[0], batch_size=self.eval_batch_size)

    

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
                 batch_size=U.DEFAULT_BS, eval_batch_size=U.DEFAULT_BS, 
                 workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, workers=workers, use_multiprocessing=use_multiprocessing, multigpu=multigpu)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        return

    
    def fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1, 
            lr_decay=1, checkpoint_folder = None, early_stopping=None,
            verbose=1, class_weight=None, callbacks=[]):
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
        class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
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

        # set call backs
        kcallbacks = callbacks if callbacks else None
        kcallbacks = self._cb_sgdr(lr, 
                                  np.ceil(len(x_train)/self.batch_size),
                                  cycle_len, cycle_mult, lr_decay, callbacks=kcallbacks)
        kcallbacks = self._cb_checkpoint(checkpoint_folder, callbacks=kcallbacks)
        kcallbacks = self._cb_earlystopping(early_stopping, callbacks=kcallbacks)
        sgdr = [cb for cb in kcallbacks if type(cb).__name__ == 'SGDRScheduler'] if kcallbacks else None
        sgdr = sgdr[0] if sgdr else None


        # train model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Check your callbacks.*')
            hist = self.model.fit(self._prepare(x_train), 
                                  self._prepare(y_train, train=False),
                                  batch_size=self.batch_size,
                                  epochs=epochs,
                                  validation_data=validation, verbose=verbose, 
                                  shuffle=True,
                                  class_weight=class_weight,
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

            obs = val[0][idx]
            join_char = ' '
            if preproc is not None: obs = preproc.undo(obs)
            if preproc is not None and isinstance(preproc, TextPreprocessor):
                if preproc.is_nospace_lang(): join_char = ''
            if type(obs) == str:
                obs = join_char.join(obs.split()[:512])
            print('----------')
            print("id:%s | loss:%s | true:%s | pred:%s)\n" % (idx, round(loss,2), truth, pred))
            print(obs)
        return



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
                 batch_size=U.DEFAULT_BS, eval_batch_size=U.DEFAULT_BS,
                 workers=1, use_multiprocessing=False, multigpu=False):
        super().__init__(model, workers=workers, use_multiprocessing=use_multiprocessing, multigpu=multigpu)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        if self.train_data:
            self.train_data.batch_size = batch_size
        if self.val_data:
            self.val_data.batch_size = eval_batch_size
        return

    
    def fit(self, lr, n_cycles, cycle_len=None, cycle_mult=1,
            lr_decay=1.0, checkpoint_folder=None, early_stopping=None, 
            class_weight=None, callbacks=[], verbose=1):
        """
        Trains the model. By default, fit is simply a wrapper for model.fit (for generators/sequences).
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
        class_weight (dict):       Optional dictionary mapping class indices (integers) to a weight (float) 
        callbacks (list):         list of Callback instances to employ during training
        verbose (boolean):       whether or not to print progress bar
        """
        # check early_stopping
        if self.val_data is None and early_stopping is not None:
            raise ValueError('early_stopping monitors val_loss but validation data not set')

        
        # handle callbacks
        num_samples = U.nsamples_from_data(self.train_data)
        train_bs = self.train_data.batch_size if hasattr(self.train_data, 'batch_size') else self.batch_size
        steps_per_epoch = math.ceil(num_samples/train_bs)
        validation_steps = None
        if self.val_data is not None:
            val_bs = self.val_data.batch_size if hasattr(self.val_data, 'batch_size') else self.batch_size
            validation_steps = math.ceil(U.nsamples_from_data(self.val_data)/val_bs)

        epochs = self._check_cycles(n_cycles, cycle_len, cycle_mult)
        self.set_lr(lr)


        # set call backs
        kcallbacks = callbacks if callbacks else None
        kcallbacks = self._cb_sgdr(lr, 
                                  steps_per_epoch,
                                  cycle_len, cycle_mult, lr_decay, callbacks=kcallbacks)
        kcallbacks = self._cb_checkpoint(checkpoint_folder, callbacks=kcallbacks)
        kcallbacks = self._cb_earlystopping(early_stopping, callbacks=kcallbacks)
        sgdr = [cb for cb in kcallbacks if type(cb).__name__ == 'SGDRScheduler'] if kcallbacks else None
        sgdr = sgdr[0] if sgdr else None
        #if kcallbacks: print([type(cb).__name__ for cb in kcallbacks])

            
        # MNIST times per epoch on Titan V
        # workers=4, usemp=True 9 sec.
        # workers=1, usemp=True 12 sec.
        # workers=1, usemp=False 16 sec.
        # workers=4, usemp=False 30+ sec.
        #print(self.workers)
        #print(self.use_multiprocessing)

        # train model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Check your callbacks.*')
            fit_fn = self.model.fit
            hist = fit_fn(self._prepare(self.train_data),
                                        steps_per_epoch = steps_per_epoch,
                                        validation_steps = validation_steps,
                                        epochs=epochs,
                                        validation_data=self._prepare(self.val_data, train=False),
                                        workers=self.workers,
                                        use_multiprocessing=self.use_multiprocessing, 
                                        verbose=verbose,
                                        shuffle=True,
                                        class_weight=class_weight,
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
        Uses first example (example_id=0) from first batch from training set, by default.
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


    #def view_top_losses(self, n=4, preproc=None, val_data=None):
    #    """
    #    Views observations with top losses in validation set.
    #    Musta be overridden by Learner subclasses.
    #    """
    #    raise NotImplementedError('view_top_losses must be overriden by GenLearner subclass')
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
        return



#------------------------------------------------------------------------------
# Predictor functions
#------------------------------------------------------------------------------

def get_predictor(model, preproc, batch_size=U.DEFAULT_BS):
    """
    Returns a Predictor instance that can be used to make predictions on
    unlabeled examples.  Can be saved to disk and reloaded as part of a 
    larger application.

    Args
        model (Model):        A compiled instance of keras.engine.training.Model
        preproc(Preprocessor):   An instance of TextPreprocessor,ImagePreprocessor,
                                 or NERPreprocessor.
                                 These instances are returned from the data loading
                                 functions in the ktrain vision and text modules:

                                 ktrain.vision.images_from_folder
                                 ktrain.vision.images_from_csv
                                 ktrain.vision.images_from_array
                                 ktrain.text.texts_from_folder
                                 ktrain.text.texts_from_csv
                                 ktrain.text.ner.entities_from_csv
        batch_size(int):    batch size to use.  default:32
    """

    # check arguments
    if not isinstance(model, Model):
        raise ValueError('model must be of instance Model')
    if not isinstance(preproc, (ImagePreprocessor,TextPreprocessor, NERPreprocessor, NodePreprocessor, LinkPreprocessor, TabularPreprocessor)):
        raise ValueError('preproc must be instance of ktrain.preprocessor.Preprocessor')
    if isinstance(preproc, ImagePreprocessor):
        return ImagePredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, TextPreprocessor):
    #elif type(preproc).__name__ == 'TextPreprocessor':
        return TextPredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, NERPreprocessor):
        return NERPredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, NodePreprocessor):
        return NodePredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, LinkPreprocessor):
        return LinkPredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, TabularPreprocessor):
        return TabularPredictor(model, preproc, batch_size=batch_size)

    else:
        raise Exception('preproc of type %s not currently supported' % (type(preproc)))


def load_predictor(fpath, batch_size=U.DEFAULT_BS):
    """
    Loads a previously saved Predictor instance
    Args
      fpath(str): predictor path name (value supplied to predictor.save)
                  From v0.16.x, this is always the path to a folder.
                  Pre-v0.16.x, this is the base name used to save model and .preproc instance.
      batch_size(int): batch size to use for predictions. default:32
    """

    # load the preprocessor
    preproc = None
    try:
        preproc_name = os.path.join(fpath, U.PREPROC_NAME)
        with open(preproc_name, 'rb') as f: preproc = pickle.load(f)
    except:
        try:
            preproc_name = fpath +'.preproc'
            #warnings.warn('could not load .preproc file as %s - attempting to load as %s' % (os.path.join(fpath, U.PREPROC_NAME), preproc_name))
            with open(preproc_name, 'rb') as f: preproc = pickle.load(f)
        except:
            raise Exception('Could not find a .preproc file in either the post v0.16.x loction (%s) or pre v0.16.x location (%s)' % (os.path.join(fpath, U.PREPROC_NAME), fpath+'.preproc'))

    # load the model
    model = _load_model(fpath, preproc=preproc)


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
    
    # return the appropriate predictor
    if not isinstance(model, Model):
        raise ValueError('model must be of instance Model')
    if not isinstance(preproc, (ImagePreprocessor, TextPreprocessor, NERPreprocessor, NodePreprocessor, LinkPreprocessor, TabularPreprocessor)):
        raise ValueError('preproc must be instance of ktrain.preprocessor.Preprocessor')
    if isinstance(preproc, ImagePreprocessor):
        return ImagePredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, TextPreprocessor):
        return TextPredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, NERPreprocessor):
        return NERPredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, NodePreprocessor):
        return NodePredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, LinkPreprocessor):
        return LinkPredictor(model, preproc, batch_size=batch_size)
    elif isinstance(preproc, TabularPreprocessor):
        return TabularPredictor(model, preproc, batch_size=batch_size)
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


def _load_model(fpath, preproc=None, train_data=None, custom_objects=None):
    if not preproc and not train_data:
        raise ValueError('Either preproc or train_data is required.')
    if (preproc and isinstance(preproc, TransformersPreprocessor)) or \
       (train_data and U.is_huggingface(data=train_data)):
        if preproc:
            model = preproc.get_model(fpath=fpath)
        else:
            model = TransformersPreprocessor.load_model_and_configure_from_data(fpath, train_data)
        return model
    elif (preproc and (isinstance(preproc, BERTPreprocessor) or \
                    type(preproc).__name__ == 'BERTPreprocessor')) or\
       train_data and U.bert_data_tuple(train_data):
        # custom BERT model
        from keras_bert import get_custom_objects
        custom_objects = get_custom_objects()
    elif (preproc and (isinstance(preproc, NERPreprocessor) or \
                    type(preproc).__name__ == 'NERPreprocessor')) or \
        train_data and U.is_ner(data=train_data):
        from .text.ner.anago.layers import CRF
        from .text.ner import crf_loss
        custom_objects={'CRF': CRF, 'crf_loss':crf_loss}
    elif (preproc and (isinstance(preproc, NodePreprocessor) or \
                    type(preproc).__name__ == 'NodePreprocessor')) or \
        train_data and U.is_nodeclass(data=train_data):
        from stellargraph.layer import MeanAggregator
        custom_objects={'MeanAggregator': MeanAggregator}
    elif (preproc and (isinstance(preproc, LinkPreprocessor) or \
                    type(preproc).__name__ == 'LinkPreprocessor')) or \
        train_data and U.is_linkpred(data=train_data):
        from stellargraph.layer import MeanAggregator
        custom_objects={'MeanAggregator': MeanAggregator}
    custom_objects = {} if custom_objects is None else custom_objects
    custom_objects['AdamWeightDecay'] = AdamWeightDecay
    try:
        try:
            model = load_model(os.path.join(fpath, U.MODEL_NAME), custom_objects=custom_objects)
        except:
            try:
                # pre-0.16: model fpath was file name of model not folder for non-Transformer models
                #warnings.warn('could not load model as %s - attempting to load model as %s' % (os.path.join(fpath, U.MODEL_NAME), fpath))
                model = load_model(fpath, custom_objects=custom_objects)
            except:
                # for bilstm models without CRF layer on TF2 where CRF is not supported 
                model = load_model(fpath, custom_objects={'AdamWeightDecay':AdamWeightDecay})
    except Exception as e:
        print('Call to keras.models.load_model failed.  '
              'Try using the learner.model.save_weights and '
              'learner.model.load_weights instead.')
        print('Error was: %s' % (e))
        return

    # see issue https://github.com/amaiya/ktrain/issues/21
    if hasattr(model, '_make_predict_function'):
        model._make_predict_function()

    return model



