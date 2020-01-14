from ..imports import *
from .. import utils as U


class LRFinder:
    """
    Tracks (and plots) the change in loss of a Keras model as learning rate is gradually increased.
    Used to visually identify a good learning rate, given model and data.
    Reference:
        Original Paper: https://arxiv.org/abs/1506.01186
    """
    def __init__(self, model, stop_factor=4):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9
        self._weightfile = None
        self.stop_factor = stop_factor

        self.avg_loss = 0
        self.batch_num = 0
        self.beta = 0.98

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.batch_num +=1
        self.avg_loss = self.beta * self.avg_loss + (1-self.beta) *loss
        smoothed_loss = self.avg_loss / (1 - self.beta**self.batch_num)
        self.losses.append(smoothed_loss)


        # Check whether the loss got too large or NaN
        if self.batch_num > 1 and smoothed_loss > self.stop_factor * self.best_loss:
            self.model.stop_training = True
            return

        # record best loss
        if smoothed_loss < self.best_loss or self.batch_num==1:
            self.best_loss = smoothed_loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

        # stop if LR grows too large
        if lr > 10.:
            self.model.stop_training = True
            return


    def find(self, train_data, steps_per_epoch, use_gen=False,
             start_lr=1e-7, lr_mult=1.01, max_epochs=None, 
             batch_size=U.DEFAULT_BS, workers=1, use_multiprocessing=False, verbose=1):
        """
        Track loss as learning rate is increased.
        NOTE: batch_size is ignored when train_data is instance of Iterator.
        """

        # check arguments and initialize
        if train_data is None:
            raise ValueError('train_data is required')
        #U.data_arg_check(train_data=train_data, train_required=True)
        self.lrs = []
        self.losses = []

         # compute steps_per_epoch
        #num_samples = U.nsamples_from_data(train_data)
        #if U.is_iter(train_data):
            #use_gen = True
            #steps_per_epoch = num_samples // train_data.batch_size
        #else:
            #use_gen = False
            #steps_per_epoch = np.ceil(num_samples/batch_size)

        # max_epochs and lr_mult are None, set max_epochs
        # using sample size of 1500 batches
        if max_epochs is None and lr_mult is None:
            max_epochs = int(np.ceil(1500./steps_per_epoch))

        if max_epochs:
            epochs = max_epochs
            num_batches = epochs * steps_per_epoch
            end_lr = 10 if start_lr < 10 else start_lr * 10
            self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)
        else:
            epochs = 1024
            self.lr_mult = lr_mult

        # Save weights into a file
        new_file, self._weightfile = tempfile.mkstemp()
        self.model.save_weights(self._weightfile)

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))


        if use_gen:
            if version.parse(tf.__version__) < version.parse('2.0'):
                fit_fn = self.model.fit_generator
            else:
                fit_fn = self.model.fit
            
            fit_fn(train_data, steps_per_epoch=steps_per_epoch, 
                   epochs=epochs, 
                   workers=workers, use_multiprocessing=use_multiprocessing,
                   verbose=verbose,
                   callbacks=[callback])
        else:
            self.model.fit(train_data[0], train_data[1],
                            batch_size=batch_size, epochs=epochs, verbose=verbose,
                            callbacks=[callback])


        # Restore the weights to the state before model fitting
        self.model.load_weights(self._weightfile)
        self._weightfile=None

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)


        return 


    def plot_loss(self, n_skip_beginning=10, n_skip_end=1):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            highlight - will highlight numerical estimate
                        of best lr if True
        """
        fig, ax = plt.subplots()
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        ax.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        return

        
    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)

