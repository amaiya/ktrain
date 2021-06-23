Module ktrain.lroptimize.lrfinder
=================================

Classes
-------

`LRFinder(model, stop_factor=4)`
:   Tracks (and plots) the change in loss of a Keras model as learning rate is gradually increased.
    Used to visually identify a good learning rate, given model and data.
    Reference:
        Original Paper: https://arxiv.org/abs/1506.01186

    ### Methods

    `estimate_lr(self)`
    :   Generates two numerical estimates of lr: 
          1. lr associated with minum numerical gradient (None if gradient computation fails)
          2. lr associated with minimum loss divided by 10
        Args:
          tuple: (float, float)
        
          If gradient computation fails, first element of tuple will be None.

    `find(self, train_data, steps_per_epoch, use_gen=False, class_weight=None, start_lr=1e-07, lr_mult=1.01, max_epochs=None, batch_size=32, workers=1, use_multiprocessing=False, verbose=1)`
    :   Track loss as learning rate is increased.
        NOTE: batch_size is ignored when train_data is instance of Iterator.

    `find_called(self)`
    :

    `on_batch_end(self, batch, logs)`
    :

    `plot_loss(self, n_skip_beginning=10, n_skip_end=1, suggest=False, return_fig=False)`
    :   Plots the loss.
        Args:
            n_skip_beginning(int): number of batches to skip on the left.
            n_skip_end(int):  number of batches to skip on the right.
            suggest(bool): will highlight numerical estimate
                           of best lr if True - methods adapted from fastai
            return_fig(bool):  If True, return matplotlib.figure.Figure
        Returns:
          matplotlib.figure.Figure if return_fig else None

    `plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01))`
    :   Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.