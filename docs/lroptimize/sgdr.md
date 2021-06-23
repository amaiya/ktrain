Module ktrain.lroptimize.sgdr
=============================

Classes
-------

`SGDRScheduler(min_lr, max_lr, steps_per_epoch, lr_decay=0.9, cycle_length=10, mult_factor=2)`
:   Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-7,
                                     max_lr=1e-1,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=1,
                                     mult_factor=2)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Original paper: http://arxiv.org/abs/1608.03983
        Blog Post:      http://www.jeremyjordan.me/nn-learning-rate/

    ### Ancestors (in MRO)

    * tensorflow.python.keras.callbacks.Callback

    ### Methods

    `clr(self)`
    :   Calculate the learning rate.

    `on_batch_end(self, batch, logs={})`
    :   Record previous batch statistics and update the learning rate.

    `on_epoch_begin(self, epoch, logs={})`
    :   Initialize the learning rate to the minimum value at the start of training.

    `on_epoch_end(self, epoch, logs={})`
    :   Check for end of current cycle, apply restarts when necessary.

    `on_train_begin(self, logs={})`
    :   Initialize the learning rate to the minimum value at the start of training.

    `on_train_end(self, logs={})`
    :   Set weights to the values from the end of the most recent cycle for best performance.