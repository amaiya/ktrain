from ..imports import *


class SGDRScheduler(keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler with periodic restarts.
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
    """

    def __init__(
        self,
        min_lr,
        max_lr,
        steps_per_epoch,
        lr_decay=0.9,
        cycle_length=10,
        mult_factor=2,
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        """Calculate the learning rate."""
        fraction_to_restart = self.batch_since_restart / (
            self.steps_per_epoch * self.cycle_length
        )
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(fraction_to_restart * np.pi)
        )
        return lr

    def on_train_begin(self, logs={}):
        """Initialize the learning rate to the minimum value at the start of training."""
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())
        # print(K.eval(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs={}):
        """Check for end of current cycle, apply restarts when necessary."""
        # print(K.eval(self.model.optimizer.lr))
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            # no longer needed as kauto completes cycles/epochs
            # self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        """Set weights to the values from the end of the most recent cycle for best performance."""
        # no longer needed as kauto completes cycles/epochs
        # self.model.set_weights(self.best_weights)
        pass

    def on_epoch_begin(self, epoch, logs={}):
        """Initialize the learning rate to the minimum value at the start of training."""
        logs = logs or {}
