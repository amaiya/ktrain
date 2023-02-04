from .. import utils as U
from ..imports import *


class CyclicLR(keras.callbacks.Callback):
    """
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
        reduce_on_plateau (int): LR will be reduced after this many
                                 epochs with no improvement on validation loss.
                                 If zero or None, no reduction will take place
        reduce_factor(int):      LR is reduced by this factor (e.g., 2 = 1/2  = 0.5)
        monitor (str):           Value to monitor when reducing LR
        max_momentum(float):     maximum momentum when momentum is cycled
                                 If both max_momentum and min_momentum is None,
                                 default momentum for Adam is used.
                                 (only used if optimizer is Adam)
        min_momentum(float):     minimum momentum when momentum is cycled
                                 If both max_momentum and min_momentum is None,
                                 default momentum for Adam is used.
                                 (only used if optimizer is Adam)
        verbose (bool):          If True, will print information on LR reduction
    References:
        Original Paper: https://arxiv.org/abs/1803.09820
        Blog Post: https://sgugger.github.io/the-1cycle-policy.html
        Code Reference: https://github.com/bckenstler/CLR
    """

    def __init__(
        self,
        base_lr=0.001,
        max_lr=0.006,
        step_size=2000.0,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        reduce_on_plateau=0,
        monitor="val_loss",
        reduce_factor=2,
        max_momentum=0.95,
        min_momentum=0.85,
        verbose=1,
    ):
        super(keras.callbacks.Callback, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.0
        self.trn_iterations = 0.0
        self.history = {}

        # restoring weights due to CRF bug
        self.best_weights = None

        # LR reduction
        self.verbose = verbose
        self.patience = reduce_on_plateau
        self.factor = 1.0 / reduce_factor
        self.monitor = monitor
        if "acc" not in self.monitor:
            self.monitor_op = lambda a, b: np.less(a, b)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b)
            self.best = -np.Inf

        # annihalting LR
        self.overhump = False

        # cyclical momentum
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        if self.min_momentum is None and self.max_momentum:
            self.min_momentum = self.max_momentum
        elif self.min_momentum and self.max_momentum is None:
            self.max_momentum = self.min_momentum
        self.cycle_momentum = True if self.max_momentum is not None else False

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

        self.orig_base_lr = self.base_lr

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault("iterations", []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

        # annihilate learning rate
        prev_overhump = self.overhump
        iterations = (self.clr_iterations + 1) % (self.step_size * 2)
        if iterations / self.step_size > 1:
            self.overhump = True
        else:
            self.overhump = False
        if not prev_overhump and self.overhump:
            self.base_lr = self.max_lr / 1000
        elif prev_overhump and not self.overhump:
            self.base_lr = self.orig_base_lr

        # set momentum
        if self.cycle_momentum:
            if self.overhump:
                current_percentage = 1.0 - (
                    (iterations - self.step_size) / float(self.step_size)
                )
                new_momentum = self.max_momentum - current_percentage * (
                    self.max_momentum - self.min_momentum
                )
            else:
                current_percentage = iterations / float(self.step_size)
                new_momentum = self.max_momentum - current_percentage * (
                    self.max_momentum - self.min_momentum
                )
            K.set_value(self.model.optimizer.beta_1, new_momentum)
            self.history.setdefault("momentum", []).append(
                K.get_value(self.model.optimizer.beta_1)
            )

    def on_epoch_end(self, epoch, logs=None):
        # print(K.eval(self.model.optimizer.lr))

        # Stop training if training loss becomes zero or negative
        # to address bug in keras_contrib code for CRF.
        # We restore the weights from previous best epoch
        # rather than this epoch.
        crf = U.is_crf(self.model)
        if crf:
            current_loss = logs.get("loss")
            current_val_loss = logs.get("val_loss", None)
            if (current_loss is not None and current_loss <= 0.0) or (
                current_val_loss is not None and current_val_loss <= 0.0
            ):
                self.model.stop_training = True
                if crf and self.best_weights is not None:
                    if self.verbose > 0:
                        print(
                            "Restoring model weights from the end of " "the best epoch"
                        )
                    self.model.set_weights(self.best_weights)
                return

        if self.patience:
            current = logs.get(self.monitor)
            if current is None:
                raise Exception("cannot monitor %s" % (self.monitor))
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                if crf:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    min_lr = 1e-7
                    current_lr = float(K.get_value(self.model.optimizer.lr))
                    if self.max_lr > min_lr:
                        self.base_lr = self.base_lr * self.factor
                        self.max_lr = self.max_lr * self.factor
                        new_lr = current_lr * self.factor
                        new_lr = max(new_lr, min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose:
                            print(
                                "\nEpoch %05d: Reducing Max LR on Plateau: "
                                "new max lr will be %s (if not early_stopping)."
                                % (epoch + 1, self.max_lr)
                            )
                        self.wait = 0
