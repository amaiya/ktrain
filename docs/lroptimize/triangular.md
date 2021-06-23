Module ktrain.lroptimize.triangular
===================================

Classes
-------

`CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000.0, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', reduce_on_plateau=0, monitor='val_loss', reduce_factor=2, max_momentum=0.95, min_momentum=0.85, verbose=1)`
:   This callback implements a cyclical learning rate policy (CLR).
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

    ### Ancestors (in MRO)

    * tensorflow.python.keras.callbacks.Callback

    ### Methods

    `clr(self)`
    :

    `on_batch_end(self, batch, logs=None)`
    :   A backwards compatibility alias for `on_train_batch_end`.

    `on_epoch_end(self, epoch, logs=None)`
    :   Called at the end of an epoch.
        
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        
        Arguments:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`.

    `on_train_begin(self, logs={})`
    :   Called at the beginning of training.
        
        Subclasses should override for any actions to run.
        
        Arguments:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.