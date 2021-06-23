Module ktrain.lroptimize.optimization
=====================================
Functions and classes related to optimization (weight updates).

Functions
---------

    
`create_optimizer(init_lr: float, num_train_steps: int, num_warmup_steps: int, min_lr_ratio: float = 0.0, adam_epsilon: float = 1e-08, weight_decay_rate: float = 0.0, include_in_weight_decay: Union[List[str], NoneType] = None)`
:   Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.
    Args:
        init_lr (:obj:`float`):
            The desired learning rate at the end of the warmup phase.
        num_train_step (:obj:`int`):
            The total number of training steps.
        num_warmup_steps (:obj:`int`):
            The number of warmup steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0):
            The final learning rate at the end of the linear decay will be :obj:`init_lr * min_lr_ratio`.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon to use in Adam.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to use.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters except bias and layer norm parameters.

Classes
-------

`AdamWeightDecay(learning_rate: Union[float, tensorflow.python.keras.optimizer_v2.learning_rate_schedule.LearningRateSchedule] = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-07, amsgrad: bool = False, weight_decay_rate: float = 0.0, include_in_weight_decay: Union[List[str], NoneType] = None, exclude_from_weight_decay: Union[List[str], NoneType] = None, name: str = 'AdamWeightDecay', **kwargs)`
:   Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in
    `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`__.
    Instead we want ot decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.
    Args:
        learning_rate (:obj:`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`, defaults to 1e-3):
            The learning rate to use or a schedule.
        beta_1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (:obj:`float`, `optional`, defaults to 1e-7):
            The epsilon paramenter in Adam, which is a small constant for numerical stability.
        amsgrad (:obj:`bool`, `optional`, default to `False`):
            Wheter to apply AMSGrad varient of this algorithm or not, see
            `On the Convergence of Adam and Beyond <https://arxiv.org/abs/1904.09237>`__.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in :obj:`exclude_from_weight_decay`).
        exclude_from_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            :obj:`include_in_weight_decay` is passed, the names in it will supersede this list.
        name (:obj:`str`, `optional`, defaults to 'AdamWeightDecay'):
            Optional name for the operations created when applying gradients.
        kwargs:
            Keyward arguments. Allowed to be {``clipnorm``, ``clipvalue``, ``lr``, ``decay``}. ``clipnorm`` is clip
            gradients by norm; ``clipvalue`` is clip gradients by value, ``decay`` is included for backward
            compatibility to allow time inverse decay of learning rate. ``lr`` is included for backward compatibility,
            recommended to use ``learning_rate`` instead.
    
    Create a new Optimizer.
    
    This must be called by the constructors of subclasses.
    Note that Optimizer instances should not bind to a single graph,
    and so shouldn't keep Tensors as member variables. Generally
    you should be able to use the _set_hyper()/state.get_hyper()
    facility instead.
    
    This class in stateful and thread-compatible.
    
    Args:
      name: A non-empty string.  The name to use for accumulators created
        for the optimizer.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    
    Raises:
      ValueError: If name is malformed.

    ### Ancestors (in MRO)

    * tensorflow.python.keras.optimizer_v2.adam.Adam
    * tensorflow.python.keras.optimizer_v2.optimizer_v2.OptimizerV2
    * tensorflow.python.training.tracking.base.Trackable

    ### Static methods

    `from_config(config)`
    :   Creates an optimizer from its config with WarmUp custom object.

    ### Methods

    `apply_gradients(self, grads_and_vars, name=None, **kwargs)`
    :   Apply gradients to variables.
        
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        
        The method sums gradients from all replicas in the presence of
        `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
        passing `experimental_aggregate_gradients=False`.
        
        Example:
        
        ```python
        grads = tape.gradient(loss, vars)
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
        # Processing aggregated gradients.
        optimizer.apply_gradients(zip(grads, vars),
            experimental_aggregate_gradients=False)
        
        ```
        
        Args:
          grads_and_vars: List of (gradient, variable) pairs.
          name: Optional name for the returned operation. Default to the name passed
            to the `Optimizer` constructor.
          experimental_aggregate_gradients: Whether to sum gradients from different
            replicas in the presense of `tf.distribute.Strategy`. If False, it's
            user responsibility to aggregate the gradients. Default to True.
        
        Returns:
          An `Operation` that applies the specified gradients. The `iterations`
          will be automatically increased by 1.
        
        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.

    `get_config(self)`
    :   Returns the config of the optimizer.
        
        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.
        
        Returns:
            Python dictionary.

`GradientAccumulator()`
:   Gradient accumulation utility.
    When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and
    without synchronization. Users should then call ``.gradients``, scale the
    gradients if required, and pass the result to ``apply_gradients``.
    
    Initializes the accumulator.

    ### Instance variables

    `gradients`
    :   The accumulated gradients on the current replica.

    `step`
    :   Number of accumulated steps.

    ### Methods

    `reset(self)`
    :   Resets the accumulated gradients on the current replica.

`WarmUp(initial_learning_rate: float, decay_schedule_fn: Callable, warmup_steps: int, power: float = 1.0, name: str = None)`
:   Applies a warmup schedule on a given learning rate decay schedule.
    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.

    ### Ancestors (in MRO)

    * tensorflow.python.keras.optimizer_v2.learning_rate_schedule.LearningRateSchedule

    ### Methods

    `get_config(self)`
    :