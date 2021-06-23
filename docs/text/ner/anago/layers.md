Module ktrain.text.ner.anago.layers
===================================

Functions
---------

    
`crf_accuracy(y_true, y_pred)`
:   Ge default accuracy based on CRF `test_mode`.

    
`crf_loss(y_true, y_pred)`
:   General CRF loss function depending on the learning mode.
    
    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.
    
    # Returns
        If the CRF layer is being trained in the join mode, returns the negative
        log-likelihood. Otherwise returns the categorical crossentropy implemented
        by the underlying Keras backend.
    
    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.

    
`crf_marginal_accuracy(y_true, y_pred)`
:   Use time-wise marginal argmax as prediction.
    `y_pred` must be an output from CRF with `learn_mode="marginal"`.

    
`crf_nll(y_true, y_pred)`
:   The negative log-likelihood for linear chain Conditional Random Field (CRF).
    
    This loss function is only used when the `layers.CRF` layer
    is trained in the "join" mode.
    
    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.
    
    # Returns
        A scalar representing corresponding to the negative log-likelihood.
    
    # Raises
        TypeError: If CRF is not the last layer.
    
    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.

    
`crf_viterbi_accuracy(y_true, y_pred)`
:   Use Viterbi algorithm to get best path, and compute its accuracy.
    `y_pred` must be an output from CRF.

    
`to_tuple(shape)`
:   This functions is here to fix an inconsistency between keras and tf.keras.
    
    In tf.keras, the input_shape argument is an tuple with `Dimensions` objects.
    In keras, the input_shape is a simple tuple of ints or `None`.
    
    We'll work with tuples of ints or `None` to be consistent
    with keras-team/keras. So we must apply this function to
    all input_shapes of the build methods in custom layers.

Classes
-------

`CRF(units, learn_mode='join', test_mode=None, sparse_target=False, use_boundary=True, use_bias=True, activation='linear', kernel_initializer='glorot_uniform', chain_initializer='orthogonal', bias_initializer='zeros', boundary_initializer='zeros', kernel_regularizer=None, chain_regularizer=None, boundary_regularizer=None, bias_regularizer=None, kernel_constraint=None, chain_constraint=None, boundary_constraint=None, bias_constraint=None, input_dim=None, unroll=False, **kwargs)`
:   An implementation of linear chain conditional random field (CRF).
    
    An linear chain CRF is defined to maximize the following likelihood function:
    
    $$ L(W, U, b; y_1, ..., y_n) := rac{1}{Z}
    \sum_{y_1, ..., y_n} \exp(-a_1' y_1 - a_n' y_n
        - \sum_{k=1^n}((f(x_k' W + b) y_k) + y_1' U y_2)), $$
    
    where:
        $Z$: normalization constant
        $x_k, y_k$:  inputs and outputs
    
    This implementation has two modes for optimization:
    1. (`join mode`) optimized by maximizing join likelihood,
    which is optimal in theory of statistics.
       Note that in this case, CRF must be the output/last layer.
    2. (`marginal mode`) return marginal probabilities on each time
    step and optimized via composition
       likelihood (product of marginal likelihood), i.e.,
       using `categorical_crossentropy` loss.
       Note that in this case, CRF can be either the last layer or an
       intermediate layer (though not explored).
    
    For prediction (test phrase), one can choose either Viterbi
    best path (class indices) or marginal
    probabilities if probabilities are needed.
    However, if one chooses *join mode* for training,
    Viterbi output is typically better than marginal output,
    but the marginal output will still perform
    reasonably close, while if *marginal mode* is used for training,
    marginal output usually performs
    much better. The default behavior and `metrics.crf_accuracy`
    is set according to this observation.
    
    In addition, this implementation supports masking and accepts either
    onehot or sparse target.
    
    If you open a issue or a pull request about CRF, please
    add 'cc @lzfelix' to notify Luiz Felix.
    
    
    # Examples
    
    ```python
        from keras_contrib.layers import CRF
        from keras_contrib.losses import crf_loss
        from keras_contrib.metrics import crf_viterbi_accuracy
    
        model = Sequential()
        model.add(Embedding(3001, 300, mask_zero=True)(X)
    
        # use learn_mode = 'join', test_mode = 'viterbi',
        # sparse_target = True (label indice output)
        crf = CRF(10, sparse_target=True)
        model.add(crf)
    
        # crf_accuracy is default to Viterbi acc if using join-mode (default).
        # One can add crf.marginal_acc if interested, but may slow down learning
        model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    
        # y must be label indices (with shape 1 at dim 3) here,
        # since `sparse_target=True`
        model.fit(x, y)
    
        # prediction give onehot representation of Viterbi best path
        y_hat = model.predict(x_test)
    ```
    
    The following snippet shows how to load a persisted
    model that uses the CRF layer:
    
    ```python
        from tensorflow.keras.models import load_model
        from keras_contrib.losses import import crf_loss
        from keras_contrib.metrics import crf_viterbi_accuracy
    
        custom_objects={'CRF': CRF,
                        'crf_loss': crf_loss,
                        'crf_viterbi_accuracy': crf_viterbi_accuracy}
    
        loaded_model = load_model('<path_to_model>',
                                  custom_objects=custom_objects)
    ```
    
    # Arguments
        units: Positive integer, dimensionality of the output space.
        learn_mode: Either 'join' or 'marginal'.
            The former train the model by maximizing join likelihood while the latter
            maximize the product of marginal likelihood over all time steps.
            One should use `losses.crf_nll` for 'join' mode
            and `losses.categorical_crossentropy` or
            `losses.sparse_categorical_crossentropy` for
            `marginal` mode.  For convenience, simply
            use `losses.crf_loss`, which will decide the proper loss as described.
        test_mode: Either 'viterbi' or 'marginal'.
            The former is recommended and as default when `learn_mode = 'join'` and
            gives one-hot representation of the best path at test (prediction) time,
            while the latter is recommended and chosen as default
            when `learn_mode = 'marginal'`,
            which produces marginal probabilities for each time step.
            For evaluating metrics, one should
            use `metrics.crf_viterbi_accuracy` for 'viterbi' mode and
            'metrics.crf_marginal_accuracy' for 'marginal' mode, or
            simply use `metrics.crf_accuracy` for
            both which automatically decides it as described.
            One can also use both for evaluation at training.
        sparse_target: Boolean (default False) indicating
            if provided labels are one-hot or
            indices (with shape 1 at dim 3).
        use_boundary: Boolean (default True) indicating if trainable
            start-end chain energies
            should be added to model.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        chain_initializer: Initializer for the `chain_kernel` weights matrix,
            used for the CRF chain energy.
            (see [initializers](../initializers.md)).
        boundary_initializer: Initializer for the `left_boundary`,
            'right_boundary' weights vectors,
            used for the start/left and end/right boundary energy.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        chain_regularizer: Regularizer function applied to
            the `chain_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        boundary_regularizer: Regularizer function applied to
            the 'left_boundary', 'right_boundary' weight vectors
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        chain_constraint: Constraint function applied to
            the `chain_kernel` weights matrix
            (see [constraints](../constraints.md)).
        boundary_constraint: Constraint function applied to
            the `left_boundary`, `right_boundary` weights vectors
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        unroll: Boolean (default False). If True, the network will be
            unrolled, else a symbolic loop will be used.
            Unrolling can speed-up a RNN, although it tends
            to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    
    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.
    
    # Output shape
        3D tensor with shape `(nb_samples, timesteps, units)`.
    
    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    ### Ancestors (in MRO)

    * tensorflow.python.keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.training.tracking.tracking.AutoTrackable
    * tensorflow.python.training.tracking.base.Trackable
    * tensorflow.python.keras.utils.version_utils.LayerVersionSelector

    ### Static methods

    `shift_left(x, offset=1)`
    :

    `shift_right(x, offset=1)`
    :

    `softmaxNd(x, axis=-1)`
    :

    ### Instance variables

    `accuracy`
    :

    `loss_function`
    :

    `marginal_acc`
    :

    `viterbi_acc`
    :

    ### Methods

    `add_boundary_energy(self, energy, mask, start, end)`
    :

    `backward_recursion(self, input_energy, **kwargs)`
    :

    `build(self, input_shape)`
    :   Creates the variables of the layer (optional, for subclass implementers).
        
        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.
        
        This is typically used to create the weights of `Layer` subclasses.
        
        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

    `call(self, X, mask=None)`
    :   This is where the layer's logic lives.
        
        Note here that `call()` method in `tf.keras` is little bit different
        from `keras` API. In `keras` API, you can pass support masking for
        layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
        method to support masking.
        
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments. Currently unused.
        
        Returns:
            A tensor or list/tuple of tensors.

    `compute_mask(self, input, mask=None)`
    :   Computes an output mask tensor.
        
        Arguments:
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.
        
        Returns:
            None or a tensor (or list of tensors,
                one per output tensor of the layer).

    `compute_output_shape(self, input_shape)`
    :   Computes the output shape of the layer.
        
        If the layer has not been built, this method will call `build` on the
        layer. This assumes that the layer will later be used with inputs that
        match the input shape provided here.
        
        Arguments:
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.
        
        Returns:
            An input shape tuple.

    `forward_recursion(self, input_energy, **kwargs)`
    :

    `get_config(self)`
    :   Returns the config of the layer.
        
        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).
        
        Returns:
            Python dictionary.

    `get_energy(self, y_true, input_energy, mask)`
    :   Energy = a1' y1 + u1' y1 + y1' U y2 + u2' y2 + y2' U y3 + u3' y3 + an' y3

    `get_log_normalization_constant(self, input_energy, mask, **kwargs)`
    :   Compute logarithm of the normalization constant Z, where
        Z = sum exp(-E) -> logZ = log sum exp(-E) =: -nlogZ

    `get_marginal_prob(self, X, mask=None)`
    :

    `get_negative_log_likelihood(self, y_true, X, mask)`
    :   Compute the loss, i.e., negative log likelihood (normalize by number of time steps)
        likelihood = 1/Z * exp(-E) ->  neg_log_like = - log(1/Z * exp(-E)) = logZ + E

    `recursion(self, input_energy, mask=None, go_backwards=False, return_sequences=True, return_logZ=True, input_length=None)`
    :   Forward (alpha) or backward (beta) recursion
        
        If `return_logZ = True`, compute the logZ, the normalization constant:
        
        \[ Z = \sum_{y1, y2, y3} exp(-E) # energy
          = \sum_{y1, y2, y3} exp(-(u1' y1 + y1' W y2 + u2' y2 + y2' W y3 + u3' y3))
          = sum_{y2, y3} (exp(-(u2' y2 + y2' W y3 + u3' y3))
          sum_{y1} exp(-(u1' y1' + y1' W y2))) \]
        
        Denote:
            \[ S(y2) := sum_{y1} exp(-(u1' y1 + y1' W y2)), \]
            \[ Z = sum_{y2, y3} exp(log S(y2) - (u2' y2 + y2' W y3 + u3' y3)) \]
            \[ logS(y2) = log S(y2) = log_sum_exp(-(u1' y1' + y1' W y2)) \]
        Note that:
              yi's are one-hot vectors
              u1, u3: boundary energies have been merged
        
        If `return_logZ = False`, compute the Viterbi's best path lookup table.

    `step(self, input_energy_t, states, return_logZ=True)`
    :

    `viterbi_decoding(self, X, mask=None)`
    :