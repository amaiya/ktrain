Module ktrain.vision.wrn
========================

Functions
---------

    
`conv1_block(input, k=1, dropout=0.0)`
:   

    
`conv2_block(input, k=1, dropout=0.0)`
:   

    
`conv3_block(input, k=1, dropout=0.0)`
:   

    
`create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, activation='softmax', dropout=0.0, verbose=1)`
:   Creates a Wide Residual Network with specified parameters
    
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:

    
`expand_conv(init, base, k, strides=(1, 1))`
:   

    
`initial_conv(input)`
: