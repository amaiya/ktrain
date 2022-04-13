from .imports import *


def bn_drop_lin(inp, n_out, bn=True, p=0.0, actn=None):
    out = inp
    if bn:
        out = keras.layers.BatchNormalization()(out)
    if p > 0:
        out = keras.layers.Dropout(p)(out)
    use_bias = False if bn else True
    out = keras.layers.Dense(n_out, activation=actn, use_bias=use_bias)(out)
    return out
