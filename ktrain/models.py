from .imports import *

def bn_drop_lin(inp, n_out,  bn=True, p=0., actn=None):
    out = inp
    if bn:
        out = BatchNormalization()(out)
    if p>0:
        out = Dropout(p)(out)
    out = Dense(n_out, activation=actn)(out)
    return out

