Module ktrain.text.shallownlp.classifier
========================================

Classes
-------

`NBSVM(alpha=1, C=1, beta=0.25, fit_intercept=False)`
:   Base class for all estimators in scikit-learn
    
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    ### Ancestors (in MRO)

    * sklearn.base.BaseEstimator
    * sklearn.linear_model._base.LinearClassifierMixin
    * sklearn.base.ClassifierMixin
    * sklearn.linear_model._base.SparseCoefMixin

    ### Methods

    `fit(self, X, y)`
    :