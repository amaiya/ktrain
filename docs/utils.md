Module ktrain.utils
===================

Functions
---------

    
`add_headers_to_df(fname_in, header_dict, fname_out=None)`
:   

    
`bad_data_tuple(data)`
:   Checks for standard tuple or BERT-style tuple

    
`bert_data_tuple(data)`
:   checks if data tuple is BERT-style format

    
`check_array(X, y=None, X_name='X', y_name='targets')`
:   

    
`data_arg_check(train_data=None, val_data=None, train_required=False, val_required=False, ndarray_only=False)`
:   

    
`download(url, filename)`
:   

    
`get_default_optimizer(lr=0.001, wd=0.01)`
:   

    
`get_hf_model_name(model_id)`
:   

    
`get_ktrain_data()`
:   

    
`get_random_colors(n, name='hsv', hex_format=True)`
:   Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    
`is_classifier(model)`
:   checks for classification and mutlilabel from model

    
`is_crf(model)`
:   checks for CRF sequence tagger.

    
`is_huggingface(model=None, data=None)`
:   check for hugging face transformer model
    from  model and/or data

    
`is_huggingface_from_data(data)`
:   

    
`is_huggingface_from_model(model)`
:   

    
`is_imageclass_from_data(data)`
:   

    
`is_iter(data, ignore=False)`
:   

    
`is_linkpred(model=None, data=None)`
:   

    
`is_multilabel(data)`
:   checks for multilabel from data

    
`is_ner(model=None, data=None)`
:   

    
`is_ner_from_data(data)`
:   

    
`is_nodeclass(model=None, data=None)`
:   

    
`is_regression_from_data(data)`
:   checks for regression task from data

    
`is_tabular_from_data(data)`
:   

    
`is_tf_keras()`
:   

    
`list2chunks(a, n)`
:   

    
`loss_fn_from_model(model)`
:   

    
`metrics_from_model(model)`
:   

    
`nclasses_from_data(data)`
:   

    
`nsamples_from_data(data)`
:   

    
`ondisk(data)`
:   

    
`plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=<matplotlib.colors.LinearSegmentedColormap object>)`
:   This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    
`plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None)`
:   

    
`shape_from_data(data)`
:   

    
`vprint(s=None, verbose=1)`
:   

    
`y_from_data(data)`
:   

Classes
-------

`YTransform(class_names=[], label_encoder=None)`
:   Cheks and transforms array of targets. Targets are transformed in place.
    Args:
      class_names(list):  labels associated with targets (e.g., ['negative', 'positive'])
                     Only used/required if:
                     1. targets are one/multi-hot-encoded
                     2. targets are integers and represent class IDs for classification task
                     Not required if:
                     1. targets are numeric and task is regression
                     2. targets are strings and task is classification (class_names are populated automatically)
      label_encoder(LabelEncoder): a prior instance of LabelEncoder.  
                                   If None, will be created when train=True

    ### Descendants

    * ktrain.utils.YTransformDataFrame

    ### Methods

    `apply(self, targets, train=True)`
    :

    `apply_test(self, targets)`
    :

    `apply_train(self, targets)`
    :

    `get_classes(self)`
    :

    `set_classes(self, class_names)`
    :

`YTransformDataFrame(label_columns=[], is_regression=False)`
:   Checks and transforms label columns in DataFrame. DataFrame is modified in place
    Args:
      label_columns(list): list of columns storing labels 
      is_regression(bool): If True, task is regression and integer targets are treated as numeric dependent variable.
                           IF False, task is classification and integer targets are treated as class IDs.

    ### Ancestors (in MRO)

    * ktrain.utils.YTransform

    ### Methods

    `apply(self, df, train=True)`
    :

    `apply_test(self, df)`
    :

    `apply_train(self, df)`
    :

    `get_label_columns(self, squeeze=True)`
    :   Returns label columns of transformed DataFrame