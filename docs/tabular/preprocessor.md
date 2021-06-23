Module ktrain.tabular.preprocessor
==================================

Functions
---------

    
`add_cyclic_datepart(df: pandas.core.frame.DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False, add_linear: bool = False)`
:   Helper function that adds trigonometric date/time features to a date in the column `field_name` of `df`.

    
`add_datepart(df: pandas.core.frame.DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False, return_added_columns=True)`
:   Helper function that adds columns relevant to a date in the column `field_name` of `df`.

    
`clean_df(train_df, val_df=None, pc=[], lc=[], check_labels=True, return_types=False)`
:   

    
`cont_cat_split(df, max_card=20, label_columns=[])`
:   Helper function that returns column names of cont and cat variables from given df.

    
`cyclic_dt_feat_names(time: bool = True, add_linear: bool = False) ‑> List[str]`
:   Return feature names of date/time cycles as produced by `cyclic_dt_features`.

    
`cyclic_dt_features(d, time: bool = True, add_linear: bool = False) ‑> List[float]`
:   Calculate the cos and sin of date/time cycles.

    
`ifnone(a, b)`
:   `a` if `a` is not None, otherwise `b`.

    
`make_date(df, date_field)`
:   Make sure `df[field_name]` is of the right date type.
    Reference: https://github.com/fastai/fastai

    
`pd_data_types(df, return_df=False)`
:   infers data type of each column in Pandas DataFrame
    Args:
      df(pd.DataFrame): pandas DataFrame
      return_df(bool): If True, returns columns and types in DataFrame. 
                       Otherwise, a dictionary is returned.

Classes
-------

`Categorify(cat_names, cont_names)`
:   A processor for tabular dataframes.

    ### Ancestors (in MRO)

    * ktrain.tabular.preprocessor.TabularProc

`FillMissing(cat_names, cont_names, fill_strategy='median', add_col=True, fill_val=0.0)`
:   Fill the missing values in continuous columns.

    ### Ancestors (in MRO)

    * ktrain.tabular.preprocessor.TabularProc

    ### Methods

    `apply_test(self, df)`
    :   Fill missing values in `self.cont_names` like in `apply_train`.

`Normalize(cat_names, cont_names)`
:   Normalize the continuous variables.

    ### Ancestors (in MRO)

    * ktrain.tabular.preprocessor.TabularProc

    ### Methods

    `apply_test(self, df)`
    :   Normalize `self.cont_names` with the same statistics as in `apply_train`.

    `apply_train(self, df)`
    :   Compute the means and stds of `self.cont_names` columns to normalize them.

    `revert(self, df)`
    :   Undoes normalization and returns reverted dataframe

`TabularDataset(df, cat_columns, cont_columns, label_columns, batch_size=32, shuffle=False)`
:   Base class for custom datasets in ktrain.
    
    If subclass of Dataset implements a method to to_tfdataset
    that converts the data to a tf.Dataset, then this will be
    invoked by Learner instances just prior to training so
    fit() will train using a tf.Dataset representation of your data.
    Sequence methods such as __get_item__ and __len__
    must still be implemented.
    
    The signature of to_tfdataset is as follows:
    
    def to_tfdataset(self, training=True)
    
    See ktrain.text.preprocess.TransformerDataset as an example.

    ### Ancestors (in MRO)

    * ktrain.data.SequenceDataset
    * ktrain.data.Dataset
    * tensorflow.python.keras.utils.data_utils.Sequence

    ### Methods

    `get_y(self)`
    :

    `nsamples(self)`
    :

    `on_epoch_end(self)`
    :   Method called at the end of every epoch.

`TabularPreprocessor(predictor_columns, label_columns, date_columns=[], is_regression=False, procs=[], max_card=20)`
:   Tabular preprocessing base class

    ### Ancestors (in MRO)

    * ktrain.preprocessor.Preprocessor
    * abc.ABC

    ### Instance variables

    `na_names`
    :

    ### Methods

    `denormalize(self, df)`
    :

    `get_classes(self)`
    :

    `get_preprocessor(self)`
    :

    `preprocess(self, df)`
    :

    `preprocess_test(self, df, verbose=1)`
    :   preprocess test set

    `preprocess_train(self, df, mode='train', verbose=1)`
    :   preprocess training set

    `preprocess_valid(self, df, verbose=1)`
    :   preprocess validation set

`TabularProc(cat_names, cont_names)`
:   A processor for tabular dataframes.

    ### Descendants

    * ktrain.tabular.preprocessor.Categorify
    * ktrain.tabular.preprocessor.FillMissing
    * ktrain.tabular.preprocessor.Normalize

    ### Methods

    `apply_test(self, df)`
    :   Function applied to `df` if it's the test set.

    `apply_train(self, df)`
    :   Function applied to `df` if it's the train set.