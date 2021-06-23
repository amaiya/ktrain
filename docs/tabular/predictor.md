Module ktrain.tabular.predictor
===============================

Classes
-------

`TabularPredictor(model, preproc, batch_size=32)`
:   predictions for tabular data

    ### Ancestors (in MRO)

    * ktrain.predictor.Predictor
    * abc.ABC

    ### Methods

    `explain(self, test_df, row_index=None, row_num=None, class_id=None, background_size=50, nsamples=500)`
    :   Explain the prediction of an example using SHAP.
        Args:
          df(pd.DataFrame): a pd.DataFrame of test data is same format as original training data DataFrame
                            The DataFrame does NOT need to contain all the original label columns
                            (e.g., the Survived column in Kaggle's Titatnic dataset) but  MUST contain
                            all the original predictor columns (e.g., un-normalized numerical variables, categorical
                            variables as strings).
          row_index(int): index of row in DataFrame to explain (e.g., PassengerID in Titanic dataset).
                          mutually-exclusive with row_id
          row_num(int): raw row number in DataFrame to explain (i.e., 0=first row, 1=second rows, etc.)
                         mutually-exclusive with row_index
          class_id(int): Only required for classification
          background_size(int): size of background data (SHAP parameter)
          nsamples(int): number of samples (SHAP parameter)

    `get_classes(self)`
    :

    `predict(self, df, return_proba=False)`
    :   Makes predictions for a test dataframe
        Args:
          df(pd.DataFrame):  a pandas DataFrame in same format as DataFrame used for training model
          return_proba(bool): If True, return probabilities instead of predicted class labels