from .. import utils as U
from ..imports import *
from ..predictor import Predictor
from .preprocessor import TabularPreprocessor


class TabularPredictor(Predictor):
    """
    ```
    predictions for tabular data
    ```
    """

    def __init__(self, model, preproc, batch_size=U.DEFAULT_BS):
        if not isinstance(model, keras.Model):
            raise ValueError("model must be of instance keras.Model")
        if (
            not isinstance(preproc, TabularPreprocessor)
            and type(preproc).__name__ != "TabularPreprocessor"
        ):
            raise ValueError("preproc must be a TabularPreprocessor object")
        self.model = model
        self.preproc = preproc
        self.c = self.preproc.get_classes()
        self.batch_size = batch_size

    def get_classes(self):
        return self.c

    def predict(self, df, return_proba=False, verbose=0):
        """
        ```
        Makes predictions for a test dataframe
        Args:
          df(pd.DataFrame):  a pandas DataFrame in same format as DataFrame used for training model
          return_proba(bool): If True, return probabilities instead of predicted class labels
          verbose(int): verbosity: 0 (silent), 1 (progress bar), 2 (single line)
        ```
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pd.DataFrame")
        df = df.copy()

        classification, multilabel = U.is_classifier(self.model)

        # get predictions
        tseq = self.preproc.preprocess_test(df, verbose=0)
        tseq.batch_size = self.batch_size
        preds = self.model.predict(tseq, verbose=verbose)
        result = (
            preds
            if return_proba or multilabel or not self.c
            else [self.c[np.argmax(pred)] for pred in preds]
        )
        if multilabel and not return_proba:
            result = [list(zip(self.c, r)) for r in result]
        return result

    def _predict_shap(self, X):
        n_cats = len(self.preproc.cat_names)
        n_conts = len(self.preproc.cont_names)

        # reformat for model
        batch_x = [X[:, i : i + 1] for i in range(n_cats)] + [X[:, -n_conts:]]
        result = self.model.predict(batch_x)
        return result

    def explain(
        self,
        test_df,
        row_index=None,
        row_num=None,
        class_id=None,
        background_size=50,
        nsamples=500,
    ):
        """
        ```
        Explain the prediction of an example using SHAP.
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
        ```
        """
        try:
            import shap
        except ImportError:
            msg = (
                "TabularPredictor.explain requires shap library. Please install with: pip install shap. "
                + "Conda users should use this command instead: conda install -c conda-forge shap"
            )
            warnings.warn(msg)
            return

        classification, multilabel = U.is_classifier(self.model)
        if classification and class_id is None:
            raise ValueError(
                "For classification models, please supply the class_id of the class you would like to explain."
                + "It should be an index into the list returned by predictor.get_classes()."
            )

        f = self._predict_shap

        # prune dataframe
        df_display = test_df.copy()
        df_display = df_display[self.preproc.pc]

        # add synthetic labels
        for lab in self.preproc.lc:
            df_display[lab] = np.zeros(df_display.shape[0], dtype=int)

        # convert DataFrame to TabularDataset with processed/normalized independent variables
        tabseq = self.preproc.preprocess_test(df_display, verbose=0)
        tabseq.batch_size = df_display.shape[0]
        df = pd.DataFrame(
            data=np.concatenate(tabseq[0][0], axis=1),
            columns=tabseq.cat_columns + tabseq.cont_columns,
            index=df_display.index,
        )

        # add new auto-engineered feature columns
        for col in [self.preproc.na_names + self.preproc.date_names]:
            df_display[col] = df[col]

        # sort display df correctly
        df_display = df_display[tabseq.cat_columns + tabseq.cont_columns]

        # select row
        if row_num is not None and row_index is not None:
            raise ValueError(
                "row_num and row_index are mutually exclusive with eachother."
            )

        if row_index is not None:
            df_row = df[df.index.isin([row_index])].iloc[0, :]
            df_display_row = df_display[df_display.index.isin([row_index])].iloc[0, :]
            r_key = "row_index" if df.index.name is None else df.index.name
            r_val = row_index
        elif row_num is not None:
            df_row = df.iloc[row_num, :]
            df_display_row = df_display.iloc[row_num, :]
            r_key = "row_num"
            r_val = row_num
        # print(df_row)
        # print(df_display_row)

        # shap
        explainer = shap.KernelExplainer(f, df.iloc[:background_size, :])
        shap_values = explainer.shap_values(df_row, nsamples=nsamples, l1_reg="aic")
        expected_value = explainer.expected_value

        if not np.issubdtype(type(explainer.expected_value), np.floating):
            expected_value = explainer.expected_value[
                0 if class_id is None else class_id
            ]
        if type(shap_values) == list:
            shap_values = shap_values[0 if class_id is None else class_id]

        if classification:
            print(
                "Explanation for class = %s (%s=%s): "
                % (self.get_classes()[class_id], r_key, r_val)
            )
        plt.show(
            shap.force_plot(
                expected_value, shap_values, df_display_row, matplotlib=True
            )
        )
