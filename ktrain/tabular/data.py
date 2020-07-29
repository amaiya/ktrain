from ..imports import *
from .. import utils as U
from . import preprocessor as pp

def tabular_from_df(train_df, label_columns=[], date_columns=[], val_df=None, val_pct=0.1, 
                    is_regression=False, max_card=20, random_state=None, verbose=1):


    train_df = train_df.copy()

    # strip space from string columns and check supplied val_df
    train_type_dict = pp.clean_df(train_df, val_df=val_df, return_types=True)

    # check label_columns
    if label_columns is None or (isinstance(label_columns, (list, np.ndarray)) and len(label_columns) == 0):
        raise ValueError('label_columns is required')
    if isinstance(label_columns, (list, np.ndarray)) and len(label_columns) == 1:
        label_columns = label_columns[0]

    # define original predictor_columns
    predictor_columns = [col for col in train_df.columns.values if col not in label_columns]

    # create validation set
    if val_df is None:
        if val_pct:
            df = train_df.copy()
            prop = 1-val_pct
            if random_state is not None: np.random.seed(42)
            msk = np.random.rand(len(df)) < prop
            train_df = df[msk]
            val_df = df[~msk]
    else:
        val_df = val_df.copy()

    procs = [pp.FillMissing, pp.Categorify, pp.Normalize]
    preproc = pp.TabularPreprocessor(predictor_columns, label_columns, date_columns=date_columns, 
                                     is_regression=is_regression, procs=procs, max_card=max_card)
    trn = preproc.preprocess_train(train_df, verbose=verbose)
    if verbose:
        integer_cats = []
        for col in preproc.cat_names:
            if train_type_dict.get(col, None) == 'integer':  integer_cats.append(col)
        if integer_cats:
            print(f'\nThe following integer column(s) are being treated as categorical variables:\n{integer_cats}\n' +\
                  'To treat any of these column(s) as numerical, cast the column to float in DataFrame or CSV\n and re-run tabular_from* function.\n')
    val = None if val_df is None else preproc.preprocess_test(val_df, verbose=verbose)
    return (trn, val, preproc)



def tabular_from_csv(train_csv, label_columns=[], date_columns=[], val_csv=None, val_pct=0.1, 
                     index_col=None, is_regression=False, max_card=20, random_state=None):
    """
    Loads tabular data from CSV file
    """

    # read in dataset
    train_df = pd.read_csv(train_csv, index_col=index_col)
    val_df = None
    if val_csv is not None:
        val_df = pd.read_csv(val_csv, index_col=index_col)
    return tabular_from_df(train_df, label_columns=label_columns, date_columns=date_columns, val_df=val_df, val_pct=val_pct, 
                          is_regression=is_regression, max_card=max_card, random_state=random_state)
 


