from ..imports import *
from .. import utils as U
from . import preprocessor as pp

def table_from_df(train_df, label_columns=[], date_columns=[], val_df=None, val_pct=0.1, is_regression=False, random_state=None):

    # TODO: this code is similar to images_from_df: must refactor and cleaned up

    # check label_columns
    if label_columns is None or (isinstance(label_columns, (list, np.ndarray)) and len(label_columns) == 0):
        raise ValueError('label_columns is required')
    if isinstance(label_columns, (list, np.ndarray)) and len(label_columns) == 1:
        label_columns = label_columns[0]

    # check for regression
    peek = train_df[label_columns].iloc[0]
    if isinstance(label_columns, str) and peek.isdigit() and not is_regression:
        warnings.warn('Targets are integers, but is_regression=False. Task treated as classification instead of regression.')
    if isinstance(peek, str) and is_regression:
        train_df[label_columns] = train_df[label_columns].astype('float32')
        if val_df is not None:
            val_df[label_columns] = val_df[label_columns].astype('float32')
    peek = train_df[label_columns].iloc[0]


    # convert date fields
    for field in date_columns:
        train_df = pp.add_datepart(train_df, field)

    # define predictor_columns
    predictor_columns = [col for col in train_df.columns.values if col not in label_columns]

    # convert to dataframes
    if val_df is None:
        if val_pct:
            df = train_df.copy()
            prop = 1-val_pct
            if random_state is not None: np.random.seed(42)
            msk = np.random.rand(len(df)) < prop
            train_df = df[msk]
            val_df = df[~msk]

    # process class labels
    if isinstance(label_columns, (list, np.ndarray)): label_columns.sort()
    if isinstance(label_columns, str) or \
       (isinstance(label_columns, (list, np.ndarray)) and len(label_columns) == 1):
        label_col_name = label_columns if isinstance(label_columns, str) else label_columns[0]
        if not is_regression:
            le = LabelEncoder()
            train_labels = train_df[label_col_name].values
            le.fit(train_labels)
            y_train = to_categorical(le.transform(train_labels))
            y_val = to_categorical(le.transform(val_df[label_col_name].values))
            label_columns = list(le.classes_)
            train_df = train_df[predictor_columns]
            for i, col in enumerate(label_columns):
                train_df[col] = y_train[:,i]
            val_df = val_df[predictor_columns]
            for i, col in enumerate(label_columns):
                val_df[col] = y_val[:,i]

    return (train_df, val_df)


def table_from_csv(train_csv, label_columns=[], date_columns=[], val_csv=None, val_pct=0.1, is_regression=False, random_state=None):
    """
    Loads tabular data from CSV file
    """

    # read in dataset
    train_df = pd.read_csv(train_csv, index_col=0)
    val_df = None
    if val_csv is not None:
        val_df = pd.read_csv(val_csv, index_col=0)
    return table_from_df(train_df, label_columns=label_columns, date_columns=date_columns, val_df=val_df, val_pct=val_pct, 
                         is_regression=is_regression, random_state=random_state)
 


