from ..imports import *
from .. import utils as U
from . import preprocessor as pp


def table_from_csv(fname, target_column, datetime_columns=[]):
    """
    Loads tabular data from CSV file
    """
    # read in dataset
    df = pd.read_csv(fname, index_col=0)

    # convert specified fields to dates
    for field in datetime_columns:
        df = pp.add_datepart(df, field)
    #print(U.pd_data_types(df))
    return df



