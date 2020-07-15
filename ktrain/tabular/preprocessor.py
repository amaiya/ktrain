from ..imports import *
from .. import utils as U
from ..preprocessor import Preprocessor


class TabularPreprocessor(Preprocessor):
    """
    Tabular preprocessing base class
    """

    def __init__(self, class_names=[], target_columns=[], date_columns=[]):
        self.c = class_names
        self.target_columns = target_columns
        self.date_columns = date_columns


    def get_preprocessor(self):
        return (self.target_columns, self.date_columns)


    def get_classes(self):
        return self.c


    def preprocess(self, df):
        return self.preprocess_test(df)



    def preprocess_train(self, df):
        """
        preprocess training set
        """
        pass

    def preprocess_valid(self, df):
        """
        preprocess validation set
        """
        return self.preprocess_test(df)



    def preprocess_test(self, df):
        """
        preprocess test set
        """
        pass



def add_missing(df, include_present=False):
    cols_missing = [col for col in df.columns if df[col].isnull().any()]
    for col in cols_missing:
        df["MISSING_%s" % (col)] = df[col].isnull().astype(int)
    if include_present:
        cols_present = [col for col in df.columns if not df[col].isnull().any()]
        for col in cols_present:
            df["MISSING_%s" % (col)] = 0
    return df



#--------------------------------------------------------------------
# These are helper functions adapted from fastai:
# https://github.com/fastai/fastai
# -------------------------------------------------------------------

from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace


def ifnone(a,b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def make_date(df, date_field):
    """
    Make sure `df[field_name]` is of the right date type.
    Reference: https://github.com/fastai/fastai
    """
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
    return


def cont_cat_split(df, max_card=20, dep_var=None):
    "Helper function that returns column names of cont and cat variables from given df."
    cont_names, cat_names = [], []
    for label in df:
        if label == dep_var: continue
        if df[label].dtype == int and df[label].unique().shape[0] > max_card or df[label].dtype == float: cont_names.append(label)
        else: cat_names.append(label)
    return cont_names, cat_names


def add_datepart(df:pd.DataFrame, field_name:str, prefix:str=None, drop:bool=True, time:bool=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower())
    df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


def cyclic_dt_feat_names(time:bool=True, add_linear:bool=False)->List[str]:
    "Return feature names of date/time cycles as produced by `cyclic_dt_features`."
    fs = ['cos','sin']
    attr = [f'{r}_{f}' for r in 'weekday day_month month_year day_year'.split() for f in fs]
    if time: attr += [f'{r}_{f}' for r in 'hour clock min sec'.split() for f in fs]
    if add_linear: attr.append('year_lin')
    return attr


def cyclic_dt_features(d, time:bool=True, add_linear:bool=False)->List[float]:
    "Calculate the cos and sin of date/time cycles."
    tt,fs = d.timetuple(), [np.cos, np.sin]
    day_year,days_month = tt.tm_yday, calendar.monthrange(d.year, d.month)[1]
    days_year = 366 if calendar.isleap(d.year) else 365
    rs = d.weekday()/7, (d.day-1)/days_month, (d.month-1)/12, (day_year-1)/days_year
    feats = [f(r * 2 * np.pi) for r in rs for f in fs]
    if time and isinstance(d, datetime) and type(d) != date:
        rs = tt.tm_hour/24, tt.tm_hour%12/12, tt.tm_min/60, tt.tm_sec/60
        feats += [f(r * 2 * np.pi) for r in rs for f in fs]
    if add_linear:
        if type(d) == date: feats.append(d.year + rs[-1])
        else:
            secs_in_year = (datetime(d.year+1, 1, 1) - datetime(d.year, 1, 1)).total_seconds()
            feats.append(d.year + ((d - datetime(d.year, 1, 1)).total_seconds() / secs_in_year))
    return feats

def add_cyclic_datepart(df:pd.DataFrame, field_name:str, prefix:str=None, drop:bool=True, time:bool=False, add_linear:bool=False):
    "Helper function that adds trigonometric date/time features to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    series = field.apply(partial(cyclic_dt_features, time=time, add_linear=add_linear))
    columns = [prefix + c for c in cyclic_dt_feat_names(time, add_linear)]
    df_feats = pd.DataFrame([item for item in series], columns=columns, index=series.index)
    for column in columns: df[column] = df_feats[column]
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

