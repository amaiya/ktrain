from ..imports import *
from .. import utils as U
from ..preprocessor import Preprocessor
from ..data import SequenceDataset


class TabularPreprocessor(Preprocessor):
    """
    Tabular preprocessing base class
    """

    def __init__(self, predictor_columns, label_columns, date_columns=[], 
                 is_regression=False, procs=[], max_card=20):
        self.is_regression=is_regression
        self.c  = None
        self.pc = predictor_columns
        self.lc = label_columns
        self.lc = [self.lc] if isinstance(self.lc, str) else self.lc
        self.dc = date_columns
        self.label_columns = None
        self.cat_names = []
        self.cont_names = []
        self.date_names = []
        self.label_transform = None
        self.procs = procs
        self.max_card = max_card

    @property
    def na_names(self):
        return [n for n in self.cat_names if n[-3:] == '_na']
        
    def get_preprocessor(self):
        return (self.label_transform, self.procs)


    def get_classes(self):
        return self.label_columns if not self.is_regression else []


    def preprocess(self, df):
        return self.preprocess_test(df)


    def _validate_columns(self, df):
        missing_columns = []
        for col in df.columns.values:
            if col not in self.lc and col not in self.pc:
                missing_columns.append(col)
        if len(missing_columns) > 0:
            raise ValueError('df is missing columns: %s' % (missing_columns))
        return


    def denormalize(self, df):
        normalizer = None
        for proc in self.procs:
            if type(proc).__name__ == 'Normalize':
                normalizer = proc
                break
        if normalizer is None: return df
        return normalizer.revert(df)

    #def codify(self, df):
        #df = df.copy()
        #for lab in self.lc:
            #df[lab] = df[lab].cat.codes
        #return df



    def preprocess_train(self, df, mode='train', verbose=1):
        """
        preprocess training set
        """
        df = df.copy()

        clean_df(df, pc=self.pc, lc=self.lc, check_labels=mode=='train')

        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pd.DataFrame')

        # validate columns
        self._validate_columns(df)

        # validate mode
        #if mode != 'train' and self.label_transform is None:
            #raise ValueError('self.label_transform is None but mode is %s: are you sure preprocess_train was invoked first?' % (mode))

        # verbose
        if verbose:
            print('processing %s: %s rows x %s columns' % (mode, df.shape[0], df.shape[1]))

        # convert date fields
        for field in self.dc:
            df = df.copy()  # TODO: fix this
            df, date_names = add_datepart(df, field)
            self.date_names = date_names

        # preprocess labels and data
        if mode == 'train':
            label_columns = self.lc[:]
            #label_columns.sort() # leave label columns sorted in same order as in DataFrame
            self.label_transform = U.YTransformDataFrame(label_columns, is_regression=self.is_regression)
            df = self.label_transform.apply_train(df)
            self.label_columns = self.label_transform.get_classes() if not self.is_regression else self.label_transform.label_columns
            self.cont_names, self.cat_names = cont_cat_split(df, label_columns=self.label_columns, max_card=self.max_card)
            self.procs = [proc(self.cat_names, self.cont_names) for proc in self.procs] # "objectivy"
        else:
            df = self.label_transform.apply_test(df)
        for proc in self.procs: proc(df, test=mode!='train')  # apply processors

        return TabularDataset(df, self.cat_names, self.cont_names, self.label_columns)



    def preprocess_valid(self, df, verbose=1):
        """
        preprocess validation set
        """
        return self.preprocess_train(df, mode='valid', verbose=verbose)



    def preprocess_test(self, df, verbose=1):
        """
        preprocess test set
        """
        return self.preprocess_train(df, mode='test', verbose=verbose)



class TabularDataset(SequenceDataset):
    def __init__(self, df, cat_columns, cont_columns, label_columns, batch_size=32, shuffle=False):
        # error checks
        if not isinstance(df, pd.DataFrame): raise ValueError('df must be pandas Dataframe')
        all_columns = cat_columns + cont_columns + label_columns
        missing_columns = []
        for col in df.columns.values:
            if col not in all_columns: missing_columns.append(col)
        if len(missing_columns) > 0: raise ValueError('df is missing these columns: %s' % (missing_columns))

        # set variables
        super().__init__(batch_size=batch_size)
        self.indices = np.arange(df.shape[0])
        self.df = df
        self.cat_columns = cat_columns
        self.cont_columns = cont_columns
        self.label_columns = label_columns
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(self.df.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        df = self.df[self.cat_columns+self.cont_columns].iloc[inds]
        for cat_name in self.cat_columns:
            codes = np.stack([c.cat.codes.values for n,c in df[[cat_name]].items()], 1).astype(np.int64) + 1
            batch_x.append(codes)
        if len(self.cont_columns) > 0:
            conts = np.stack([c.astype('float32').values for n,c in df[self.cont_columns].items()], 1)
            batch_x.append(conts)
        batch_y = self.df[self.label_columns].iloc[inds].values
        batch_x = batch_x[0] if len(batch_x)==1 else tuple(batch_x)
        return batch_x, batch_y

    def nsamples(self):
        return self.df.shape[0]

    def get_y(self):
        return self.df[self.label_columns].values

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

    def xshape(self):
        return self.df.shape

    def nclasses(self):
        return self.get_y().shape[1]


def pd_data_types(df, return_df=False):
    """
    infers data type of each column in Pandas DataFrame
    Args:
      df(pd.DataFrame): pandas DataFrame
      return_df(bool): If True, returns columns and types in DataFrame. 
                       Otherwise, a dictionary is returned.
    """

    infer_type = lambda x: pd.api.types.infer_dtype(x, skipna=True)
    df.apply(infer_type, axis=0)

    # DataFrame with column names & new types
    df_types = pd.DataFrame(df.apply(pd.api.types.infer_dtype, axis=0)).reset_index().rename(columns={'index': 'column', 0: 'type'})
    if return_df: return df_types
    cols = list(df_types['column'].values)
    col_types = list(df_types['type'].values)
    return dict(list(zip(cols, col_types)))





def clean_df(train_df, val_df=None, pc=[], lc=[], check_labels=True, return_types=False):
    train_type_dict = pd_data_types(train_df)
    for k,v in train_type_dict.items():
        if v != 'string': continue
        train_df[k] = train_df[k].str.strip()
        if val_df is not None:
            if k not in val_df.columns: raise ValueError('val_df is missing %s column' % (k))
            val_df[k] = val_df[k].str.strip()
    if (pc and not lc) or (not pc and lc): raise ValueError('pc and lc: both or neither must exist')
    if pc and lc:
        inp_cols = train_df.columns.values if check_labels else [col for col in train_df.columns.values if col not in lc]
        original_cols = pc + lc if check_labels else pc
        if set(original_cols) != set(inp_cols):
            raise ValueError('DataFrame is either missing columns or includes extra columns: \n'+\
                    'expected: %s\nactual: %s' % (original_cols, inp_cols))
    if return_types: return train_type_dict
    return


#--------------------------------------------------------------------
# These are helper functions adapted from fastai:
# https://github.com/fastai/fastai
# -------------------------------------------------------------------


from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace


from pandas.api.types import is_numeric_dtype, is_categorical_dtype


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


def cont_cat_split(df, max_card=20, label_columns=[]):
    "Helper function that returns column names of cont and cat variables from given df."
    cont_names, cat_names = [], []
    for col in df:
        if col in label_columns: continue
        if df[col].dtype == int and df[col].unique().shape[0] > max_card or df[col].dtype == float: cont_names.append(col)
        else: cat_names.append(col)
    return cont_names, cat_names


def add_datepart(df:pd.DataFrame, field_name:str, prefix:str=None, drop:bool=True, time:bool=False, return_added_columns=True):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    added_columns = []
    for n in attr: 
        df[prefix + n] = getattr(field.dt, n.lower())
        added_columns.append(prefix+n)
    df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop: df.drop(field_name, axis=1, inplace=True)
    if return_added_columns:
        return (df, added_columns)
    else:
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



class TabularProc():
    "A processor for tabular dataframes."

    def __init__(self, cat_names, cont_names):
        self.cat_names = cat_names
        self.cont_names = cont_names

    def __call__(self, df, test=False):
        "Apply the correct function to `df` depending on `test`."
        func = self.apply_test if test else self.apply_train
        func(df)

    def apply_train(self, df):
        "Function applied to `df` if it's the train set."
        raise NotImplementedError
    def apply_test(self, df):
        "Function applied to `df` if it's the test set."
        self.apply_train(df)


class Categorify(TabularProc):
    def __init__(self, cat_names, cont_names):
        super().__init__(cat_names, cont_names)
        self.categories = None

    def apply_train(self, df):
        self.categories = {}
        for n in self.cat_names:
            df.loc[:,n] = df.loc[:,n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df):
        for n in self.cat_names:
            df.loc[:,n] = pd.Categorical(df[n], categories=self.categories[n], ordered=True)

FILL_MEDIAN = 'median'
FILL_CONSTANT = 'constant'
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    def __init__(self, cat_names, cont_names, fill_strategy=FILL_MEDIAN, add_col=True, fill_val=0.):
        super().__init__(cat_names, cont_names)
        self.fill_strategy = fill_strategy
        self.add_col = add_col
        self.fill_val = fill_val
        self.na_dict = None

    def apply_train(self, df):
        self.na_dict = {}
        self.filler_dict = {}
        for name in self.cont_names:
            if self.fill_strategy == FILL_MEDIAN: filler = df[name].median()
            elif self.fill_strategy == FILL_CONSTANT: filler = self.fill_val
            else: filler = df[name].dropna().value_counts().idxmax()
            self.filler_dict[name] = filler
            if pd.isnull(df[name]).sum():
                if self.add_col:
                    df[name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df[name] = df[name].fillna(filler)
                self.na_dict[name] = True

    def apply_test(self, df):
        "Fill missing values in `self.cont_names` like in `apply_train`."
        for name in self.cont_names:
            if name in self.na_dict:
                if self.add_col:
                    df[name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df[name] = df[name].fillna(self.filler_dict[name])
            elif pd.isnull(df[name]).sum() != 0:
                warnings.warn(f"""There are nan values in field {name} but there were none in the training set. 
                Filled with {self.fill_strategy}.""")
                df[name] = df[name].fillna(self.filler_dict[name])
                #raise Exception(f"""There are nan values in field {name} but there were none in the training set. 
                #Please fix those manually.""")
           

class Normalize(TabularProc):
    "Normalize the continuous variables."
    def __init__(self, cat_names, cont_names):
        super().__init__(cat_names, cont_names)
        self.means = None
        self.stds = None

    def apply_train(self, df):
        "Compute the means and stds of `self.cont_names` columns to normalize them."
        self.means,self.stds = {},{}
        for n in self.cont_names:
            assert is_numeric_dtype(df[n]), (f"""Cannot normalize '{n}' column as it isn't numerical.
                Are you sure it doesn't belong in the categorical set of columns?""")
            self.means[n],self.stds[n] = df[n].mean(),df[n].std()
            df[n] = (df[n]-self.means[n]) / (1e-7 + self.stds[n])

    def apply_test(self, df):
        "Normalize `self.cont_names` with the same statistics as in `apply_train`."
        for n in self.cont_names:
            df[n] = (df[n]-self.means[n]) / (1e-7 + self.stds[n])

    def revert(self,df):
        """
        Undoes normalization and returns reverted dataframe
        """
        out_df = df.copy()
        for n in self.cont_names:
            out_df[n] =  (df[n] * (1e-7 + self.stds[n])) + self.means[n]
        return out_df



