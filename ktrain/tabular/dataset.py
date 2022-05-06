from ..dataset import SequenceDataset
from ..imports import *


class TabularDataset(SequenceDataset):
    def __init__(
        self, df, cat_columns, cont_columns, label_columns, batch_size=32, shuffle=False
    ):
        # error checks
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be pandas Dataframe")
        all_columns = cat_columns + cont_columns + label_columns
        missing_columns = []
        for col in df.columns.values:
            if col not in all_columns:
                missing_columns.append(col)
        if len(missing_columns) > 0:
            raise ValueError("df is missing these columns: %s" % (missing_columns))

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
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        df = self.df[self.cat_columns + self.cont_columns].iloc[inds]
        for cat_name in self.cat_columns:
            codes = (
                np.stack(
                    [c.cat.codes.values for n, c in df[[cat_name]].items()], 1
                ).astype(np.int64)
                + 1
            )
            batch_x.append(codes)
        if len(self.cont_columns) > 0:
            conts = np.stack(
                [c.astype("float32").values for n, c in df[self.cont_columns].items()],
                1,
            )
            batch_x.append(conts)
        batch_y = self.df[self.label_columns].iloc[inds].values
        batch_x = batch_x[0] if len(batch_x) == 1 else tuple(batch_x)
        return batch_x, batch_y

    def nsamples(self):
        return self.df.shape[0]

    def get_y(self):
        return self.df[self.label_columns].values

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def xshape(self):
        return self.df.shape

    def nclasses(self):
        return self.get_y().shape[1]
