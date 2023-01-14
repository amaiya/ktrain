from ..dataset import SequenceDataset
from ..imports import *


class TransformerDataset(SequenceDataset):
    """
    ```
    Wrapper for Transformer datasets.
    ```
    """

    def __init__(self, x, y, batch_size=1, use_token_type_ids=True):
        if type(x) not in [list, np.ndarray]:
            raise ValueError("x must be list or np.ndarray")
        if type(y) not in [list, np.ndarray]:
            raise ValueError("y must be list or np.ndarray")
        if type(x) == list:
            x = np.array(x)
        if type(y) == list:
            y = np.array(y)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.use_token_type_ids = use_token_type_ids

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return (batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def to_tfdataset(self, train=True):
        """
        ```
        convert transformer features to tf.Dataset
        ```
        """
        if train:
            shuffle = True
            repeat = True
        else:
            shuffle = False
            repeat = False

        if len(self.y.shape) == 1:
            yshape = []
            ytype = tf.float32
        else:
            yshape = [None]
            ytype = tf.int64

        if self.use_token_type_ids:

            def gen():
                for idx, data in enumerate(self.x):
                    yield (
                        {
                            "input_ids": data[0],
                            "attention_mask": data[1],
                            "token_type_ids": data[2],
                        },
                        self.y[idx],
                    )

            tfdataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    ytype,
                ),
                (
                    {
                        "input_ids": tf.TensorShape([None]),
                        "attention_mask": tf.TensorShape([None]),
                        "token_type_ids": tf.TensorShape([None]),
                    },
                    tf.TensorShape(yshape),
                ),
            )
        else:

            def gen():
                for idx, data in enumerate(self.x):
                    yield (
                        {
                            "input_ids": data[0],
                            "attention_mask": data[1],
                        },
                        self.y[idx],
                    )

            tfdataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                    },
                    ytype,
                ),
                (
                    {
                        "input_ids": tf.TensorShape([None]),
                        "attention_mask": tf.TensorShape([None]),
                    },
                    tf.TensorShape(yshape),
                ),
            )

        if shuffle:
            tfdataset = tfdataset.shuffle(self.x.shape[0])
        tfdataset = tfdataset.batch(self.batch_size)
        if repeat:
            tfdataset = tfdataset.repeat(-1)
        return tfdataset

    def get_y(self):
        return self.y

    def nsamples(self):
        return len(self.x)

    def nclasses(self):
        return self.y.shape[1]

    def xshape(self):
        return (len(self.x), self.x[0].shape[1])
