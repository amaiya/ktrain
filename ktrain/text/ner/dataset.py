from ...dataset import SequenceDataset
from ...imports import *


class NERSequence(SequenceDataset):
    def __init__(self, x, y, batch_size=1, p=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.p = p
        self.prepare_called = False

    def prepare(self):
        if self.p is not None and not self.prepare_called:
            self.x, self.y = self.p.fix_tokenization(self.x, self.y)
        self.prepare_called = True
        return

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        return self.p.transform(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def get_lengths(self, idx):
        x_true, y_true = self[idx]
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def nsamples(self):
        return len(self.x)

    def get_y(self):
        return self.y

    def xshape(self):
        return (len(self.x), self[0][0][0].shape[1])

    def nclasses(self):
        return len(self.p._label_vocab._id2token)
