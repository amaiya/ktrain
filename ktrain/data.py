from .imports import *


class Dataset(Sequence):
   def __init__(self, batch_size=32):
       self.batch_size = batch_size

   def __len__(self):
       raise NotImplemented

   def __getitem__(self, idx):
       raise NotImplemented

   def on_epoch_end(self):
       raise NotImplemented

   def xshape(self):
       raise NotImplemented

   def nsamples(self):
       raise NotImplemented

   def nclasses(self):
       raise NotImplemented

    def get_y(self):
        raise NotImplemented

   def ondisk(self):
       return False



class ArrayDataset(Dataset):
    def __init__(self, x, y, batch_size=32):
        if type(x) != np.ndarray or type(y) != np.ndarray:
            raise ValueError('x and y must be numpy arrays')
        if len(x.shape) != 3:
            raise valueError('x must have 3 dimensions')
        super().__init__(batch_size=batch_size)
        self.x, self.y = x, y
        self.indices = np.arange(self.x[0].shape[0])
        self.n_inputs = x.shape[0]


    def __len__(self):
        return math.ceil(self.x[0].shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for i in range(self.n_inputs):
            batch_x.append(self.x[i][inds])
        batch_y = self.y[inds]
        return tuple(batch_x), batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def xshape(self):
        return self.x.shape

    def nsamples(self):
        if self.n_inputs == 1:
            return self.x.shape[0]
        else:
            return self.x.shape[1]

    def nclasses(self):
        return self.y.shape[1]

    def get_y(self):
        return self.y

    def ondisk(self):
        return False

