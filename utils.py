import numpy as np


class LookupTable(dict):
    def set_missing(self, value):
        self.value = value

    def __missing__(self, key):
        return self.value


# Fast discrete sampling method.
# References:
# https://en.wikipedia.org/wiki/Alias_method
# https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
class AliasTable:
    def __init__(self, probs):
        '''
            probs: 1D numpy array with summation equals to 1
        '''
        self.N = len(probs)
        smaller = []
        larger = []
        self.U = self.N * probs
        self.K = np.zeros(self.N, dtype='i')
        for idx, p in enumerate(self.U):
            if p < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.K[small] = large
            self.U[large] -= (1.0 - self.U[small])
            if self.U[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def sample(self, *shape):
        '''
            args should be a list of postive integers
        '''
        idx = np.floor(np.random.rand(*shape) * self.N).astype('i')
        mask = np.random.rand(*shape) > self.U[idx]
        idx[mask] = self.K[idx][mask]

        return idx


def sliding_window(arr, size, axis=0):
    '''
        Create a sliding window view of the original array.
        This function has to be used with extreme care!
    '''
    shape = list(arr.shape)
    shape[axis] = arr.shape[axis] - size + 1
    shape.append(size)
    strides = list(arr.strides)
    strides.append(arr.strides[axis])
    return np.lib.stride_tricks.as_strided(
        arr, shape=shape, strides=strides, writeable=False
    )


def np2tor(arr):
    return torch.from_numpy(arr)


def sum_normalize(arr):
    ''' normalize the array to 1. arr: numpy array '''
    return arr / arr.sum()
