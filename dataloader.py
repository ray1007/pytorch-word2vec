import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import AliasTable, sliding_window, np2tor, sum_normalize


NEG_SAMPLES = 5


class SentenceDataset(Dataset):
    def __init__(self, filepath, vocab_map):
        '''
            filepath: path to the corpus (one line per sentence)
            vocab_map: LookupTable
        '''
        self.vocab_map = vocab_map
        self.sents = []
        with open(filepath, 'r') as f:
            for line in f:
                tokens = list(filter(
                    lambda x: x is not None,
                    (self.vocab_map.get(t, None) for t in line.strip().split())
                ))
                self.sents.append(np.array(tokens, dtype='i'))

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sents)


class CBOWLoaderIter:
    def __init__(self, loader):
        self.loader = loader
        self.shuffle = loader.shuffle
        self.batch_size = loader.batch_size
        self.window_size = loader.window_size
        self.padding_index = loader.padding_index

        neg_prob = sum_normalize(loader.idx_count ** loader.neg_power)
        self.neg_table = AliasTable(neg_prob)
        ratio = loader.sub_threshold / sum_normalize(loader.idx_count)
        self.sub_prob = np.sqrt(ratio) + ratio

        self.in_iter = iter(loader.in_loader)
        self.queue = []

    def gen_context(self, sent):
        size = self.window_size
        # pad zeros
        pad_word_idx = np.pad(sent, (size, size), 'constant',
                              constant_values=self.padding_index)
        # following is correct but not easy to understand
        wds = sliding_window(pad_word_idx, size)
        ctx = np.concatenate((wds[:-(size+1)], wds[(size+1):]), axis=1)
        return ctx

    def __iter__(self):
        return self

    def __next__(self):
        n_sample = sum(len(x) for x, y in self.queue)
        while n_sample < self.batch_size:
            try:
                sent = next(self.in_iter)[0]
            except StopIteration:
                break

            # subsampling
            mask = self.sub_prob[sent] > np.random.random_sample(len(sent))
            sent = sent[mask]
            if len(sent) < 2:
                continue
            # generate context
            ctx = self.gen_context(sent)
            n_sample += len(sent)
            self.queue.append((sent, ctx))

        if n_sample == 0:
            raise StopIteration

        bound = n_sample - self.batch_size
        remaining = None
        if bound > 0:
            n_sample = self.batch_size
            last_sent, last_ctx = self.queue.pop()
            remaining = (last_sent[-bound:], last_ctx[-bound:])
            self.queue.append((last_sent[:-bound], last_ctx[:-bound]))

        sent, ctx = [np.concatenate(arr) for arr in zip(*self.queue)]

        self.queue.clear()
        if remaining is not None:
            self.queue.append(remaining)

        word_idx = np2tor(sent).long()
        ctx_idxs = np2tor(ctx).long()
        neg_idxs = np2tor(self.neg_table.sample(n_sample, NEG_SAMPLES)).long()
        return (word_idx, ctx_idxs, neg_idxs)


class CBOWLoader:
    def __init__(self, dataset, window_size, idx_count, padding_index,
                 neg_power=0.75, sub_threshold=1e-5,
                 batch_size=1, shuffle=False, num_workers=0):
        '''
            window_size: int
            idx_count: numpy array (1D)
        '''
        self.dataset = dataset
        self.window_size = window_size
        self.idx_count = idx_count
        self.padding_index = padding_index
        self.neg_power = neg_power
        self.sub_threshold = sub_threshold
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.in_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                    num_workers=num_workers,
                                    collate_fn=lambda x: x)

    def __iter__(self):
        return CBOWLoaderIter(self)
