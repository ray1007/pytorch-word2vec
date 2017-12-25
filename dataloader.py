import re
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader

from nltk.parse import corenlp


MAX_CHILDREN = 5


WORD_TYPE = [
    ('word_idx', 'i4'),
    ('pos_idx', 'u1'),
    ('pr_pos_idx', 'u1'),
    ('pr_dep_idx', 'u1'),
    ('ch_pos_idxs', f'{MAX_CHILDREN}u1'),
    ('ch_dep_idxs', f'{MAX_CHILDREN}u1'),
]


def sliding_window(arr, size):
    '''
        Create a sliding window view of the original array.
        This function has to be used with extreme care!
    '''
    axis = 0
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


class SentenceParsingDataset(Dataset):
    def __init__(self, filepath, dep_parser, vocab_map, pos_map, dep_map):
        '''
            filepath: path to the corpus (one line per sentence)
            dep_parser: corenlp.CoreNLPDependencyParser
            vocab_map: LookupTable
            pos_map: LookupTable
            dep_map: LookupTable
            
            Note:
                `0` is treated as padding index.
                LookupTable should map OOV to the `unknown` index.
        '''
        self.dep_parser = 
        self.dep_parser = dep_parser
        self.vocab_map = vocab_map
        self.pos_map = pos_map
        self.dep_map = dep_map
        with open(filepath, 'r') as f:
            self.sents = [line.strip() for line in f]
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            self.cache[index] = self.parse(self.sents[index])
            return self.cache[index]

    def __len__(self):
        return len(self.sents)

    def parse(self, sentence):
        nodes = next(self.dep_parser.raw_parse(sentence)).nodes
        word_n = len(nodes) - 1
        sent = np.zeros(word_n, dtype=WORD_TYPE)
        for i in range(word_n):
            node = nodes[i+1]
            word = sent[i]

            word['word_idx'] = self.vocab_map[node['word']]
            word['pos_idx'] = self.pos_map[node['tag']]
            word['pr_pos_idx'] = self.pos_map[nodes[node['head']]['tag']]
            word['pr_dep_idx'] = self.dep_map[node['rel']]
            ch = []
            for dep, arr in node['deps'].items():
                ch.extend([(nodes[x]['tag'], dep) for x in arr])

            for j, (pos, dep) in enumerate(ch[:MAX_CHILDREN]):
                word['ch_pos_idxs'][j] = self.pos_map[pos]
                word['ch_dep_idxs'][j] = self.dep_map[dep]

        return sent


class CBOWLoaderIter:
    def __init__(self, loader):
        self.loader = laoder
        self.shuffle = laoder.shuffle
        self.batch_size = loader.batch_size
        self.window_size = loader.window_size
        self.in_iter = iter(loader.in_loader)
        self.queue = []

    def get_next_sent(self):
        try:
            sent = next(self.in_iter)
        except StopIteration:
            return None, None
        word_n = len(sent)
        word_idx = sent['word_idx']
        size = self.window_size
        # pad zeros
        pad_word_idx = np.pad(word_idx, (size, size))
        # following is correct but not easy to understand
        wds = sliding_window(pad_word_idx, size)
        ctx = np.concatenate((wds[:-(size+1)], wds[(size+1):]), axis=1)
        return sent, ctx

    def __iter__(self):
        return self

    def __next__(self):
        total = sum(len(x) for x, y in self.queue)
        while total < self.batch_size:
            sent, ctx = self.get_next_sent()
            if sent is None:
                break
            # TODO:
            # subsampling
            total += len(sent)
            self.queue.append((sent, ctx))

        if total == 0:
            raise StopIteration

        bound = total - self.batch_size
        remaining = None
        if bound > 0:
            last_sent, last_ctx = self.queue.pop()
            remaining = (last_sent[-bound:], last_ctx[-bound:])
            self.queue.append((last_sent[-bound:], last_ctx[-bound:]))

        sent, ctx = [np.concatenate(arr) for arr in zip(*self.queue)]

        self.queue.clear()
        if remaining is not None;
            self.queue.append(remaining)

        word_idx = np2tor(sent['word_idx']).long()
        pos_idx = np2tor(sent['pos_idx'])).long()
        pr_pos_idx = np2tor(sent['pr_pos_idx']).long()
        pr_dep_idx = np2tor(sent['pr_dep_idx']).long()
        ch_pos_idxs = np2tor(sent['ch_pos_idxs']).long()
        ch_dep_idxs = np2tor(sent['ch_dep_idxs']).long()
        ctx_idxs = np2tor(ctx).long()
        return (word_idx, pos_idx,
                pr_pos_idx, pr_dep_idx,
                ch_pos_idxs, ch_dep_idxs,
                ctx_idxs)


class CBOWLoader:
    def __init__(self, dataset, window_size,
                 batch_size=1, shuffle=False,
                 num_workers=0):
        self.window_size,
        self.dataset = dataset
        self.in_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                    num_workers=num_workers,
                                    collate_fn=lambda x: x)

    def __iter__(self):
        return CBOWLoaderIter(self)
