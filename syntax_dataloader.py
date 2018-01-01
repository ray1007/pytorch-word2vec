import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nltk.parse.corenlp import CoreNLPDependencyParser


MAX_CHILDREN = 5
NEG_SAMPLES = 5


WORD_TYPE = [
    ('word_idx', 'i4'),
    ('pos_idx', 'u1'),
    ('pr_pos_idx', 'u1'),
    ('pr_dep_idx', 'u1'),
    ('ch_pos_idxs', f'{MAX_CHILDREN}u1'),
    ('ch_dep_idxs', f'{MAX_CHILDREN}u1'),
]


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


class SentenceParsingDataset(Dataset):
    def __init__(self, filepath, dep_parser, vocab_map, pos_map, dep_map):
        '''
            filepath: path to the corpus (one line per sentence)
            dep_parser: CoreNLPDependencyParser
            vocab_map: LookupTable
            pos_map: LookupTable
            dep_map: LookupTable
            
            Note:
                `0` is treated as padding index.
                LookupTable should map OOV to the `unknown` index.
        '''
        self.dep_parser = dep_parser
        self.vocab_map = vocab_map
        self.pos_map = pos_map
        self.dep_map = dep_map
        with open(filepath, 'r') as f:
            self.sents = [line.strip() for line in f]

    def __getitem__(self, index):
        return self.parse(self.sents[index])

    def __len__(self):
        return len(self.sents)

    def parse(self, sentence):
        token_arr = list(self.dep_parser.tokenize(sentence))
        nodes = self.dep_parser.parse_one(token_arr).nodes
        n_word = len(nodes) - 1
        sent = np.zeros(n_word, dtype=WORD_TYPE)
        for i, token in enumerate(token_arr):
            node = nodes[i+1]
            word = sent[i]

            word['word_idx'] = self.vocab_map[token]
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


class SentenceParsedDataset:
    def __init__(self, filepath):
        self.sents = np.load(filepath)

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

        idx_count = loader.idx_count
        arr = idx_count ** loader.neg_power
        self.neg_prob = np2tor(arr / arr.sum())
        ratio = loader.sub_threshold / (idx_count / idx_count.sum())
        self.sub_prob = np.sqrt(ratio) + ratio

        self.in_iter = iter(loader.in_loader)
        self.queue = []

    def gen_context(self, sent):
        word_idx = sent['word_idx']
        size = self.window_size
        # pad zeros
        pad_word_idx = np.pad(word_idx, (size, size), 'constant')
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
                # get the next sentence, use cache if available
                sent = next(self.in_iter)[0]
            except StopIteration:
                break

            # subsampling
            mask = self.sub_prob[sent['word_idx']] > np.random.random_sample(len(sent))
            sent = sent[mask]
            if len(sent) < 2:
                continue
            # generate context
            ctx = self.gen_context(sent)
            n_sample += len(sent)
            self.queue.append((sent, ctx))

        if n_sample == 0:
            # end of iteration, set cached to True
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

        word_idx = np2tor(sent['word_idx']).long()
        pos_idx = np2tor(sent['pos_idx']).long()
        pr_pos_idx = np2tor(sent['pr_pos_idx']).long()
        pr_dep_idx = np2tor(sent['pr_dep_idx']).long()
        ch_pos_idxs = np2tor(sent['ch_pos_idxs']).long()
        ch_dep_idxs = np2tor(sent['ch_dep_idxs']).long()
        ctx_idxs = np2tor(ctx).long()
        ctx_len = (ctx == self.padding_index).sum(1)
        neg_idxs = self.neg_prob.multinomial(n_sample * NEG_SAMPLES)
        neg_idxs = neg_idxs.view(n_sample, -1)
        # neg_mask = (neg_idxs == word_idx.unsqueeze(-1)).float()
        return (word_idx, pos_idx,
                pr_pos_idx, pr_dep_idx,
                ch_pos_idxs, ch_dep_idxs,
                ctx_idxs, ctx_len, neg_idxs)


class CBOWLoader:
    def __init__(self, dataset, window_size, idx_count,
                 neg_power=0.75, sub_threshold=1e-5,
                 batch_size=1, shuffle=False, num_workers=0):
        '''
            window_size: int
            idx_count: numpy array (1D)
        '''
        self.dataset = dataset
        self.window_size = window_size
        self.idx_count = idx_count
        self.neg_power = neg_power
        self.sub_threshold = sub_threshold
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.in_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                    num_workers=num_workers,
                                    collate_fn=lambda x: x)

    def __iter__(self):
        return CBOWLoaderIter(self)
