#!/usr/bin/env python3

import argparse
from collections import Counter
from multiprocessing import set_start_method
import pdb
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import data_producer
import dataloader
from utils import LookupTable

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="", help="training file")
parser.add_argument("--output", type=str, default="vectors.txt", help="output word embedding file")
parser.add_argument("--size", type=int, default=300, help="word embedding dimension")
parser.add_argument("--cbow", type=int, default=1, help="1 for cbow, 0 for skipgram")
parser.add_argument("--window", type=int, default=5, help="context window size")
parser.add_argument("--sample", type=float, default=1e-4, help="subsample threshold")
parser.add_argument("--negative", type=int, default=10, help="number of negative samples")
parser.add_argument("--min_count", type=int, default=5, help="minimum frequency of a word")
parser.add_argument("--processes", type=int, default=4, help="number of processes")
parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data processsing")
parser.add_argument("--iter", type=int, default=5, help="number of iterations")
parser.add_argument("--lr", type=float, default=-1.0, help="initial learning rate")
parser.add_argument("--batch_size", type=int, default=100, help="(max) batch size")
parser.add_argument("--cuda", action='store_true', default=False, help="enable cuda")
parser.add_argument("--output_ctx", action='store_true', default=False, help="output context embeddings")

MAX_SENT_LEN = 1000

# Build the vocabulary.
def file_split(f, delim=' \t\n', bufsize=1024):
    prev = ''
    while True:
        s = f.read(bufsize)
        if not s:
            break
        tokens = re.split('['+delim+']{1,}', s)
        if len(tokens) > 1:
            yield prev + tokens[0]
            prev = tokens[-1]
            for x in tokens[1:-1]:
                yield x
        else:
            prev += s
    if prev:
        yield prev

def build_vocab(args):
    vocab = Counter()
    word_count = 0
    for word in file_split(open(args.train)):
        vocab[word] += 1
        word_count += 1
        if word_count % 10000 == 0:
            sys.stdout.write('%d\r' % len(vocab))
    freq = {k:v for k,v in vocab.items() if v >= args.min_count}
    word_count = sum([freq[k] for k in freq])
    word_list = sorted(freq, key=freq.get, reverse=True)
    word2idx = {}
    vocab_map = LookupTable()
    idx_count = np.zeros((len(word_list),))
    for i,w in enumerate(word_list):
        vocab_map[w] = i
        word2idx[w] = i
        idx_count[i] = freq[w]

    print("Vocab size: %ld" % len(word2idx))
    print("Words in train file: %ld" % word_count)
    vars(args)['vocab_size'] = len(word2idx)
    vars(args)['train_words'] = word_count

    padding_index = len(word2idx)
    vocab_map.set_missing(padding_index)


    #return word2idx, word_list, freq
    return vocab_map, word_list, idx_count

class CBOWMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lens):
        ctx.save_for_backward(x)
        x = torch.sum(x, 1, keepdim=True)
        x = x.permute(1,2,0) / lens
        return x.permute(2,0,1)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_variables
        return g.expand_as(x), None

class CBOW(nn.Module):
    def __init__(self, args):
        super(CBOW, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.window = args.window
        self.negative = args.negative
        self.pad_idx = args.vocab_size

    def forward(self, word_idx, ctx_inds, ctx_lens, neg_inds):
        w_embs = self.emb1_lookup(word_idx)
        c_embs = self.emb0_lookup(ctx_inds)
        n_embs = self.emb1_lookup(neg_inds)

        c_embs = CBOWMean.apply(c_embs, ctx_lens)
        #c_embs = torch.mean(c_embs, 1, keepdim=True)
        #c_embs = torch.sum(c_embs, 1, keepdim=True)

        pos_ips = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_ips = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) )
        #neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )

        return pos_loss + neg_loss

class SG(nn.Module):
    def __init__(self, args):
        super(SG, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb1_lookup.weight.data.zero_()

        self.window = args.window
        self.negative = args.negative
        self.pad_idx = args.vocab_size

    def forward(self, data):
        word_idx = data[:, 0]
        ctx_idx = data[:, 1]
        neg_indices = data[:, 2:2+self.negative]
        neg_mask = data[:, 2+self.negative:].float()

        w_embs = self.emb0_lookup(word_idx)
        c_embs = self.emb1_lookup(ctx_idx)
        n_embs = self.emb1_lookup(neg_indices)

        pos_ips = torch.sum(w_embs * c_embs, 1)
        neg_ips = torch.bmm(n_embs, torch.unsqueeze(w_embs,1).permute(0,2,1))[:,:,0]

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )

        return pos_loss + neg_loss

# Initialize model.
def init_net(args):
    if args.cbow == 1:
        if args.lr == -1.0:
            vars(args)['lr'] = 0.05
        return CBOW(args)
    elif args.cbow == 0:
        if args.lr == -1.0:
            vars(args)['lr'] = 0.025
        return SG(args)

# Training
def train_process(p_id, word_count_actual, vocab_map, idx_count, args, model):
    dataset = dataloader.SentenceDataset(
        "%s.%d" % (args.train, p_id),
        vocab_map,
    )
    loader = dataloader.CBOWLoader(dataset, args.window, idx_count,
                                   padding_index=len(idx_count),
                                   sub_threshold=args.sample,
                                   batch_size=args.batch_size,
                                   num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    #st = time.monotonic()
    prev_cnt = 0
    cnt = 0
    for it in range(args.iter):
        for batch in loader:
            cnt += len(batch[0])
            if cnt - prev_cnt > 10000:
                with word_count_actual.get_lock():
                    word_count_actual.value += cnt - prev_cnt
                delta = time.monotonic() - args.t_start
                #print('\rtotal words: {0}, time spent: {1:.6f}, speed: {2:.6f}'.format(cnt, delta, cnt / delta) ,end='')
                print('\rtotal words: {0}, time spent: {1:.6f}, speed: {2:.6f}'.format(word_count_actual.value, delta, word_count_actual.value / delta) ,end='')
                prev_cnt = cnt

            if args.cbow == 1:
                word_idx = Variable(batch[0].cuda(), requires_grad=False)
                ctx_inds = Variable(batch[1].cuda(), requires_grad=False)
                ctx_lens = Variable(batch[2].cuda(), requires_grad=False)
                neg_inds = Variable(batch[3].cuda(), requires_grad=False)

                optimizer.zero_grad()
                loss = model(word_idx, ctx_inds, ctx_lens, neg_inds)
                loss.backward()
                optimizer.step()
                model.emb0_lookup.weight.data[args.vocab_size].fill_(0)
            elif args.cbow == 0:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()

if __name__ == '__main__':
    set_start_method('forkserver')

    args = parser.parse_args()
    print("Starting training using file %s" % args.train)
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()

    #word2idx, word_list, freq = build_vocab(args)
    vocab_map, word_list, idx_count = build_vocab(args)

    word_count_actual = mp.Value('L', 0)

    model = init_net(args)
    model.share_memory()
    if args.cuda:
        model.cuda()

    vars(args)['t_start'] = time.monotonic()
    processes = []
    for p_id in range(args.processes):
        p = mp.Process(target=train_process, args=(p_id, word_count_actual, vocab_map, idx_count, args, model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # output vectors
    if args.cuda:
        embs = model.emb0_lookup.weight.data.cpu().numpy()
    else:
        embs = model.emb0_lookup.weight.data.numpy()

    data_producer.write_embs(args.output, word_list, embs, args.vocab_size, args.size)
    print("")
    print(time.monotonic() - args.t_start)

