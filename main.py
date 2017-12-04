#!/usr/bin/env python3

import argparse
from collections import Counter
import copy
import pdb
import re
import sys
import threading
import time
import queue

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import data_producer

from multiprocessing import set_start_method

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
    #train_file = open(args.train, 'r')
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
    for i,w in enumerate(word_list):
        word2idx[w] = i

    print("Vocab size: %ld" % len(word2idx))
    print("Words in train file: %ld" % word_count)
    vars(args)['vocab_size'] = len(word2idx)
    vars(args)['train_words'] = word_count

    return word2idx, word_list, freq

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

class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        cond1 = (input > 6.0).float()
        cond2 = (input > -6.0).float()
        ret = cond1 + (1-cond1) * input.sigmoid()
        ret = cond2 * ret
        return ret
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

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

    def forward(self, data):
        #self.emb0_lookup.weight.data[self.pad_idx].fill_(0)

        ctx_indices = data[:, 0:2*self.window]
        ctx_lens = data[:, 2*self.window].float()
        word_idx = data[:, 2*self.window+1]
        neg_indices = data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = data[:, 2*self.window+2+self.negative:].float()

        c_embs = self.emb0_lookup(ctx_indices)
        w_embs = self.emb1_lookup(word_idx)
        n_embs = self.emb1_lookup(neg_indices)

        c_embs = CBOWMean.apply(c_embs, ctx_lens)

        pos_ips = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_ips = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]
        '''
        pos_logits = MySigmoid.apply(pos_ips)
        neg_logits = MySigmoid.apply(neg_ips)
        neg_logits = neg_logits * neg_mask
        pos_loss = torch.sum( 0.5 * torch.pow(1-pos_logits, 2) )
        neg_loss = torch.sum( 0.5 * torch.pow(0-neg_logits, 2) )

        '''
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)) )
        neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )

        return pos_loss + neg_loss

class SG(nn.Module):
    def __init__(self, args):
        super(SG, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        #self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.zero_()
        self.window = args.window
        self.negative = args.negative
        self.pad_idx = args.vocab_size

    def forward(self, data):
        #self.emb0_lookup.weight.data[self.pad_idx].fill_(0)

        word_idx = data[:, 0]
        ctx_idx = data[:, 1]
        neg_indices = data[:, 2:2+self.negative]
        neg_mask = data[:, 2+self.negative:].float()

        w_embs = self.emb0_lookup(word_idx)
        c_embs = self.emb1_lookup(ctx_idx)
        n_embs = self.emb1_lookup(neg_indices)

        pos_ips = torch.sum(w_embs * c_embs, 1)
        neg_ips = torch.bmm(n_embs, torch.unsqueeze(w_embs,1).permute(0,2,1))[:,:,0]
        '''
        pos_logits = MySigmoid.apply(pos_ips)
        neg_logits = MySigmoid.apply(neg_ips)
        neg_logits = neg_logits * neg_mask
        pos_loss = torch.sum( 0.5 * torch.pow(1-pos_logits, 2) )
        neg_loss = torch.sum( 0.5 * torch.pow(0-neg_logits, 2) )

        neg_ips = neg_ips * neg_mask
        '''

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
def train_process_sent_producer(p_id, data_queue, word_count_actual, word2idx, word_list, freq, args):
    if args.negative > 0:
        table_ptr_val = data_producer.init_unigram_table(word_list, freq, args.train_words)

    train_file = open(args.train)
    file_pos = args.file_size * p_id // args.processes
    train_file.seek(file_pos, 0)
    while True:
        try:
            train_file.read(1)
        except UnicodeDecodeError:
            file_pos -= 1
            train_file.seek(file_pos, 0)
        else:
            train_file.seek(file_pos, 0)
            break

    batch_count = 0
    if args.cbow == 1:
        batch_placeholder = np.zeros((args.batch_size, 2*args.window+2+2*args.negative), 'int64')
    else:
        batch_placeholder = np.zeros((args.batch_size, 2+2*args.negative), 'int64')

    for it in range(args.iter):
        train_file.seek(file_pos, 0)

        last_word_cnt = 0
        word_cnt = 0
        sentence = []
        prev = ''
        eof = False
        while True:
            #if eof or word_cnt > args.train_words / args.processes:
            if eof or train_file.tell() > file_pos + args.file_size / args.processes:
                break

            while True:
                s = train_file.read(1)
                if not s:
                    eof = True
                    #print(train_file.tell(), args.file_size)
                    break
                elif s == ' ' or s == '\t':
                    if prev in word2idx:
                        sentence.append(prev)
                    prev = ''
                    if len(sentence) >= MAX_SENT_LEN:
                        break
                elif s == '\n':
                    if prev in word2idx:
                        sentence.append(prev)
                    prev = ''
                    break
                else:
                    prev += s

            if len(sentence) > 0:
                # subsampling
                sent_id = []
                if args.sample != 0:
                    sent_len = len(sentence)
                    i = 0
                    while i < sent_len:
                        word = sentence[i]
                        f = freq[word] / args.train_words
                        pb = (np.sqrt(f / args.sample) + 1) * args.sample / f;

                        if pb > np.random.random_sample():
                            sent_id.append( word2idx[word] )
                        i += 1

                if len(sent_id) < 2:
                    word_cnt += len(sentence)
                    sentence.clear()
                    continue

                next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
                if args.cbow == 1: # train CBOW
                    chunk = data_producer.cbow_producer(sent_id, len(sent_id), table_ptr_val,
                                args.window, args.negative, args.vocab_size, args.batch_size, next_random)
                elif args.cbow == 0: # train skipgram
                    chunk = data_producer.sg_producer(sent_id, len(sent_id), table_ptr_val,
                                args.window, args.negative, args.vocab_size, args.batch_size, next_random)

                chunk_pos = 0
                while chunk_pos < chunk.shape[0]:
                    remain_space = args.batch_size - batch_count
                    remain_chunk = chunk.shape[0] - chunk_pos

                    if remain_chunk < remain_space:
                        take_from_chunk = remain_chunk
                    else:
                        take_from_chunk = remain_space

                    batch_placeholder[batch_count:batch_count+take_from_chunk, :] = chunk[chunk_pos:chunk_pos+take_from_chunk, :]
                    batch_count += take_from_chunk

                    if batch_count == args.batch_size:
                        data_queue.put(batch_placeholder)
                        batch_count = 0

                    chunk_pos += take_from_chunk

                word_cnt += len(sentence)
                if word_cnt - last_word_cnt > 10000:
                    with word_count_actual.get_lock():
                        word_count_actual.value += word_cnt - last_word_cnt
                    last_word_cnt = word_cnt
                sentence.clear()

        with word_count_actual.get_lock():
            word_count_actual.value += word_cnt - last_word_cnt
        #print("p_id: %d, word_cnt:%d" % (p_id, word_cnt))

    if batch_count > 0:
        data_queue.put(batch_placeholder[:batch_count,:])
    data_queue.put(None)
    #print(p_id, file_pos, train_file.tell(), args.file_size)

def train_process(p_id, word_count_actual, word2idx, word_list, freq, args, model):
    data_queue = mp.SimpleQueue()
    #data_queue = mp.Queue(100000)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    t = mp.Process(target=train_process_sent_producer, args=(p_id, data_queue, word_count_actual, word2idx, word_list, freq, args))
    t.start()

    # get from data_queue and feed to model
    prev_word_cnt = 0
    while True:
        d = data_queue.get()
        if d is None:
            break
        else:
            # lr anneal & output
            if word_count_actual.value - prev_word_cnt > 10000:
                lr = args.lr * (1 - word_count_actual.value / (args.iter * args.train_words))
                if lr < 0.0001 * args.lr:
                    lr = 0.0001 * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Words/sec: %f" % (lr, word_count_actual.value / (args.iter * args.train_words) * 100, word_count_actual.value / (time.monotonic() - args.t_start)))
                sys.stdout.flush()
                prev_word_cnt = word_count_actual.value

            if args.cuda:
                data = Variable(torch.LongTensor(d).cuda(), requires_grad=False)
            else:
                data = Variable(torch.LongTensor(d), requires_grad=False)

            if args.cbow == 1:
                optimizer.zero_grad()
                loss = model(data)
                loss.backward()
                optimizer.step()
                model.emb0_lookup.weight.data[args.vocab_size].fill_(0)
            elif args.cbow == 0:
                optimizer.zero_grad()
                loss = model(data)
                loss.backward()
                optimizer.step()

    t.join()

if __name__ == '__main__':
    set_start_method('forkserver')

    args = parser.parse_args()
    print("Starting training using file %s" % args.train)
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()

    word2idx, word_list, freq = build_vocab(args)

    word_count_actual = mp.Value('i', 0)

    model = init_net(args)
    model.share_memory()
    if args.cuda:
        model.cuda()

    vars(args)['t_start'] = time.monotonic()
    processes = []
    for p_id in range(args.processes):
        p = mp.Process(target=train_process, args=(p_id, word_count_actual, word2idx, word_list, freq, args, model))
        #p = mp.Process(target=train_process, args=(p_id, word_count_actual, word2idx, word_list, freq, args, model, optimizer))
        #p = threading.Thread(target=train_process, args=(p_id, word_count_actual, word2idx, freq, args, model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("\nStage 1, ", time.monotonic() - args.t_start, " secs, ", word_count_actual.value)

    # output vectors
    if args.cuda:
        embs = model.emb0_lookup.weight.data.cpu().numpy()
    else:
        embs = model.emb0_lookup.weight.data.numpy()

    data_producer.write_embs(args.output, word_list, embs, args.vocab_size, args.size)
    print("")

