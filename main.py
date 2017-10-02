#!/usr/bin/env python3

import argparse
from collections import Counter
import copy
import pdb
import re
import sys
import threading
import time

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
parser.add_argument("--cuda", action='store_true', default=False, help="enable cuda")


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
    freq = {k:v for k,v in vocab.items() if v > args.min_count}
    word_count = sum([freq[k] for k in freq])
    word_freq_list = sorted(freq.items(), key=lambda x:x[1], reverse=True)
    word2idx = {}
    for i,(w,f) in enumerate(word_freq_list):
        word2idx[w] = i

    print("Vocab size: %ld" % len(word2idx))
    print("Words in train file: %ld" % word_count)
    vars(args)['vocab_size'] = len(word2idx)
    vars(args)['train_words'] = word_count
    return word2idx, freq

class CBOW(nn.Module):
    def __init__(self, args):
        super(CBOW, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size)
        self.emb1_lookup = nn.Embedding(args.vocab_size+1, args.size)
        self.window = args.window

    def forward(self, data):
        ctx_indices = data[:, 0:2*self.window]
        word_idx = data[:, 2*self.window]
        neg_indices = data[:, 2*self.window+1:]

        c_embs = self.emb0_lookup(ctx_indices)
        c_embs = torch.mean(c_embs, 1, keepdim=True)
        w_embs = self.emb1_lookup(word_idx)
        n_embs = self.emb1_lookup(neg_indices)

        pos_logits = torch.sum(c_embs[:,0,:] * w_embs, 1, keepdim=True)
        neg_logits = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]

        return torch.cat((pos_logits, neg_logits), 1)
#class Word2vec(nn.Module):


# Initialize model.
def init_net(args):
    if args.cbow == 1:
        vars(args)['lr'] = 0.05
        return CBOW(args)
    elif args.cbow == 0:
        pass

# Training
def train_process_worker(sent_queue, data_queue, word2idx, freq, args):
    while True:
        sent = sent_queue.get()
        if sent is None:
            data_queue.put(None)
            break

        #print(sent)
        sent_id = []
        tStart = time.process_time()
        # subsampling
        if args.sample != 0:
            sent_len = len(sent)
            i = 0
            while i < sent_len:
                word = sent[i]
                f = freq[word] / args.train_words
                pb = np.sqrt(f / args.sample + 1) * args.sample / f;

                if pb < np.random.random_sample():
                    del sent[i]
                    sent_len -= 1
                else:
                    i += 1
                    sent_id.append( word2idx[word] )
        print("@train_pc_worker-part1: %f" % (time.process_time() - tStart))
        tStart = time.process_time()

        # train cbow architecture
        if args.cbow == 1:
            data = data_producer.cbow_producer(sent_id, len(sent_id), args.window, args.negative, args.vocab_size)
            data_queue.put(data)
        elif args.cbow == 0:
            for i in range(len(sent)):
                word_idx = word2idx[ sent[i] ]
                for j in range(i-args.window, i+args.window+1):
                    if j < 0 or j >= len(sent) or j == i:
                        continue
                    ctx_idx = word2idx[ sent[j] ]
                    # data = (word_idx, ctx_idx)
                    data_queue.put((word_idx, ctx_idx, 1))

                    # negative sampling
                    for n in range(args.negative):
                        neg_idx = np.random.randint(args.vocab_size)
                        data_queue.put((word_idx, neg_idx, 0))
        print("@train_pc_worker-part2: %f" % (time.process_time() - tStart))

def train_process_sent_producer(p_id, sent_queue, data_queue, word_count_actual, word2idx, freq, args):
    train_file = open(args.train)
    file_pos = args.file_size / args.processes * p_id
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

    workers = []
    for i in range(args.num_workers):
        w = mp.Process(target=train_process_worker, args=(sent_queue, data_queue, word2idx, freq, args))
        w.start()
        workers.append(w)

    cnt = 0
    last_word_cnt = 0
    word_cnt = 0
    sentence = []
    prev = ''
    while True:
        if word_cnt > args.train_words / args.processes:
            break

        s = train_file.read(1)
        if not s:
            break
        elif s == ' ':
            if prev in word2idx:
                sentence.append(prev)
            prev = ''
        elif s == '\n':
            if prev in word2idx:
                sentence.append(prev)
            prev = ''
            sent_queue.put(copy.deepcopy(sentence))
            #print(sentence)
            word_cnt += len(sentence)
            with word_count_actual.get_lock():
                word_count_actual.value += word_cnt - last_word_cnt
            last_word_cnt = word_cnt
            #sys.stdout.write("Progess: %0.2f\r" % ( 100- (float(end_pos - current_pos) / chunk_size * 100)))
            #sys.stdout.flush()
            sentence.clear()

            print("Progess: %0.2f, Words/sec: %f" % (word_count_actual.value / args.train_words * 100, word_count_actual.value / (time.monotonic() - args.t_start)))

            #cnt += 1
            #if cnt > 400:
            #    break
        else:
            prev += s

    for i in range(args.num_workers):
        sent_queue.put(None)
    for w in workers:
        w.join()

def train_process(p_id, word_count_actual, word2idx, freq, args, model, loss_fn, optimizer):
    sent_queue = mp.SimpleQueue()
    data_queue = mp.SimpleQueue()

    #t = mp.Process(target=train_process_sent_producer, args=(sent_queue, data_queue, train_file, word_count_actual, word2idx, freq, args))
    t = mp.Process(target=train_process_sent_producer, args=(p_id, sent_queue, data_queue, word_count_actual, word2idx, freq, args))
    t.start()
    #workers.append(t)

    # get from data_queue and feed to model
    cnt = 0
    none_cnt = 0
    while True:
        #try:
        d = data_queue.get()
        if d is None:
            none_cnt += 1
            if none_cnt >= args.num_workers:
                break
        else:
            cnt += 1
            tStart = time.process_time()
            if args.cbow == 1:
                if args.cuda:
                    data = Variable(torch.LongTensor(d).cuda())
                else:
                    data = Variable(torch.LongTensor(d))
                #    print(type(data))
                #print(type(data))
                print("@train_process-part1: %f" % (time.process_time() - tStart))
                tStart = time.process_time()
                #print(data.size())
                #pass

                optimizer.zero_grad()

                logits = model(data)
                #logit = model(ctx_indices, word_idx)
                #print(ctx_indices.size(), word_idx.size())
                #print(logit.size(), label.size())

                #loss = loss_fn(logit, label)

                #loss.backward()
                #optimizer.step()
                print("@train_process-part2: %f" % (time.process_time() - tStart))
    print('@train_process: got %d data' % cnt)

    t.join()
    print("@train_process: end")

if __name__ == '__main__':
    #set_start_method('spawn')
    set_start_method('forkserver')

    args = parser.parse_args()
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()
    #vars(args)['t_start'] = time.time()
    vars(args)['t_start'] = time.monotonic()

    word2idx, freq = build_vocab(args)
    word_count_actual = mp.Value('i', 0)

    model = init_net(args)
    if args.cuda:
        model.cuda()
    #loss_fn = nn.BCELoss()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #pdb.set_trace()

    processes = []
    for p_id in range(args.processes):
        p = mp.Process(target=train_process, args=(p_id, word_count_actual, word2idx, freq, args, model, loss_fn, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
