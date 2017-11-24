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
parser.add_argument("--size", type=int, default=300, help="word embedding dimension")
parser.add_argument("--window", type=int, default=5, help="context window size")
parser.add_argument("--sample", type=float, default=1e-4, help="subsample threshold")
parser.add_argument("--negative", type=int, default=10, help="number of negative samples")
parser.add_argument("--delta", type=float, default=0.15, help="create new sense for a type if similarity lower than this value.")
parser.add_argument("--min_count", type=int, default=5, help="minimum frequency of a word")
#parser.add_argument("--processes", type=int, default=4, help="number of processes")
#parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data processsing")
parser.add_argument("--iter", type=int, default=5, help="number of iterations")
parser.add_argument("--lr", type=float, default=-1.0, help="initial learning rate")
parser.add_argument("--batch_size", type=int, default=100, help="(max) batch size")
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


class CSV(nn.Module):
    def __init__(self, args):
        super(CSV, self).__init__()
        self.global_embs = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.sense_embs = nn.Embedding(args.vocab_size*3, args.size, sparse=True)
        self.word2sense = [ [i] for i in range(args.vocab_size) ]
        self.ctx_weight = torch.nn.Parameter(torch.zeros(2*args.window, args.size))

        self.global_embs.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.sense_embs.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        #self.ctx_weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.ctx_weight.data.normal_(0,1)

        self.n_senses = args.vocab_size
        self.sense_capacity = args.vocab_size*3
        self.batch_size = args.batch_size
        self.window = args.window
        self.negative = args.negative
        self.pad_idx = args.vocab_size

    def get_context_feats(self, ctx_type_indices):
        ctx_type_embs = self.global_embs(ctx_type_indices)
        ctx_feats = torch.sum(ctx_type_embs * self.ctx_weight, 1)
        return ctx_feats

    def get_possible_sense_embs(self, type_indices):
        sense_indices = []
        sense2idx = {}
        #sense_indices = set()
        for type_id in type_indices:
            for s_id in self.word2sense[type_id]:
                #sense_indices.add(s_id)
                if s_id not in sense2idx:
                    sense2idx[s_id] = len(sense_indices)
                    sense_indices.append( s_id )
        sense_indices = np.array(sense_indices)

        sense_embs = self.sense_embs(Variable(torch.LongTensor(sense_indices).cuda()))
        
        return sense2idx, sense_embs.cpu().data.numpy()

    def add_sense_embs(self, created_sense_embs):
        n_created = np.array(created_sense_embs).shape[0]
        if n_created == 0:
            return
        created_sense_embs = torch.FloatTensor(np.array(created_sense_embs))
        self.sense_embs.weight.data[self.n_senses:self.n_senses+n_created, :] = created_sense_embs
        self.n_senses += n_created

        # max number of created_sense_emb = batch_size
        # reallocate sense_embs if needed to ensure enough capacity.
        if self.n_senses + self.batch_size > self.sense_capacity:
            new_capacity = self.sense_capacity * 3 // 2
            new_embs = nn.Embedding(new_capacity, args.size, sparse=True)
            new_embs.weight.data[:self.n_senses, :] = self.sense_embs.weight.data[:self.n_senses, :]
            self.sense_embs = new_embs
            self.sense_capacity = new_capacity

    def forward(self, data, with_neg=True):
        ctx_type_indices = data[:, 0:2*self.window]
        pos_sense_idx = data[:, 2*self.window+1]
        if with_neg:
            neg_sense_indices = data[:, 2*self.window+2:2*self.window+2+self.negative]
            neg_mask = data[:, 2*self.window+2+self.negative:].float()

        ctx_type_embs = self.global_embs(ctx_type_indices)
        pos_sense_embs = self.sense_embs(pos_sense_idx)
        if with_neg:
            neg_sense_embs = self.sense_embs(neg_sense_indices)

        ctx_feats = torch.sum(ctx_type_embs * self.ctx_weight, 1, keepdim=True)

        # Neg Log Likelihood
        pos_ips = torch.sum(ctx_feats[:,0,:] * pos_sense_embs, 1)
        pos_loss = torch.sum( -F.logsigmoid(torch.clamp(pos_ips,max=10,min=-10)))
        if with_neg:
            neg_ips = torch.bmm(neg_sense_embs, ctx_feats.permute(0,2,1))[:,:,0]
            neg_loss = torch.sum( -F.logsigmoid(torch.clamp(-neg_ips,max=10,min=-10)) * neg_mask )
            return pos_loss, neg_loss
        else:
            return pos_loss


# Initialize model.
def init_net(args):
    if args.lr == -1.0:
        vars(args)['lr'] = 0.05
    return CSV(args)

def save_model(model, args, word2idx):
    torch.save({
        'word2idx':word2idx,
        'args':args,
        'word2sense': model.word2sense,
        'n_senses': model.n_senses,
        'params': model.state_dict()
    }, 'test.pth.tar')
        
def load_model(filename):
    checkpoint = torch.load(filename)
    word2idx = checkpoint['word2idx']
    args = checkpoint['args']
    model = CSV(args)
    if args.cuda:
        model.cuda()

    model.global_embs.weight.data = checkpoint['params']['global_embs.weight']
    model.sense_embs.weight.data = checkpoint['params']['sense_embs.weight']
    model.ctx_weight.data = checkpoint['params']['ctx_weight']
    model.word2sense = checkpoint['word2sense']
    model.n_senses = checkpoint['n_senses']

    return model, word2idx

if __name__ == '__main__':
    args = parser.parse_args()
    print("Starting training using file %s" % args.train)
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()

    word2idx, word_list, freq = build_vocab(args)
    model = init_net(args)
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.negative > 0:
        table_ptr_val = data_producer.init_unigram_table(word_list, freq, args.train_words)
    
    vars(args)['t_start'] = time.monotonic()
    # stage 1
    print("\nBegin stage 1")
    train_file = open(args.train)
    train_file.seek(0, 0)
    
    word_count_actual = 0
    for it in range(args.iter):
        train_file.seek(0, 0)

        sentence_cnt = 0
        last_word_cnt = 0
        word_cnt = 0
        sentence = []
        prev = ''
        while True:
            if word_cnt > args.train_words:
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
                if len(sentence) > 0:
                    sent_id = []
                    # subsampling
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
                        continue
                    
                    next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
                    
                    # train 
                    for chunk in data_producer.cbow_producer(sent_id, len(sent_id), table_ptr_val, args.window, args.negative, args.vocab_size, args.batch_size, next_random):
                        #t_start = time.process_time()
                        # feed to model
                        if args.cuda:
                            data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                        else:
                            data = Variable(torch.LongTensor(chunk), requires_grad=False)
                        # 0.002s
                        optimizer.zero_grad()
                            
                        pos_loss, neg_loss = model(data)
                        loss = pos_loss + neg_loss
                        loss.backward()
                        optimizer.step()
                        model.global_embs.weight.data[args.vocab_size].fill_(0)
                        #print("train stage 1: ", time.process_time() - t_start)
                        #print("loop: ", time.process_time() - t_start)

                word_cnt += len(sentence)
                sentence_cnt += 1
                sentence.clear()
                # lr anneal & output    
                if word_cnt - last_word_cnt > 10000:
                    word_count_actual += word_cnt - last_word_cnt
                    last_word_cnt = word_cnt
                    lr = args.lr * (1 - word_count_actual / (args.iter * args.train_words))
                    if lr < 0.0001 * args.lr:
                        lr = 0.0001 * args.lr 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Words/sec: %f" % (lr, word_count_actual / (args.iter * args.train_words) * 100, word_count_actual / (time.monotonic() - args.t_start)))
                    sys.stdout.flush()
            else:
                prev += s
        word_count_actual += word_cnt - last_word_cnt
    
    #  stage 2
    vars(args)['t_start'] = time.monotonic()
    model.global_embs.requires_grad = False
    model.ctx_weight.requires_grad = False
    print("\nBegin stage 2")
    train_file = open(args.train)
    train_file.seek(0, 0)
    
    word_count_actual = 0
    for it in range(1):
        train_file.seek(0, 0)

        sentence_cnt = 0
        last_word_cnt = 0
        word_cnt = 0
        sentence = []
        prev = ''
        while True:
            if word_cnt > args.train_words:
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
                if len(sentence) > 0:
                    sent_id = []
                    # subsampling
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
                        continue
                    
                    next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
                    
                    # train 
                    for chunk in data_producer.cbow_producer(sent_id, len(sent_id), table_ptr_val, args.window, 0, args.vocab_size, args.batch_size, next_random):
                        #t_start = time.process_time()
                        # feed to model
                        if args.cuda:
                            data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                        else:
                            data = Variable(torch.LongTensor(chunk), requires_grad=False)
                        sense2idx, sense_embs = model.get_possible_sense_embs(chunk[:, 2*args.window+1].tolist())
                        zero = np.zeros((chunk.shape[0], args.size),'float32')
                        #print("get tensor: ", time.process_time() - t_start)

                        #t_start = time.process_time()
                        # get type_idx from chunk, and do sense selection here.
                        context_feats = model.get_context_feats(data[:, :2*args.window])             
                        context_feats = context_feats.cpu().data.numpy()
                        #print("ctx_feat: ", time.process_time() - t_start)
                        #t_start = time.process_time()
                        #type_ids = data[:, -1:].cpu().data.numpy()

                        chunk, created_sense_embs = data_producer.create_n_select_sense(chunk, context_feats, sense2idx, 
                                                        np.concatenate((sense_embs,zero),0), model.word2sense, chunk.shape[0], 
                                                        sense_embs.shape[0], args.size, args.window, args.delta, model.n_senses)
                        #print("create_n_select: ", time.process_time() - t_start)

                        # update n_senses & sense_embs
                        #t_start = time.process_time()
                        model.add_sense_embs(created_sense_embs)
                        #print("add_sense: ", time.process_time() - t_start)

                        #t_start = time.process_time()
                        # feed selected_sense_ids to model
                        if chunk.shape[0] > 0:
                            if args.cuda:
                                data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                            else:
                                data = Variable(torch.LongTensor(chunk), requires_grad=False)
                         
                            optimizer.zero_grad()   
                            loss = model(data, False)
                            loss.backward()
                            optimizer.step()
                        #print("train: ", time.process_time() - t_start)
                        #pdb.set_trace()
                        

                word_cnt += len(sentence)
                sentence_cnt += 1
                sentence.clear()
                # lr anneal & output    
                if word_cnt - last_word_cnt > 10000:
                    word_count_actual += word_cnt - last_word_cnt
                    last_word_cnt = word_cnt
                    lr = args.lr * (1 - word_count_actual / args.train_words)
                    if lr < 0.0001 * args.lr:
                        lr = 0.0001 * args.lr 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Words/sec: %f" % (lr, word_count_actual / args.train_words * 100, word_count_actual / (time.monotonic() - args.t_start)))
                    sys.stdout.flush()
                    #print(model.emb0_lookup.weight.data.cpu().numpy()[0,:20])
            else:
                prev += s
        word_count_actual += word_cnt - last_word_cnt
    
    #  stage 3
    vars(args)['t_start'] = time.monotonic()
    model.global_embs.requires_grad = True
    model.ctx_weight.requires_grad = True
    print("\nBegin stage 3")
    train_file = open(args.train)
    train_file.seek(0, 0)
    
    word_count_actual = 0
    for it in range(args.iter):
        train_file.seek(0, 0)

        sentence_cnt = 0
        last_word_cnt = 0
        word_cnt = 0
        sentence = []
        prev = ''
        while True:
            if word_cnt > args.train_words:
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
                if len(sentence) > 0:
                    sent_id = []
                    # subsampling
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
                        continue
                    
                    next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
                    
                    # train 
                    for chunk in data_producer.cbow_producer(sent_id, len(sent_id), table_ptr_val, args.window, args.negative, args.vocab_size, args.batch_size, next_random):
                        # feed to model
                        if args.cuda:
                            data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                        else:
                            data = Variable(torch.LongTensor(chunk), requires_grad=False)
                        
                        type_ids = chunk[:, 2*args.window+1:2*args.window+2+2*args.negative]
                        type_ids = np.reshape(type_ids, (type_ids.shape[0] * type_ids.shape[1]))
                        sense2idx, sense_embs = model.get_possible_sense_embs(type_ids.tolist())

                        # get type_idx from chunk, and do sense selection here.
                        context_feats = model.get_context_feats(data[:, :2*args.window])             
                        context_feats = context_feats.cpu().data.numpy()

                        chunk = data_producer.select_sense(chunk, context_feats, sense2idx, sense_embs,
                                        model.word2sense, chunk.shape[0], args.size, args.window, args.negative)

                        if args.cuda:
                            data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                        else:
                            data = Variable(torch.LongTensor(chunk), requires_grad=False)
                         
                        optimizer.zero_grad()   
                        loss = model(data, False)
                        loss.backward()
                        optimizer.step()
                        #pdb.set_trace()
                            
                word_cnt += len(sentence)
                sentence_cnt += 1
                sentence.clear()
                # lr anneal & output    
                if word_cnt - last_word_cnt > 10000:
                    word_count_actual += word_cnt - last_word_cnt
                    last_word_cnt = word_cnt
                    lr = args.lr * (1 - word_count_actual / (args.iter * args.train_words))
                    if lr < 0.0001 * args.lr:
                        lr = 0.0001 * args.lr 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Words/sec: %f" % (lr, word_count_actual / (args.iter * args.train_words) * 100, word_count_actual / (time.monotonic() - args.t_start)))
                    sys.stdout.flush()
            else:
                prev += s
        word_count_actual += word_cnt - last_word_cnt
     

    save_model(model, args)
    print("")

