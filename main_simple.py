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
#parser.add_argument("--processes", type=int, default=4, help="number of processes")
#parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data processsing")
parser.add_argument("--iter", type=int, default=5, help="number of iterations")
parser.add_argument("--lr", type=float, default=-1.0, help="initial learning rate")
parser.add_argument("--batch_size", type=int, default=100, help="(max) batch size")
parser.add_argument("--cuda", action='store_true', default=False, help="enable cuda")
parser.add_argument("--output_ctx", action='store_true', default=False, help="output context embeddings")


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

'''
class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #cond1 = (input > 6.0).float()
        #cond2 = (input > -6.0).float()
        #ret = cond1 + (1-cond1) * input.sigmoid()
        #ret = cond2 * ret 
        #return ret
        return input.sigmoid()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
'''

class mCBOW(nn.Module):
    def __init__(self, args):
        super(mCBOW, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.zero_()
        self.window = args.window
        self.negative = args.negative
        self.use_cuda = args.cuda
        self.pad_idx = args.vocab_size

    def forward(self, data):
        self.emb0_lookup.weight.data[self.pad_idx].fill_(0)

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
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(pos_ips) )
        neg_loss = torch.sum( -F.logsigmoid(-neg_ips) )

        return pos_loss, neg_loss


class CBOW(nn.Module):
    def __init__(self, args):
        super(CBOW, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        #self.emb1_lookup = nn.Embedding(args.vocab_size+1, args.size, padding_idx=args.vocab_size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.zero_()
        self.window = args.window
        self.negative = args.negative
        self.use_cuda = args.cuda
        self.pad_idx = args.vocab_size

    def forward(self, data):
        self.emb0_lookup.weight.data[self.pad_idx].fill_(0)

        ctx_indices = data[:, 0:2*self.window]
        ctx_lens = data[:, 2*self.window].float()
        word_idx = data[:, 2*self.window+1]
        neg_indices = data[:, 2*self.window+2:2*self.window+2+self.negative]
        neg_mask = data[:, 2*self.window+2+self.negative:].float()

        c_embs = self.emb0_lookup(ctx_indices)
        w_embs = self.emb1_lookup(word_idx)
        n_embs = self.emb1_lookup(neg_indices)
        '''
        if self.use_cuda:
            c_embs = c_embs.cuda()
            w_embs = w_embs.cuda()
            n_embs = n_embs.cuda()
        '''
        #c_embs = torch.mean(c_embs, 1, keepdim=True)
        #print('ctx_ind')
        #print(ctx_indices)
        #print('ctx_lens')
        #print(ctx_lens)
        #print('ctx_ind')
        #print(ctx_indices[0])
        #print('word_ind')
        #print(word_idx)
        #print('neg_ind')
        #print(neg_indices)
        #print('neg_mask')
        #print(neg_mask)
        #print(word_idx[:3])
        #print('ctx_emb')
        #print(c_embs)
        #print('')
        '''
        c_embs = torch.sum(c_embs, 1, keepdim=True)
        c_embs = c_embs.permute(1,2,0) / ctx_lens
        c_embs = c_embs.permute(2,0,1)
        '''
        c_embs = CBOWMean.apply(c_embs, ctx_lens)

        #print('neg_ind')
        #print(neg_indices[:3, :])
        #pos_logits = torch.sum(c_embs[:,0,:] * w_embs, 1, keepdim=True)

        #print(c_embs.size())
        #print(w_embs.size())
        #print(n_embs.size())
        # my sigmoid
        pos_ips = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_ips = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]
        pos_logits = MySigmoid.apply(pos_ips)
        neg_logits = MySigmoid.apply(neg_ips)
        neg_logits = neg_logits * neg_mask
        #pos_logits = F.sigmoid(pos_ips)
        #neg_logits = F.sigmoid(neg_ips)
        #print(pos_logits[0])
        #print(neg_logits[0])

        # discrete sigmoid
        '''
        pos_ips = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_ips = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]
        cond1 = (pos_ips > 6.0).float()
        cond2 = (pos_ips > -6.0).float()
        pos_logits = cond1 + (1-cond1)*F.sigmoid(pos_ips)
        pos_logits = cond2*pos_logits 
        cond1 = (neg_ips > 6.0).float()
        cond2 = (neg_ips > -6.0).float()
        neg_logits = cond1 + (1-cond1)*F.sigmoid(neg_ips)
        neg_logits = cond2*neg_logits 
        '''

        # vanilla
        '''
        pos_logits = torch.sum(c_embs[:,0,:] * w_embs, 1)
        neg_logits = torch.bmm(n_embs, c_embs.permute(0,2,1))[:,:,0]
        pos_logits = torch.clamp(pos_logits, -10, 10)
        neg_logits = torch.clamp(neg_logits, -10, 10)
        '''
        #print(pos_logits)
        #print(neg_logits)
        #print(pos_logits.size())
        #print(neg_logits.size())


        #return torch.cat((pos_logits, neg_logits), 1)
        #ones = Variable(torch.ones(pos_logits.data.size()).cuda(), requires_grad=False) 
        #print(F.sigmoid(pos_logits)[0])
        #print(F.sigmoid(neg_logits)[0])
        
        # Init Loss func
        #pos_loss = torch.mean( 1 - F.sigmoid(pos_logits) )
        #pos_loss = torch.mean( F.sigmoid(-pos_logits) )

        '''
        # Neg Log Likelihood
        pos_loss = torch.mean( -F.logsigmoid(pos_logits) )
        neg_loss = torch.mean( torch.mean(-F.logsigmoid(-neg_logits), 1) )
        return pos_loss + neg_loss
        '''

        #print(pos_logits.size())
        #print(neg_logits.size())
        # Mean Squared Error
        pos_loss = torch.mean( 0.5 * torch.pow(1-pos_logits, 2) )
        neg_loss = torch.mean( torch.sum(0.5 * torch.pow(0-neg_logits, 2), 1) )
        #neg_loss = torch.mean( 0.5 * torch.pow(0-neg_logits, 2) )
        #loss = pos_loss + neg_loss
        #loss.backward()
        return pos_loss, neg_loss
        #return pos_loss 

class SG(nn.Module):
    def __init__(self, args):
        super(SG, self).__init__()
        self.emb0_lookup = nn.Embedding(args.vocab_size+1, args.size, sparse=True)
        self.emb1_lookup = nn.Embedding(args.vocab_size, args.size, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/args.size, 0.5/args.size)
        #self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.zero_()
        self.window = args.window
        self.negative = args.negative
        #self.use_cuda = args.cuda
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
        neg_ips = neg_ips * neg_mask

        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(pos_ips) )
        neg_loss = torch.sum( -F.logsigmoid(-neg_ips) )

        return pos_loss, neg_loss

# Initialize model.
def init_net(args):
    if args.cbow == 1:
        if args.lr == -1.0:
            vars(args)['lr'] = 0.05
        return mCBOW(args)
    elif args.cbow == 0:
        if args.lr == -1.0:
            vars(args)['lr'] = 0.025
        return SG(args)

if __name__ == '__main__':
    args = parser.parse_args()
    print("Starting training using file %s" % args.train)
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()

    #word2idx, word_list, freq, table_ptr_val = build_vocab(args)
    word2idx, word_list, freq = build_vocab(args)
    #vars(args)['table_ptr_val'] = table_ptr_val
    #pdb.set_trace()
    #neg_sample_table = mp.Array('i', neg_sample_table)
    #pdb.set_trace()

    #word_count_actual = mp.Value('i', 0)
    
    #data_producer.init_rng(args.processes)

    model = init_net(args)
    
    #model.share_memory()
    if args.cuda:
        model.cuda()

    #loss_fn = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.negative > 0:
        table_ptr_val = data_producer.init_unigram_table(word_list, freq, args.train_words)
    #pdb.set_trace()
    
    vars(args)['t_start'] = time.monotonic()

    train_file = open(args.train)
    train_file.seek(0, 0)
    
    word_count_actual = 0
    for it in range(args.iter):
        #print("iter: %d" % it)
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
                    #sent_queue.put(copy.deepcopy(sentence))
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
                    
                    # train cbow architecture
                    if args.cbow == 1:
                        for chunk in data_producer.cbow_producer(sent_id, len(sent_id), table_ptr_val, args.window, args.negative, args.vocab_size, args.batch_size, next_random):
                            #data_queue.put(chunk)
                            # feed to model
                            if args.cuda:
                                data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                            else:
                                data = Variable(torch.LongTensor(chunk), requires_grad=False)

                            optimizer.zero_grad()
                            ''' 
                            model.emb0_lookup.weight.data[model.pad_idx].fill_(0)

                            ctx_indices = data[:, 0:2*args.window]
                            ctx_lens = data[:, 2*args.window].float()
                            word_idx = data[:, 2*args.window+1]
                            neg_indices = data[:, 2*args.window+2:2*args.window+2+args.negative]
                            neg_mask = data[:, 2*args.window+2+args.negative:].float()

                            c_embs = model.emb0_lookup(ctx_indices)
                            w_embs = model.emb1_lookup(word_idx)
                            n_embs = model.emb1_lookup(neg_indices)

                            mean_c_embs = CBOWMean.apply(c_embs, ctx_lens)

                            pos_ips = torch.sum(mean_c_embs[:,0,:] * w_embs, 1)
                            neg_ips = torch.bmm(n_embs, mean_c_embs.permute(0,2,1))[:,:,0]
                            pos_logits = MySigmoid.apply(pos_ips)
                            neg_logits = MySigmoid.apply(neg_ips)
                            neg_logits = neg_logits * neg_mask

                            c_embs.retain_grad()
                            w_embs.retain_grad()
                            n_embs.retain_grad()
                            pos_ips.retain_grad()
                            neg_ips.retain_grad()
                            pos_logits.retain_grad()
                            neg_logits.retain_grad()
                            
                            mpos_loss = torch.mean( 0.5 * torch.pow(1-pos_logits, 2) )
                            mneg_loss = torch.mean( torch.sum(0.5 * torch.pow(0-neg_logits, 2), 1) )
                            mloss = mpos_loss + mneg_loss
                            '''
                            
                            #if word_idx.data[0] == 0:
                            #    pdb.set_trace()
                            #if pos_logits.data[0] == 1 or 0 in (neg_logits+(1-neg_mask)).data[0]:
                            #    pdb.set_trace()

                            #if chunk[0,-1] == 0:
                            #    pdb.set_trace()

                            #loss = model(data)
                            #print("\n\n from nn.module:\n")
                            pos_loss, neg_loss = model(data)
                            loss = pos_loss + neg_loss
                            loss.backward()
                            optimizer.step()
                    elif args.cbow == 0:
                        for chunk in data_producer.sg_producer(sent_id, len(sent_id), table_ptr_val, args.window, args.negative, args.vocab_size, args.batch_size, next_random):
                            if args.cuda:
                                data = Variable(torch.LongTensor(chunk).cuda(), requires_grad=False)
                            else:
                                data = Variable(torch.LongTensor(chunk), requires_grad=False)

                            optimizer.zero_grad()
                            '''
                            word_idx = data[:, 0]
                            ctx_idx = data[:, 1]
                            neg_indices = data[:, 2:2+args.negative]
                            neg_mask = data[:, 2+args.negative:].float()

                            w_embs = model.emb0_lookup(word_idx)
                            c_embs = model.emb1_lookup(ctx_idx)
                            n_embs = model.emb1_lookup(neg_indices)

                            pos_ips = torch.sum(w_embs * c_embs, 1)
                            neg_ips = torch.bmm(n_embs, torch.unsqueeze(w_embs,2))[:,:,0]
                            neg_ips = neg_ips * neg_mask

                            # Neg Log Likelihood
                            pos_loss = torch.sum( -F.logsigmoid(pos_ips) )
                            neg_loss = torch.sum( -F.logsigmoid(-neg_ips) )
                            
                            pdb.set_trace()
                            '''
                            pos_loss, neg_loss = model(data)
                            loss = pos_loss + neg_loss
                            loss.backward()
                            optimizer.step()

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
                    #print(model.emb0_lookup.weight.data.cpu().numpy()[0,:20])
            else:
                prev += s
        word_count_actual += word_cnt - last_word_cnt
    print("")
    print(sentence_cnt)
    # output vectors
    if args.cuda:
        embs = model.emb0_lookup.weight.data.cpu().numpy()
    else:
        embs = model.emb0_lookup.weight.data.numpy()

    data_producer.write_embs(args.output, word_list, embs, args.vocab_size, args.size)
    print("")

