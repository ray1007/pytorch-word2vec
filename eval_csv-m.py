#!/usr/bin/env python3

import argparse
import pdb
import re

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.autograd import Variable

from csv import load_model
import data_producer

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='', help='saved model file')
args = parser.parse_args()

#simlex999 = []

model, word2idx = load_model(args.model)
model.cpu()
model_sense_embs = model.sense_embs.weight.data.numpy()
vocab_size = model.global_embs.weight.size()[0] - 1

score_list = []
cos_list = []
with open('ratings.txt') as f:
    head = True
    for line in f:
        tokens = line.strip().lower().split('\t')
        w1 = tokens[1]
        w2 = tokens[3]
        c1 = tokens[5]
        c2 = tokens[6]
        rating = tokens[7]

        left_end = re.search('<b>', c1).span()[0]
        right_begin = re.search('</b>', c1).span()[1]
        left_contexts = re.sub('[.,;!()]', '', c1[:left_end]).split()
        right_contexts = re.sub('[.,;!()]', '', c1[right_begin:]).split()
        c1 = left_contexts[-model.window:] + right_contexts[:model.window]

        left_end = re.search('<b>', c2).span()[0]
        right_begin = re.search('</b>', c2).span()[1]
        left_contexts = re.sub('[.,;!()]', '', c2[:left_end]).split()
        right_contexts = re.sub('[.,;!()]', '', c2[right_begin:]).split()
        c2 = left_contexts[-model.window:] + right_contexts[:model.window]

        chunk = np.ones((2, 2*model.window+2))
        chunk *= vocab_size
        for idx,c in enumerate(c1):
            if c in word2idx:
                chunk[0,idx] = word2idx[c]
        chunk[0, 2*model.window+1] = 0
        if w1 in word2idx:
            chunk[0, 2*model.window+1] = word2idx[w1]
        for idx,c in enumerate(c2):
            if c in word2idx:
                chunk[1,idx] = word2idx[c]
        chunk[1, 2*model.window+1] = 0
        if w2 in word2idx:
            chunk[1, 2*model.window+1] = word2idx[w2]

        chunk = np.array(chunk, 'int64')
        ctx_feats = model.get_context_feats(Variable(torch.LongTensor(chunk[:,:2*model.window])))
        ctx_feats = ctx_feats.data.numpy()

        type_ids = chunk[:,2*model.window+1].tolist()
        #if w1 in word2idx:
        #    type_ids.append(word2idx[w1])
        #if w2 in word2idx:
        #    type_ids.append(word2idx[w2])

        try:
            sense2idx, sense_embs = model.get_possible_sense_embs(type_ids, cuda=False)
        except:
            pdb.set_trace()

        #pdb.set_trace()
        chunk = data_producer.select_sense(chunk, ctx_feats, sense2idx, sense_embs,
                    model.word2sense, chunk.shape[0], model.size, model.window, 0)
        #pdb.set_trace()

        sense_inds = np.array(chunk[:,-1])

        if w1 in word2idx and w2 in word2idx:
            emb1 = model_sense_embs[sense_inds[0], :]
            emb2 = model_sense_embs[sense_inds[1], :]
            cos = np.dot(emb1, emb2) / np.linalg.norm(emb1) / np.linalg.norm(emb2)
            cos_list.append(cos)
        else:
            print("GG")
            cos_list.append(0.0)

        score_list.append(rating)


rho, pvalue = spearmanr(score_list, cos_list)
print(rho)

pdb.set_trace()



