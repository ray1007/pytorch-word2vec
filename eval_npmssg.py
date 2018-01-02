#!/usr/bin/env python3

import argparse
import pdb
import re

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.autograd import Variable

from npmssg import load_model

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='', help='saved model file')
args = parser.parse_args()

#simlex999 = []

model, word2idx = load_model(args.model)
model.cpu()
global_embs = model.global_embs.weight.data.numpy()
cluster_embs = model.cluster_embs.weight.data.numpy()
sense_embs = model.sense_embs.weight.data.numpy()
word2sense = model.word2sense.data.numpy()
word_sense_cnts = model.word_sense_cnts.data.numpy()
vocab_size = model.global_embs.weight.size()[0] - 1
dim = model.size

score_list = []
cos_list = []
with open('../../pytorch-word2vec/ratings.txt') as f:
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
        c1_emb = np.zeros((dim))
        for c in c1:
            if c in word2idx:
                c1_emb += global_embs[ word2idx[c] ]

        left_end = re.search('<b>', c2).span()[0]
        right_begin = re.search('</b>', c2).span()[1]
        left_contexts = re.sub('[.,;!()]', '', c2[:left_end]).split()
        right_contexts = re.sub('[.,;!()]', '', c2[right_begin:]).split()
        c2 = left_contexts[-model.window:] + right_contexts[:model.window]
        c2_emb = np.zeros((dim))
        for c in c2:
            if c in word2idx:
                c2_emb += global_embs[ word2idx[c] ]


        s1_emb = np.zeros((dim))
        if w1 in word2idx:
            max_sim = -10.0
            max_sense = -1
            for s_id in range(word_sense_cnts[ word2idx[w1] ]):
                s_id = word2sense[word2idx[w1], s_id]
                sim = np.dot(c1_emb, cluster_embs[s_id]) / np.linalg.norm(c1_emb) / np.linalg.norm(cluster_embs[s_id])
                if sim > max_sim:
                    max_sim = sim
                    max_sense = s_id
            s1_emb = sense_embs[s_id]

        s2_emb = np.zeros((dim))
        if w2 in word2idx:
            max_sim = -10.0
            max_sense = -1
            for s_id in range(word_sense_cnts[ word2idx[w2] ]):
                s_id = word2sense[word2idx[w2], s_id]
                sim = np.dot(c2_emb, cluster_embs[s_id]) / np.linalg.norm(c2_emb) / np.linalg.norm(cluster_embs[s_id])
                if sim > max_sim:
                    max_sim = sim
                    max_sense = s_id
            s2_emb = sense_embs[s_id]

        if w1 in word2idx and w2 in word2idx:
            cos = np.dot(s1_emb, s2_emb) / np.linalg.norm(s1_emb) / np.linalg.norm(s2_emb)
            cos_list.append(cos)
        else:
            print("GG")
            cos_list.append(0.0)

        score_list.append(rating)


rho, pvalue = spearmanr(score_list, cos_list)
print(rho)

pdb.set_trace()



