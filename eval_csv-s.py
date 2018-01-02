#!/usr/bin/env python3

import argparse
import pdb

import numpy as np
from scipy.stats import spearmanr
import torch

from csv import load_model

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='', help='saved model file')
args = parser.parse_args()

#simlex999 = []

model, word2idx = load_model(args.model)
model.cpu()
sense_embs = model.sense_embs.weight.data.numpy()

score_list = []
cos_list = []
with open('SimLex-999/SimLex-999.txt') as f:
    head = True 
    for line in f:
        if head:
            head = False
            continue

        tokens = line.strip().split()
        w1,w2,score = tokens[0],tokens[1],tokens[3]
        score_list.append(score)

        if w1 in word2idx and w2 in word2idx:
            emb1 = sense_embs[word2idx[w1], :]
            emb2 = sense_embs[word2idx[w2], :]

            cos = np.dot(emb1, emb2) / np.linalg.norm(emb1) / np.linalg.norm(emb2)
            cos_list.append(cos)
        else:
            cos_list.append(0.0)
        #simlex999.append((w1,w2,score))

rho, pvalue = spearmanr(score_list, cos_list)
print(rho)

pdb.set_trace()



