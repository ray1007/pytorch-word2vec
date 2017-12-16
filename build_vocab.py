#!/usr/bin/env python3

import argparse
from collections import Counter
#import pdb
import pickle
import re
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="", help="training file")

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
    for i,w in enumerate(word_list):
        word2idx[w] = i

    print("Vocab size: %ld" % len(word2idx))
    print("Words in train file: %ld" % word_count)
    vars(args)['vocab_size'] = len(word2idx)
    vars(args)['train_words'] = word_count

    # FAQ page: https://nlp.stanford.edu/software/parser-faq.shtml
    # POS tags: https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf
    # Dep rels: http://universaldependencies.org/u/dep/index.html

    pos2idx = {}
    with open('pos_tag_set.txt') as f:
        idx = 0
        for line in f:
            pos2idx[ line.strip() ] = idx
            idx += 1

    dep2idx = {}
    with open('dep_rel_set.txt') as f:
        idx = 0
        for line in f:
            dep2idx[ line.strip() ] = idx
            idx += 1

    return word2idx, word_list, freq, pos2idx, dep2idx


if __name__ == '__main__':

    args = parser.parse_args()
    print("Starting training using file %s" % args.train)
    train_file = open(args.train)
    train_file.seek(0, 2)
    vars(args)['file_size'] = train_file.tell()

    # objs = word2idx, word_list, freq, pos2idx, dep2idx
    objs = build_vocab(args)

    with open('vocab_dict.pkl', 'wb') as f:
        pickle.dump(objs, f, pickle.HIGHEST_PROTOCOL)


