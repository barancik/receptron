import re, collections
import pickle
import os
from collections import Counter

def create_vocab():
    freq = Counter()
    for f in os.listdir("../data/small_files/"):
        with open("../data/small_files/"+f,"r") as file:
           print(f)
           for line in file:
               words = [w for w in line.strip().split()]
               freq += Counter([" ".join(list(a)) for a in words])
    return freq

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def save_dict(vocab,name):
    translate = {re.sub(" ","",x):x  for x in vocab.keys()}
    pickle.dump(translate,open(name,"wb"))

if __name__ == "__main__":
    vocab = create_vocab() 
    characters = len(set([z for y in vocab.keys()  for z in list(y)]))
    num_merges = 8000 - characters

    print("Characters: %s" % characters)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print("%s: %s (%s)" % (i, best,pairs[best]))
    save_dict(vocab,"word_pieces.dict_new_8000")
