# -*- coding: utf-8 -*-

import re
from random import shuffle

def is_slovakian(sentence):
    if "múk" in sentence:
        return True
    if "minút" in sentence:
        return True
    if "zemiak" in sentence:
        return True
    if "cibuľ" in sentence:
        return True
    if "mäso" in sentence:
        return True
    if "paradajk" in sentence:
        return True
    if "cesnak" in sentence:
        return True
    if "premieš" in sentence:
        return True
    if "mäkk" in sentence:
        return True
    if "poliev" in sentence:
        return True
    words = sentence.strip().split(" ")
    if "alebo" in words:
        return True
    return False


def get_trigrams():
    slovak = set([x.strip() for x in open("../data/top_slovakian_trigrams", "r")])
    not_slovak = set([x.strip() for x in open("../data/top_not_slovakian_trigrams", "r")])
    prunik = slovak.intersection(not_slovak)
    slovak.difference_update(prunik)
    not_slovak.difference_update(prunik)
    return slovak,not_slovak


def is_slovak(sentence,slovak,not_slovak):
    trigrams = [t for w in sentence.strip().split() for t in generate_n_grams(w)]
    a = len([x for x in trigrams if x in not_slovak])
    b = len([x for x in trigrams if x in slovak])
    if a+b == 0:
       return True
    return (b/(b+a) > 0.6 )


def generate_n_grams(word,n=3):
    if len(word) < 3:
        return []
    return [word[i:i+3] for i in range(len(word)-n+1)]

def recipe_legth(sentence):
    recipe =  re.search("<r> (.*) <e>",sentence)
    return len(recipe.group(1))

out = []
with open("processed_data", "r") as phil:
#with open("slovakian", "r") as phil:
    slovak, not_slovak = get_trigrams()
    for line in phil:
        if recipe_legth(line) < 50:
            continue
        if is_slovak(line,slovak, not_slovak) or is_slovakian(line):
            continue
        out.append(line.strip())
shuffle(out)
with open("processed_data_filtered","w") as phil:
    phil.write("\n".join(out))

