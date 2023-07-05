import pandas as pd
from collections import defaultdict 
import json 
import wn
from inflection import singularize, pluralize 
import os, sys
from wn.similarity import path
ewn = wn.Wordnet('ewn:2020')


def read_cohyponyms(path = '../log/word_to_cohyponyms.txt'):
    if os.path.exists(path):
        print(f"reading cohyponyms: {path}")
        df = pd.read_csv(path)
        word_to_cohyponym = dict(zip(df['word'], df['cohyponyms']))
        return word_to_cohyponym
    print(f"{path} not found")

    
def get_shortest_path_score(word1, word2):
    synsets1 = ewn.synsets(word1, pos='n')    
    synsets2 = ewn.synsets(word2, pos='n')
    if len(synsets1)==0 or len(synsets2)==0: return 0 
    score = path(synsets1[0] , synsets2[0]  )
    return score 

def write_cohyponym_scores(word_to_cohyponyms_score, output_path):
    with open(output_path, 'w') as fout:
        json.dump(word_to_cohyponyms_score, fout, indent=4)
    print(output_path)

def get_path_score_for_cohyponyms(word_to_cohyponyms, debug=False):
    word_to_cohyponyms_score = defaultdict()
    if debug:    
        query_words = ['dog', 'wolf', 'corn', 'car']
    else: 
        query_words = word_to_cohyponyms.keys()
        
    for k in query_words: 
        k_pl = pluralize(k)
        cur_cohyponyms_to_score = defaultdict()
        if len(word_to_cohyponyms[k])==0: continue 
        for v in eval(word_to_cohyponyms[k]):
            score = get_shortest_path_score(k, v)
            v = pluralize(v)
            cur_cohyponyms_to_score[v] = round(score, 4)
        cur_cohyponyms_to_score = dict(sorted(cur_cohyponyms_to_score.items(), key=lambda x: x[1], reverse=True))
        word_to_cohyponyms_score[k_pl] = cur_cohyponyms_to_score
    return word_to_cohyponyms_score


if __name__=='__main__':
    debug=False 
    # debug=True 
    word_to_cohyponyms = read_cohyponyms()
    word_to_cohyponyms_score = get_path_score_for_cohyponyms(word_to_cohyponyms, debug=debug)
    output_path = '../log/word_to_cohyponyms_score_pl.json'
    write_cohyponym_scores(word_to_cohyponyms_score, output_path)