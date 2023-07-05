import pandas as pd
from collections import defaultdict 
import json 
import wn
from inflection import singularize, pluralize 
import os, sys
from wn.similarity import path
ewn = wn.Wordnet('ewn:2020')


def read_cohyponyms(path = '../log/word_to_cohyponyms.txt', chunk_data=False, start=None, end=None):
    if os.path.exists(path):
        print(f"reading cohyponyms: {path}")
        df = pd.read_csv(path)
        print(f"all instances: {len(df.index)}")
        if chunk_data and start!=None and end!=None: 
            df = df.iloc[start:end]
        print("#instances chunk:", len(df.index))
        word_to_cohyponym = dict(zip(df['word'], df['cohyponyms']))
        return word_to_cohyponym 
        

    print(f"{path} not found")
    

    
def get_shortest_path_score(word1, word2, top_k_senses=2):
    synsets1 = ewn.synsets(word1, pos='n')    
    synsets2 = ewn.synsets(word2, pos='n')
    if len(synsets1)==0 or len(synsets2)==0: return 0 
    scores = []
    for synset1 in synsets1[:top_k_senses]:
        for synset2 in synsets2[:top_k_senses]:
            scores.append(path(synset1, synset2  ))
    return max(scores) 

def write_cohyponym_scores(word_to_cohyponyms_score, output_path):
    with open(output_path, 'w') as fout:
        json.dump(word_to_cohyponyms_score, fout, indent=4)
    print(f"save {output_path}")

def get_path_score_for_cohyponyms(word_to_cohyponyms, debug=False, top_k_senses=2):
    word_to_cohyponyms_score = defaultdict()
    if debug:    
        query_words = ['dog', 'wolf', 'corn', 'car', 'tiger']
    else: 
        query_words = word_to_cohyponyms.keys()
        
    for i, k in enumerate(query_words): 
        if len(word_to_cohyponyms[k])==0: continue 
        k_pl = pluralize(k)
        cur_cohyponyms_to_score = defaultdict()
        for v in eval(word_to_cohyponyms[k]):
            score = get_shortest_path_score(k, v, top_k_senses=2)
            v = pluralize(v)
            cur_cohyponyms_to_score[v] = round(score, 4)
        cur_cohyponyms_to_score = dict(sorted(cur_cohyponyms_to_score.items(), key=lambda x: x[1], reverse=True))
        word_to_cohyponyms_score[k_pl] = cur_cohyponyms_to_score
        if k_pl!=k: 
            word_to_cohyponyms_score[k] = cur_cohyponyms_to_score
        if i%1000==0:
            print(i) 
    return word_to_cohyponyms_score


def word_to_plural(word_to_cohyponym):
    vocab = set()
    for k,v  in word_to_cohyponym.items():
        vocab +=set(k)
        vocab += set(v)
    
    word_to_pl = defaultdict()
    for word in vocab: 
        word_to_pl[word] =  pluralize(word)
    return word_to_pl  


if __name__=='__main__':
    debug=False 
    # debug=True 
    chunk_data=sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]
    output_path = f'../log/cohyponyms/word_to_cohyponyms_score_{start}_{end}.json'

    #word_to_cohyponyms  = read_cohyponyms(path = '../log/word_to_cohyponyms.txt')
    word_to_cohyponyms  = read_cohyponyms(path = '../log/word_to_cohyponyms.txt', chunk_data=chunk_data, start=int(start), end= int(end))
    word_to_cohyponyms_score = get_path_score_for_cohyponyms(word_to_cohyponyms, debug=debug)
    write_cohyponym_scores(word_to_cohyponyms_score, output_path)
