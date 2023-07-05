from nltk.corpus import wordnet as wn
import pandas as pd
import os, sys 
import wn
import json
from wn.similarity import path
ewn = wn.Wordnet('ewn:2020')


def read_cohyponyms(path = '../log/word_to_cohyponyms.txt'):
    if os.path.exists(path):
        print(f"reading cohyponyms: {path}")
        df = pd.read_csv(path)
        df['cohyponyms'] = df['cohyponyms'].apply(lambda x: eval(x))
        word_to_cohyponym = dict(zip(df['word'].to_list(), df['cohyponyms'].to_list()))
        return word_to_cohyponym
    print(f"{path} not found")

def read_cohyponyms_with_score(path = '../log/word_to_cohyponyms_score.json', return_score=False):
    if os.path.exists(path):
        print(f"reading cohyponyms: {path}")
        word_to_cohyponym_score = json.load(open(path, 'r'))
        return word_to_cohyponym_score  if return_score else {k: list(v.keys()) for k,v in word_to_cohyponym_score.items()}
        # if return_score:
            # word_to_cohyponym = {k: list(v.keys()) for k,v in word_to_cohyponym_score.items()}
        # return word_to_cohyponym
    print(f"{path} not found")


def get_path_using_hypernym_dict(hypernym,hypernym_dict,synsets):
    '''
    Return the path between a sense and a specified hypernym 
    Core idea: starting from the hypernym, find its child sense (iterate this process, DFS)
    
    hypernym: a synset, which is a hypernym   
    hypernym_dict: a {hypernym: sense} dict 
    sensets: a synsets 
    '''
    
    path = [hypernym]
    current_synset = hypernym_dict[hypernym]
    #     print(hypernym, current_synset)
    
    while current_synset not in synsets: #stop criteria: when find a sense belonging to the synsets
    #         print(current_synset)
        path.append(current_synset)
        current_synset =  hypernym_dict[current_synset]
    path.append(current_synset)
    return path
    
# hypernym_dict = get_hypernym_path_dict(wn.synsets("book","n"))
# print(get_path_using_hypernym_dict(wn.synset('physical_entity.n.01'),hypernym_dict,wn.synsets("book","n")))
        

def get_hypernym_path_dict(synsets):
    '''
    get the hypernyms for all synsets of a given word 
    return hypernym_dict: key is the hypernym and value is the synset
    '''
    hypernym_dict = {}
    synsets_to_expand = synsets
    while synsets_to_expand:
        new_synsets_to_expand = set()
        for synset in synsets_to_expand:
            for hypernym in synset.hypernyms():
                if hypernym not in hypernym_dict:  # this ensures we get the shortest path
                    hypernym_dict[hypernym] = synset
                    new_synsets_to_expand.add(hypernym)
        synsets_to_expand = new_synsets_to_expand
    return hypernym_dict
        
def get_wordnet_shortest_path_between(word1,word2):
    '''
    get the shorttest path betwen two words
    question: what to return if a word is not in the wordnet
    '''

    synsets1 = wn.synsets(word1, "n") #""n"" is added in 230107
    synsets2 = wn.synsets(word2, "n")
    if len(synsets1)==0 or len(synsets2) ==0: return [] #retrun [] means no path exsit 
    # added these two lines to catch situation where word1 and word2 share a synset, the distance is 1, 1/path=1
    match = set(synsets1).intersection(set(synsets2))
    if match: return [list(match)[0]]

    hypernym_dict1 = get_hypernym_path_dict(synsets1)
    hypernym_dict2 = get_hypernym_path_dict(synsets2)
    best_path = []
    for hypernym in hypernym_dict1:
        if hypernym in hypernym_dict2 and hypernym_dict1[hypernym] != hypernym_dict2[hypernym]: #find the LCS
            path1 = get_path_using_hypernym_dict(hypernym,hypernym_dict1,synsets1)
            path2 = get_path_using_hypernym_dict(hypernym,hypernym_dict2,synsets2)
            
            if not best_path or len(path1) + len(path2) - 1 < len(best_path):
                path1.reverse()
                best_path = path1 + path2[1:]
    return best_path

    
def get_wordnet_shortest_path_length_between(word1,word2, oov_path_len=100):
    '''
    oov_path: the path for out-of-vocabulary words
    '''
    best_path = get_wordnet_shortest_path_between(word1, word2)    
    return len(best_path) if len(best_path)>0 else oov_path_len 

# def get_wordnet_shortest_path_score_between(word1,word2):
#     '''
#     path_score = 1/short_path_length(word2, word2)
#     path_score=0 if word1/word2 is not in wordnet 
#     '''
#     best_path = get_wordnet_shortest_path_between(word1, word2)    
#     return 1/len(best_path) if len(best_path)>0 else 0


# def get_wordnet_shortest_path_score_between(word1, word2):
#     synsets1 = ewn.synsets(word1, pos='n')    
#     synsets2 = ewn.synsets(word2, pos='n')
#     if len(synsets1)==0 or len(synsets2)==0: return 0 
#     score = path(synsets1[0] , synsets2[0]  )
#     return score 

def get_wordnet_shortest_path_score_between(word1, word2):
    synsets1 = ewn.synsets(word1, pos='n')    
    synsets2 = ewn.synsets(word2, pos='n')
    if len(synsets1)==0 or len(synsets2)==0: return 0 
    scores = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            scores.append(path(synset1, synset2  ))
    return max(scores) 


def get_wordnet_avg_path_score_between_sub_and_anchors(sub_labels, subj_anchors):
    '''
    evalaute the quality of anchors by measuring their average shortest paths score in WordNet
    path_score: [0, 1] ; 1 means those anchors are all very close to the sub_labels , 0 indicates that those anchors are far away or not exist in WordNet

    args:
        sub_labels: a list of sub_labels 
        subj_anchors: a list of dict (anchor: score), each sub_label have N anchors 
    '''
    path_scores = []
    for i, (sub_label, subj_anchors) in enumerate(zip(sub_labels, subj_anchors)): 
        path_score  = sum([get_wordnet_shortest_path_score_between(sub_label, subj_anchor) for subj_anchor in subj_anchors])/len(subj_anchors)
        path_lens.append(path_score)
    return path_scores 


def get_sister_terms(word, distance_to_hypernym=1):
    '''
    "Coordinate (sister) terms: share the same hypernym"
    "The sister relation is the usual one encountered when working with tree structures: sisters are word forms (either simple words or collocations) that are both immediate hyponyms of the same node"
    
    Args:
        word: the input word
        hop: the hops to hypernyms, default is 1, which means take the top 1 hypernym of x
    '''
    sister_terms = set()
    for synset in wn.synsets(word ,"n"):
        for hypernym in synset.hypernyms()[:distance_to_hypernym]:
#             print(hypernym)
            sister_synsets = hypernym.hyponyms()
            for sister_synset in sister_synsets:
                sister_names = [x.name() for x in sister_synset.lemmas()]
                sister_names_selected = [name.lower() for name in sister_names if len(name.split("_"))==1 and  len(name.split("-"))==1  and name!=word]
                sister_terms = sister_terms.union(set(sister_names_selected))
    return list(sister_terms)




# if __name__ == "__main__":
#     path_score = get_wordnet_shortest_path_score_between("dog", "animal")
#     print(path_score)



