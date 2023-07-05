import pandas as pd 
import json 
from collections import Counter, defaultdict 
import os, sys 
from copy import copy, deepcopy

from inflection import singularize,  pluralize 
import spacy
# import pyinflect
# import lemminflect

nlp = spacy.load('en_core_web_sm')
def word_to_pos(words):
    word_to_pos_dict = defaultdict()
    for word in words:
        if word == 'null': continue
        pos = nlp(str(word))[0].pos_ 
        word_to_pos_dict[word] = pos
    return word_to_pos_dict


def sort_similar_words(df, word_to_pos_dict=None, top_k = 10):
    '''
    sort simialr words and return the top_k
    '''
    cue_to_similar_words = defaultdict()
    for i, (k,v) in enumerate(df.to_dict().items()):
        if k=='null': continue
        #k_pos = word_to_pos_dict.get(k)
        #if k_pos !='NOUN': continue 

        v_sorted = Counter(v).most_common()
        v_sorted_noun = []
        for v in v_sorted:
            if len( str(v[0]).split())>1: continue 
            #v_pos = word_to_pos_dict.get(v[0])
            #if v_pos!='NOUN': continue 
            v_sorted_noun.append(v)
        cue_to_similar_words[k] = dict(v_sorted_noun[1:top_k+1])
    print('dog: ', cue_to_similar_words['dog'])
    return cue_to_similar_words

# path = '../output/2018/S_RW.R1.csv'


def get_similar_words(path):
    
    df1 = pd.read_csv(path)

    df1.index = df1["Unnamed: 0"]
    df1 = df1.drop(columns=["Unnamed: 0"], axis=1)
    #word_to_pos_dict = word_to_pos(list(df1.index))
    print("sorting similar words ..")
    cue_to_similar_words = sort_similar_words(df1) #1.head(20)) #, word_to_pos_dict )
    return cue_to_similar_words 
    #cue_to_similar_words['bicycle']
# cue_to_similar_words.keys()


def load_cue_to_similar_words(path, path_rw='../../data/S_RW.R123.csv'):
    if os.path.exists(path):
        print(f"loading {path}...")
        cue_to_similar_words = json.load(open(path))
    else:
        print(f"loading {path_rw} to get similar words ...")
        cue_to_similar_words = get_similar_words(path_rw)
        save_similar_words(cue_to_similar_words, path)
    return cue_to_similar_words 


def save_similar_words(cue_to_similar_words, output_file, indent=4):
    
    with open(output_file, 'w') as fout:
        json.dump(cue_to_similar_words, fout, indent=indent)
    print(f"save {output_file}")
    
def save_anchors_from_swow(json_path = '../../data/swow/swow.en.similar_words.json',  path_rw='../../data/swow/S_RW.R123.csv', debug=False):
    cue_to_similar_words_score = load_cue_to_similar_words(json_path, path_rw)
    
    vocab_cues = set(cue_to_similar_words_score.keys())
    vocab_res = set()
    for k,v in cue_to_similar_words_score.items():
        vocab_res.update(v.keys() )
    vocab = vocab_cues.union(vocab_res)
    vocab = set({str(k) for k in vocab})

    vocab_to_singular = {k: singularize(k) for k in deepcopy(vocab)}
    vocab_to_plural = {k: pluralize(k) for k in deepcopy(vocab)}

    cue_to_similar_words_score_sgpl = defaultdict()
    cue_to_similar_words_score_sg = defaultdict()
    for k,v in deepcopy(cue_to_similar_words_score).items():
        k = vocab_to_singular.get(k)
        v = { vocab_to_singular.get(k1): v1 for k1, v1 in v.items()}
        cue_to_similar_words_score_sg[k] = v 
        cue_to_similar_words_score_sgpl[k] =v 

    cue_to_similar_words_score_pl = defaultdict()
    for k,v in deepcopy(cue_to_similar_words_score).items():
        k = vocab_to_plural.get(k)
        v = { vocab_to_plural.get(k1): v1 for k1, v1 in v.items()}
        cue_to_similar_words_score_pl[k] = v 
        cue_to_similar_words_score_sgpl[k] =v 

    #return dic_sub_to_anchors_singular,  dic_sub_to_anchors_plural


def save_foward_associaitons_from_swow():
    path = '../../data/swow/strength.SWOW-EN.R123.csv'
    df = pd.read_csv(path, sep='\t')
    df.head()

    df['cue'] = df['cue'].apply(lambda x: str(x).strip().lower())
    df['response'] = df['response'].apply(lambda x: str(x).strip().lower())
    df['cue_token_num'] = df['cue'].apply(lambda x: len(str(x).split(" ")))
    df['response_token_num'] = df['response'].apply(lambda x: len(str(x).split(" ")))
    df = df.query("response_token_num ==1 and cue_token_num==1")    

    cue_association_dict = defaultdict()
    for name, group in df.groupby(['cue']):
        cue = name
        response_strength =  dict(zip(group['response'], group['R123.Strength']))
        cue_association_dict[cue] = response_strength

    vocab_cues = set(cue_association_dict.keys())
    vocab_res = set()
    for k,v in cue_association_dict.items():
        vocab_res.update(v.keys() )

    vocab = vocab_cues.union(vocab_res)
    vocab_to_singular = {k: singularize(k) for k in vocab}
    vocab_to_plural = {k: pluralize(k) for k in vocab}

    cue_association_dict_sgpl = defaultdict()
    for k,v in cue_association_dict.items():
        cue_association_dict_sgpl[vocab_to_singular[k]] = v
        cue_association_dict_sgpl[vocab_to_plural[k]] = v 


    output_file = '../../data/swow/swow.en.strength.R123.json'
    with open(output_file, 'w') as fout:
        json.dump(cue_association_dict_sgpl, fout, indent=4)
    print(f"save {output_file}")


if __name__ == '__main__':
    association_type =  sys.argv[1]
    if association_type == 'similar':
        print("generating the related/similar words in swow")
        json_path = '../../data/swow/swow.en.similar_words.json'
        save_anchors_from_swow(json_path)

        # cue_to_similar_words_score = load_cue_to_similar_words(json_path, path_rw='../../data/S_RW.R123.csv')
        #print("test: ")
        #print( dic_sub_to_anchors_singular['bicycle'] )
        #print( dic_sub_to_anchors_plural['bicycles'] )
        #print( dic_sub_to_anchors_singular['cartoon'] )
        #print( dic_sub_to_anchors_plural['cartoons'] )
    elif association_type == 'strength':
        print('generating forward associations')
        save_foward_associaitons_from_swow()

