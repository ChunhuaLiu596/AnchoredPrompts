import pandas as pd 
import json 
import copy
import re 
from pathlib import Path

from collections import Counter, defaultdict
from copy import deepcopy

pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',500)
from tqdm import tqdm
tqdm.pandas()


from nltk.corpus import wordnet as wn
from inflection import singularize, pluralize 



def get_sister_terms(word):
    '''
    "Coordinate (sister) terms: share the same hypernym"
    "The sister relation is the usual one encountered when working with tree structures: sisters are word forms (either simple words or collocations) that are both immediate hyponyms of the same node"
    '''
    sister_terms = set()
    for synset in wn.synsets(word ,"n"):
        for hypernym in synset.hypernyms()[:1]:
            sister_synsets = hypernym.hyponyms()
            for sister_synset in sister_synsets:
                sister_names = [x.name() for x in sister_synset.lemmas()]
                sister_names_selected = [name.lower() for name in sister_names if len(name.split("_"))==1 and  len(name.split("-"))==1  and name!=word]
                sister_terms = sister_terms.union(set(sister_names_selected))
#                 print(sister_synset)
#                 print(sister_terms )
#                 print()
    return list(sister_terms)

# read the singular data
path_anchor_sg = './log/bert-large-uncased/lm_diagnostic_extended/singular/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_5_anchor_scorer_probAvg.csv'
query_cols = ['sub_label', 'obj_label', 'subj_anchors', 'uuid']
df_sg  = pd.read_csv(path_anchor_sg)[query_cols]
df_sg['subj_anchors'] = df_sg['subj_anchors'].apply(lambda x: eval(x))

df_sg['subj_anchors'] = df_sg['subj_anchors'].progress_apply(lambda x: [singularize(word) for word in x])
df_sg['subj_sisters'] = df_sg['sub_label'].progress_apply(lambda x: get_sister_terms(x))


# read the plural data
path_anchor_pl = './log/bert-large-uncased/lm_diagnostic_extended/plural/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_5_anchor_scorer_probAvg_filter_obj_True_wnp_True_cpt_False.LM_DIAGNOSTIC_EXTENDED.csv'
df_pl  = pd.read_csv(path_anchor_pl)[query_cols]
df_pl['subj_anchors'] = df_pl['subj_anchors'].apply(lambda x: eval(x))
df_pl['subj_anchors'] = df_pl['subj_anchors'].apply(lambda x: [singularize(word) for word in x])


# df_sgpl.columns 
df_sgpl = pd.merge(df_sg, df_pl, on ='uuid', suffixes=('_sg', '_pl'))
df_sgpl[['sub_label_sg', 'sub_label_pl', 'obj_label_sg', 'obj_label_pl', 'subj_anchors_sg', 'subj_anchors_pl']].head(20)
df_sgpl[['sub_label_sg', 'sub_label_pl', 'obj_label_sg', 'obj_label_pl', 'subj_anchors_sg', 'subj_anchors_pl', 'subj_sisters']].head(20)

df_sgpl['anchor_sg_in_sisters'] = df_sgpl[['subj_anchors_sg', 'subj_sisters']].apply(lambda x: set(x[0]).intersection(set(x[1])), axis=1)
df_sgpl['anchor_pl_in_sisters'] = df_sgpl[['subj_anchors_sg', 'subj_sisters']].apply(lambda x: set(x[0]).intersection(set(x[1])), axis=1)

df_sgpl['anchor_sg_in_sisters_num'] =  df_sgpl['anchor_sg_in_sisters'].apply(lambda x: len(x))  
df_sgpl['anchor_pl_in_sisters_num'] =  df_sgpl['anchor_pl_in_sisters'].apply(lambda x: len(x))  

# df_sgpl.head()
print(df_sgpl[['anchor_sg_in_sisters_num', 'anchor_pl_in_sisters_num']].describe())
print(df_sgpl[['anchor_sg_in_sisters_num', 'anchor_pl_in_sisters_num']].mean())

df_sgpl.to_csv("./log/anchors/lm_diagnostic_extended.anchors.csv")
# plot(kind='hist')