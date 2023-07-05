#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import json 
import copy
import re 
from pathlib import Path
import os, sys

from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
import pathlib

pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',500)
from tqdm import tqdm
tqdm.pandas()
import string
from inflection import pluralize, singularize
from util_wordnet import get_sister_terms
from transformers import pipeline

import spacy
en = spacy.load('en_core_web_sm')
STOP_WORDS = en.Defaults.stop_words

from IPython.display import display
from df_to_latex import DataFrame2Latex


# ## HELPER FUNCTION

# In[6]:


def _get_article(word):
    if word[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    return 'a'


def save_dict_to_json(examples, output_path):
    ''' 
    save a list of dicts into otuput_path, orient='records' (each line is a dict) 
    examples: a list of dicts
    output_path: 
    '''

    with open(output_path, 'w') as fout:
        for example in examples:
            json.dump(example, fout)
            fout.write("\n")
        print(f"save {output_path} with {len(examples)} lines")

def add_period_at_the_end_of_sentence(sentence):
    last_token = sentence[-1]
    if last_token != '.': 
        return sentence + '.'
    return [sentence]

def get_unmasker(model, device, targets=None):
    if targets is None: 
        unmasker = pipeline('fill-mask', model=model)# 'bert-large-uncased') #initialize the masker
    else:
        unmasker = pipeline('fill-mask', model=model, targets=targets )# 'bert-large-uncased') #initialize the masker
    return unmasker



def remove_noisy_test_data(df):
  ''' 
  relation="hasproperty"
  why? some data points don't belong to this relation types 
  case1., sub_label=number, such as "10 is ten."  We don't say ten is the property of 10
  case2, sub_label = 'person_name' and obj_label = 'nuts;, such as ""Andrew is [MASK].", [MASK]=nuts
  '''
  sub_labels_to_exclude = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '30', '5', '50', '60', '7', '70', '70s', '80', '9', '90']
  obj_labels_to_exclude  = ['nuts']
  df = df.query(f"sub_label not in {sub_labels_to_exclude}")
  df = df.query(f"sub_label not in {obj_labels_to_exclude}")
  return  df.reset_index(drop=True)

def locate_sub_obj_position(ent, sentence, index_not_in) :
  ''' 
  function: find the index of ent in a sentence, the result will be used to filter instances whose ent cannot be find at their sentences
  args: 
    sentence: the sentnces to mask, could be the string or a list of tokens 
    ent: the ent to be found (sub_label) 
    index_not_in: the default index for failed instances (an ent not in a sentence)
  ''' 

  if isinstance(sentence, list):
    if ent not in sentence:
      return index_not_in
    return sentence.index(ent)  
  else:
    sentence = copy.deepcopy(sentence).lower()
    if isinstance(sentence, str):
      try:
        index = sentence.index(ent)
        return  index 
      except: 
        print(f"NOT FOUND sub_label: {ent} -> in sentence: {sentence}")
        return index_not_in
      
        print(ent, sentence)
        return index_not_in

def load_data(filepath, clean_test=True, tokenize=False):
  '''
  return the cleaned data
  args:
    tokenize: if True: the maksed_sentences will be tokenzied (this is slwoers); 
            otherwise, we use the string match to filter the failed sentences
    clean_test: default is True. We filter out some noisy samples spoted by huamns 
               Note that this is relation specific 

  '''
  index_not_in = 10000

  with open(filepath, 'r', encoding='utf-8') as fin:
    data = fin.readlines()
    data = [eval(x) for x in data]
    df = pd.DataFrame(data)

    df['obj_label'] = df['obj_label'].apply(lambda x: [x] if isinstance(x, str) else x)

  if tokenize:
    df['masked_sentence_tokens'] = df['masked_sentences'].apply(lambda x: tokenize_sentence(x[0]))
    df['sub_position'] = df[['sub_label', 'masked_sentence_tokens']].apply(lambda x: locate_sub_obj_position(x[0], x[1], index_not_in=index_not_in), axis=1)

  if clean_test: 
    df = remove_noisy_test_data(df)
    df['sub_position'] = df[['sub_label', 'masked_sentences']].apply(lambda x: locate_sub_obj_position(x[0], x[1][0], index_not_in), axis=1)
    df = df.query(f"sub_position !={index_not_in}") #.reset_index() #cue can not be matched in the sentence

  print(f"#Test_instances: {len(df.index)}")
  return df.reset_index(drop=True)

def get_unmasker(model, targets=None):
    if targets is None: 
        unmasker = pipeline('fill-mask', model=model)# 'bert-large-uncased') #initialize the masker
    else:
        unmasker = pipeline('fill-mask', model=model, targets=targets )# 'bert-large-uncased') #initialize the masker
    return unmasker


def get_highest_mrr_among_labels(label, pred):
    '''
    return the highest rank among the multiple labels. This is applicable to single labels as well, if we the single label is put in a list

    pred: a list of words (candidates)
    label: the true labels, which is a list (different forms of a word, e.g., singular or plurs, like animal and animals)
    '''
    mrr = 0 
    if pred is None: return mrr 

    rank_list = [ pred.index(item) + 1 for item in label if item in pred] 
    if len(rank_list)>0:
        mrr = 1/min(rank_list)
    return mrr 


def get_predictions(input_words, outputs, filter_objects_flag=True, filter_objects_with_input=True):
    '''
    excluding x from outputs
    '''
    filled_tokens = list()
    filled_scores = defaultdict()
    for i, output in enumerate(outputs):
#         print(output)
        filled_token = output['token_str'].strip().lower()
        filled_score = output['score']
        if filter_objects_flag:
            
            #####Add conditions to filter unwanted ################
            # filter the repetation of a concept in the explanation. See the the following example
            # [MASK] is the capability to do a particular job . -> capacity 
            if not filled_token.isalpha(): continue
            if filled_token in STOP_WORDS: continue 
            if len(filled_token)<=1: continue 
            if filter_objects_with_input:
                if filled_token in [input_words]: continue
                # [re.sub("\s+", '', x) for x in input_word.split()]: continue #filter out the target in input  
            if filled_token.startswith("#"): continue
            #####Add conditions to filter unwanted ################

            filled_tokens.append(filled_token)
            filled_scores[filled_token] = filled_score
        else:
            filled_tokens.append(filled_token)
            filled_scores[filled_token] = filled_score
    
    return pd.Series((filled_tokens, filled_scores))




dataset_to_jsonl_path={
    "EVAL": "../data/hypernymysuite/data/hypernymsuite/EVAL/IsA.jsonl",
    "BLESS": "../data/hypernymysuite/data/hypernymsuite/BLESS/IsA.jsonl",
    "LEDS": "../data/hypernymysuite/data/hypernymsuite/LEDS/IsA.jsonl",
    "LMDIAG": "../data/probe-generalization/Syntagmatic/LM-Diagnostic-Extended/singular/IsA.jsonl",
    "CLSB": "../data/CLSB/single_label/IsA.jsonl",
    "SHWARTZ": "../data/hypernymysuite/data/hypernymsuite/SHWARTZ/IsA.jsonl",
    }


# In[7]:


def layout_table(df, dataset_list =['BLESS','LMDIAG', 'CLSB', 'SHWARTZ', 'EVAL', 'LEDS']):
    '''
    format the output with desired dataset layout and metrics 
    '''
    df_groups = []
    for dataset in dataset_list: 
       
        df_group = df.query(f"dataset == '{dataset}'")
        df_group = df_group.pivot(index="pattern_id", columns=['dataset'], values=['MRR', 'P@K'])
        df_group = df_group.swaplevel(0, 1, axis=1)
        df_groups.append(df_group)

    df_groups = pd.concat(df_groups, axis=1)
    return df_groups

def merge_predictions_in_concept_level(uniform_funcion, words, top_k=None ):
    '''
    uniform_function: either signualarize or pluralize 
    '''
    words_uniformed = [uniform_funcion(word) for word in words]
    concepts = list(OrderedDict.fromkeys(words_uniformed))
    return concepts[:top_k] if top_k is not None else concepts


# # Definitional Patterns (baseline)

# In[8]:



def_sap_id_to_patterns = {
         "1": "[X] is a [Y].", 
         "2": "[X] is a type of [Y].", 
         "3": "[X] is a kind [Y].", 
        }

unmasker = unmasker = pipeline('fill-mask', model= 'bert-large-uncased', device=0)
top_k=10
batch_size = 200 
df_res_def = []
debug =  False #True 
# debug =  True

for dataset, filepath in dataset_to_jsonl_path.items():
    # if dataset !="LMDIAG": continue 
    df = load_data(filepath)
    
    for idx, pattern in def_sap_id_to_patterns.items():
        df[f'masked_sentences_{idx}'] = df['sub_label'].apply(lambda x: [pattern.replace("[Y]", "[MASK]").replace("[X]", f"{_get_article(x)} {x}")])
    
    if debug: 
        df = df.head(5)
#         display(df.head())    
    for idx in range(1, len(def_sap_id_to_patterns.keys())+1 ):
        df[f'outputs_{idx}']  = unmasker(df[f'masked_sentences_{idx}'].to_list(), top_k= 2*top_k, batch_size=batch_size)
        df[[f'pred_{idx}', f'pred_{idx}_score']] = df[['sub_label',f'outputs_{idx}']].apply(lambda x: get_predictions(input_words=x[0], outputs=x[1], 
                                                                                           filter_objects_flag=True, 
                                                                                           filter_objects_with_input=True), 
                                                                                      axis=1)
        
        df[f'pred_{idx}']= df[f'pred_{idx}'].apply(lambda x: merge_predictions_in_concept_level(uniform_funcion=singularize, words=x, top_k=top_k))
        df['obj_label_sg'] = df['obj_label'].apply(lambda x: [singularize(x[0])])
        
        df[f'p@{top_k}_{idx}'] = df[['obj_label_sg', f'pred_{idx}']].apply(lambda x: 1 if x[0][0] in x[1]  else 0, axis=1)
        df[f'mrr@{top_k}_{idx}'] = df[['obj_label_sg', f'pred_{idx}']].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1]), axis=1)
        
        p_at_k = df[f'p@{top_k}_{idx}'].sum()/len(df.index)
        mrr = df[f'mrr@{top_k}_{idx}'].sum()/len(df.index)
        df_res_def.append({"dataset": dataset, "pattern_id": idx, "P@K": round(p_at_k, 3)*100, 'MRR': round(mrr,3)*100 })
        
df_res_def = pd.DataFrame(df_res_def)
df_res_def_pivot = layout_table(df_res_def, dataset_list =['BLESS','LMDIAG', 'CLSB',  'SHWARTZ','EVAL', 'LEDS']) 
display(df_res_def_pivot)

DataFrame2Latex(df= df_res_def_pivot , label=f'tab:def_single_pattern_ablation', 
            caption=f'Experimental results on definitional single patterns.', 
            output_file= None , #'../log/paper_results/latex.test.tex',
            adjustbox_width = 'textwidth',
            precision = 1,
            column_format='l|ll|ll|ll|ll|ll|ll',
            multicolumn_format='c|'
            )


# # Lexico-Synatactic Patterns (baseline)

# In[6]:



lsp_sap_id_to_patterns = {
         "1": "[Y] such as [X].", 
         "2": "[Y], including [X].", 
         "3": "[Y], especially [X].", 
         "4": "[X] or other [Y].", 
         "5": "[X] and other [Y].", 
         "6": "such [Y] as [X].", 
        }


debug =  False #True 
# debug = True 
unmasker = unmasker = pipeline('fill-mask', model= 'bert-large-uncased', device=0)
top_k=10

df_res_lsp = []
for dataset, filepath in dataset_to_jsonl_path.items():
    df = load_data(filepath)
    df['sub_label_pl'] = df['sub_label'].apply(lambda x: pluralize(x))
    df['obj_label_pl'] = df['obj_label'].apply(lambda x: [pluralize(x[0])])
    for idx, pattern in lsp_sap_id_to_patterns.items():
        df[f'masked_sentences_{idx}'] = df['sub_label_pl'].apply(lambda x: [pattern.replace("[Y]", "[MASK]").replace("[X]", x)])
    
    if debug: 
        df = df.head(5)
#         display(df.head(5))
    
    for idx, pattern in lsp_sap_id_to_patterns.items():
        df[f'outputs_{idx}']  = unmasker(df[f'masked_sentences_{idx}'].to_list(), top_k= 2*top_k, batch_size=batch_size)
        df[[f'pred_{idx}', f'pred_{idx}_score']] = df[['sub_label_pl',f'outputs_{idx}']].apply(lambda x: get_predictions(input_words=x[0], outputs=x[1], 
                                                                                           filter_objects_flag=True, 
                                                                                           filter_objects_with_input=True), 
                                                                                      axis=1)
        
        df[f'pred_{idx}']= df[f'pred_{idx}'].apply(lambda x: merge_predictions_in_concept_level(uniform_funcion=pluralize, words=x, top_k=top_k))
        df[f'p@{top_k}_{idx}'] = df[['obj_label_pl', f'pred_{idx}']].apply(lambda x: 1 if x[0][0] in x[1] else 0, axis=1)
        df[f'mrr@{top_k}_{idx}'] = df[['obj_label_pl', f'pred_{idx}']].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1]), axis=1)
        
        p_at_k = df[f'p@{top_k}_{idx}'].sum()/len(df.index)
        mrr = df[f'mrr@{top_k}_{idx}'].sum()/len(df.index)
        df_res_lsp.append({"dataset": dataset, "pattern_id": idx, "P@K": round(p_at_k, 3) *100 , 'MRR': round(mrr, 3)*100})


df_res_lsp = pd.DataFrame(df_res_lsp)
df_res_lsp_pivot = layout_table(df_res_lsp)
display(df_res_lsp_pivot)


DataFrame2Latex(df= df_res_lsp_pivot , label=f'tab:lsp_single_pattern_ablation', 
            caption=f'Experimental results on LSP single patterns.', 
            output_file= None , #'../log/paper_results/latex.test.tex',
            adjustbox_width = 'textwidth',
            precision = 1,
            column_format='l|ll|ll|ll|ll|ll|ll',
            multicolumn_format='c|'
            )

output_path = '../log/df_res_df.csv'
df_res_def.to_csv(output_path)
print(f"save {output_path}")

output_path = '../log/df_res_lsp.csv'
print(f"save {output_path}")
df_res_lsp.to_csv(output_path)
# ## subset vs group
# - we want to examine what's the effect of pattern numbers to the final performance
# - plot: x is the number of patterns, y is the performance on the subset of the patterns with corresponding number on x 
# - 

# In[ ]:




