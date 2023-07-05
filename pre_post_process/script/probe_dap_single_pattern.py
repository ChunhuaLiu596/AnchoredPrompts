#!/usr/bin/env python
# coding: utf-8

# # Do anchors help each SAP pattern? 
# - the best single defp: A(n) X is a type of Y.
# - the best single lsp: Y such as X and Z. 
# 
# insert anchors into the two patterns and compare their performance with (without) anchors

# In[1]:


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


# In[2]:


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


# In[3]:


from inflection import singularize, pluralize

def read_anchors(data_path, anchor_source, debug=False):
    '''
    read the anchor files mined from singualr and plural
    
    args: 
        anchor_soure: using the anchors mined from singular probe or plural probe
        
    return: 
        dic_sub_to_anchors_singular: both sub_label and subj_anchors are singular 
        dic_sub_to_anchors_plural: both sub_label and subj_anchors are plural 
    '''
    df = pd.read_csv(data_path)
    
    if debug: df = df.head(5)
    df['subj_anchors_sg'] = df['subj_anchors_sg'].apply(lambda x: eval(x))
    df['subj_anchors_pl'] = df['subj_anchors_pl'].apply(lambda x: eval(x))
        
    dic_sub_to_anchors_singular = dict(zip(df['sub_label_sg'], df['subj_anchors_sg']))
    dic_sub_to_anchors_plural = dict(zip(df['sub_label_pl'], df['subj_anchors_pl']))
    
    return dic_sub_to_anchors_singular, dic_sub_to_anchors_plural

def insert_anchors(dic_sub_to_anchors, df, mask_col, sub_col, anchor_col, probe_type, article_for_z=False):
    
    df[anchor_col] = df[sub_col].apply(lambda x: dic_sub_to_anchors.get(x) )
#     display(df[mask_col].head())
    
    if probe_type =='plural':
        df[mask_col] =  df[[anchor_col, mask_col]].apply(lambda x: [ x[1].replace('[Z]', anchor)  for anchor in x[0]], axis=1)
    elif probe_type == 'singular':
        if article_for_z: 
           df[mask_col] =  df[[anchor_col, mask_col]].apply(lambda x: [ x[1].replace('[Z]', "{} {}".format(_get_article(anchor), anchor))  for anchor in x[0]], axis=1) 
        else:
            df[mask_col] =  df[[anchor_col, mask_col]].apply(lambda x: [ x[1].replace('[Z]', anchor)  for anchor in x[0]], axis=1) 
    return df 


def save_hypernym_vocab(df, vocab_path, y_singular=True, y_plural=True):
    vocab_sg = set(x[0] for x in df['obj_label_singular'])
    vocab_pl = set(x[0] for x in df_def_sap['obj_label_plural'])
    
    if y_singular and y_plural: 
        vocab = vocab_sg.union(vocab_pl)
        df_vocab = pd.DataFrame(data=list(vocab))
        with open(vocab_path, 'w') as fout:
            df_vocab.to_csv(vocab_path, header=None, index=None, sep=' ', mode='a')
    print(f"save {vocab_path}")
    
def aggregate_token_scores(input_word, token2probs, scorer, top_k, sort_flag=True ):
    ''' 
    goal: we want the best scorer to consider:
        (1) frequency: a token that are elicited by multiple promptso
        (2) the probability: higher overall probability 
        (3)

    token2prob: dictionary mapping a token to a list of probs 
    anchor_anchor_scorer_list = ['freqProbSum', 'probMultiply', 'probMultiplyAvg', 'freqProbMultiply', 'freq', 'probSum', 'probAvg'] #TODO: rank based


    test case:
    token2probs = {'achieve': [0.2, 0.1, 0.03, 0.006], 'tried': [0.008, 0.006, 0.003, 0.001], 'perform':[0.08], 'prevent': [0.06], 'use': [0.02], 'accomplished': [0.1], 'produce':[0.06]}
    for scorer in  [ 'freqProbSum', 'probMultiply', 'probMultiplyAvg', 'freqProbMultiply' ]: #'freq', 'probSum', 'probAvg',
        token2prob = aggregate_token_scores(token2probs, scorer, top_k=7, sort_flag=True)
        print(scorer)
        print(f"\t{token2prob}" )
        print()

    '''
    token2prob = defaultdict()
    all_count = sum([len(item) for item in token2probs.values()])
    for token, probs in token2probs.items(): #rank_score = w * p, w is the frequency weight, p is the probability
            count = len(probs)
            
            freq_weight = count/all_count
            
            new_score = 0 
            
            if scorer=='freq':
                new_score =  freq_weight 

            elif scorer=='probSum':
                new_score = sum(probs)

            elif scorer=='probAvg': #this ignore the frequency factor [not ideal]
                new_score = sum(probs)/ len(probs)

            elif scorer=='freqProbSum': #[close to ideal]
                new_score = freq_weight * sum(probs)
                # print(token, freq_weight,sum(probs), new_score )
            elif scorer=='probLogSum':
                probs_valid = [item for item in probs if item>0]
                if len(probs_valid )==0:
                    new_score= 0
                else:
                    new_score =  sum([math.log(item, 2) for item in probs_valid ])/len(probs_valid)
                    # new_score =  sum([math.log(item, 2) for item in probs if item>0])/len(probs)

            elif scorer=='freqProbLogSum': #[close to ideal, requires a token to be (1) frequent (2) high probs across prompts]
                probs_valid = [item for item in probs if item>0]
                if len(probs_valid )==0:
                    new_score= 0
                else:
                    new_score =   sum([math.log(item*freq_weight, 2) for item in probs_valid ])/len(probs_valid)

            token2prob[token] = new_score
    return token2prob


def filter_outputs_with_probs(inputs, outputs, filter_objects_flag=True, return_probs=True, top_k=None, scorer='freqProbSum', filter_objects_with_input=True, add_wordnet_path_score=False, add_cpt_score=False, cpt_unmasker=None, mask_string=None, cpt_only=False):
    '''
    inputs: the original inputs, for example [A] is a type of [B], A is the input
    outputs: the candidates returned by PTLMs

    filter: True 
        filter: non-alpha tokens); 

    top_k: take the top_k outputs. This is important when using multiple prompts for each sub 
    add_wordnet_path_score: add wordnet path score into the output scoring function 
    add_cpt_score: add concept-positioning test score into the output scoring function 

    '''
    anchor_list = []
    anchor_scores = [] 
        
    for input_words, top_outputs in zip(inputs, outputs):  #iterate through the samples (sub)
        input_words = [re.sub("\s+", '', x) for x in input_words.split()]
        input_word  = input_words[0]
        filled_tokens  = defaultdict(int) #filter/accumulate predictions for each sample 
        filled_scores = defaultdict(list) #a list of token:[score1, score2, ...]   
        token2cpt  = defaultdict(list) #filter/accumulate predictions for each sample 

        if isinstance(top_outputs[0], list):
            flatten_output = [item for top_k_output  in top_outputs for item in top_k_output]
        else:
            flatten_output = [item for item  in top_outputs]

        for i, output in enumerate(flatten_output):
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
                    if filled_token in input_words: continue
                if filled_token.startswith("#"): continue
                #####Add conditions to filter unwanted ################
                
                filled_tokens[filled_token] +=1
                filled_scores[filled_token].append(filled_score)
            else:
                filled_tokens[filled_token] +=1
                filled_scores[filled_token] += filled_score

        if len(filled_tokens) ==0: 
            filled_tokens={'MISSING':1}
            filled_scores['MISSING'] = [0]

        # feed the input into the agrregate _token_scores() so that we can calcuate the 
        token2probs = aggregate_token_scores(input_word, token2probs=filled_scores, scorer=scorer, top_k=top_k, sort_flag=True)
            
        if top_k is not None and isinstance(top_k, int):
            token2probs = sorted(token2probs.items(), key=lambda x: x[1], reverse=True )
            token2probs = token2probs[:top_k] 
            token2probs = dict(token2probs)
        anchor_list.append(list(token2probs.keys())) 
        anchor_scores.append(token2probs) 

        # print("-"*60)
    return anchor_list if not return_probs  else pd.Series((anchor_list,anchor_scores))


# ## Data HELPER

# In[4]:


def get_dataset_to_respath(print_flag=False):
    # remote path 
    dataset_to_respath = {'hypernymsuite-BLESS': 'log/bert-large-uncased/hypernymsuite/BLESS/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv', 
                          'lm_diagnostic_extended-singular': 'log/bert-large-uncased/lm_diagnostic_extended/singular/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.LM_DIAGNOSTIC_EXTENDED.csv',
                          'clsb-singular': 'log/bert-large-uncased/clsb/singular/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.CLSB.csv', 
                          'hypernymsuite-LEDS': 'log/bert-large-uncased/hypernymsuite/LEDS/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv', 
                          'hypernymsuite-EVAL': 'log/bert-large-uncased/hypernymsuite/EVAL/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv', 
                          'hypernymsuite-SHWARTZ': 'log/bert-large-uncased/hypernymsuite/SHWARTZ/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv'}

    source_dir = 'spartan:~/cogsci/DAP/'
    target_dir = '../../'
    dataset_to_localpath = defaultdict()
    dataset_rename = {
        'hypernymsuite-BLESS': 'BLESS', 'lm_diagnostic_extended-singular': 'LMDIAG', 'clsb-singular':'CLSB', 'hypernymsuite-LEDS': 'LEDS', 'hypernymsuite-EVAL': 'EVAL', 'hypernymsuite-SHWARTZ': 
        "SHWARTZ"
    }
    dataset_name_to_relpath = defaultdict()
    for dataset, path in dataset_to_respath.items():
        path = path.replace(".tsv", ".csv")
        source_path = source_dir + path 
        dataset_l1 = dataset.split("-")[0]
        dataset_l2 = dataset.split("-")[1] 
        target_path = target_dir + path
        
        scp_string = f"!scp {source_path} {target_path}"
        if print_flag:
            print(scp_string)
            print()
#         print(target_path)
        dataset_to_localpath[dataset_rename[dataset]] = target_path 
        dataset_name_to_relpath[dataset_rename[dataset]] = "/".join(dataset.split('-'))
#     print(dataset_to_localpath)
    return dataset_to_localpath, dataset_name_to_relpath


def read_data(filepath):
    '''
    the dataformat is the *.jsonl file we used to probe LMs
    '''
    if '.jsonl' in filepath:
        df = load_json_to_df(filepath)
        df['sub_label_sg'] = df['sub_label'].apply(lambda x: singularize(x))
        df['sub_label_pl'] = df['sub_label'].apply(lambda x: pluralize(x))
    elif ".csv" in filepath:
        df = pd.read_csv(filepath)
        for col in ['obj_label']:
            df[col] = df[col].apply(lambda x: eval(x))
    
    #df['obj_label'] = df['obj_label'].apply(lambda x: x[0])

    df['relation'] = 'IsA'
    df['uuid'] = df.index + 1
    df['obj_label_sg'] =  df['obj_label'].apply(lambda x: [singularize(x[0])])
    df['obj_label_pl'] =  df['obj_label'].apply(lambda x: [pluralize(x[0])])


#     df = df[['sub_label_sg', 'obj_label_sg', 
#              'sub_label_pl', 'obj_label_pl', 
#              'uuid', 'relation', 'obj_label']]
    return df 

# dataset_to_localpath, dataset_name_to_relpath = get_dataset_to_respath()
# # print(dataset_to_localpath)
# df_test = read_data(dataset_to_localpath['BLESS'])
# # df_test.head()
# dataset_to_localpath['BLESS']
# dic_sub_to_anchors_singular,dic_sub_to_anchors_plural = read_anchors(data_path=dataset_to_localpath['BLESS'], anchor_source='plural', debug=False)


# ## Definitional Patterns (DAP)

# In[12]:


# def_sap_id_to_patterns = {
#          "1": "[X] or [Z] is a [Y].", 
#          "2": "[X] or [Z] is a type of [Y].", 
#          "3": "[X] or [Z] is a kind [Y].", 
#         }
def_sap_id_to_patterns = {
         "1": "[X] is a [Y]. So is [Z].", 
         "2": "[X] is a type of [Y]. So is [Z].", 
         "3": "[X] is a kind [Y]. So is [Z].", 
        }
article_for_z=True 
unmasker = unmasker = pipeline('fill-mask', model= 'bert-large-uncased', device=0)
top_k=10
batch_size = 100 
df_res_def = []
debug =  False #True 
# debug =  True
scorer_target_N_prompts = 'probAvg' #'probLogSum'

dataset_to_localpath, dataset_name_to_relpath = get_dataset_to_respath()

for dataset, data_path in dataset_to_localpath.items():
#     if dataset !="BLESS": continue 
    dic_sub_to_anchors_singular,dic_sub_to_anchors_plural = read_anchors(data_path, anchor_source='plural', debug=False)
    df = read_data(data_path)
    
    
    for idx, pattern in def_sap_id_to_patterns.items():
        df[f'masked_sentences_{idx}'] = df['sub_label_sg'].apply(lambda x: pattern.replace("[Y]", "[MASK]").replace("[X]", f"{_get_article(x)} {x}"))
        
        df = insert_anchors(dic_sub_to_anchors=dic_sub_to_anchors_singular, 
                                    df= df, 
                                    mask_col = f'masked_sentences_{idx}', 
                                    sub_col = 'sub_label_sg', 
                                    anchor_col='subj_anchors_sg', 
                                    probe_type='singular', 
                                    article_for_z=article_for_z)

    if debug: 
        df = df.head(5)
        display(df)
        
    for idx in range(1, len(def_sap_id_to_patterns.keys())+1 ):
        df[f'outputs_{idx}'] = [unmasker(x, top_k= 2*top_k) for x in tqdm(df[f'masked_sentences_{idx}'].to_list())]
        df[[f'pred_{idx}', f'pred_{idx}_score']] = filter_outputs_with_probs(df.subj_anchors_combined.to_list(), 
                                                                                                        df[f'outputs_{idx}'],  
                                                                                                        return_probs=True, 
                                                                                                        top_k= 2* top_k, 
                                                                                                        scorer= scorer_target_N_prompts,
                                                                                                        filter_objects_flag = True,
                                                                                                        filter_objects_with_input = True 
                                                                                                        )
        
        df[f'pred_{idx}']= df[f'pred_{idx}'].apply(lambda x: merge_predictions_in_concept_level(uniform_funcion=singularize, words=x, top_k=top_k))
        df['obj_label_sg'] = df['obj_label'].apply(lambda x: [singularize(x[0])])
        
        df[f'p@{top_k}_{idx}'] = df[['obj_label_sg', f'pred_{idx}']].apply(lambda x: 1 if x[0][0] in x[1]  else 0, axis=1)
        df[f'mrr@{top_k}_{idx}'] = df[['obj_label_sg', f'pred_{idx}']].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1]), axis=1)
        
        p_at_k = df[f'p@{top_k}_{idx}'].sum()/len(df.index)
        mrr = df[f'mrr@{top_k}_{idx}'].sum()/len(df.index)
        cur_res = {"dataset": dataset, "pattern_id": idx, "P@K": round(p_at_k, 3)*100, 'MRR': round(mrr,3)*100 }
        df_res_def.append(cur_res)
        print(cur_res)
        
df_res_def = pd.DataFrame(df_res_def)
df_res_def_pivot = layout_table(df_res_def, dataset_list =['BLESS','LMDIAG', 'CLSB', 'SHWARTZ', 'EVAL', 'LEDS']) 
display(df_res_def_pivot)

DataFrame2Latex(df= df_res_def_pivot , label=f'tab:def_single_pattern_ablation', 
            caption=f'Experimental results on definitional single patterns.', 
            output_file= None , #'../log/paper_results/latex.test.tex',
            adjustbox_width = 'textwidth',
            precision = 1,
            column_format='l|ll|ll|ll|ll|ll|ll',
            multicolumn_format='c|'
            )


# 	MRR	P@K
# pattern_id		
# 1	13.5	23.0
# 2	16.9	28.3
# 3	15.7	30.8
# 
# \begin{table*}[!h]
# \centering
# \begin{adjustbox}{width=\textwidth}
# \label{tab:def_single_pattern_ablation}
# \begin{tabular}{l|ll|ll|ll|ll|ll|ll}
# \toprule
# dataset & \multicolumn{2}{c|}{BLESS} \\
#  & MRR & P@K \\
# pattern_id &  &  \\
# \midrule
# 1 & 13.5 & 23.0 \\
# 2 & \textbf{16.9} & 28.3 \\
# 3 & 15.7 & \textbf{30.8} \\
# \bottomrule
# \end{tabular}
#  \end{adjustbox}
# \caption{Experimental results on definitional single patterns.}
# \end{table*}
# 

# ## Lexico-Synatactic Patterns (DAP)

# In[14]:



lsp_sap_id_to_patterns = {
         "1": "[Y] such as [X] and [Z].", 
         "2": "[Y], including [X] and [Z].", 
         "3": "[Y], especially [X] and [Z].", 
         "4": "[X], [Z] or other [Y].", 
         "5": "[X], [Z] and other [Y].", 
         "6": "such [Y] as [X] and [Z].", 
        }

debug =  False #True 
# debug = True 
unmasker = unmasker = pipeline('fill-mask', model= 'bert-large-uncased', device=0)
top_k=10
scorer_target_N_prompts = 'probAvg' #'probLogSum'

dataset_to_localpath, dataset_name_to_relpath = get_dataset_to_respath()
df_res_lsp = []
for dataset, data_path in dataset_to_localpath.items():
#     if dataset !="LMDIAG": continue
    dic_sub_to_anchors_singular,dic_sub_to_anchors_plural = read_anchors(data_path, anchor_source='plural', debug=False)
    df = read_data(data_path)
    if debug: 
        df = df.head(5)
    for idx, pattern in lsp_sap_id_to_patterns.items():
        df[f'masked_sentences_{idx}'] = df['sub_label_pl'].apply(lambda x: pattern.replace("[Y]", "[MASK]").replace("[X]", x))
        
        df = insert_anchors(dic_sub_to_anchors=dic_sub_to_anchors_plural, 
                                    df= df, 
                                    mask_col = f'masked_sentences_{idx}', 
                                    sub_col = 'sub_label_pl', 
                                    anchor_col='subj_anchors_pl', 
                                    probe_type='plural', 
                                    article_for_z=False)

    if debug: display(df)
        
    for idx in range(1, len(lsp_sap_id_to_patterns.keys())+1 ):
        df[f'outputs_{idx}'] = [unmasker(x, top_k=2*top_k) for x in tqdm(df[f'masked_sentences_{idx}'].to_list())]
        df[[f'pred_{idx}', f'pred_{idx}_score']] = filter_outputs_with_probs(df.subj_anchors_combined.to_list(), 
                                                                                                        df[f'outputs_{idx}'],  
                                                                                                        return_probs=True, 
                                                                                                        top_k= 2* top_k, 
                                                                                                        scorer= scorer_target_N_prompts,
                                                                                                        filter_objects_flag = True,
                                                                                                        filter_objects_with_input = True 
                                                                                                        )
        
        df[f'pred_{idx}']= df[f'pred_{idx}'].apply(lambda x: merge_predictions_in_concept_level(uniform_funcion=pluralize, words=x, top_k=top_k))
#         df['obj_label_pl'] = df['obj_label'].apply(lambda x: [pluralize(x[0])])
        
        df[f'p@{top_k}_{idx}'] = df[['obj_label_pl', f'pred_{idx}']].apply(lambda x: 1 if x[0][0] in x[1]  else 0, axis=1)
        df[f'mrr@{top_k}_{idx}'] = df[['obj_label_pl', f'pred_{idx}']].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1]), axis=1)
        
        p_at_k = df[f'p@{top_k}_{idx}'].sum()/len(df.index)
        mrr = df[f'mrr@{top_k}_{idx}'].sum()/len(df.index)
        df_res_lsp.append({"dataset": dataset, "pattern_id": idx, "P@K": round(p_at_k, 3)*100, 'MRR': round(mrr,3)*100 })
        
df_res_lsp = pd.DataFrame(df_res_lsp)
df_res_lsp_pivot = layout_table(df_res_lsp, dataset_list =['BLESS','LMDIAG', 'CLSB', 'SHWARTZ', 'EVAL', 'LEDS'])
display(df_res_lsp_pivot)

DataFrame2Latex(df= df_res_lsp_pivot , label=f'tab:lsp_single_pattern_ablation', 
            caption=f'Experimental results on lexico-syntactic single patterns.', 
            output_file= None , #'../log/paper_results/latex.test.tex',
            adjustbox_width = 'textwidth',
            precision = 1,
            column_format='l|ll|ll|ll|ll|ll|ll',
            multicolumn_format='c|'
            )


# In[ ]:





# In[ ]:




