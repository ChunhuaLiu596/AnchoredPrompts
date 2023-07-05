import os, sys 
import pandas as pd
pd.options.display.max_columns=500
pd.options.display.max_colwidth=1000
import matplotlib.pyplot as plt
from tqdm import tqdm 
tqdm.pandas()
import re 
from collections import Counter, defaultdict, OrderedDict
import seaborn as sns
from copy import deepcopy
from tabulate import tabulate, simple_separated_format
from inflection import singularize, pluralize
from df_to_latex import DataFrame2Latex
from collections import Counter, defaultdict
from inflection import singularize, pluralize

from utils_path import dataset_to_respath
# # Using WN, WN.taxonomy to retrieve path distance 
# - Pointers: 
#     - https://wn.readthedocs.io/en/latest/setup.html
#     - https://wn.readthedocs.io/en/latest/api/wn.html
# 
# - Installation: 
# ```
# !pip install wn
# !pip install wn[web]
# wn.download('ewn:2020')
# ```

# get the co-hyponyms from WordNet (Shick and Schutze, 2020)
# - get the hypernyms, maxinum d(x,y) is 2
# - get the top 2 most frequent senses of each hypernym 
# - get hyponyms of each hypernyms, maxinum distance d(y,z) is 4 
# - constrain the depeth of hypernyms to be 6

# In[201]:


import re 
import wn, wn.taxonomy
ewn = wn.Wordnet('ewn:2020')

def debug_hypernyms():
    '''
    why debug? some common words such as mountain, grain don't have cohyponyms. 
    explanation: the original algorithms contrains the depth of the hypernyms to be larger than 6; which is violated when the depth of input word is smaller than 6
    '''
    word = 'mountain'
    # word ='dog'
    min_taxo_depth = 6 
    max_path_hyper = 2
    hyper_synsets = []
    print_flag=True
    for i, synset in enumerate(wn.synsets(word, pos='n')[:k_synset]): #top K senses of word 
        print(f"{word} synset {i+1}: {synset}")
        for j, path in enumerate(wn.taxonomy.hypernym_paths(synset)): #retrieve the hyper path for each synset 
            print(f"path {j}: {path}")
            for i, ss in enumerate(path[:max_path_hyper]): # get the hypernyms within max_path_hyper
                ss_min_txo_depth = wn.taxonomy.min_depth(ss, simulate_root=False)
                if ss_min_txo_depth< min_taxo_depth: 
                    print(f"{ss} {ss.lemmas()[0]} {ss_min_txo_depth}")
                    print(wn.taxonomy.max_depth(ss, simulate_root=False))
                    print(wn.taxonomy.max_depth(synset, simulate_root=False))
                    
                    continue  #remove general concepts like "entity", 'physical entity'
                hyper_synsets.append(ss)
                if print_flag: 
                    print(' ' * i, ss, ss.lemmas()[0], ss_min_txo_depth)

def test_min_depth():
    for word in ['concept', 'thought', 'living thing', 'whole', 'psychological feature', 'unit', 'artifact', 'abstraction', 'object','physical entity', 'entity']:
        synset = wn.synsets(word, pos='n')[0]
        min_depth = wn.taxonomy.min_depth(synset, simulate_root=False)
        print(word, min_depth)


def get_inherited_hypernyms(word, k_synset, max_path_hyper, min_taxo_depth=6, print_flag=False):
    '''
    k_synset: the most frequent k_synset of word
    max_path_hyper: up to k level of hypernyms, e.g., 2 level higher than word 
    min_taxo_depth=6: concept, exluded hypernyms: unit, object, artifact, entity
    '''
    
    hyper_synsets = []
    for i, synset in enumerate(wn.synsets(word, pos='n')[:k_synset]): #top K senses of word 
        #print(f"{word} synset {i+1}")
        for j, path in enumerate(wn.taxonomy.hypernym_paths(synset)): #retrieve the hyper path for each synset 
            #print(f"path {j}")
            for i, ss in enumerate(path[:max_path_hyper]): # get the hypernyms within max_path_hyper
                ss_min_txo_depth = wn.taxonomy.min_depth(ss, simulate_root=False)
                
                if ss_min_txo_depth< min_taxo_depth: continue  #remove general concepts like "entity", 'physical entity'
                hyper_synsets.append(ss)
                if print_flag: 
                    print(' ' * i, ss, ss.lemmas()[0], ss_min_txo_depth)
                    
    return hyper_synsets

def get_direct_hyonyms(synsets):
    '''
    Return the direct hyponyms of a given list of synsets
    ''' 
    sister_synsets = []
    for synset in synsets: 
        sister_synsets.extend(synset.hyponyms() )
    return sister_synsets


def get_inherited_hyponyms(initial_synsets, max_path_hypo):
    synsets = initial_synsets
    synsets_hyponyms = []
    
    while max_path_hypo>0:
        synsets = get_direct_hyonyms(synsets)
        synsets_hyponyms.extend(synsets)
        max_path_hypo -=1
        #print(dist)
        #print(synsets)
        #print("-"*80)
    #print(Counter(synsets_all).most_common())
    return synsets_hyponyms


def filter_cohyponyms(word, synsets_cohyponyms, top_k=50):
    cohyponyms = []
    for synset in synsets_cohyponyms:
        for lemma in synset.lemmas():
            if lemma == word: continue 
            if len(lemma.split(" ")) >1 or len(lemma.split("-")) >1: continue 
            cohyponyms.append(lemma.lower())
    cohyponyms = Counter(cohyponyms)
     #if top_k !=None:        
    return cohyponyms.most_common(top_k)
    #else:
    #    return dict(cohyponyms.most_common())

    
def get_cohyponyms(word, top_k_cohyonyms=50, top_k_word_synset=2, max_path_hyper=2, max_path_hypo =4, print_flag=False):
    
    hyper_synsets = get_inherited_hypernyms(word, k_synset=top_k_word_synset, max_path_hyper = max_path_hyper)
    if print_flag:
        for synset in hyper_synsets:
            print(synset, synset.lemmas())

    synsets_cohyponyms = get_inherited_hyponyms(hyper_synsets, max_path_hypo= max_path_hypo)

    concept_cohyponyms = filter_cohyponyms(word, synsets_cohyponyms, top_k=top_k_cohyonyms)
    return list(dict(concept_cohyponyms).keys())
   

def test_get_cohyponyms(word, test_cohyponyms):
    '''
    word = 'corn'
    test_cohyponyms = ['bean', 'potato', 'barley', 'wheat', 'pea'] 
    word = 'train'
    test_cohyponyms = ['bus', 'plane', 'car', 'tram', 'truck']
    test_get_cohyponyms(word,test_cohyponyms )
    '''
    top_k_cohyonyms = None #200 
    top_k_word_synset = 2
    max_path_hyper = 2
    max_path_hypo = 4

    concept_cohyponyms  = get_cohyponyms(word, top_k_cohyonyms=top_k_cohyonyms, 
                                         top_k_word_synset=top_k_word_synset, 
                                         max_path_hyper=max_path_hyper, max_path_hypo = max_path_hypo)
    
    for query in test_cohyponyms:
        if query in concept_cohyponyms:
            print(query, 'yes')
        else:
            print(query, 'no')
    print(len(concept_cohyponyms), concept_cohyponyms)



# # Evaluation 

# In[225]:


def merge_predictions_in_concept_level(words, uniform_funcion=None, top_k=None ):
    '''
    uniform_function: either signualarize or pluralize 
    '''
    words_uniformed = [uniform_funcion(word) for word in words] if uniform_funcion !=None else words
    concepts = list(OrderedDict.fromkeys(words_uniformed))
    return concepts[:top_k] if top_k is not None else concepts

def concept_evaluation(label, pred):
    '''
    
    label: a list with the singualr and plural labels (e.g., ['tool', 'tools'])
    pred: the top K prediction list 

    return:
        1 if label share with pred else 0  
    '''
    if not isinstance(label, list):
        label = eval(label)
        
    if not isinstance(pred, list):
        pred = eval(pred)

    shared = set(label).intersection(set(pred))
    return 1 if len(shared)>0 else 0 
    # return len(shared)/len(pred)
    

def get_precision_at_k_concept(df, relation, pred_cols, label_col, k_list, pred_col_suffix='obj_mask_'):
    '''
    evalaute model predictions in concept level, ignoring the morphology affects (singular, plural)
    '''

    p_at_x = [] #defaultdict() 
    for pred_col in pred_cols: 
        suffix = pred_col.replace(pred_col_suffix, "")
        prec_cur = defaultdict()
        prec_cur['mask_type'] = suffix
        for k in k_list: 
            df[f'p{k}_{suffix}'] = df[[label_col, pred_col]].apply(lambda x: concept_evaluation(x[0], eval(x[1])[:k] if isinstance(x[1], str) else x[1][:k]), axis=1 )
            prec_cur[f'p@{k}'] = round(df[f'p{k}_{suffix}'].mean() , 3)*100

        p_at_x.append(prec_cur)  
        

    # aggregate the average precision across k 
    df_res = pd.DataFrame(p_at_x) #, columns=['mask_type', 'mAP'])
    df_res['relation'] = [relation]*len(df_res)
    return df_res

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


def get_mrr(df, relation, pred_cols, label_col, pred_col_suffix):
    '''
    mrr is calculated based on the top_k rank, all elements in obj_col are used
    '''

    mrr = [] 
    for i, pred_col in enumerate(pred_cols):
        cur_mrr = defaultdict()
        suffix = pred_col.replace(pred_col_suffix, "")

        df[f'mrr_{suffix}'] = df[[label_col, pred_col]].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1]), axis=1 ) 
        
        cur_mrr['mask_type'] = suffix
        cur_mrr[f"mrr"] = round(df[f'mrr_{suffix}'].mean(), 3)*100
        mrr.append(cur_mrr)

    mrr_df =  pd.DataFrame(data = mrr) #, columns=['mask_type', 'mrr'])
    # mrr_df['mask_type']= mrr_df['mask_type'].apply(lambda x: x.replace(""))
    mrr_df['relation'] = relation
    return mrr_df 


# In[202]:


def get_dataset_to_respath(dataset_to_respath, print_flag=False):
    # remote path 
#     dataset_to_respath = {'hypernymsuite-BLESS': 'log/bert-large-uncased/hypernymsuite/BLESS/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv', 'lm_diagnostic_extended-singular': 'log/bert-large-uncased/lm_diagnostic_extended/singular/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.LM_DIAGNOSTIC_EXTENDED.csv', 'clsb-singular': 'log/bert-large-uncased/clsb/singular/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.CLSB.csv', 'hypernymsuite-LEDS': 'log/bert-large-uncased/hypernymsuite/LEDS/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv', 'hypernymsuite-EVAL': 'log/bert-large-uncased/hypernymsuite/EVAL/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv', 'hypernymsuite-SHWARTZ': 'log/bert-large-uncased/hypernymsuite/SHWARTZ/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_True_cpt_False.HYPERNYMSUITE.csv'}

    source_dir = 'spartan:~/cogsci/DAP/'
    target_dir = '../../'
    dataset_to_localpath = defaultdict()
    dataset_rename = {
        'hypernymsuite-BLESS': 'BLESS', 'lm_diagnostic_extended-singular': 'DIAG', 'clsb-singular':'CLSB', 'hypernymsuite-LEDS': 'LEDS', 'hypernymsuite-EVAL': 'EVAL', 'hypernymsuite-SHWARTZ': 
        "SHWARTZ"
    }
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
#     print(dataset_to_localpath)
    return dataset_to_localpath
dataset_to_localpath = get_dataset_to_respath(dataset_to_respath)


from nltk.corpus import wordnet 
wn_lemmas = set(wordnet.all_lemma_names())
def check_word_in_wordnet(word, wn_lemmas):
    '''
    1 if word in wordnet else 0
    '''
    return 1 if word in wn_lemmas else 0 


def get_all_vocab(dataset_to_localpath):
    dataset_to_df = defaultdict()
    vocab_sub = set()
    for dataset, path in dataset_to_localpath.items(): 
        if debug:
           if dataset!='DIAG': continue 
        print("dataset", dataset)
        df = pd.read_csv(path)
        vocab_sub.update(df['sub_label_sg'].to_list())
        print(len(vocab_sub))
    return list(vocab_sub)




## Config for getting co-hyponyms from WordNet
top_k_cohyonyms = None #200 
top_k_word_synset = 2
max_path_hyper = 2
max_path_hypo = 4

#config for evaluation 
pred_col_suffix=''
label_col = 'sub_sister_new'
pred_cols = ['subj_anchors_all_sg']
relation='co-hyponyms'
debug= False #True #eval(sys.argv[1])
print("debug", debug)

vocab_sub = get_all_vocab(dataset_to_localpath)
word_to_cohyponyms = defaultdict(list)
for i, word in enumerate(vocab_sub):
    if i%10==0: print(i)
    if not check_word_in_wordnet(word, wn_lemmas):
        word_to_cohyponyms[word] = []
    else:
        cohyponyms = get_cohyponyms(word, top_k_cohyonyms=top_k_cohyonyms, 
                                 top_k_word_synset=top_k_word_synset, 
                                 max_path_hyper=max_path_hyper, 
                                 max_path_hypo = max_path_hypo)
        word_to_cohyponyms[word] = cohyponyms
    if debug:
        print(word, cohyponyms)
       
df = pd.DataFrame(word_to_cohyponyms.items(), columns=['word', 'cohyponyms'])
output_path = '../log/word_to_cohyponyms.txt'
df.to_csv(output_path, index=False)
print(f"save {output_path}")



