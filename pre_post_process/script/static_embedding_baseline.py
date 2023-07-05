#!/usr/bin/env python
# coding: utf-8

# # Hypernym in KNN (Word2Vec trained on 300d GoogleNews)
# 
# Pointer: 
# - Word2Vec (Home Page) https://code.google.com/archive/p/word2vec/ 
# - GoogleNews word2vec: The archive is available here: GoogleNews-vectors-negative300.bin.gz.  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# In[ ]:


import pandas as pd
import copy 
from tqdm import tqdm 
from collections import Counter, defaultdict, OrderedDict

import gensim 
from gensim.models import Word2Vec, KeyedVectors 
tqdm.pandas()
from IPython.display import display
pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',500)

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# ## HELPER FUNCTIONS

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
    data_raw = fin.readlines()
    data = []
    for x in data_raw:
        x = eval(x)
        if not x['sub_label'] in model.key_to_index: continue  ### exluding OOV; or if word not in model
        data.append(x)
        #data = [eval(x) for x in data]
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


def layout_table(df, dataset_list = ['BLESS','LMDIAG', 'CLSB',  'SHWARTZ', 'EVAL', 'LEDS']):
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

dataset_to_jsonl_path={
    "EVAL": "../data/hypernymysuite/data/hypernymsuite/EVAL/IsA.jsonl",
    "BLESS": "../data/hypernymysuite/data/hypernymsuite/BLESS/IsA.jsonl",
    "LEDS": "../data/hypernymysuite/data/hypernymsuite/LEDS/IsA.jsonl",
    "LMDIAG": "../data/probe-generalization/Syntagmatic/LM-Diagnostic-Extended/singular/IsA.jsonl",
    "CLSB": "../data/CLSB/single_label/IsA.jsonl",
    "SHWARTZ": "../data/hypernymysuite/data/hypernymsuite/SHWARTZ/IsA.jsonl",
    }



# ## Get Hypernyms in KNN

# In[4]:


def get_knn(model, word, topn=10, return_score=False):
    neighbours = model.most_similar([model[word]], topn=topn+1)[1:]
    if return_score: 
        return neighbours
    return [x[0] for x in neighbours]

neighbours = get_knn(model, 'dog')
print(neighbours)


# In[ ]:



df_res = []
top_k = 10
debug=False #True
# debug=True

for dataset, filepath in dataset_to_jsonl_path.items():
    print(dataset)
    if dataset !='SHWARTZ': continue 
    df = load_data(filepath)
    print(f"#Test_instances: {len(df.index)} (excluding OOV)")
    if debug:
        df = df.head(5)
        
    df['sub_neighbours'] = df['sub_label'].progress_apply(lambda x: get_knn(model=model, word=x, topn=top_k))
    df[f'p@{top_k}'] = df[['obj_label', 'sub_neighbours']].apply(lambda x: 1 if x[0][0] in x[1]  else 0, axis=1)
    df[f'mrr@{top_k}'] = df[['obj_label', 'sub_neighbours']].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1]), axis=1)

    p_at_k = df[f'p@{top_k}'].sum()/len(df.index)
    mrr = df[f'mrr@{top_k}'].sum()/len(df.index)
    df_res.append({"dataset": dataset, "P@K": p_at_k, 'MRR': mrr})
    print(f"P@K: {p_at_k} MRR:{mrr}")
    print()
df_res = pd.DataFrame(df_res)
display(df_res)
df_res.to_csv("../log/221217_baseline_word2vec.csv")


# DataFrame2Latex(df= df_res , label=f'tab:baseline_word2vec', 
#             caption=f'Experimental results on extracting hypernyms with word2vec.', 
#             output_file= None , #'../log/paper_results/latex.test.tex',
#             adjustbox_width = 'textwidth',
#             precision = 1,
#             column_format='l|ll|ll|ll|ll|ll|ll',
#             multicolumn_format='c|'
#             )


# In[ ]:



# dog = model['dog']
# print(dog.shape)
# print(dog[:10])

# # Deal with an out of dictionary word: Михаил (Michail)
# if 'Михаил' in model:
#     print(model['Михаил'].shape)
# else:
#     print('{0} is an out of dictionary word'.format('Михаил'))

# # Some predefined functions that show content related information for given words
# print(model.most_similar(positive=['woman', 'king'], negative=['man']))
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))
# print(model.similarity('woman', 'man'))

