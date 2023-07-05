import re 

def get_revserse_prompt(stimulus_word , output, unmasker, mask_string, top_k=10):
    '''
    Test whether two words are mutually predicted by LMs 
    For example: 
        such as apple and ___ [cherry, banana, lemon, grape, grapes]
        such as cherry and ___ [apple, lime, lemon, orange ]
        such as banana and ___ [coconut, coffee, rice, orange ]
        such as grapes and __ [vegetables, fruits, wine]
    
    args:
        stimulus_word: the stimulus word 
        output: one output of a LM, a dict with keys 'token_str', 'sequence', 'token_score'
        unmasker: the LM query the reverse prompt 
        mask_string: each LM has a packed mask_string, "<mask>" for roberta, "[MASK]" for bert 

    return 1 if they are mutually predicted other 0

    Future extension: we can also measure the probabilty gap if two words are mutually predicted  
    '''

    # print("stimulus_word:", stimulus_word)
    # print("output:", output)

    # if output['token_str'].strip().lower()==stimulus_word:
        # return "" 

    # construct the reverse prompts by: 
    # (1) replacing the predicted word (token_str) with a mask_string 
    # (2) replacing the stimulus_word with the predicted word 
    # do (1) first as the stimulus_word might be a substring of token_str (e.g., apple vs apples) 
    # it will give to mask_strings if we do (2) first 
    sequence_list   = re.findall(r"[\w']+|[.,!?;]", output['sequence'])
    # sequence_list = sequence.split()
    stimulus_position = sequence_list.index(stimulus_word)
    output_position = sequence_list.index(output['token_str'])

    sequence_list[stimulus_position] =  output['token_str'] 
    sequence_list[output_position] =  mask_string 

    reverse_prompt = " ".join(sequence_list)
    return reverse_prompt



# def concpet_positioning_test(stimulus_word ,token2reverse_prompts, unmasker, mask_string, top_k=10):
#     outputs = unmasker(reverse_prompts.values(), top_k= top_k)
    
#     scores = []
#     for output in outputs:
#         reverse_tokens = set([item['token_str'].strip().lower() for item in output])
#         scores.append(1) if stimulus_word in reverse_outputs else scores.append(0)

#     # score = [1 for item in reverse_outputs] if stimulus_word in set(item['token_str'].strip().lower()) else 0]
#     # return 1 if stimulus_word in reverse_tokens else 0
#     return dict(zip(token2reverse_prompts.keys(), scores))



def concpet_positioning_test(stimulus_word , output, unmasker, mask_string, top_k=10):
    '''
    Test whether two words are mutually predicted by LMs 
    For example: 
        such as apple and ___ [cherry, banana, lemon, grape, grapes]
        such as cherry and ___ [apple, lime, lemon, orange ]
        such as banana and ___ [coconut, coffee, rice, orange ]
        such as grapes and __ [vegetables, fruits, wine]
    
    args:
        stimulus_word: the stimulus word 
        output: one output of a LM, a dict with keys 'token_str', 'sequence', 'token_score'
        unmasker: the LM query the reverse prompt 
        mask_string: each LM has a packed mask_string, "<mask>" for roberta, "[MASK]" for bert 

    return 1 if they are mutually predicted other 0

    Future extension: we can also measure the probabilty gap if two words are mutually predicted  
    '''

    # print("stimulus_word:", stimulus_word)
    # print("output:", output)

    if output['token_str'].strip().lower()==stimulus_word:
        return 0

    # construct the reverse prompts by: 
    # (1) replacing the predicted word (token_str) with a mask_string 
    # (2) replacing the stimulus_word with the predicted word 
    # do (1) first as the stimulus_word might be a substring of token_str (e.g., apple vs apples) 
    # it will give to mask_strings if we do (2) first 
    sequence_list   = re.findall(r"[\w']+|[.,!?;]", output['sequence'])
    # sequence_list = sequence.split()
    stimulus_position = sequence_list.index(stimulus_word)
    output_position = sequence_list.index(output['token_str'])

    sequence_list[stimulus_position] =  output['token_str'] 
    sequence_list[output_position] =  mask_string 

    reverse_prompt = " ".join(sequence_list)

    # reverse_prompt = output['sequence'].replace(output['token_str'], mask_string).replace(stimulus_word, output['token_str'])
    # print(reverse_prompt)

    reverse_outputs = unmasker(reverse_prompt, top_k= top_k)
    # print(reverse_outputs)
    reverse_tokens = set([item['token_str'] for item in reverse_outputs])
    return 1 if stimulus_word in reverse_tokens else 0


def test_concpet_positioning_test():
    print("-"*40, "singular", "-"*40)
    mask_string = '[MASK]'
    dfe_all = []
    cue = 'apple'
    text = f'such as {cue} and {mask_string},'
    outputs = unmasker(text, top_k= 10)
    dfe =  pd.DataFrame(outputs)

    for output in outputs:
        print(output) 
        cpt = concpet_positioning_test(cue, output, unmasker, mask_string)
        print(cue, output['token_str'], cpt)
        print("-"*80)


def filter_outputs_with_cpt(inputs, outputs, unmasker, mask_string, filter=True, return_probs=True,\
                            top_k=None, scorer='freqProbSum', filter_inputs=True, add_wordnet_path_score=False):
    '''
    

    inputs: the original inputs, for example [A] is a type of [B], A is the input
    outputs: the candidates returned by PTLMs

    filter: True 
        filter: non-alpha tokens); 

    top_k: take the top_k outputs. This is important when using multiple prompts for each sub 

    '''
    anchor_list = []
    anchor_scores = [] 
        
    for input_word, top_outputs in zip(inputs, outputs):  #iterate through the samples (sub)
        token2cpt  = defaultdict(int) #filter/accumulate predictions for each sample 

        filled_scores = defaultdict(list) #a list of token:[score1, score2, ...]   

        if isinstance(top_outputs[0], list):
            flatten_output = [item for top_k_output  in top_outputs for item in top_k_output]
        else:
            flatten_output = [item for item  in top_outputs]
        # print("flatten_output", flatten_output)

        for i, output in enumerate(flatten_output):
            filled_token = output['token_str'].strip().lower()
            filled_score = output['score']
            
            if filter:
                #####Add pre-conditions to filter unwanted ################
                # filter the repetation of a concept in the explanation. See the the following example
                # [MASK] is the capability to do a particular job . -> capacity 
                if not filled_token.isalpha(): continue
                if filled_token in STOP_WORDS: continue 
                if len(filled_token)<=1: continue 
                if filter_inputs:
                    if filled_token.strip() in [re.sub("\s+", '', x) for x in input_word.split()]: continue #filter out the target in input  
                if filled_token.startswith("#"): continue
                #####Add pre-conditions to filter unwanted ################
                token2cpt[filled_token] = concpet_positioning_test(stimulus_word=input_word , output= output, unmasker=unmasker, mask_string=mask_string)
            else:
                # token2cpt.append(filled_token)  
                token2cpt[filled_token] +=1
                filled_scores[filled_token] += filled_score
                # filled_scores.append(filled_score)
        # row.filled_subj_anchor = token2cpt 
        if len(token2cpt) ==0: 
            token2cpt={'MISSING':1}
            filled_scores['MISSING'] = [0]
        # feed the input into the agrregate _token_scores() so that we can calcuate the 
        filled_scores_aggregated = aggregate_token_scores(input_word, token2probs=filled_scores, scorer=scorer, top_k=top_k,  add_wordnet_path_score=add_wordnet_path_score, sort_flag=True)
        anchor_list.append(list(filled_scores_aggregated.keys())) 
        anchor_scores.append(filled_scores_aggregated) 
        # print("-"*60)
    return anchor_list if not return_probs  else pd.Series((anchor_list,anchor_scores))



