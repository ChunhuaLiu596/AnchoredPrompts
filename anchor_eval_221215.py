
import pandas as pd

path1 = 'log/bert-large-uncased/BLESS/coordinate/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_False_wnp_False_cpt_False.BLESS.csv'
path2 = 'log/bert-large-uncased/BLESS/coordinate/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_False_wnp_False_cpt_True.BLESS.csv'
path3 =  'log/bert-large-uncased/BLESS/coordinate/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_False_wnp_True_cpt_False.BLESS.csv'

# '../log/exp_data_results_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_False_wnp_True_cpt_False.BLESS.csv'

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


for path in [path1, path2, path3]:
    df = pd.read_csv(path)
    df['sub_sister'] = df['sub_sister'].apply(lambda x: eval(x))
    df['subj_anchors'] = df['subj_anchors'].apply(lambda x: eval(x))
    df['subj_anchors_sg'] = df['subj_anchors_sg'].apply(lambda x: eval(x))

    df_groups = []
    for name, group in df.groupby('sub_label_sg'):
        df_groups.append({"sub_label": name, 'sub_sister': group['sub_sister'].values, 'subj_anchors_sg': group['subj_anchors_sg'].values[0], 'subj_anchors': group['subj_anchors'].values[0],  })
    # df.head()
    df_groups = pd.DataFrame(df_groups)
    df_groups['sub_sister'] = df_groups['sub_sister'].apply(lambda x: [item[0] for i, item in enumerate(x) if i!=0])
    df_groups['subj_anchors_sg'] = df_groups['subj_anchors_sg'].apply(lambda x: [item for i, item in enumerate(x) if i!=0])
    df_groups['subj_anchors'] = df_groups['subj_anchors_sg'].apply(lambda x: [item for i, item in enumerate(x) if i!=0])
    # display(df_groups)


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
        

    for k in [1, 5, 10] :
        df_groups[f'p{k}'] = df_groups[['sub_sister', 'subj_anchors']].apply(lambda x: concept_evaluation(x[0], x[1][:k]), axis=1)
    df_groups['mrr'] = df_groups[['sub_sister', 'subj_anchors']].apply(lambda x: get_highest_mrr_among_labels(x[0], x[1][:k]), axis=1)
    print(df_groups[['p1', 'p5', 'p10', 'mrr']].mean())
    # display(df_groups)