import pandas as pd
import re
from collections import defaultdict
from utils_wordnet import get_sister_terms



def get_results_log_path(line):
    # line="save log/bert-large-uncased/lm_diagnostic_extended/singular/df_all_use_global_dap_True_anchor_type_Coordinate_remove_Y_PUNC_FULL_concate_or_single_max_anchor_num_10_anchor_scorer_probAvg_filter_obj_True_filter_objects_with_input_True_wnp_False_cpt_False.LM_DIAGNOSTIC_EXTENDED.tsv"
    res_path = line.split(" ")[1].strip()
#     print(res_path)
    return res_path
    
def get_dataset_name(line):
    line_list = line.strip().split(", ") 
    for item in line_list:
        item_split = item.split(":")
        if 'data_dir' in item_split[0]:
            dataset_name = "-".join(item_split[1].split("/")[1:-1])
    return dataset_name

def get_dataset_to_respath(path):
    '''
    parse the log to get the dataset name and the result path 
    '''
    dataset_to_respath = defaultdict()
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
            # if line.startswith("save") and  and '.tsv' in line:
                # res_path = get_results_log_path(line)
                # dataset_to_respath[dataset] = res_path
            if line.startswith("save") and '.csv' in line:
                res_path = get_results_log_path(line)
                dataset_to_respath[dataset] = res_path

    return dataset_to_respath


def 

filter_anchor_to_logpaths = { 'WN': 'log/221207_all_datasets_wn.log',}
                            # 'CPT': 'log/221207_all_datasets_cpt.log'}

anchor_col =  'subj_anchors'
anchor_col_sg =  'subj_anchors_sg'
anchor_col_pl =  'subj_anchors_pl'
# filter_anchor_to_logpaths = {"BLESS": 'log/221207_bless_anchor_scoers_ablation.log'} 
dfs = []
for filter_anchor, log_path in filter_anchor_to_logpaths.items():
    dataset_to_respath = get_dataset_to_respath(log_path)
    print(dataset_to_respath)
    for dataset, path in dataset_to_respath.items(): 
        df = pd.read_csv(path, sep='\t')
        df['dataset'] = dataset
         df = df.loc[:, ~df.columns.str.contains('^Unnamed*')]


        if 'sub_sister' in not in df.columns:
            df['sub_sister'] = df['sub_label'].apply(lambda x: get_sister_terms(x, distance_to_hypernym=6) )

        print("-"*40,"anchor evaluation", "-"*40)
        pred_col_suffix=''
        label_col = 'sub_sister'
        pred_cols = [anchor_col_sg] #['subj_anchors']
        df_prec = get_precision_at_k_concept(df, relation, pred_cols, label_col, k_list=[1, 5, max_anchor_num],pred_col_suffix=pred_col_suffix ) ##note that this would be super slow when top_k is large (>1000) 
        df_mAP = get_mean_average_precision_at_k(df, relation, pred_cols,label_col, k_list=[1, 5, max_anchor_num], pred_col_suffix=pred_col_suffix)
        df_recall = get_recall_at_k(df, relation, pred_cols, label_col, k_list=[1, 5, max_anchor_num], pred_col_suffix=pred_col_suffix)
        for k in [max_anchor_num]:
            df_prec[f'mAP@{k}'] = df_prec['mask_type'].apply(lambda x:  df_mAP.loc[df_mAP['mask_type']==x, f'mAP@{k}'].values[0])
            df_prec[f'recall@{k}'] = df_recall['mask_type'].apply(lambda x:  df_recall.loc[df_mAP['mask_type']==x, f'recall@{k}'].values[0])

        # add WordNet path score for evaluation 
        anchor_wordnet_avg_path, anchor_wordnet_coverage = get_wordnet_avg_path_between_sub_and_anchors(df, oov_path_len = 100)
        df_prec['anchor_wordnet_avg_path'] = anchor_wordnet_avg_path
        df_prec['anchor_wordnet_coverage'] = anchor_wordnet_coverage

        df_prec_display = df_prec[["mask_type", "p@1", "p@5", f"p@{max_anchor_num}", "relation",  
                                   "mAP@1", "mAP@5" , f"mAP@{max_anchor_num}", "recall@1", "recall@5", f"recall@{max_anchor_num}", 
                                    "anchor_wordnet_avg_path", "anchor_wordnet_coverage"]]
    
        # print(tabulate(df_prec_display, headers='firstrow', tablefmt='simple'))
        print(tabulate(df_prec_display, tablefmt='latex', headers=df_prec_display.columns).replace("\\", "").replace("&", "\t"))
        # display(df_prec_display)


        query_mask_type = ['def_sap', 'def_dap', 'lsp_sap', 'lsp_dap']
        df = df.query(f"mask_type in {query_mask_type}")
       
        # df.index = df['mask_type']
        # df = df.reindex(query_mask_type)
        df['filter_anchors'] = filter_anchor
        dfs.append(df)
        
dfs = pd.concat(dfs)
# output_file = f'log/221207_all_results.csv'
# output_file = f'log/221207_bless.csv'
dfs.to_csv(output_file)
print(output_file)