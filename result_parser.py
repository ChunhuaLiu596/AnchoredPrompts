import pandas as pd
import re
import json
from collections import defaultdict



def get_results_path(line):
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

def get_top_k_anchors(line):
    line_list = line.strip().split(", ") 
    for item in line_list:
        item_split = item.split(":")
        if 'top_k_anchors' in item_split[0]:
            top_k_anchors = item_split[1]
    return top_k_anchors

def get_dataset_to_respath(path):
    '''
    parse the log to get the dataset name and the result path 
    '''
    dataset_to_respath = defaultdict()
    dataset_to_top_k_anchors = list() 
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
                dataset_to_top_k_anchors.append(dataset_to_top_k_anchors) 
            if '.tsv' in line:
                res_path = get_results_path(line)
                dataset_to_respath[dataset] = res_path

    return dataset_to_respath 

# def parse_results_log(path)

def get_bless_true_anchor_pattern_ablation_results():
    filter_anchor_to_logpath = 'log/221208_bless_true_anchors.log'
    filter_anchor_type = ["RAW", 'CPT', 'WN'] 

    dfs = []
    # for filter_anchor, log_path in filter_anchor_to_logpaths.items():
    count = 0
    with open(filter_anchor_to_logpath , 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
                top_k_anchors = get_top_k_anchors(line)
            if '.tsv' in line:
                res_path = get_results_path(line)
                df = pd.read_csv(res_path, sep='\t')
                df['dataset'] = dataset
                df['top_k_anchors '] = top_k_anchors 
                query_mask_type = ['def_sap', 'def_dap', 'lsp_sap', 'lsp_dap']
                df = df.query(f"mask_type in {query_mask_type}")
                df = df.loc[:, ~df.columns.str.contains('^Unnamed*')]
                # df.index = df['mask_type']
                # df = df.reindex(query_mask_type)
                print(count)
                df['filter_anchors']  = [filter_anchor_type[count] ] * len(df.index)
                dfs.append(df)
                count  +=1
            
    dfs = pd.concat(dfs)
    # output_file = f'log/221207_all_results.csv'
    output_file = f'log/221208_bless_true_anchors.csv'
    dfs.to_csv(output_file)
    print(output_file)





def get_bless_pred_anchor_pattern_ablation_results():
    filter_anchor_to_logpath = 'log/221209_bless_anchor_scorers_ablation.log'
    filter_anchor_type = ["RAW", 'CPT', 'WN'] 

    dfs = []
    # for filter_anchor, log_path in filter_anchor_to_logpaths.items():
    count = 0
    with open(filter_anchor_to_logpath , 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
            if '.tsv' in line:
                res_path = get_results_path(line)
                df = pd.read_csv(res_path, sep='\t')
                df['dataset'] = dataset
                query_mask_type = ['def_sap', 'def_dap', 'lsp_sap', 'lsp_dap', 'anchor_col_all']
                df = df.query(f"mask_type in {query_mask_type}")
                df = df.loc[:, ~df.columns.str.contains('^Unnamed*')]
                # df.index = df['mask_type']
                # df = df.reindex(query_mask_type)
                print(count)
                df['filter_anchors']  = [filter_anchor_type[count] ] * len(df.index)
                dfs.append(df)
                count  +=1
            
    dfs = pd.concat(dfs)
    # output_file = f'log/221207_all_results.csv'
    output_file = f'log/221209_bless_anchor_scorers_ablation.csv'
    dfs.to_csv(output_file)
    print(output_file)

def get_hyper_results_all_datasets(filter_anchor_to_logpaths, output_file = f'log/221208_all_datasets_log.csv'):

    dfs = []
    for filter_anchor, log_path in filter_anchor_to_logpaths.items():
        dataset_to_respath = get_dataset_to_respath(log_path)

        print(json.dumps(dataset_to_respath, indent=4))
        for i, (dataset, path) in enumerate(dataset_to_respath.items()): 
            df = pd.read_csv(path, sep='\t')
            df['dataset'] = dataset
            query_mask_type = ['def_sap', 'def_dap', 'lsp_sap', 'lsp_dap']
            df = df.query(f"mask_type in {query_mask_type}")
            df = df.loc[:, ~df.columns.str.contains('^Unnamed*')]
            # df.index = df['mask_type']
            # df = df.reindex(query_mask_type)
            df['filter_anchors'] = filter_anchor
            dfs.append(df)
            
    dfs = pd.concat(dfs)
    # output_file = f'log/221207_all_results.csv'
    
    dfs.to_csv(output_file)
    print(output_file)



def get_anchor_results_all_datasets(filter_anchor_to_logpaths, output_file):

    # filter_anchor_to_logpaths = {
    #                             'RAW': 'log/221208_all_datasets_raw.log', 
    #                             'WN': 'log/221208_all_datasets_wn.log',
    #                             'CPT': 'log/221208_all_datasets_cpt.log'
    #                             }

    dfs = []
    for filter_anchor, log_path in filter_anchor_to_logpaths.items():
        dataset_to_respath = get_dataset_to_respath(log_path)
        print(dataset_to_respath)
        for dataset, path in dataset_to_respath.items(): 
            df = pd.read_csv(path, sep='\t')
            df['dataset'] = dataset
            query_mask_type =['anchor_col_all'] # ['def_sap', 'def_dap', 'lsp_sap', 'lsp_dap']
            df = df.query(f"mask_type in {query_mask_type}")
            df = df.loc[:, ~df.columns.str.contains('^Unnamed*')]
            # df.index = df['mask_type']
            # df = df.reindex(query_mask_type)
            cols = ['dataset', 'mask_type', 'p@1', 'p@5', 'p@10', 'mrr', 'anchor_wordnet_avg_path', 'anchor_wordnet_coverage']
            df = df[cols]
            df['filter_anchors'] = filter_anchor
            dfs.append(df)
            
    dfs = pd.concat(dfs)
    # output_file = f'log/221207_all_results.csv'
    # output_file = f'log/221208_all_datasets_anchor_log.csv'
    dfs.to_csv(output_file)
    print(output_file)




def get_anchor_num_ablation_results():
    filter_anchor_to_logpath = 'log/221209_lmdai_anchor_num_ablation.log'
    # filter_anchor_type = ["RAW", 'CPT', 'WN'] 

    dfs = []
    # for filter_anchor, log_path in filter_anchor_to_logpaths.items():
    count = 0
    with open(filter_anchor_to_logpath , 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
                top_k_anchors = get_top_k_anchors(line)
            if '.tsv' in line:
                res_path = get_results_path(line)
                df = pd.read_csv(res_path, sep='\t')
                df['dataset'] = dataset
                df['top_k_anchors'] = top_k_anchors 
                query_mask_type = ['def_sap', 'def_dap', 'lsp_sap', 'lsp_dap', 'anchor_col_all']
                df = df.query(f"mask_type in {query_mask_type}")
                df = df.loc[:, ~df.columns.str.contains('^Unnamed*')]
                # df.index = df['mask_type']
                # df = df.reindex(query_mask_type)
                print(count)
                # df['filter_anchors']  = [filter_anchor_type[count] ] * len(df.index)
                dfs.append(df)
                count  +=1
            
    dfs = pd.concat(dfs)
    # output_file = f'log/221207_all_results.csv'
    output_file = filter_anchor_to_logpath.replace(".log", ".csv") 
    dfs.to_csv(output_file)
    print(output_file)




def get_all_tsv_or_csv_filepath(path, file_suffix):
    '''
    file_suffix: ".tsv" or ".csv"
    '''
    dataset_to_respath = defaultdict()
    dfs = []
    dataset = None
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
                # dataset_to_top_k_anchors.append(dataset_to_top_k_anchors) 
            if file_suffix in line:
                res_path = get_results_path(line)
                if dataset is None:
                    dataset = "-".join(res_path.split("/")[2:-3])
                    dataset_to_respath[dataset] = res_path
                    dataset = None
    # output_file = f'log/221207_all_results.csv'
    print(json.dumps(dataset_to_respath, indent=4))



def get_consistency_pair(path):
    dfs = []
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if '--- config --' in line and lines[i+1].startswith("{"): 
                line = lines[i+1]
                dataset= get_dataset_name(line)
                # dataset_to_top_k_anchors.append(dataset_to_top_k_anchors) 
            if '.tsv' in line:
                res_path = get_results_path(line)
                df = pd.read_csv(res_path)
                dfs.append(df) 
    dfs = pd.concat(dfs) 
    # display(dfs)
    out_path = f'{path.replace(".log", ".csv")}'
    dfs.to_csv(out_path)
    print(f"save {out_path}")


if __name__=='__main__':
    # get_bless_true_anchor_pattern_ablation_results()

    # get_bless_pred_anchor_pattern_ablation_results()

    filter_anchor_to_logpaths = {
                                # 'RAW': 'log/221208_all_datasets_raw.log', 
                                # 'WN': 'log/221208_all_datasets_wn.log',
                                # 'CPT': 'log/221208_all_datasets_cpt.log'
                                # 'WN': 'log/221221_all_datasets_wn_top_5_anchors.log'
                                'RAW': 'log/230116_all_datasets_pure_lm.log',
                                'WN': 'log/230107_all_datasets_wn_top_5_anchors.log',
                                'CPT': 'log/230116_all_datasets_cpt.log', 
                                }
    # filter_anchor_to_logpaths = {
    # #                             'WN': 'log/221209_lmdai_anchor_num_ablation.log',
    # #                             }


    get_anchor_results_all_datasets(filter_anchor_to_logpaths, output_file='log/230116_all_datasets_anchor_log.csv')


    # output_file = f'log/221208_all_datasets_log.csv'
    output_file = f'log/230116_all_datasets_log.csv'
    get_hyper_results_all_datasets(filter_anchor_to_logpaths, output_file=output_file)
    get_all_tsv_or_csv_filepath('log/230116_all_datasets_pure_lm.log', file_suffix='.csv')

    # get_all_tsv_or_csv_filepath('log/221221_all_datasets_wn_top_5_anchors.log', file_suffix='.csv')
    # get_anchor_num_ablation_results()

    # get_all_tsv_filepath('log/221208_all_datasets_wn.log')
    # get_consistency_pair("log/221213_consistency_pair_all.log")
    # get_consistency_pair("log/221213_consistency_group_all.log")

    # get_consistency_pair("log/230110_consistency_pair_all_datasets.log")
    print("Pairwise consistency Probes")
    get_consistency_pair("log/230119_consistency_pair_all_datasets.log")
    get_all_tsv_or_csv_filepath('log/230119_consistency_pair_all_datasets.log', file_suffix='.csv')
    print("-"*80)


    #get_consistency_pair("log/230110_consistency_group_all_datasets.log")
    print("Group consistency Probes")
    get_consistency_pair("log/230119_consistency_group_all_datasets.log")
    get_all_tsv_or_csv_filepath('log/230119_consistency_group_all_datasets.log', file_suffix='.csv')
    print("-"*80)
    # get_consistency_pair("log/230106_consistency_group_dfp.log")
    # get_all_tsv_filepath("log/221213_consistency_group_all.log")