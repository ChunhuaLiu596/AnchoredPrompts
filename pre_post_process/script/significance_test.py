import pandas as pd 
from scipy import stats 



def significance_test_single_dataset(df, dataset, metrics, min_pvalue =0.05, display_res=False):
    df_ttest_rel = []
    # metrics = [['p1_def_sap', 'p1_def_dap'], ['p1_lsp_sap','p1_lsp_dap'], ['p1_def_dap', 'p1_lsp_dap']]
    for metric in metrics: 
        statistic, pvalue = stats.ttest_rel(df[metric[0]], df[metric[1]])
        reject_np = True if pvalue < min_pvalue else False 
        df_ttest_rel.append({'metric': ' .vs '.join(metric), 'dataset': dataset, 'statistic': statistic, 'pvalue': pvalue, 'significantly different': reject_np})
    df_ttest_rel = pd.DataFrame(df_ttest_rel)
    if display_res:
        for name, group in df_ttest_rel.groupby('metric'):
            display(group.sort_values('significantly different'))
    return df_ttest_rel
# np: a and b are similar 

def significance_test_multiple_datasets(dataset_to_df, metrics, min_pvalue =0.05, display_res=False):
    df_ttest_rel = []
    for dataset,df in dataset_to_df.items(): 
        # metrics = [['p1_def_sap', 'p1_def_dap'], ['p1_lsp_sap','p1_lsp_dap'], ['p1_def_dap', 'p1_lsp_dap']]
        for metric in metrics: 
            statistic, pvalue = stats.ttest_rel(df[metric[0]], df[metric[1]])
            reject_np = True if pvalue < min_pvalue else False 
            df_ttest_rel.append({'metric': ' .vs '.join(metric), 'dataset': dataset, 'statistic': statistic, 'pvalue': pvalue, 'significantly different': reject_np})
    #         print(dataset, statistic, pvalue, reject_np)

    df_ttest_rel = pd.DataFrame(df_ttest_rel)
    # display(df_ttest_rel.sort_values(by=['metric', 'significantly different']))
    if display_res: 
        for name, group in df_ttest_rel.groupby('metric'):
            display(group.sort_values('significantly different'))
    return df_ttest_rel
