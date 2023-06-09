#from evaluation import mean_average_precision, average_precision_at_k, precision_at_k, recall_at_k

# def get_mean_average_precision_at_k(df, relation, pred_cols, label_col, k_list, pred_col_suffix='obj_mask_'):
#     # get the avearage precision per query
#     map_at_x = []
#     for pred_col in pred_cols:
#         suffix = pred_col.replace(pred_col_suffix, "")
#         map_cur = defaultdict()
#         map_cur['mask_type'] = suffix

#         for k in k_list:
#             df[f'ap{k}_{suffix}'] = df[[label_col, pred_col]].apply(lambda x: average_precision_at_k(y_true=np.array([x[0]]) if isinstance(x[0], str) else np.array(x[0]) , y_pred= np.array(x[1]), k=k), axis=1)
#             map_cur[f'mAP@{k}'] = round(df[f'ap{k}_{suffix}'].mean(), 3)*100
#         map_at_x.append(map_cur)

#     # aggregate the average precision across k
#     df_res = pd.DataFrame(map_at_x) #, columns=['mask_type', 'mAP'])
#     df_res['relation'] = [relation]*len(df_res)
#     return df_res




import numpy as np




################### mAP #####################
#reference
# 1. https://www.kaggle.com/code/debarshichanda/understanding-mean-average-precision/notebook (main)
# 2. https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#f9ce

def rel_at_k(y_true, y_pred, k=10):
    """ Computes Relevance at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Relevance at k
    """
#     print(f"y_pred:{y_pred}")
#     print(f"y_true:{y_true}")

    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0

def precision_at_k(y_true, y_pred, k=10):
    """ Computes Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Precision at k
    """
#     intersection = np.intersect1d(y_true, y_pred[:k])
#     if len(y_pred)==0 or len(y_true)==0:
        # return 0
    intersection = set(y_true).intersection(set(y_pred[:k]))
    return len(intersection) / k


def mean_average_precision(y_true, y_pred, k=10):
    """ Computes MAP at k

    Parameters
    __________
    y_true: list
            2D list of correct recommendations (Order doesn't matter)
    y_pred: np.list
            2D list of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           MAP at k

        return np.mean([average_precision_at_k(np.array(gt), np.array(pred), k) \
                        for gt, pred in zip(y_true, y_pred)])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean([average_precision_at_k(gt, pred, k) \
                        for gt, pred in zip(y_true, y_pred)])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])



def recall_at_k(y_true, y_pred, k=10):
    """ Computes Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Precision at k
    """
    if len(y_true)==0: return 0
    intersection = set(y_true).intersection(set(y_pred[:k]))
    return len(intersection) / len(y_true) 




def test_mean_average_precision():
    gt = np.array(['a', 'b', 'c', 'd', 'e' ])
    preds1 = np.array(['a', 'x', 'y', 'd', 'e'])
    preds2 = np.array(['a', 'b', 'c', 'd', 'e'])
    preds3 = np.array(['f', 'b', 'c', 'd', 'e'])
    preds4 = np.array(['a', 'f', 'e', 'g', 'b'])
    preds5 = np.array(['a', 'f', 'c', 'g', 'b'])
    preds6 = np.array(['d', 'c', 'b', 'a', 'e'])

    # y_true = np.array([gt, gt, gt, gt, gt, gt])
    # y_pred = np.array([preds1, preds2, preds3, preds4, preds5, preds6])

    y_true = np.array([gt])
    y_pred = np.array([preds1])
    mean_ap = mean_average_precision(y_true, y_pred, k=5)

    print(mean_ap)
    print("{k}\t{p_at_k}\t{rel}\t{ap}")
    for k in [1, 2,3, 4, 5]:
            ap = average_precision_at_k(gt, preds1, k=k)
            rel = rel_at_k(gt, preds1, k=k)
            p_at_k = precision_at_k(gt, preds1, k=k)

            # y_true = gt
            # y_pred = preds1
            # ap = 0.0
            # for i in range(1, k+1):
            #         ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
            # print('before:', ap)
            # # ap = ap / min(k, len(y_true))
            # # ap = ap / min(k, len(y_true))
            # ap = ap / sum(rel_at_k(y_true, y_pred, i))
            print(f"{k}\t{p_at_k}\t{rel}\t{ap}")

            # intersection = np.intersect1d(y_true, y_pred[:k])
    #     return len(intersection) / k



def average_precision_at_k(y_true, y_pred, k=10):
    """ Computes Average Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    indicators = []
#     for i in range(1, k+1):
        # https://stackoverflow.com/questions/46374405/precision-at-k-when-fewer-than-k-documents-are-retrieved
        # Precision measured at various doc level cutoffs in the ranking.
        #If the cutoff is larger than the number of docs retrieved, then
        #it is assumed nonrelevant docs fill in the rest.  Eg, if a method
        #retrieves 15 docs of which 4 are relevant, then P20 is 0.2 (4/20).
        #Precision is a very nice user oriented measure, and a good comparison
        #number for a single topic, but it does not average well. For example,
        #P20 has very different expected characteristics if there 300
        #total relevant docs for a topic as opposed to 10.
    p_at_k = []
    ap = 0.0
    for i in range(1, min( len(y_pred)+1, k+1)): # revised range to fix the cornor case: when K < len(y_pred)
        indicator = rel_at_k(y_true, y_pred, i)
        p_at_k = precision_at_k(y_true, y_pred, i)
        ap +=  p_at_k* indicator 
        indicators.append(indicator)
    return ap/sum(indicators)  if sum(indicators)>0 else 0 #min(k, len(y_pred))


def test_map():
        y_true = ['tree', 'plant']
        y_pred_sap = ['tree', 'wood', 'plant', 'shrub', 'forest', 'woodland', 'vegetable', 'herb', 'thing', 'specie']
        y_pred_dap = ['tree', 'wood', 'forest', 'woodland', 'shrub', 'plant', 'oak', 'bush', 'specie', 'herb']

        y_true = np.array(y_true)
        y_pred_sap = np.array(y_pred_sap)
        y_pred_dap = np.array(y_pred_dap)
        # 0.833333	1.000000	['oaks', 'pines', 'cedars', 'willows', 'beeches']
        ap1 = average_precision_at_k(y_true, y_pred_sap, k=5)
        ap2 = average_precision_at_k(y_true, y_pred_dap, k=5)
        print(ap1)
        print(ap2)
