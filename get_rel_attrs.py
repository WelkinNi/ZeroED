import numpy as np
import pandas as pd
import fasttext.util
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mutual_info_score
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from functools import partial

def cal_mutual_information(col1, col2):
    mask = (col1 != 'nan') & (col2 != 'nan')
    col1, col2 = col1[mask], col2[mask]
    return mutual_info_score(col1, col2)

def cal_entropy(column):
    column = column[column != 'nan']
    _, counts = np.unique(column, return_counts=True)
    probabilities = counts / len(column)
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def cal_nmi(column1, column2):
    mi = cal_mutual_information(column1, column2)
    entropy1 = cal_entropy(column1)
    entropy2 = cal_entropy(column2)
    
    if entropy1 == 0 and entropy2 == 0:
        return 1.0
    elif entropy1 == 0 or entropy2 == 0:
        return 0.0
    else:
        return 2 * mi / (entropy1 + entropy2)

def cal_all_column_nmi(df):
    columns = df.columns
    results = {}
    
    for col1, col2 in combinations(columns, 2):
        nmi = cal_nmi(df[col1], df[col2])
        results[(col1, col2)] = nmi
    
    return results


def cal_strong_res_column_nmi(nmi_results, rel_top=1, threshold=0):
    threshold = 0
    results = defaultdict(dict)
    for (col1, col2), nmi in nmi_results.items(): 
        if nmi > threshold:
            results[col1][col2] = nmi
            results[col2][col1] = nmi
    # Select top 2 col2 of col1 considering nmi
    top_results = defaultdict(dict)
    for col1, related_cols in results.items():
        sorted_cols = sorted(related_cols.items(), key=lambda item: item[1], reverse=True)
        top_results[col1] = dict(sorted_cols[:rel_top])
    return top_results
