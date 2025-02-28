import concurrent
import csv
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fasttext.util
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utility import default_dict_of_lists

max_workers = 128

def count_attribute_value_pairs(csv_filepath):
    co_occur_dict = defaultdict(lambda: defaultdict(int))
    attr_val_dict = defaultdict(int)
    with open(csv_filepath, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        attr_list = reader.fieldnames

        csv_file_list = []
        for row in reader:
            for key, value in row.items():
                if value == '':  # or other condition identifying NaN values
                    row[key] = 'nan'
            csv_file_list.append(row)

        def process_row(row):
            local_attr_val_dict = defaultdict(int)
            local_co_occur_dict = defaultdict(lambda: defaultdict(int))
            for i in range(len(attr_list)):
                attr_i = attr_list[i]
                if row[attr_i] == 'nan':
                    continue
                local_attr_val_dict[(attr_i, row[attr_i])] += 1
                for j in range(len(attr_list)):
                    if j == i:
                        continue
                    attr_j = attr_list[j]
                    if row[attr_j] == 'nan':
                        continue
                    co_occur_key = (attr_i, row[attr_i])
                    local_co_occur_dict[co_occur_key][(attr_j, row[attr_j])] += 1
            return local_attr_val_dict, local_co_occur_dict

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_row, csv_file_list), total=len(csv_file_list), ncols=90, desc="[Count the occurance]"))

        for local_attr_val_dict, local_co_occur_dict in results:
            for key, value in local_attr_val_dict.items():
                attr_val_dict[key] += value
            for key, inner_dict in local_co_occur_dict.items():
                for inner_key, inner_value in inner_dict.items():
                    co_occur_dict[key][inner_key] += inner_value
    return attr_val_dict, co_occur_dict


def execute_func(function_code, val, attr):
    # Define a local scope to execute our function
    local_scope = {}
    exec(function_code, globals(), local_scope)
    function_name = list(local_scope.keys())[0]
    function = local_scope[function_name]
    return function(val, attr)


funcs_with_errors = set()
def handle_func_exec(func, val, attr):
    try:
        result = execute_func(func, val, attr)
    except Exception as err:
        func_str = f"Error: {err}\n" + f"Value: {val}, Attribute: {attr}\nFunc: {func}\n"
        funcs_with_errors.add(func_str)
        return -1  # Returning -1 to indicate failure
    return 1 if result else 0  # Returning 1 for True, 0 for False


def L1_str_agg(value):
    pattern = re.compile(r'(\w+)')
    def symbol_replacer(match):
        text = match.group(0)
        if text.isalnum():
            return r'\A-{}'.format(len(text))
        else:
            return text
    symbol_string = pattern.sub(symbol_replacer, value)
    return symbol_string


def L2_str_agg(s):
    result = []
    current_char_type = None
    count = 0
    for char in s:
        if char.isdigit():
            char_type = 'D'
        elif char.isalpha():
            char_type = 'L'
        else:
            char_type = 'S'
        if char_type != current_char_type:
            if current_char_type is not None:
                result.append(f'\\{current_char_type}-{count}')
            current_char_type = char_type
            count = 1
        else:
            count += 1
    if current_char_type is not None:
        result.append(f'\\{current_char_type}[{count}]')
    return ''.join(result)


def L3_str_agg(value):
    result = []
    prev_char_type = ""
    count = 0

    def append_l2(char_type, count):
        if count > 0:
            return f"{char_type}-{count}"
        return ""

    for char in value:
        if char.isdigit():
            char_type = "\\D"
        elif char.islower():
            char_type = "\\Ll"
        elif char.isupper():
            char_type = "\\Lu"
        elif not char.isalnum():
            char_type = "\\S"
        else:
            continue

        if char_type != prev_char_type and prev_char_type != "":
            result.append(append_l2(prev_char_type, count))
            count = 1
        else:
            count += 1

        prev_char_type = char_type
    result.append(append_l2(prev_char_type, count))
    return ''.join(result)

def str_agg(s):
    return [s, L3_str_agg(s), L2_str_agg(s), L1_str_agg(s)]


def feat_gen_single(dirty_csv, co_occur_dict, cell_pat_dict, cell_pat_stats, fasttext_list, pre_funcs_feat, row, col, val, resp_path):
    feature = []
    all_attrs = list(dirty_csv.columns)
    attr = all_attrs[col]
    clean_co_occur = co_occur_dict.get((attr, val), {})
    occur_cnt_feat = []
    for i, attr_i in enumerate(all_attrs):
        if i != col:
            occur_cnt = clean_co_occur.get((attr_i, dirty_csv.iloc[row, i]), 0)
            occur_cnt_feat.append(occur_cnt)
    feature.extend(occur_cnt_feat)
    pat_list = cell_pat_dict.get((row, col), str_agg(val))
    pat_stats_feature = []
    for pat in pat_list:
        pat_stats_feature.append(cell_pat_stats.get(attr).get(pat, 0))
    feature.extend(pat_stats_feature)
    feature.extend(fasttext_list[row][col])
    feature.extend(pre_funcs_feat[row][col])
    
    feat_single_dict = defaultdict(dict)
    feat_single_dict[(row, col)] = {
        'occur_cnt_feat': occur_cnt_feat,
        'pat_stats_feat': pat_stats_feature,
        'fasttext_feat': fasttext_list[row][col],
        'pre_funcs_feat': pre_funcs_feat[row][col]
    }
    return feature, feat_single_dict


def process_row(row, dirty_csv, all_attrs, col_num):
    row_pat_dict = {}
    row_pat_stats = {}
    # for col, attr in enumerate(all_attrs):
    attr = all_attrs[col_num]
    str_agg_list = str_agg(dirty_csv.iloc[row, col_num])
    row_pat_dict[(row, col_num)] = str_agg_list
    if attr not in row_pat_stats:
        row_pat_stats[attr] = {}
    for pat in str_agg_list:
        if pat not in row_pat_stats[attr]:
            row_pat_stats[attr][pat] = 0
        row_pat_stats[attr][pat] += 1
    return row_pat_dict, row_pat_stats

def feat_gen(dataset, col_num, col_name, related_attrs_dict, pre_funcs_for_attr, resp_path):
    base_dir = '.'
    err_rate = '01'
    dirty_path = base_dir + '/data/' + dataset + '_error-' + err_rate + '.csv'
    dirty_csv = pd.read_csv(dirty_path).astype(str).fillna('nan')
    all_attrs = list(dirty_csv.columns)
    _, co_occur_dict = count_attribute_value_pairs(dirty_path)
    cell_pat_dict = {}
    cell_pat_stats = {attr: {} for attr in all_attrs}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_row, range(len(dirty_csv)), 
                                         [dirty_csv]*len(dirty_csv), 
                                         [all_attrs]*len(dirty_csv),
                                         [col_num]*len(dirty_csv)
                                         ), 
                            total=len(dirty_csv), ncols=90, 
                            desc="[Generating patterns for each cell]"))

    for row_pat_dict, row_pat_stats in results:
        cell_pat_dict.update(row_pat_dict)
        for attr, pat_dict in row_pat_stats.items():
            for pat, count in pat_dict.items():
                if pat not in cell_pat_stats[attr]:
                    cell_pat_stats[attr][pat] = 0
                cell_pat_stats[attr][pat] += count

    # get fasttext_list
    model = fasttext.load_model('./cc.en.300.bin')
    fasttext_dimension = len(all_attrs)
    fasttext.util.reduce_model(model, fasttext_dimension)
    fasttext_list = {}
    related_attrs = related_attrs_dict[col_name]
    related_attrs = [all_attrs.index(attr) for attr in related_attrs]
    pre_funcs_feat = {}
    for row in tqdm(range(len(dirty_csv)), ncols=90, desc="[Generating fasttext_vector for each cell]"):
        if row not in fasttext_list:
            fasttext_list[row] = {}
        if row not in pre_funcs_feat:
            pre_funcs_feat[row] = {}
        fasttext_vector = model.get_word_vector(str(dirty_csv.iloc[row, col_num]))
        fasttext_list[row][col_num] = fasttext_vector
        if len(pre_funcs_for_attr) > 0:
            row_val = [dirty_csv.loc[row, [col_name]+[all_attrs[i] for i in related_attrs]].to_dict()]
            pre_funcs_feat_temp = np.array([handle_func_exec(func, row_val[0], attr) for func in pre_funcs_for_attr[attr]['clean']])
            pre_funcs_feat[row][col_num] = pre_funcs_feat_temp
        else:
            pre_funcs_feat[row][col_num] = np.array([])
        for attr_rel_idx in related_attrs:
            rel_attr_vec = model.get_word_vector(str(dirty_csv.iloc[row, attr_rel_idx]))
            fasttext_list[row][col_num] = np.append(fasttext_list[row][col_num], rel_attr_vec)
            attr_rel = all_attrs[attr_rel_idx]
            attr_rel_related_attrs = [all_attrs.index(attr) for attr in related_attrs_dict[attr_rel]]
            if len(pre_funcs_for_attr) > 0:
                row_val = [dirty_csv.loc[row, [attr_rel]+[all_attrs[i] for i in attr_rel_related_attrs]].to_dict()]
                pre_funcs_feat_temp = np.array([handle_func_exec(func, row_val[0], attr) for func in pre_funcs_for_attr[attr_rel]['clean']])
                pre_funcs_feat[row][col_num] = np.append(pre_funcs_feat[row][col_num], pre_funcs_feat_temp)


    feature_list = []
    feature_all_dict = defaultdict(default_dict_of_lists)
    for row in tqdm(range(len(dirty_csv)), ncols=90, desc="Formulating Features: "):
        feature, feat_single_dict_tmp = feat_gen_single(dirty_csv, co_occur_dict, cell_pat_dict, cell_pat_stats, fasttext_list, pre_funcs_feat, row, col_num,
                                    dirty_csv.iloc[row, col_num], resp_path)
        feature_list.append(feature)
        feature_all_dict.update(feat_single_dict_tmp)
    scaler = MinMaxScaler()
    feature_list = scaler.fit_transform(np.array(feature_list))
    # feature_dict[col] = np.array(feature_list)  
    return np.array(feature_list), feature_all_dict


def cluster(dataset, cluster_method, n_method, col_num, related_attrs_dict, pre_funcs_for_attr, resp_path):
    base_dir = '.'
    err_rate = '01'
    dirty_path = base_dir + '/data/' + dataset + '_error-' + err_rate + '.csv'
    dirty_csv = pd.read_csv(dirty_path).astype(str).fillna('nan')
    all_attrs = list(dirty_csv.columns)
    col_name = all_attrs[col_num]
    related_attrs = list(related_attrs_dict[col_name])
    feat, feature_all_dict = feat_gen(dataset, col_num, col_name, related_attrs_dict, pre_funcs_for_attr, resp_path)

    if isinstance(n_method, str) and '%' in n_method:
        n_method = float(n_method.strip('%')) / 100
    n_clusters = int(dirty_csv.shape[0] * n_method)  # label 5%
    if n_clusters > len(dirty_csv.loc[:, [col_name] + related_attrs].drop_duplicates()):
        n_clusters = len(dirty_csv.loc[:, [col_name] + related_attrs].drop_duplicates())

    if cluster_method == 'KMeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(feat)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(cluster_centers, feat)
        clusters = [np.where(labels == i)[0].tolist() for i in range(n_clusters)]
    elif cluster_method == 'RANDOM':
        np.random.seed(0)
        closest = np.random.choice(len(feat), size=n_clusters, replace=False)
        distances = pairwise_distances(feat, feat[closest])
        labels = np.argmin(distances, axis=1)
        clusters = [np.where(labels == i)[0].tolist() for i in range(n_clusters)]
    elif cluster_method == 'DBSCAN':
        dbscan = DBSCAN()
        labels = dbscan.fit_predict(feat)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        closest = []
        clusters = []
        
        for i in range(n_clusters):
            # Get points in current cluster
            cluster_points = feat[labels == i]
            cluster_indices = np.where(labels == i)[0]
            clusters.append(cluster_indices.tolist())
            
            # Find point closest to cluster mean
            cluster_mean = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - cluster_mean, axis=1)
            closest_idx_in_cluster = cluster_indices[np.argmin(distances)]
            closest.append(closest_idx_in_cluster)
            
        # Handle noise points if any
        if -1 in labels:
            noise_points = np.where(labels == -1)[0]
            # Add each noise point as its own cluster
            for noise_idx in noise_points:
                closest.append(noise_idx)
                clusters.append([noise_idx])
    else:
        print('AgglomerativeClustering...')
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg_clustering.fit_predict(feat)
        def get_closest_points_per_cluster(features, labels, n_clusters):
            closest_points = []
            for cluster_num in range(n_clusters):
                cluster_points = features[labels == cluster_num]
                distances = pairwise_distances(cluster_points)
                closest_idx_in_cluster = np.argmin(np.sum(distances, axis=1))
                original_idx = np.where(labels == cluster_num)[0][closest_idx_in_cluster]
                closest_points.append(original_idx)
            return closest_points

        closest = get_closest_points_per_cluster(np.array(feat), labels, n_clusters)
        clusters = [np.where(labels == i)[0].tolist() for i in range(n_clusters)]

    print("FINISH CLUSTERING")
    val_feat_dict = {}
    for idx, feat_val in enumerate(feat):
        key = dirty_csv.iloc[idx, col_num]
        if key in val_feat_dict:
            val_feat_dict[key].append(feat_val)
        else:
            val_feat_dict[key] = [feat_val]
    return col_num, closest, clusters, val_feat_dict, feature_all_dict
