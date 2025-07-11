import argparse
import ast
import json
import multiprocessing
import os
import pickle
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import fasttext.util
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import yaml
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from distri_analys import LLMDataDistrAnalyzer
from feature import cluster
from get_rel_attrs import (cal_all_column_nmi, cal_strong_res_column_nmi)
from measure import measure_detect
from prompt_gen import (create_err_gen_inst_prompt, err_clean_func_prompt,
                        error_check_prompt, guide_gen_prompt, pre_func_prompt)
from utility import (Logger, Timer, copy_file, copy_read_files_in_dir,
                     default_dict_of_lists, get_ans_from_llm, query_base,
                     rag_query, split_list_to_sublists, get_read_paths)


def subtask_det_initial(val_list, attr_name):
    str_list = [str(a_val) for a_val in val_list]
    vals_str = '\n'.join(str_list)
    prompt = error_check_prompt(vals_str, attr_name)
    if GUIDE_USE:
        response = rag_query(prompt, guide_content[attr_name])
    else:
        response = query_base(prompt)
    error_check_prompt_file = open(os.path.join(error_checking_res_directory, f'prompt_error_checking_{attr_name}.txt'), 'w', encoding='utf-8')
    error_check_prompt_file.write(prompt + '\n\n')
    error_check_prompt_file.close()
    return response 


def extract_func(text_content):
    try:
        code_blocks = re.findall(r'```(.*?)```', text_content, re.DOTALL)
    except re.error as e:
        print(f"Regex error: {e}")
        return [], []
    clean_func_list = []
    dirty_func_list = []
    for code_block in code_blocks:
        functions = re.findall(r'def \w+\(.*?\):\n(?:[ \t]*\n)*(?: .*\n)+', code_block)
        for function in functions:
            try:
                function_name = re.findall(r'def (\w+)', function)[0]
            except IndexError:
                print("Function name not found in the function definition.")
                continue
            if 'is_clean' in function_name:
                clean_func_list.append(function)
            elif 'is_dirty' in function_name:
                dirty_func_list.append(function)
    return clean_func_list, dirty_func_list


def extract_err_info(text, attr):
    information = []
    attr_name = attr
    lines = text.split('\n')
    for line in lines:
        err_info = []
        match = re.search(r'\[(.*?)\]', line)
        if match:
            try:
                data = match.group().replace("Reason: '", "'Reason: ")
                parsed_data = ast.literal_eval(data)
                err_info.append(attr_name)
                for i, content in enumerate(parsed_data):
                    if i != len(parsed_data) - 1 and i != 0:
                        err_info.append(str(content))
                    elif i == len(parsed_data) - 1:
                        err_info.append(content)
            except Exception as e:
                print("\n\nWhen processing error_err_info():" + line + "--" + attr)
                print(e)
        information.append(err_info)
    information = list(filter(None, information))
    return information


def gen_dirty_funcs(attr, clean_info, errs_info):
    dirty_str = "\n"
    clean_info = '\n'.join([str(i) for i in clean_info])
    try:
        dirty_str = dirty_str + '\n'.join([str(i) for i in errs_info])
    except Exception as e:
        print(f"Error: {e}\n When handling {errs_info}\n")
        dirty_str = dirty_str + str(errs_info)
        dirty_str = dirty_str + "\n"
    func_gen_prompt = err_clean_func_prompt(attr, clean_info, dirty_str)
    llm_gen_func = get_ans_from_llm(func_gen_prompt, api_use=API_USE)
    temp_clean_flist, dirty_flist = extract_func(llm_gen_func)
    return temp_clean_flist, dirty_flist, func_gen_prompt, llm_gen_func


def subtask_func_gen(attr_name, err_list, func_file_num, right_values_list):
    temp_clean_flist, dirty_flist, func_gen_prompt, llm_gen_func = gen_dirty_funcs(attr_name, right_values_list, err_list)
    funcs_for_attr = defaultdict(default_dict_of_lists)
    funcs_for_attr[attr_name]['clean'].extend(list(set(temp_clean_flist)))
    funcs_for_attr[attr_name]['dirty'].extend(list(set(dirty_flist)))
    with open(os.path.join(funcs_directory, f"prompt_funcs_zgen_{attr_name}{func_file_num}.txt"), 'w', encoding='utf-8') as prom_file:
        prom_file.write(func_gen_prompt)
    with open(os.path.join(funcs_directory, f"funcs_zgen_{attr_name}{func_file_num}.txt"), 'w', encoding='utf-8') as func_file:
        func_file.write("\n".join(list(set(temp_clean_flist))))
    return attr_name, funcs_for_attr


def process_gen_err_data(ERR_GEN_USE, ERR_GEN_READ, read_err_gen_path, err_gen_directory, dirty_csv, all_attrs, related_attrs_dict, center_index_value_label_dict, err_gen_dict, logger):
    if ERR_GEN_USE and ERR_GEN_READ:
        copy_read_files_in_dir(err_gen_directory, read_err_gen_path)
        for attr in all_attrs:
            if os.path.exists(os.path.join(err_gen_directory, f'err_gen_res_{attr}.txt')):
                with open(os.path.join(err_gen_directory, f'err_gen_res_{attr}.txt'), 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            err_dict = json.loads(line.strip())
                            err_gen_dict[attr]['dirty'].append(err_dict)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON for attribute {attr}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error for attribute {attr}: {e}")
                            continue

    elif ERR_GEN_USE and not ERR_GEN_READ:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_gen_err_data, attr, dirty_csv, center_index_value_label_dict, related_attrs_dict, err_gen_dict) for attr in all_attrs]
            outputs = [result.result() for result in results]


def task_gen_err_data(attr, dirty_csv, center_index_value_label_dict, related_attrs_dict, err_gen_dict):
    related_attrs = list(related_attrs_dict[attr]) 
    err_gen_prompt_file = open(os.path.join(err_gen_directory, f"prompt_ans_error_gen_{attr}.txt"), 'w', encoding='utf-8')
    err_gen_file = open(os.path.join(err_gen_directory, f"error_gen_{attr}.txt"), 'w', encoding='utf-8')
    wrong_values = []
    right_values = []
    used_idx_list = {}
    for idx, _, label in center_index_value_label_dict[attr]:
        if label == 1:  
            wrong_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1
        elif label == 0:  
            right_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1 
    max_vals = 20
    if len(wrong_values) > max_vals:  
        wrong_values_tmp = wrong_values[:max_vals]
    else:
        wrong_values_tmp = wrong_values
    if len(right_values) > max_vals:  
        right_values_tmp = right_values[:max_vals]
    else:
        right_values_tmp = right_values
    err_gen_prompt = create_err_gen_inst_prompt(right_values_tmp, wrong_values_tmp, attr, num_errors=(len(right_values)))
    err_gen_ans = get_ans_from_llm(err_gen_prompt, api_use=API_USE)
    err_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + err_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + err_gen_ans + '\n\n\n\n\n\n')
    err_gen_file.write(err_gen_ans)
    err_gen_prompt_file.close()
    err_gen_file.close()
    err_info = extract_err_info(err_gen_ans, attr)
    filtered_error = []
    filtered_error.extend(wrong_values)
    for err in err_info:
        try:
            if err[0] in all_attrs and str(err[-1]).strip() not in right_values and str(
                    err[-1]).strip() not in wrong_values:
                err_gen_dict[attr]['dirty'].append(err[3])
                filtered_error.extend([f"{err[3]}, {err[2]}"])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {err}\n Processing attribute: {attr}\n")
    err_gen_res_file = open(os.path.join(err_gen_directory, f"err_gen_res_{attr}.txt"), 'w', encoding='utf-8')
    for err_dict in err_gen_dict[attr]['dirty']:
        json.dump(err_dict, err_gen_res_file)
        err_gen_res_file.write('\n')
    err_gen_res_file.close()


def gen_err_funcs(attr, err_gen_dict):  
    related_attrs = list(related_attrs_dict[attr])  
    err_gen_prompt_file = open(os.path.join(funcs_directory, f"prompt_ans_error_gen_{attr}.txt"), 'a', encoding='utf-8')
    err_gen_file = open(os.path.join(funcs_directory, f"error_gen_{attr}.txt"), 'w', encoding='utf-8')
    wrong_values = []
    right_values = []
    used_idx_list = {}
    for idx, _, label in center_index_value_label_dict[attr]:
        if label == 1:  # wrong 
            wrong_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1
        elif label == 0:  # right
            right_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1
    filtered_error = [str(vals) for vals in wrong_values]
    if len(filtered_error) == 0:
        return False
    max_err_num = 20
    if max_err_num > (int(len(filtered_error)/2)+1):
        max_err_num = int(len(filtered_error)/2)+1
    filtered_error_sublists = split_list_to_sublists(filtered_error, max_err_num)
    if len(filtered_error_sublists) > 2:
       filtered_error_sublists = filtered_error_sublists[:2]
    funcs_for_attr = {}
    max_err_num = min(max_err_num, len(right_values))
    with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as a_executor:
        a_results = [a_executor.submit(subtask_func_gen, attr, filtered_error_sublists[temp_idx], temp_idx, random.sample(right_values, max_err_num)) for temp_idx in range(len(filtered_error_sublists))]
        for a_future in as_completed(a_results):  
            attr_name, funcs_for_attr_gen = a_future.result()
            funcs_for_attr.update(funcs_for_attr_gen)
    func_extract_file = open(os.path.join(funcs_directory, f"funcs_zgen_{attr}.txt"), 'w', encoding='utf-8')
    temp_clean_flist_str = "\n".join(funcs_for_attr[attr]['clean'])
    func_extract_file.write(temp_clean_flist_str)
    func_extract_file.close()
    return funcs_for_attr


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


def task_guide_gen(attr_name, uni_vals, distri_analy_content, prompt_content, guide_content):
    attr_analy_content = distri_analy_content[attr_name]
    prompt = guide_gen_prompt(attr_name, dataset, uni_vals, dirty_csv, attr_analy_content)
    while True:
        try:
            res_content = get_ans_from_llm(prompt, api_use=API_USE)
            break
        except Exception as eee:
            print(eee, f'while guide_gen {attr_name}')
    prompt_content[attr_name] = prompt
    guide_content[attr_name] = res_content
    with open(os.path.join(guide_directory, f'prompt_{attr_name}.txt'), 'w', encoding='utf-8') as file:
        file.write(prompt)
    with open(os.path.join(guide_directory, f'guide_{attr_name}.txt'), 'w', encoding='utf-8') as file:
        file.write(res_content)


def task_func_gen(attr_name, err_gen_dict):
    funcs_for_attr = gen_err_funcs(attr_name, err_gen_dict)
    if funcs_for_attr:
        para_file.write(f"{attr_name} func_num:{len(funcs_for_attr[attr_name]['clean'])}\n")
        return funcs_for_attr
    else:
        return {attr_name: {'clean': [], 'dirty': []}}


def task_det_initial(attr_name, error_checking_res_directory):
    error_checking_file = open(os.path.join(error_checking_res_directory, f'error_checking_{attr_name}.txt'), 'w', encoding='utf-8')
    related_attrs = list(related_attrs_dict[attr_name])
    center_idx = cluster_index_dict[attr_name][0]    
    df_center_idx = ["{" + ",".join(f'"{col}":"{dirty_csv.loc[idx, col]}"' for col in [attr_name] + related_attrs) + "}" for idx in center_idx]
    split_center_values = split_list_to_sublists(df_center_idx, err_check_val_num_per_query)
    error_response = ''
    with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as a_executor:
        a_results = [a_executor.submit(subtask_det_initial, sub_list_values, attr_name) for sub_list_values in split_center_values]
        for a_future in as_completed(a_results):
            error_response += a_future.result() + '\n'
    error_checking_file.write(error_response)
    error_checking_file.close()


def normalize_string(s):
    return str(s.replace(" \\", "\\")
               .replace("\\\\", "\\")
               .replace("\\", "")
               .replace(", ", ",")
               .replace(": ", ":")
               .replace("'", '"'))
    
    
def process_attr_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, err_gen_dict, funcs_for_attr, feature_all_dict, resp_path):
    fasttext_model = fasttext.load_model('./cc.en.300.bin')
    fasttext_dimension = len(dirty_csv.columns) 
    fasttext.util.reduce_model(fasttext_model, fasttext_dimension)  
    feature_list, label_list = prep_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, err_gen_dict, funcs_for_attr, fasttext_model, feature_all_dict, resp_path)
    return attr, feature_list, label_list


def task_distri_analys(attr, analyzer, dist_dir):
    output_file = os.path.join(dist_dir, f'ori_distri_analys_{attr}.txt')
    distr_prompt_file = os.path.join(dist_dir, f'prompt_distri_analys_{attr}.txt')
    llm_prompt, examples = analyzer.generate_llm_prompt(attr)
    llm_response = get_ans_from_llm(llm_prompt, api_use=API_USE)
    analyze_content = analyzer.analyze_data(attr, llm_response, output_file)
    with open(distr_prompt_file, 'w', encoding='utf-8') as f:
        f.write(llm_prompt)
    with open(os.path.join(dist_dir, f'distri_analys_{attr}.txt'), 'w', encoding='utf-8') as f:
        f.write(analyze_content)
    return attr, analyze_content


def single_val_feat(val, fasttext_m, funcs_for_attr, attr, idx, all_attrs, feature_all_dict, resp_path):
    feature = [handle_func_exec(func, val, attr) for func in funcs_for_attr[attr]['clean']]
    if idx == -1:
        for a_val in val.values():
            feature.extend(fasttext_m.get_word_vector(str(a_val)))
        return feature
    else:
        if feature_all_dict is not None:
            fasttext_feat = feature_all_dict[(idx, all_attrs.index(attr))].get('fasttext_feat', [])
            if len(fasttext_feat) == 0 or len(fasttext_feat) < len(all_attrs):
                fasttext_feat = []
                fasttext_m = fasttext.load_model('./cc.en.300.bin')
                fasttext_dimension = len(all_attrs)  
                fasttext.util.reduce_model(fasttext_m, fasttext_dimension)  
                for a_val in val.values():
                    fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            feature.extend(fasttext_feat)
        else:
            fasttext_m = fasttext.load_model('./cc.en.300.bin')
            fasttext_dimension = len(all_attrs)  
            fasttext.util.reduce_model(fasttext_m, fasttext_dimension)  
            fasttext_feat = []
            for a_val in val.values():
                fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            feature.extend(fasttext_feat)
        return idx, feature


def prep_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, err_gen_dict, funcs_for_attr, fasttext_m, feature_all_dict, resp_path):
    feature_list = []
    label_list = []
    related_attrs = list(related_attrs_dict[attr])
    right_values = [(idx, dirty_csv.loc[idx, [attr]+related_attrs].to_dict()) for idx, a in det_right_list if a == attr]
    wrong_values = [(idx, dirty_csv.loc[idx, [attr]+related_attrs].to_dict()) for idx, a in det_wrong_list if a == attr]
    
    for idx, val in tqdm(right_values, ncols=120, desc=f"Processing {attr} right values"):
        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(0)
    for idx, val in tqdm(wrong_values, ncols=120, desc=f"Processing {attr} wrong values"):
        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(1)
    for val in tqdm(err_gen_dict[attr]['dirty'], ncols=120, desc=f"Processing {attr} generated errors"):
        if len(err_gen_dict[attr]['dirty']) == 0 or len(err_gen_dict[attr]['dirty'][0].keys()) < len([attr]+related_attrs):
            continue
        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(1)
    return feature_list, label_list


def make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path):
    if attr not in model_col.keys():
        return []    
    model = model_col[attr]
    test_feat_list = []
    related_attrs = list(related_attrs_dict[attr])
    columns = list(dirty_csv.columns)
    
    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = []
        for idx in range(len(dirty_csv)):
            cell_val = dirty_csv.loc[idx, [attr]+related_attrs].to_dict()
            future = executor.submit(single_val_feat, 
                cell_val, None, funcs_for_attr, attr, idx, columns, feature_all_dict, resp_path)
            futures.append(future)
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            
    sorted_results = sorted([(r[0], r[1]) for r in results])
    test_feat_list = [feat for idx, feat in sorted_results]

    test_feat_np = np.array(test_feat_list)
    pred_prob_list = model.predict(test_feat_np)
    wrong_cells = []
    for idx, cell_val in dirty_csv.iloc[:, col].items():
        pred_prob = pred_prob_list[idx]
        if pred_prob == 1:
            wrong_cells.append((idx, attr))
    return wrong_cells


def train_model(attr, feature_list, label_list, num_epochs):
    if feature_list is None:
        return attr, None, 'mlp', 'optimizer', "None", 500
    elif len(feature_list) == 0 or len(feature_list[0]) == 0:
        return attr, None, 'mlp', 'optimizer', "None", 500
    
    feat_np = np.array(feature_list)
    label_np = np.array(label_list)
    
    input_dim = feat_np.shape[1]  
    
    model = MLPClassifier(
        hidden_layer_sizes=(2 * input_dim, input_dim),  
        activation='relu',           
        solver='adam',               
        max_iter=num_epochs,         
        random_state=42,
        n_iter_no_change=10,         
        verbose=True                 
    )
    
    model.fit(feat_np, label_np)
    return attr, model, 'mlp', 'optimizer', model, num_epochs


def process_related_attr(RELATED_ATTRS, RELATED_ATTRS_READ, REL_TOP, read_path, resp_path, clean_csv, dirty_csv, all_attrs):
    related_attrs_dict = {}
    gt_wrong_dict = {}
    if RELATED_ATTRS and RELATED_ATTRS_READ:
        with open(os.path.join(read_path, 'related_attrs_dict.json'), 'r', encoding='utf-8') as f:
            related_attrs_dict = json.load(f)
        copy_file(read_path, resp_path, 'related_attrs_dict.json')
    elif RELATED_ATTRS and not RELATED_ATTRS_READ:
        nmi_results = cal_all_column_nmi(dirty_csv)
        related_attrs_dict = cal_strong_res_column_nmi(nmi_results, rel_top=REL_TOP)
        with open(os.path.join(resp_path, 'related_attrs_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(related_attrs_dict, f, ensure_ascii=False, indent=4)
    elif not RELATED_ATTRS:
        for attr in all_attrs:
            related_attrs_dict[attr] = []
        with open(os.path.join(resp_path, 'related_attrs_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(related_attrs_dict, f, ensure_ascii=False, indent=4)

    for attr in all_attrs:
        related_attrs = list(related_attrs_dict[attr])
        if attr not in gt_wrong_dict:
            gt_wrong_dict[attr] = set()
        for i in range(len(dirty_csv)):
            if str(dirty_csv.loc[i, attr]) != str(clean_csv.loc[i, attr]) or str(clean_csv.loc[i, attr]) == 'nan':
                wrong_tuple = str(dirty_csv.loc[i, [attr] + related_attrs].to_dict())
                gt_wrong_dict[attr].add(wrong_tuple)
    return related_attrs_dict, gt_wrong_dict


def process_cluster(n_method, CLUSTER_READ, dataset, read_path, resp_path, dirty_csv, all_attrs, related_attrs_dict, pre_funcs_for_attr):
    cluster_index_dict = {}  
    center_value_dict = {}  
    feature_all_dict = defaultdict(default_dict_of_lists)
    if not CLUSTER_READ:
        with multiprocessing.Pool(len(all_attrs)) as pool:
            results = [pool.apply_async(cluster, args=(dataset, 'RANDOM', n_method, col, related_attrs_dict, pre_funcs_for_attr, resp_path)) for col in range(len(all_attrs))]
            for result in results:
                col, center_list, cluster_list, val_feat_dict, feature_dict_attr = result.get()
                cluster_list.insert(0, center_list)
                cluster_index_dict[all_attrs[col]] = cluster_list
                feature_all_dict.update(feature_dict_attr)
        for key, value in cluster_index_dict.items():
            temp_list = []
            related_attrs = list(related_attrs_dict[key])
            for ind in value[0]:
                temp_list.append(dirty_csv.loc[ind, [key] + related_attrs].to_dict())
            center_value_dict[key] = temp_list
        with open(os.path.join(resp_path, 'center_value_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(center_value_dict, f, ensure_ascii=False, indent=4)
        serializable_cluster_index_dict = {
                    attr: [[int(idx) for idx in cluster] for cluster in clusters]
                    for attr, clusters in cluster_index_dict.items()
                }
        with open(os.path.join(resp_path, 'cluster_index_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable_cluster_index_dict, f, ensure_ascii=False, indent=4)
        with open(os.path.join(resp_path, 'cluster_feat_dict.pkl'), 'wb') as f:
            pickle.dump(feature_all_dict, f)
    elif CLUSTER_READ:
        with open(os.path.join(read_path, 'center_value_dict.json'), 'r', encoding='utf-8') as src_file:
            center_value_dict = json.load(src_file)
        with open(os.path.join(resp_path, 'center_value_dict.json'), 'w', encoding='utf-8') as dst_file:
            json.dump(center_value_dict, dst_file, ensure_ascii=False, indent=4)
                    
        with open(os.path.join(read_path, 'cluster_index_dict.json'), 'r', encoding='utf-8') as f:
            cluster_index_dict = json.load(f)
        with open(os.path.join(resp_path, 'cluster_index_dict.json'), 'w', encoding='utf-8') as dst_file:
           json.dump(cluster_index_dict, dst_file, ensure_ascii=False, indent=4)
        cluster_index_dict = {
                    attr: [[int(idx) for idx in cluster] for cluster in clusters]
                    for attr, clusters in cluster_index_dict.items()
                }
        if os.path.exists(os.path.join(read_path, 'cluster_feat_dict.pkl')):
            copy_file(read_path, resp_path, 'cluster_feat_dict.pkl')
            with open(os.path.join(read_path, 'cluster_feat_dict.pkl'), 'rb') as f:
                feature_all_dict = pickle.load(f)
    return cluster_index_dict, center_value_dict, feature_all_dict


def process_distri_analys(DISTRI_ANALYSIS, DISTRI_ANALYSIS_READ, read_path, resp_path, dirty_csv, all_attrs):
    dist_dir = os.path.join(resp_path, 'distri_analys')
    os.makedirs(dist_dir, exist_ok=True)
    distri_analy_content = {}
    if DISTRI_ANALYSIS and DISTRI_ANALYSIS_READ:
        distri_analy_read_dir = os.path.join(read_path, 'distri_analys')
        copy_read_files_in_dir(dist_dir, distri_analy_read_dir)
        for attr in all_attrs:
            dist_dir_file = os.path.join(dist_dir, f'distri_analys_{attr}.txt')
            with open(dist_dir_file, 'r', encoding='utf-8') as file:
                distri_analy_content[attr] = file.read()
    elif DISTRI_ANALYSIS and not DISTRI_ANALYSIS_READ:
        analyzer = LLMDataDistrAnalyzer(dirty_csv)
        with multiprocessing.Pool(len(all_attrs)) as pool:
            results = [pool.apply_async(task_distri_analys, args=(attr, analyzer, dist_dir)) for attr in all_attrs]
            for result in results:
                attr, content = result.get()
                distri_analy_content[attr] = content
    else:
        for attr in all_attrs:
            distri_analy_content[attr] = ''

    return distri_analy_content


def process_guidlines(GUIDE_USE, GUIDE_READ, dataset, read_path, read_guide_path, resp_path, dirty_csv, all_attrs, guide_directory, cluster_index_dict, distri_analy_content):
    guide_content = {}
    prompt_content = {}
    if GUIDE_USE and GUIDE_READ:
        copy_read_files_in_dir(guide_directory, read_guide_path)
        for attr in all_attrs:
            attr_analy_content = distri_analy_content[attr]
            file_path = os.path.join(read_guide_path, f'{dataset}_{attr}_ref_knowledge.txt')
            prompt = guide_gen_prompt(attr, dataset, cluster_index_dict[attr][0], dirty_csv, attr_analy_content)
            prompt_content[attr] = prompt
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    guide_content[attr] = file.read()
            elif os.path.exists(os.path.join(read_path, f'guide/guide_{attr}.txt')):
                with open(os.path.join(read_path, f'guide/guide_{attr}.txt'), 'r', encoding='utf-8') as file:
                    guide_content[attr] = file.read()
            else:
                continue
    elif GUIDE_USE:
        guide_content = {}
        prompt_content = {}
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_guide_gen, attr, cluster_index_dict[attr][0], distri_analy_content, prompt_content, guide_content) for attr in all_attrs]
            for future in as_completed(results):  
                result = future.result()
    return guide_content


def process_error_checking(ERROR_CHECKING_READ, read_error_checking_path, all_attrs, error_checking_res_directory):
    if ERROR_CHECKING_READ:
        copy_read_files_in_dir(error_checking_res_directory, read_error_checking_path)

    else:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_det_initial, attr, error_checking_res_directory) for attr in all_attrs]
            outputs = [result.result() for result in as_completed(results)]


def measure_llm_label(resp_path, clean_csv, all_attrs, related_attrs_dict, gt_wrong_dict, center_index_value_label_dict):
    llm_label_eval_file = open(os.path.join(resp_path, 'llm_label_results.txt'), 'w', encoding='utf-8')
    overall_wrong_label_num = 0
    overall_lwrong_num = 0
    overall_lright_num = 0
    overall_miss_wrong_num = 0
    for attr in all_attrs:
        llm_label_eval_file.write('\n' + '*'*30 + attr + '*'*30 + '\n\n')
        wrongly_llm_det = []
        missing_llm_det = []
        llm_wrong_label_num = 0
        llm_lwrong_num = 0
        llm_lright_num = 0
        llm_miss_wrong_num = 0
        for idx, llm_lstr, llm_label in center_index_value_label_dict[attr]:
            if llm_label == 1:
                llm_lwrong_num += 1
                overall_lwrong_num += 1
                if str(llm_lstr) not in gt_wrong_dict[attr]:
                    llm_wrong_label_num += 1
                    overall_wrong_label_num += 1
                    wrongly_llm_det.append((idx, str(llm_lstr)))
            elif llm_label == 0:
                llm_lright_num += 1
                overall_lright_num += 1
                if str(llm_lstr) in gt_wrong_dict[attr]:
                    llm_miss_wrong_num += 1
                    overall_miss_wrong_num += 1
                    missing_llm_det.append((idx, str(llm_lstr)))
        llm_label_eval_file.write(f"Wrong data labeling accuracy: {1-llm_wrong_label_num/(llm_lwrong_num+1e-6)} ({llm_lwrong_num-llm_wrong_label_num}/{llm_lwrong_num})\n")
        llm_label_eval_file.write(f"Right data labeling accuracy: {1-llm_miss_wrong_num/(llm_lright_num+1e-6)} ({llm_lright_num-llm_miss_wrong_num}/{llm_lright_num})\n\n")
        llm_label_eval_file.write('-'*30 + "Wrongly Detected Values" + '-'*30 + '\n\n')
        for idx, llm_lstr in wrongly_llm_det:
            llm_label_eval_file.write('\nDirty: ' + llm_lstr)
            llm_label_eval_file.write('\nClean: ' + str(clean_csv.loc[int(idx), [attr] + list(related_attrs_dict[attr])].to_dict()) + '\n')
                
        llm_label_eval_file.write('\n' + '-'*30 + "Missing Erroneous Values" + '-'*30 + '\n\n')
        for idx, llm_lstr in missing_llm_det:
            llm_label_eval_file.write('\nDirty: ' + llm_lstr)
            llm_label_eval_file.write('\nClean: ' + str(clean_csv.loc[int(idx), [attr] + list(related_attrs_dict[attr])].to_dict()) + '\n\n')

    llm_label_eval_file.write('*'*30 + "Overall Evaluation" + '*'*30 + '\n\n')
    llm_label_eval_file.write(f"Overall Wrong data labeling accuracy: {1-overall_wrong_label_num/(overall_lwrong_num+1e-6)} ({overall_lwrong_num-overall_wrong_label_num}/{overall_lwrong_num})\n")
    llm_label_eval_file.write(f"Overall Right data labeling accuracy: {1-overall_miss_wrong_num/(overall_lright_num)+1e-6} ({overall_lright_num-overall_miss_wrong_num}/{overall_lright_num})\n\n")
    llm_label_eval_file.close()
    return 'Done'


def err_pat_in_text_attr(attr):
    pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*true'
    return pattern


def right_pat_in_text_attr(attr):
    pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*false'
    return pattern


def extract_llm_label_res(all_attrs, error_checking_res_directory, cluster_index_dict, center_value_dict):
    all_extracted_values = defaultdict(list)
    for attr in all_attrs:
        content = ""
        with open(os.path.join(error_checking_res_directory, f'error_checking_{attr}.txt'), 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('\\+', '').replace('\\n', '\n')
        wrong_pattern = err_pat_in_text_attr(attr)
        matches = re.finditer(wrong_pattern, content)
        all_extracted_values[attr].extend([match.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'") for match in matches])
        all_extracted_values[attr] = [normalize_string(match).replace('"{', '{', 1)[:-1] for match in all_extracted_values[attr]]
        all_extracted_values[attr] = list(set(all_extracted_values[attr])) 
        # # try handling conflictions
        right_pattern = right_pat_in_text_attr(attr)
        right_matches = re.finditer(right_pattern, content)
        right_matches = [match.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'") for match in right_matches]
        right_matches = [normalize_string(match).replace('"{', '{', 1)[:-1] for match in right_matches]
        all_extracted_values[attr] = [extr_vals for extr_vals in all_extracted_values[attr] if extr_vals not in right_matches]
    for key, value in center_value_dict.items():
        temp_list = []
        for i in range(len(value)): 
            if normalize_string(str(value[i])) in all_extracted_values[key]:
                temp_list.append((cluster_index_dict[key][0][i], value[i], 1))
            else:
                temp_list.append((cluster_index_dict[key][0][i], value[i], 0))
        center_index_value_label_dict[key] = temp_list
    return center_index_value_label_dict


def label_prop(resp_path, dirty_path, clean_path, cluster_index_dict, center_index_value_label_dict):
    det_wrong_list = []    
    det_right_list = []    
    for key, value in cluster_index_dict.items():
        for center_index in value[0]:
            temp_cluster = []
            for i in range(1, len(value)):
                if center_index in value[i]:
                    temp_cluster.extend(value[i])
                    break
            temp_label = -1
            for triple_set in center_index_value_label_dict[key]:
                if triple_set[0] == center_index:
                    temp_label = triple_set[2]
                    break
            if temp_label == 0:
                for index in temp_cluster:
                    det_right_list.append((index, key))
            elif temp_label == 1:
                for index in temp_cluster:
                    det_wrong_list.append((index, key))
    return det_wrong_list, det_right_list


def process_gen_err_funcs(FUNC_USE, FUNC_READ, read_path, read_func_path, read_error_path, resp_path, funcs_directory, dirty_csv, all_attrs, para_file, related_attrs_dict, center_index_value_label_dict, det_wrong_list, det_right_list):
    err_gen_dict = defaultdict(default_dict_of_lists)
    funcs_for_attr = defaultdict(default_dict_of_lists)
    if FUNC_USE and FUNC_READ:
        for attr in all_attrs:
            file_names = os.listdir(read_func_path)
            for file_name in sorted(file_names):
                if file_name == f'funcs_zgen_{attr}.txt':
                    with open(os.path.join(read_func_path, file_name), 'r', encoding='utf-8') as file:
                        func_gen_str = file.read()
                    with open(os.path.join(funcs_directory, file_name), 'w', encoding='utf-8') as file:
                        file.write(func_gen_str)
                    try:
                        clean_flist, _ = extract_func('```python+\n' + func_gen_str + '\n```')
                        clean_flist = list(set(clean_flist))
                        funcs_for_attr[attr]['clean'].extend(clean_flist)
                    except Exception as e:
                        print(f"Error: {e}")
            para_file.write(f"{attr} func_num:{len(funcs_for_attr[attr]['clean'])}\n")

    elif FUNC_USE:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_func_gen, attr, err_gen_dict) for attr in all_attrs]
            outputs = [result.result() for result in results]
            for output in outputs:
                funcs_for_attr.update(output)
    return err_gen_dict, funcs_for_attr


def process_gen_clean_funcs(PRE_FUNC_USE, PRE_FUNC_READ, read_pre_func_path, funcs_pre_directory, dirty_csv, all_attrs, related_attrs_dict, logger):
    pre_funcs_for_attr = defaultdict(default_dict_of_lists)
    if PRE_FUNC_USE and PRE_FUNC_READ:
        copy_read_files_in_dir(funcs_pre_directory, read_pre_func_path)
        for attr in all_attrs:
            file_names = os.listdir(read_pre_func_path)
            for file_name in sorted(file_names):
                if file_name == f'pre_funcs_zgen_{attr}.txt':
                    with open(os.path.join(read_pre_func_path, file_name), 'r', encoding='utf-8') as file:
                        func_gen_str = file.read()
                    try:
                        flist, _ = extract_func('```python+\n' + func_gen_str + '\n```')
                        flist = list(set(flist))
                        pre_funcs_for_attr[attr]['clean'].extend(flist)
                    except Exception as e:
                        print(f"Error: {e}")
    elif PRE_FUNC_USE:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(gen_clean_funcs, attr, dirty_csv, funcs_pre_directory, related_attrs_dict, logger) for attr in all_attrs]
            outputs = [result.result() for result in results]
            for output in outputs:
                pre_funcs_for_attr.update(output)
    elif not PRE_FUNC_USE:
        for attr in all_attrs:
            pre_funcs_for_attr[attr] = {'clean': []}
    return pre_funcs_for_attr


def gen_clean_funcs(attr, dirty_csv, funcs_pre_directory, related_attrs_dict, logger):  
    related_attrs = list(related_attrs_dict[attr])  
    sample_rows = []
    total_rows = len(dirty_csv)
    max_samp_num = 20
    if total_rows > 0:
        sample_indices = random.sample(range(total_rows), min(max_samp_num, total_rows))
        for idx in sample_indices:
            row_dict = dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
            sample_rows.append(row_dict)
    sample_rows_str = '\n'.join([str(row) for row in sample_rows])
    
    if len(sample_rows) == 0:
        logger.error("The Data is EMPTY!!!")
    prompt = pre_func_prompt(attr, sample_rows_str)
    pre_func_response = get_ans_from_llm(prompt, api_use=API_USE)
    flist, _ = extract_func(pre_func_response)
    with open(os.path.join(funcs_pre_directory, f"prompt_pre_funcs_zgen_{attr}.txt"), 'w', encoding='utf-8') as prom_file:
        prom_file.write(prompt)
    with open(os.path.join(funcs_pre_directory, f"pre_funcs_zgen_{attr}.txt"), 'w', encoding='utf-8') as func_file:
        func_file.write("\n".join(list(set(flist))))
    funcs_for_attr = {attr: {'clean': flist}}
    return funcs_for_attr


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Model settings
    n_method = config['model']['n_method']
    API_USE = config['model']['api_use']
    RELATED_ATTRS = config['model']['related_attrs']
    DISTRI_ANALYSIS = config['model']['distri_analysis']
    GUIDE_USE = config['model']['guide_use']
    PRE_FUNC_USE = config['model']['pre_func_use']
    FUNC_USE = config['model']['func_use']
    ERR_GEN_USE = config['model']['err_gen_use']
    REL_TOP = config['model']['rel_top']
    
    # Read settings
    PRE_FUNC_READ = config['read']['pre_func']
    DISTRI_ANALYSIS_READ = config['read']['distri_analysis']
    RELATED_ATTRS_READ = config['read']['related_attrs']
    CLUSTER_READ = config['read']['cluster']
    GUIDE_READ = config['read']['guide']
    ERROR_CHECKING_READ = config['read']['error_checking']
    FUNC_READ = config['read']['func']
    ERR_GEN_READ = config['read']['err_gen']
    
    # Dataset settings
    base_dir = config['data']['base_dir']
    err_rate_list = config['data']['err_rate_list']
    all_set_num = config['data']['all_set_num']
    dataset_list = config['data']['datasets'] * all_set_num
    result_dir = config['data']['result_dir']
    dataset_list = sorted(dataset_list)
    set_num_list = [i % all_set_num + 1 for i in range(len(dataset_list))]
    
    err_check_val_num_per_query = config['data']['err_check_val_num_per_query']
    
    read_path_dict = config['read_paths']
    READ_IN_BATCH = config['read']['read_in_batch']
    if READ_IN_BATCH:
        read_path_dict = get_read_paths(config['read']['start_time'], config['read']['end_time'], base_dir)
    
    date_time = datetime.now().strftime("%m-%d")
    run_info = config['model']['run_info']
    info_path = f"{base_dir}/result/{result_dir}/{date_time} {run_info}"
    os.makedirs(info_path, exist_ok=True)
    with open(os.path.join(info_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    for set_num, dataset in zip(set_num_list, dataset_list):
        for err_rate in err_rate_list:
            read_path = ''
            if any([PRE_FUNC_READ, DISTRI_ANALYSIS_READ, RELATED_ATTRS_READ, CLUSTER_READ, GUIDE_READ, ERROR_CHECKING_READ, FUNC_READ]):
                read_path = read_path_dict[f'{dataset}{err_rate}-{set_num}']
            read_guide_path = os.path.join(read_path, 'guide_refine')
            read_error_checking_path = os.path.join(read_path, 'error_checking')
            read_func_path = os.path.join(read_path, 'funcs')
            read_pre_func_path = os.path.join(read_path, 'funcs_pre')
            read_err_gen_path = os.path.join(read_path, 'err_gen')
            read_error_path = read_path + 'funcs'
            
            date_time = datetime.now().strftime("%m-%d")
            resp_path = f"{base_dir}/result/{result_dir}/{date_time} {dataset}{err_rate}-{n_method}-set{set_num}"
            guide_directory = f'{resp_path}/guide'
            error_checking_res_directory = f'{resp_path}/error_checking'
            funcs_directory = f'{resp_path}/funcs'
            funcs_pre_directory = f'{resp_path}/funcs_pre'
            err_gen_directory = f'{resp_path}/err_gen'
            os.makedirs(resp_path, exist_ok=True)
            os.makedirs(guide_directory, exist_ok=True)
            os.makedirs(error_checking_res_directory, exist_ok=True)
            os.makedirs(funcs_directory, exist_ok=True)
            os.makedirs(funcs_pre_directory, exist_ok=True)
            os.makedirs(err_gen_directory, exist_ok=True)
            
            dirty_path = base_dir + '/data/' + dataset + '_error-' + str(err_rate) + '.csv'
            clean_path = base_dir + '/data/' + dataset + '_clean.csv'
            clean_csv = pd.read_csv(clean_path, dtype=str).fillna('nan')
            dirty_csv = pd.read_csv(dirty_path, dtype=str).fillna('nan')
            all_attrs = list(dirty_csv.columns)
            
            para_file = open(os.path.join(resp_path, '0-parameters.txt'), 'w', encoding='utf-8')
            time_file = open(os.path.join(resp_path, '0-time.txt'), 'w', encoding='utf-8')
                    
            logger = Logger(resp_path).get_logger()
            
            parameters = {
                "executing File": os.path.abspath(__file__),
                "read_path": read_path,
                "resp_path": resp_path,
                "dirty_path": dirty_path,
                "clean_path": clean_path,
            }

            para_file.write("\n".join(f"{key}: {value}" for key, value in parameters.items()))
            para_file.write("\nConfig:\n")
            for section in ['model', 'read', 'data']:
                para_file.write(f"\n{section.title()}:\n")
                for key, value in config[section].items():
                    para_file.write(f"  {key}: {value}\n")
            para_file.write("\n")

            total_time = 0
            related_attrs_dict, gt_wrong_dict = {}, {}
            with Timer('Getting Related Attributes & gt_wrong_list', logger, time_file) as t:
                related_attrs_dict, gt_wrong_dict = process_related_attr(RELATED_ATTRS, RELATED_ATTRS_READ, REL_TOP, read_path, resp_path, clean_csv, dirty_csv, all_attrs)
            total_time += t.duration
                        
            pre_funcs_for_attr = {}
            with Timer('Preliminary Function Generation', logger, time_file) as t:
                pre_funcs_for_attr = process_gen_clean_funcs(PRE_FUNC_USE, PRE_FUNC_READ, read_pre_func_path, funcs_pre_directory, dirty_csv, all_attrs, related_attrs_dict, logger)
            total_time += t.duration
            
            cluster_index_dict, center_value_dict = {}, {}
            feature_all_dict = defaultdict(default_dict_of_lists)
            with Timer('Clustering', logger, time_file) as t:
                cluster_index_dict, center_value_dict, feature_all_dict = process_cluster(n_method, CLUSTER_READ, dataset, read_path, resp_path, dirty_csv, all_attrs, related_attrs_dict, pre_funcs_for_attr)
            total_time += t.duration
            
            distri_analy_content = {}
            with Timer('Analyzing Data Distribution', logger, time_file) as t:
                distri_analy_content = process_distri_analys(DISTRI_ANALYSIS, DISTRI_ANALYSIS_READ, read_path, resp_path, dirty_csv, all_attrs)
            total_time += t.duration
            
            guide_content = {}
            with Timer('Constructing Guidelines', logger, time_file) as t:
                guide_content = process_guidlines(GUIDE_USE, GUIDE_READ, dataset, read_path, read_guide_path, resp_path, dirty_csv, all_attrs, guide_directory, cluster_index_dict, distri_analy_content)
            total_time += t.duration
            
            with Timer('LLM Labeling', logger, time_file) as t:
                process_error_checking(ERROR_CHECKING_READ, read_error_checking_path, all_attrs, error_checking_res_directory)
            total_time += t.duration
            
            center_index_value_label_dict = {}
            with Timer('Extract Labeling Results', logger, time_file) as t:
                center_index_value_label_dict = extract_llm_label_res(all_attrs, error_checking_res_directory, cluster_index_dict, center_value_dict)
            total_time += t.duration
            
            measure_status = 'Not Done'
            with Timer('Evaluating LLM Labeling', logger, time_file) as t:
                measure_status = measure_llm_label(resp_path, clean_csv, all_attrs, related_attrs_dict, gt_wrong_dict, center_index_value_label_dict)
            total_time += t.duration
            
            para_file.write(f"LLM labeled value number: {sum(len(value) for value in cluster_index_dict.values())}\n")
            det_wrong_list, det_right_list = [], []
            with Timer('Label Propagation', logger, time_file) as t:
                det_wrong_list, det_right_list = label_prop(resp_path, dirty_path, clean_path, cluster_index_dict, center_index_value_label_dict)
            total_time += t.duration
            
            err_gen_dict, funcs_for_attr = {}, {}
            with Timer('Generating Functions', logger, time_file) as t:
                err_gen_dict, funcs_for_attr = process_gen_err_funcs(FUNC_USE, FUNC_READ, read_path, read_func_path, read_error_path, resp_path, funcs_directory, dirty_csv, all_attrs, para_file, related_attrs_dict, center_index_value_label_dict, det_wrong_list, det_right_list)
            total_time += t.duration
            
            err_gen_dict = defaultdict(default_dict_of_lists)
            with Timer('Generating Error Data', logger, time_file) as t:
                process_gen_err_data(ERR_GEN_USE, ERR_GEN_READ, read_err_gen_path, err_gen_directory, dirty_csv, all_attrs, related_attrs_dict, center_index_value_label_dict, err_gen_dict, logger)
            total_time += t.duration
            
            func_num = 0
            for attr in all_attrs:
                func_num += len(funcs_for_attr[attr]['clean'])
            para_file.write(f"ori_func_num:{func_num}\n")
            llm_label_vals_dict = defaultdict(lambda: defaultdict(list))

            init_det_right_dict = defaultdict(list)
            init_det_wrong_dict = defaultdict(list)
            for idx, attr in det_right_list:
                related_attrs = list(related_attrs_dict[attr])
                init_det_right_dict[attr].append((idx, dirty_csv.loc[idx, [attr]+related_attrs].to_dict()))
            for idx, attr in det_wrong_list:
                related_attrs = list(related_attrs_dict[attr])
                init_det_wrong_dict[attr].append((idx, dirty_csv.loc[idx, [attr]+related_attrs].to_dict()))
                
            for attr in all_attrs:
                related_attrs = list(related_attrs_dict[attr])
                wrong_val_values = []
                right_val_values = []
                for index, value, label in center_index_value_label_dict[attr]:
                    if label == 1:
                        wrong_val_values.append(dirty_csv.loc[index, [attr]+related_attrs].to_dict())
                    elif label == 0:
                        right_val_values.append(dirty_csv.loc[index, [attr]+related_attrs].to_dict())
                llm_label_vals_dict[attr]['wrong_val_values'].extend(list(wrong_val_values))
                llm_label_vals_dict[attr]['right_val_values'].extend(list(right_val_values))

            # use init_det_right_dict to filter funcs_for_attr
            for attr in all_attrs:
                for func in funcs_for_attr[attr]['clean']:
                    pass_num = 0
                    if len(init_det_right_dict[attr]) == 0:
                        continue
                    for val in init_det_right_dict[attr]:
                        if handle_func_exec(func, val[1], attr) == 1:
                            pass_num += 1
                    if float(pass_num / len(init_det_right_dict[attr])) < 0.5:
                        funcs_for_attr[attr]['clean'].remove(func)
            
            # use filtered funcs_for_attr to filter all det_right_list, with pass_num < 0.5
            for attr in all_attrs:
                for val in init_det_right_dict[attr]:
                    flag = 0
                    pass_num = 0
                    for func in funcs_for_attr[attr]['clean']:
                        if handle_func_exec(func, val[1], attr) == 1:
                            pass_num += 1
                    if float(pass_num / (len(funcs_for_attr[attr]['clean'])+1e-6)) < 0.5:
                        det_right_list.remove((val[0], attr))

            for attr in all_attrs:
                temp_func_list = []
                val_num = len(llm_label_vals_dict[attr]['right_val_values'])
                if val_num == 0:
                    continue
                for func in funcs_for_attr[attr]['clean']:
                    pass_num = 0
                    for val in llm_label_vals_dict[attr]['right_val_values']:
                        if handle_func_exec(func, val, attr) == 1:
                            pass_num += 1
                    if float(pass_num / val_num) >= 0.5:
                        temp_func_list.append(func)
                funcs_for_attr[attr]['clean'] = temp_func_list
            func_num = 0
            for attr in all_attrs:
                func_num += len(funcs_for_attr[attr]['clean'])
            para_file.write(f"after_right_val:{func_num}\n")
            
            feature_all_dict = None
            if os.path.exists(os.path.join(resp_path, f'cluster_feat_dict.pkl')):
                with open(os.path.join(resp_path, f'cluster_feat_dict.pkl'), 'rb') as f:
                    feature_all_dict = pickle.load(f)
                    
            num_epochs = 5000
            logger.info('Start Training Local Models')
            time_start = time.time()
            mp.set_start_method('spawn', force=True)
            feat_dict_train = {}
            label_dict_train = {}
            feat_pool = mp.Pool()
            results = []
            for attr in all_attrs:
                result = feat_pool.apply_async(process_attr_train_feat, args=(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, err_gen_dict, funcs_for_attr, feature_all_dict, resp_path))
                results.append(result)
            for result in results:
                attr, feature_list, label_list = result.get()
                feat_dict_train[attr] = feature_list
                label_dict_train[attr] = label_list
            feat_pool.close()
            
            model_col = {}
            
            for attr in tqdm(all_attrs, desc="Training models", ncols=120):
                attr, model, learning_rate, optimizer, model_str, epoch = train_model(attr, feat_dict_train[attr], label_dict_train[attr], num_epochs)
                if model is not None:
                    model_col[attr] = model
                    
            logger.info('Finish Generating Features & Training Models')
            para_file.close()
            det_wrong_list = []
            
            for col, attr in tqdm(enumerate(all_attrs), desc="Making predictions", ncols=120):
                wrong_cells = make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path)
                for cell in wrong_cells:
                    if cell not in det_wrong_list:
                        det_wrong_list.append(cell)
                        
            det_res_path = os.path.join(resp_path, "func_det_res.txt")
            measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)
            time_end = time.time()
            logger.info(f'Finish Local Model Training and Prediction, Using {time_end - time_start}s')
            total_time += time_end - time_start
            time_file.write(f"model_training: {time_end - time_start}\n")
            time_file.write(f"total: {total_time}\n")
        time_file.close()
        para_file.close()