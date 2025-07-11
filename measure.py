import pandas as pd

def measure_detect(clean_path, dirty_path_ori, detect_list, res_path, err_type='missing_included'):
    df_clean = pd.read_csv(clean_path, dtype=str).fillna('nan')
    df_dirty = pd.read_csv(dirty_path_ori, dtype=str).fillna('nan')
    results = ''
    all_need_detect = 0
    all_detected = 0
    correctly_detect = 0
    wrongly_detect = 0
    all_wrong_set = set()

    for index, row in df_clean.iterrows():
        for i in range(len(df_clean.columns)):
            if str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                all_need_detect += 1
                all_wrong_set.add((index, df_clean.columns[i]))
            elif str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                all_need_detect += 1
                all_wrong_set.add((index, df_clean.columns[i]))
    for i in range(len(detect_list)):
        tuple_number = int(detect_list[i][0])
        attribute = str(detect_list[i][1])
        if df_clean.iloc[tuple_number][attribute] != df_dirty.iloc[tuple_number][attribute]:
            if (tuple_number, attribute) in all_wrong_set:
                correctly_detect += 1
        else:
            wrongly_detect += 1
        all_detected += 1

    pre = correctly_detect / (all_detected + 1e-8)
    rec = correctly_detect / (all_need_detect + 1e-8)
    f1 = 2 * pre * rec / (pre + rec + 1e-8)

    results += 'all_wrong_num:' + str(all_need_detect) + '\n'
    results += 'all_detected_num:' + str(all_detected) + '\n'
    results += 'correctly_detect:' + str(correctly_detect) + '\n'
    results += 'pre:' + str(pre) + '\n'
    results += 'rec:' + str(rec) + '\n'
    results += 'f1:' + str(f1) + '\n'

    # Track counts per attribute
    attr_wrong_counts = {}  
    attr_detected_wrong = {}
    attr_missing_wrong = {} 
    
    # Count total errors per attribute
    for index, row in df_clean.iterrows():
        for i, col in enumerate(df_clean.columns):
            if (str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i])):
                attr_wrong_counts[col] = attr_wrong_counts.get(col, 0) + 1
                attr_missing_wrong[col] = attr_missing_wrong.get(col, 0) + 1

    # Count wrongly detected per attribute
    for i in range(len(detect_list)):
        tuple_number = int(detect_list[i][0])
        attribute = str(detect_list[i][1])
        if df_clean.loc[tuple_number, attribute] == df_dirty.loc[tuple_number, attribute]:
            attr_detected_wrong[attribute] = attr_detected_wrong.get(attribute, 0) + 1
        elif df_clean.loc[tuple_number, attribute] != df_dirty.loc[tuple_number, attribute]:
            attr_missing_wrong[attribute] = attr_missing_wrong.get(attribute, 0) - 1

    # Report statistics for each attribute
    results += '\nPer-attribute statistics:\n'
    for attr in df_clean.columns:
        total_errors = attr_wrong_counts.get(attr, 0)
        wrong_detects = attr_detected_wrong.get(attr, 0)
        missing_errors = attr_missing_wrong.get(attr, 0)
        results += f'\nAttribute: {attr}\n'
        results += f'Total errors: {total_errors}\n'
        results += f'Wrongly detected: {wrong_detects} ({(wrong_detects/(sum(attr_detected_wrong.values())+1e-6)*100):.2f}%)\n'
        results += f'Missing errors: {missing_errors} ({(missing_errors/(sum(attr_missing_wrong.values())+1e-6)*100):.2f}%)\n'

    results += '\nwrongly_detected:\n'
    for i in range(len(detect_list)):
        tuple_number = int(detect_list[i][0])
        attribute = str(detect_list[i][1])
        if df_clean.loc[tuple_number, attribute] == df_dirty.loc[tuple_number, attribute] and str(
                df_clean.loc[tuple_number, attribute]) != 'nan' and str(
                df_clean.loc[tuple_number, attribute]) != 'empty':
            results += str(tuple_number) + ' ' + attribute + ' ' + df_dirty.loc[tuple_number, attribute] + '\n'

    results += '\nmissing_errors:\n'
    for index, row in df_clean.iterrows():
        for i in range(len(df_clean.columns)):
            if (str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i])) and (index, list(df_clean.columns)[i]) not in detect_list:
                results += str(index) + ' ' + str(i) + ' ' + str(df_dirty.iat[index, i]) + ' --> ' + str(
                    df_clean.iat[index, i]) + '\n'

    with open(res_path, "w", encoding='utf-8') as file:
        file.write(results)
    file.close()
