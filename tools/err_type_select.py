import argparse
import pandas as pd
from err_type_det import ErrorTypeDetector
from tqdm import tqdm

def main(dataset_name):
    detector = ErrorTypeDetector()
    dirty_df_path = f"./data/{dataset_name}_error-01.csv"
    clean_df_path = f"./data/{dataset_name}_clean.csv"
    
    try:
        dirty_df = pd.read_csv(dirty_df_path, dtype=str).fillna('nan')
        clean_df = pd.read_csv(clean_df_path, dtype=str).fillna('nan')
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files for '{dataset_name}'")
        print(f"Expected files: {dirty_df_path} and {clean_df_path}")
        return
    
    dirty_df.filepath = dirty_df_path
    results = detector.detect_errors(dirty_df, clean_df)
    
    missing_err_cnts = detector.missing_err_counts
    pattern_err_cnts = detector.pattern_err_counts
    typo_err_cnts = detector.typo_err_counts
    outlier_err_cnts = detector.outlier_err_counts
    rule_err_cnts = detector.rule_err_counts
    
    all_err_sets = [set(missing_err_cnts), set(pattern_err_cnts), set(typo_err_cnts),
                    set(outlier_err_cnts), set(rule_err_cnts)]
    
    mixed_err_cnts = set()
    for i, j in detector.error_cell_idx:
        appearance_count = sum(1 for err_set in all_err_sets if (i, j) in err_set)
        if appearance_count > 2:
            mixed_err_cnts.add((i, j))
    
    missing_dirty_df = dirty_df.copy()
    typo_dirty_df = dirty_df.copy()
    pattern_dirty_df = dirty_df.copy()
    outlier_dirty_df = dirty_df.copy()
    rule_dirty_df = dirty_df.copy()
    mixed_err_dirty_df = dirty_df.copy()
    for i, j in detector.error_cell_idx:
        if clean_df.iat[i, j] == 'nan (wait to be cleaned)':
            clean_df.iat[i, j] = 'nan (but not error)'
        if clean_df.iat[i, j] == 'empty (wait to be cleaned)':
            clean_df.iat[i, j] = 'empty (but not error)'
        if (i, j) not in missing_err_cnts:
            missing_dirty_df.iat[i, j] = clean_df.iat[i, j]
        if (i, j) not in pattern_err_cnts:
            pattern_dirty_df.iat[i, j] = clean_df.iat[i, j]
        if (i, j) not in typo_err_cnts:
            typo_dirty_df.iat[i, j] = clean_df.iat[i, j]
        if (i, j) not in outlier_err_cnts:
            outlier_dirty_df.iat[i, j] = clean_df.iat[i, j]
        if (i, j) not in rule_err_cnts:
            rule_dirty_df.iat[i, j] = clean_df.iat[i, j]
        if (i, j) not in mixed_err_cnts:
            mixed_err_dirty_df.iat[i, j] = clean_df.iat[i, j]
    
    ["typos", "missing_values", "pattern_violations", "rule_violations", "outliers", "all_err"]
    missing_dirty_df.to_csv(f"./data/{dataset_name}_clean_missing_values.csv", index=False)
    typo_dirty_df.to_csv(f"./data/{dataset_name}_clean_typos.csv", index=False)
    pattern_dirty_df.to_csv(f"./data/{dataset_name}_clean_pattern_violations.csv", index=False)
    outlier_dirty_df.to_csv(f"./data/{dataset_name}_clean_outliers.csv", index=False)
    rule_dirty_df.to_csv(f"./data/{dataset_name}_clean_rule_violations.csv", index=False)
    mixed_err_dirty_df.to_csv(f"./data/{dataset_name}_clean_mixed_err.csv", index=False)
    
if __name__ == "__main__":
    dataset_list = ['rayyan', 'hospital', 'flights', 'beers', 'movies', 'billionaire']
    for dataset in tqdm(dataset_list):
        main(dataset)
