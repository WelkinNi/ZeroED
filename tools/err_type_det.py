import pandas as pd
import numpy as np
from collections import defaultdict
import re
import string
from typing import Dict, List, Tuple
import jellyfish  
from sklearn.ensemble import IsolationForest  
from spellchecker import SpellChecker  
import numpy.typing as npt
import argparse
from prettytable import PrettyTable  

class ErrorTypeDetector:
    def __init__(self):
        self.error_types = {
            'missing_values': self._check_missing_values,
            'pattern_violations': self._check_pattern_violations,
            'typos': self._check_typos,
            'outliers': self._check_outliers,
            'rule_violations': self._check_rule_violations
        }
        
    def detect_errors(self, dirty_df: pd.DataFrame, clean_df: pd.DataFrame = None) -> Dict[str, Dict[str, float]]:
        """
        Detect different types of errors in the dataset
        
        Args:
            dirty_df: DataFrame containing potentially erroneous data
            clean_df: Optional reference DataFrame containing clean data
            
        Returns:
            Dictionary containing error rates by type and column
        """
        results = defaultdict(lambda: defaultdict(float))
        total_cells = dirty_df.size  
        
        self.error_cell_idx = set()
        error_count = 0
        for index, row in clean_df.iterrows():
            for i in range(len(clean_df.columns)):
                if (str(dirty_df.iat[index, i]) != str(clean_df.iat[index, i]) or 
                    (dirty_df.columns[i] == 'Company Founded' and str(dirty_df.iat[index, i]) == '0') or 
                    (dirty_df.columns[i] == 'Demographics Age' and str(dirty_df.iat[index, i]) == '0')):
                    error_count += 1
                    self.error_cell_idx.add((index, i))
        
        for error_type, check_func in self.error_types.items():
            column_error_counts = check_func(dirty_df, clean_df)
            total_errors = sum(column_error_counts.values())
            results[error_type] = {
                'overall_rate': total_errors / total_cells,
                'column_rates': column_error_counts
            }
            
        return results

    def _check_missing_values(self, df: pd.DataFrame, _) -> Dict[str, int]:
        """Check for explicit and implicit missing values"""
        self.missing_err_counts = set()
        missing_counts = {}

        for col in df.columns:
            cnt = 0
            for i, val in enumerate(df[col]):
                if pd.isna(val) or val == 'nan' or val == 'null' or val == 'none' or val == 'na' or val == 'n/a' or val == '-':
                    self.missing_err_counts.add((i, df.columns.get_loc(col)))
                    cnt += 1
            missing_counts[col] = cnt
            
        return missing_counts

    def _check_pattern_violations(self, df: pd.DataFrame, clean_df: pd.DataFrame = None) -> Dict[str, int]:
        """Check for pattern violations by comparing with clean data patterns"""
        self.pattern_err_counts = set()
        pattern_counts = {}
        
        if clean_df is None:
            return self._check_pattern_violations_without_clean(df)
            
        for col in df.columns:
            violations = 0
            clean_patterns = set()
            
            for val in clean_df[col]:
                pattern = self._get_value_pattern(val)
                clean_patterns.add(pattern)
            
            for i, val in enumerate(df[col]):
                pattern = self._get_value_pattern(val)
                if pattern not in clean_patterns:
                    if (i, df.columns.get_loc(col)) in self.error_cell_idx:
                        self.pattern_err_counts.add((i, df.columns.get_loc(col)))
                        violations += 1
            pattern_counts[col] = violations
            
        return pattern_counts
    
    def _check_pattern_violations_without_clean(self, df: pd.DataFrame) -> Dict[str, int]:
        """Fallback method for pattern checking without clean data"""
        pattern_counts = {}
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'date': r'^\d{4}(-|/)\d{2}(-|/)\d{2}$',
            'numeric': r'^\d*\.?\d+$',
            'url': r'^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$'
        }
        for col in df.columns:
            sample_val = df[col].iloc[0]
            violations = 0
            
            if 'email' in col.lower():
                pattern = patterns['email']
            elif 'phone' in col.lower():
                pattern = patterns['phone']
            elif 'date' in col.lower():
                pattern = patterns['date']
            elif 'url' in col.lower() or 'link' in col.lower():
                pattern = patterns['url']
            elif isinstance(sample_val, (int, float)):
                pattern = patterns['numeric']
            else:
                continue
                
            violations = df[col].astype(str).str.match(pattern).value_counts().get(False, 0)
            pattern_counts[col] = violations
            
        return pattern_counts
        
    def _get_value_pattern(self, val) -> str:
        """Convert a value to its pattern representation using regex"""
        if pd.isna(val):
            return "NA"
            
        val_str = str(val)
        pattern = val_str
        pattern = re.sub(r'[A-Za-z]+', 'A', pattern)  
        pattern = re.sub(r'\d+', '0', pattern)     
                
        return pattern

    def _check_typos(self, df: pd.DataFrame, clean_df: pd.DataFrame = None) -> Dict[str, int]:
        """Check for potential typos using edit distance"""
        self.typo_err_counts = set()
        typo_counts = {}
        
        for col in df.columns:
            typos = 0
            if clean_df is not None:
                for i, (dirty_val, clean_val) in enumerate(zip(df[col], clean_df[col])):
                    if isinstance(dirty_val, str) and isinstance(clean_val, str):
                        distance = jellyfish.levenshtein_distance(dirty_val.lower(), clean_val.lower())
                        # If distance is 1 or 2 and it's a known error, count as typo
                        if len(dirty_val) == len(clean_val) and distance <= 3 and distance > 0 and (i, df.columns.get_loc(col)) in self.error_cell_idx:
                            self.typo_err_counts.add((i, df.columns.get_loc(col)))
                            typos += 1
                            
                typo_counts[col] = typos
            else:
                typo_counts[col] = 0
                        
        return typo_counts

    def _check_outliers(self, df: pd.DataFrame, _) -> Dict[str, int]:
        """Detect outliers using frequency-based approach for both numeric and categorical data"""
        self.outlier_err_counts = set()
        outlier_counts = {}
        
        for col in df.columns:
            value_counts = df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < 0.01].index
            
            outliers = 0
            for i, value in enumerate(df[col]):
                if value in rare_values and (i, df.columns.get_loc(col)) in self.error_cell_idx:
                    self.outlier_err_counts.add((i, df.columns.get_loc(col)))
                    outliers += 1
            
            if outliers > 0:
                outlier_counts[col] = outliers
                
        return outlier_counts

    def _check_rule_violations(self, df: pd.DataFrame, clean_df: pd.DataFrame = None) -> Dict[str, int]:
        """Check for functional dependency and pattern violations"""
        self.rule_err_counts = set()
        rule_counts = {}
        
        dataset_rules = {
            "hospital": {
                "functions": [
                    ["HospitalName", "ZipCode"], ["HospitalName", "PhoneNumber"],
                    ["MeasureCode", "MeasureName"], ["MeasureCode", "Stateavg"],
                    ["ProviderNumber", "HospitalName"], ["MeasureCode", "Condition"],
                    ["HospitalName", "Address1"], ["HospitalName", "HospitalOwner"],
                    ["HospitalName", "ProviderNumber"], ["City", "CountyName"],
                    ["ZipCode", "EmergencyService"], ["HospitalName", "City"],
                    ["MeasureName", "MeasureName"]
                ]
            },
            "flights": {
                "functions": [
                    ["flight", "act_dep_time"], ["flight", "sched_arr_time"],
                    ["flight", "act_arr_time"], ["flight", "sched_dep_time"],
                    ["sched_arr_time", "act_arr_time"], ["sched_dep_time", "act_dep_time"]
                ]
            },
            "beers": {
                "functions": [
                    ["brewery_name", "brewery_id"], ["brewery_id", "brewery_name"],
                    ["brewery_id", "city"], ["brewery_id", "state"],
                    ["beer_name", "brewery_name"]
                ]
            },
            "rayyan": {
                "functions": [
                    ["article_jvolumn", "article_pagination"],
                    ["article_language", "article_jcreated_at"],
                    ["journal_issn", "journal_title"]
                ]
            },
            "billionaire": {
                "functions": [["Name", "Rank"], ["Name", "Wealth How Industry"],
                              ["Company Name", "Company Founded"]],
            },
            "tax200k": {
                "functions": [["zip", "city"], ["zip", "state"], ["f_name", "gender"], ["area_code", "state"]],
            },
            "movies": {
                "functions": [],
            },
        }
        
        dataset_type = None
        for dataset in dataset_rules.keys():
            if dataset in str(df.filepath) if hasattr(df, 'filepath') else '':
                dataset_type = dataset
                break

        if dataset_type:
            for l_attribute, r_attribute in dataset_rules[dataset_type]["functions"]:
                df_cols_lower = {col.lower(): col for col in df.columns}
                l_attribute_lower = l_attribute.lower()
                r_attribute_lower = r_attribute.lower()
                
                if l_attribute_lower in df_cols_lower and r_attribute_lower in df_cols_lower:
                    l_attribute_orig = df_cols_lower[l_attribute_lower]
                    r_attribute_orig = df_cols_lower[r_attribute_lower]
                    value_dict = {}
                    for i, row in df.iterrows():
                        if row[l_attribute_orig]:
                            if row[l_attribute_orig] not in value_dict:
                                value_dict[row[l_attribute_orig]] = set()
                            if row[r_attribute_orig]:
                                value_dict[row[l_attribute_orig]].add(row[r_attribute_orig])
                
                    for i, row in df.iterrows():
                        if row[l_attribute_orig] and row[l_attribute_orig] in value_dict:
                            if len(value_dict[row[l_attribute_orig]]) > 1:  
                                l_j = df.columns.get_loc(l_attribute_orig)
                                r_j = df.columns.get_loc(r_attribute_orig)
                                if (i, r_j) in self.error_cell_idx:
                                    rule_counts[(i, r_j)] = 1
                                    self.rule_err_counts.add((i, r_j))
        
        column_counts = defaultdict(int)
        for (i, j), count in rule_counts.items():
            col = df.columns[j]
            if rule_counts.get((i, j), 0) == 1:
                if (i,j) in self.error_cell_idx:
                    column_counts[col] += 1
        
        return column_counts

def main(dataset_name):
    detector = ErrorTypeDetector()
    dirty_df_path = f"./data/{dataset_name}_clean_mixed_err.csv"
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
    total_cells = dirty_df.size
    error_count = len(detector.error_cell_idx)
    overall_error_rate = error_count / total_cells
    
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.align["Metric"] = "l"  
    summary_table.align["Value"] = "r"   
    summary_table.add_row(["Total Cells", total_cells])
    summary_table.add_row(["Error Count", error_count])
    summary_table.add_row(["Overall Error Rate", f"{overall_error_rate:.2%}"])
    
    error_table = PrettyTable()
    error_table.field_names = ["Error Type", "Error Rate"]
    error_table.align["Error Type"] = "l"  
    error_table.align["Error Rate"] = "r"  
    
    for error_type, rates in results.items():
        error_table.add_row([error_type.upper(), f"{rates['overall_rate']:.2%}"])
    
    print(f"\nDataset: {dataset_name}")
    print("\nSummary:")
    print(summary_table)
    print("\nError Types Breakdown:")
    print(error_table)
    print("\n" + "="*50)

if __name__ == "__main__":
    for dataset in ["beers"]:
        main(dataset)
