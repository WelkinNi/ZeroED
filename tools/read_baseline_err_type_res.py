import os
import pandas as pd

def read_results_file(file_path):
    """Read metrics from a results file."""
    metrics = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if any(metric in line for metric in ['pre:', 'rec:', 'f1:']):
                key, value = line.strip().split(':')
                metrics[key] = float(value)
    return metrics

def create_results_table():
    results_dir = 'result/baselines/'
    results = {
        'Algorithm': [],
        'Error Type': [],
        'Precision': [],
        'Recall': [],
        'F1': []
    }
    algos_list = ['Raha', 'KATARA', 'NADEEF', 'dBoost', 'ActiveClean']
    err_type_list = ['typos', 'missing_values', 'pattern_violations', 'rule_violations', 'outliers', 'mixed_err']
    
    for algo in algos_list:
        print(f"********{algo}********")
        for err_type in err_type_list:
            file_name = f'beers_{err_type}_{algo}_det_res.txt'
            file_path = os.path.join(results_dir, file_name)
            
            if os.path.exists(file_path):
                metrics = read_results_file(file_path)
                
                # Store results
                results['Algorithm'].append(algo)
                results['Error Type'].append(err_type)
                results['Precision'].append(metrics.get('pre', 0.0))
                results['Recall'].append(metrics.get('rec', 0.0))
                results['F1'].append(metrics.get('f1', 0.0))
                print(f"{err_type} - F1: {metrics.get('f1', 0.0)}")
                
    
    df = pd.DataFrame(results)
    
    pivot_df = pd.pivot_table(df,
                            index='Algorithm',
                            columns='Error Type',
                            values=['Precision', 'Recall', 'F1'],
                            aggfunc='first')  # Take first value in case of duplicates
    
    # Flatten column names
    pivot_df.columns = [f'{metric}_{error}' for metric, error in pivot_df.columns]
    
    return pivot_df

results_table = create_results_table()
print("\nResults Table:")