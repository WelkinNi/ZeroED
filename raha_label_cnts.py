import sys
import time
import pandas as pd
sys.path.append('./raha-master')
import raha
from measure_Sep9 import measure_detect
import os
import shutil
import csv
sys.path.append('./result/baselines')

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

def safe_remove_directory(path):
    """Safely remove a directory with retries and error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
            return
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Warning: Could not remove directory {path}: {e}")
            time.sleep(1)  # Wait before retry

class RahaLabelingExperiment:
    """
    Class to test Raha's performance with different labeling budgets over multiple runs
    """
    def __init__(self, num_runs=5):
        self.DATASETS = ["hospital", "flights", "movies", "beers", "rayyan", "billionaire"]
        self.MIN_BUDGET = 1
        self.MAX_BUDGET = 50
        self.NUM_RUNS = num_runs
        self.all_runs_results = []

    def run_experiment(self):
        """
        Run experiments for different labeling budgets on all datasets multiple times
        """
        data_dir = './data'
        print("------------------------------------------------------------------------")
        print("-------- Testing Different Labeling Budgets Over Multiple Runs --------")
        print("------------------------------------------------------------------------")

        for run in range(1, self.NUM_RUNS + 1):
            print(f"\nStarting Run {run}/{self.NUM_RUNS}")
            run_results = []  
            run_results.append(['Dataset', 'Labeling_Budget', 'Precision', 'Recall', 'F1', 'Time'])

            for dataset_name in self.DATASETS:
                print(f"  Processing dataset: {dataset_name}")

                for item in os.listdir(data_dir):
                    if item.startswith('raha-baran'):
                        safe_remove_directory(os.path.join(data_dir, item))

                dataset_dictionary = {
                    "name": dataset_name,
                    "path": f'./data/{dataset_name}_error-01.csv',
                    "clean_path": f'./data/{dataset_name}_clean.csv'
                }

                dirty_csv = pd.read_csv(dataset_dictionary['path'])
                attr_list = list(dirty_csv.columns)

                for budget in range(self.MIN_BUDGET, self.MAX_BUDGET + 1):
                    print(f"    Testing labeling budget: {budget}")
                    result_file = f'./result/labeling_budget/run_{run}/{dataset_name}_budget_{budget}_det_res.txt'
                    os.makedirs(os.path.dirname(result_file), exist_ok=True)

                    if os.path.exists(result_file):
                        metrics = read_results_file(result_file)
                        run_results.append([
                            dataset_name,
                            budget,
                            metrics.get('pre', 0.0),
                            metrics.get('rec', 0.0),
                            metrics.get('f1', 0.0),
                            0.0  # Assuming time isn't stored; modify if necessary
                        ])
                        continue

                    detector = raha.detection.Detection()
                    detector.VERBOSE = False
                    detector.LABELING_BUDGET = budget

                    start_time = time.time()
                    detection_dictionary = detector.run(dataset_dictionary)
                    end_time = time.time()
                    exec_time = end_time - start_time

                    detection_list = []
                    for (row_index, col_index), value in detection_dictionary.items():
                        detection_list.append([int(row_index), attr_list[col_index]])
                    detection_list = sorted(detection_list, key=lambda x: x[0])

                    measure_detect(
                        dataset_dictionary['clean_path'],
                        dataset_dictionary['path'],
                        detection_list,
                        result_file,
                        'error-01'
                    )

                    metrics = read_results_file(result_file)

                    run_results.append([
                        dataset_name,
                        budget,
                        metrics.get('pre', 0.0),
                        metrics.get('rec', 0.0),
                        metrics.get('f1', 0.0),
                        exec_time
                    ])

            self.all_runs_results.extend(run_results[1:])
        self.compute_and_save_mean_results()

    def compute_and_save_mean_results(self):
        """
        Compute mean metrics from all runs and save to a CSV file
        """
        df = pd.DataFrame(self.all_runs_results, columns=['Dataset', 'Labeling_Budget', 'Precision', 'Recall', 'F1', 'Time'])

        df['Labeling_Budget'] = df['Labeling_Budget'].astype(int)
        df[['Precision', 'Recall', 'F1', 'Time']] = df[['Precision', 'Recall', 'F1', 'Time']].astype(float)

        mean_df = df.groupby(['Dataset', 'Labeling_Budget']).agg({
            'Precision': 'mean',
            'Recall': 'mean',
            'F1': 'mean',
            'Time': 'mean'
        }).reset_index()

        output_file = 'raha_label_budget_mean_results.csv'
        mean_df.to_csv(output_file, index=False)

        print(f"\nMean results saved to {output_file}")

