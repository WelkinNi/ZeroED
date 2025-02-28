import os
import re
import csv
from datetime import datetime

def extract_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as file:
        content = file.read()
        metrics['pre'] = float(re.search(r'pre:([\d.]+)', content).group(1))
        metrics['rec'] = float(re.search(r'rec:([\d.]+)', content).group(1))
        metrics['f1'] = float(re.search(r'f1:([\d.]+)', content).group(1))
    return metrics

def process_folders(folder_name, results):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, 'initial_det_res.txt')
        if os.path.exists(file_path):
            metrics = extract_metrics(file_path)
            parts = folder_name.split(' ')[-1].split('-')
            dataset = parts[0]  # e.g. billionaire01
            # set_num = parts[-1].replace('set', '') # e.g. 1
            # dataset = f"{dataset}-{set_num}"  # e.g. billionaire01-1
            if dataset not in results:
                results[dataset] = []
            results[dataset].append((folder_name, metrics))
    return results

def write_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Folder', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for dataset, folders in results.items():
            mean_pre = sum(metrics['pre'] for _, metrics in folders) / len(folders)
            mean_rec = sum(metrics['rec'] for _, metrics in folders) / len(folders)
            mean_f1 = sum(metrics['f1'] for _, metrics in folders) / len(folders)
            
            std_pre = (sum((metrics['pre'] - mean_pre) ** 2 for _, metrics in folders) / len(folders)) ** 0.5
            std_rec = (sum((metrics['rec'] - mean_rec) ** 2 for _, metrics in folders) / len(folders)) ** 0.5
            std_f1 = (sum((metrics['f1'] - mean_f1) ** 2 for _, metrics in folders) / len(folders)) ** 0.5
            
            for folder, metrics in folders:
                writer.writerow({
                    'Dataset': dataset,
                    'Folder': folder,
                    'Precision': f"{metrics['pre']:.4f}",
                    'Recall': f"{metrics['rec']:.4f}",
                    'F1': f"{metrics['f1']:.4f}"
                })
            writer.writerow({
                'Dataset': f"{dataset}_mean",
                'Folder': "AVERAGE",
                'Precision': f"{mean_pre:.4f}±{std_pre:.4f}",
                'Recall': f"{mean_rec:.4f}±{std_rec:.4f}", 
                'F1': f"{mean_f1:.4f}±{std_f1:.4f}"
            })
            writer.writerow({
                'Dataset': '',
                'Folder': '',
                'Precision': '',
                'Recall': '',
                'F1': ''
            })
            
            
root_directory = './result/pipeline'
folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f))]

start_time = '02-15-15:18'
end_time = '02-15-15:21'

start_dt = datetime.strptime(start_time, '%m-%d-%H:%M')
end_dt = datetime.strptime(end_time, '%m-%d-%H:%M')
current_year = datetime.now().year
start_dt = start_dt.replace(year=current_year)
end_dt = end_dt.replace(year=current_year)

filtered_folders = []
for folder in folders:
    timestamp_match = re.match(r'(\d{2}-\d{2}-\d{2}:\d{2})', folder)
    if timestamp_match:
        folder_time = datetime.strptime(timestamp_match.group(1), '%m-%d-%H:%M')
        folder_time = folder_time.replace(year=current_year)
        if start_dt <= folder_time <= end_dt:
            filtered_folders.append(folder)

filtered_folders.sort(key=lambda x: x.split(' ')[-1].split('-')[0])

results = {}
for folder in filtered_folders:
    process_folders(folder, results)
    

output_file = os.path.join(root_directory, f'{start_time}-TO-{end_time}_metrics_results.csv')
write_to_csv(results, output_file)
