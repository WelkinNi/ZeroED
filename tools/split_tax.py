# split tax into 50k, 100k, 150k, and 200k in data/tax_error-01.csv and corresponding clean version data/tax_clean.csv
import pandas as pd
import os

error_df = pd.read_csv('data/tax_error-01.csv')
clean_df = pd.read_csv('data/tax_clean.csv')

os.makedirs('data/splits', exist_ok=True)

sizes = [50000, 100000, 150000, 200000]

for size in sizes:
    error_subset = error_df.head(size)
    clean_subset = clean_df.head(size)
    
    size_str = f'{size//1000}k'
    error_subset.to_csv(f'data/tax{size_str}_error-01.csv', index=False)
    clean_subset.to_csv(f'data/tax{size_str}_clean.csv', index=False)
    
print("Split files have been created in data/splits directory")

