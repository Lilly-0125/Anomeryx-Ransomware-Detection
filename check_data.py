import os
import json
import pandas as pd

print("="*50)
print("CHECKING CSV FILES")
print("="*50)

# Check CSV 1
print("\n--- MalBehavD-V1-dataset.csv ---")
df1 = pd.read_csv('data/MalBehavD-V1-dataset.csv')
print(f"Rows    : {df1.shape[0]}")
print(f"Columns : {df1.shape[1]}")
print(f"All column names:")
for col in df1.columns:
    print(f"  - {col} ({df1[col].dtype})")
print(f"\nFirst 2 rows:")
print(df1.head(2))

print("\n--- dll by samples ID Dataset.csv ---")
df2 = pd.read_csv('data/dll by samples ID Dataset.csv')
print(f"Rows    : {df2.shape[0]}")
print(f"Columns : {df2.shape[1]}")
print(f"All column names:")
for col in df2.columns:
    print(f"  - {col} ({df2[col].dtype})")
print(f"\nFirst 2 rows:")
print(df2.head(2))

print("\n--- lookup_table.json (small file) ---")
with open('data/lookup_table.json', 'r') as f:
    lookup = json.load(f)
print(f"Content: {lookup}")

print("\n--- benign.json (first item only) ---")
with open('data/benign.json', 'r') as f:
    benign = json.load(f)
if isinstance(benign, list):
    print(f"Total items : {len(benign)}")
    print(f"Keys in item: {list(benign[0].keys())}")
    print(f"First item  : {benign[0]}")
elif isinstance(benign, dict):
    print(f"Keys: {list(benign.keys())}")

print("="*50)
print("DONE")
print("="*50)