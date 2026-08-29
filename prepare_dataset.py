import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("STEP 1: Loading raw data...")
print("="*60)

# Load files
df = pd.read_csv('data/MalBehavD-V1-dataset.csv')
with open('data/lookup_table.json', 'r') as f:
    lookup = json.load(f)

print(f"Total samples : {df.shape[0]}")
print(f"Total columns : {df.shape[1]}")

# Show label distribution
print("\nLabel distribution:")
label_counts = df['labels'].value_counts().sort_index()
for label_id, count in label_counts.items():
    name = lookup[label_id] if label_id < len(lookup) else 'unknown'
    print(f"  Label {label_id:2d} = {name:15s} → {count} samples")

print("\n" + "="*60)
print("STEP 2: Preparing API call columns...")
print("="*60)

# API call columns are named 0, 1, 2, ... 151
api_cols = [str(i) for i in range(152) if str(i) in df.columns]
print(f"API call columns found: {len(api_cols)}")

# Drop unnamed/empty columns
unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
df = df.drop(columns=unnamed_cols)
print(f"Dropped {len(unnamed_cols)} unnamed empty columns")

# Get all unique API call names across the dataset
print("\nGathering all unique API call names...")
all_apis = set()
for col in api_cols:
    unique_vals = df[col].dropna().unique()
    all_apis.update(unique_vals)

all_apis = sorted(list(all_apis))
print(f"Unique API calls found: {len(all_apis)}")
print(f"Sample API names: {all_apis[:10]}")

print("\n" + "="*60)
print("STEP 3: Converting API calls to frequency features...")
print("="*60)

# For each sample, count how many times each API call appears
# This converts text sequence → numeric feature vector
api_to_idx = {api: i for i, api in enumerate(all_apis)}

# Create frequency matrix
print("Building frequency matrix (this may take a moment)...")
freq_matrix = np.zeros((len(df), len(all_apis)), dtype=np.float32)

for row_idx, (_, row) in enumerate(df.iterrows()):
    for col in api_cols:
        api_name = row[col]
        if pd.notna(api_name) and api_name in api_to_idx:
            freq_matrix[row_idx, api_to_idx[api_name]] += 1

print(f"Frequency matrix shape: {freq_matrix.shape}")

print("\n" + "="*60)
print("STEP 4: Creating binary labels (benign vs ransomware)...")
print("="*60)

# Strategy: benign=0, ransomware=1, skip others
# OR: benign=0, all malware=1 (more training data)

LABEL_MODE = 'all_malware'  # change to 'all_malware' if too few ransomware

if LABEL_MODE == 'ransomware_only':
    # Keep only benign (label=0) and ransomware (label=11)
    mask = df['labels'].isin([0, 11])
    X = freq_matrix[mask]
    y = (df['labels'][mask] == 11).astype(int).values
    print(f"Mode: Benign vs Ransomware only")
else:
    # Benign=0, all other malware=1
    X = freq_matrix
    y = (df['labels'] > 0).astype(int).values
    print(f"Mode: Benign vs All Malware")

print(f"Final samples: {X.shape[0]}")
print(f"  Benign     : {(y==0).sum()}")
print(f"  Ransomware : {(y==1).sum()}")

print("\n" + "="*60)
print("STEP 5: Removing zero-variance features (unused APIs)...")
print("="*60)

# Remove API columns that never appear (all zeros) - saves memory
col_sums = X.sum(axis=0)
active_cols = col_sums > 0
X = X[:, active_cols]
active_apis = [api for api, active in zip(all_apis, active_cols) if active]
print(f"Active API features: {X.shape[1]} (was {len(all_apis)})")

print("\n" + "="*60)
print("STEP 6: Saving processed dataset...")
print("="*60)

# Build final dataframe
feature_names = [f'api_{api.replace(" ", "_")}' for api in active_apis]
df_processed = pd.DataFrame(X, columns=feature_names)
df_processed['label'] = y

output_path = 'data/processed_dataset.csv'
df_processed.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print(f"Shape   : {df_processed.shape}")
print(f"Columns : {df_processed.shape[1]-1} features + 1 label")
print(f"\nSample of feature names:")
for name in feature_names[:8]:
    print(f"  - {name}")

print("\n" + "="*60)
print("DATASET READY!")
print(f"Use this in the app:")
print(f"  Dataset path : data/processed_dataset.csv")
print(f"  Label column : label")
print("="*60)