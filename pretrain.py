import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.model    import RansomwareDetector
from core.features import FeatureExtractor
from core.pso      import PSOFeatureSelector

print("="*55)
print("  RansomGuard - Pre-Training Script")
print("="*55)

print("\n[1/5] Loading dataset...")
df = pd.read_csv("data/processed_dataset.csv")
print(f"      Rows: {df.shape[0]} | Columns: {df.shape[1]}")

print("\n[2/5] Extracting features...")
extractor = FeatureExtractor()
X, y, feature_names = extractor.preprocess_dataset(df, "label")
print(f"      Features : {len(feature_names)}")
print(f"      Benign   : {(y==0).sum()}")
print(f"      Malware  : {(y==1).sum()}")

print("\n[3/5] PSO Feature Selection...")
pso = PSOFeatureSelector(n_particles=10, n_iterations=15)

def pso_cb(iteration, score):
    bar = "#" * int(iteration / 15 * 20)
    print(f"      [{bar:<20}] {iteration}/15 | Score: {score:.4f}",
          end="\r")

best_idx  = pso.run(X, y, callback=pso_cb)
X_sel     = X[:, best_idx]
sel_names = [feature_names[i] for i in best_idx]
print(f"\n      Selected {len(best_idx)} best features      ")

print("\n[4/5] Training MLP model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y)

detector = RansomwareDetector()

def train_cb(chunk, logs):
    acc = logs.get("val_accuracy", 0) * 100
    bar = "#" * int(chunk / 10 * 20)
    print(f"      [{bar:<20}] {chunk}/10 | Accuracy: {acc:.1f}%",
          end="\r")

detector.train(X_train, y_train, X_test, y_test, callback=train_cb)
print()

print("\n[5/5] Evaluating and saving...")
results = detector.evaluate(X_test, y_test)
r       = results["report"]

os.makedirs("models", exist_ok=True)
np.save("models/selected_features.npy", best_idx)
with open("models/feature_names.json", "w") as f:
    json.dump(sel_names, f)

print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)
print(f"  Accuracy  : {r.get('accuracy', 0)*100:.2f}%")
print(f"  Precision : {r.get('1',{}).get('precision',0)*100:.2f}%")
print(f"  Recall    : {r.get('1',{}).get('recall',0)*100:.2f}%")
print(f"  F1-Score  : {r.get('1',{}).get('f1-score',0)*100:.2f}%")
print("="*55)
print("  Model saved! Run: python main.py")
print("="*55)
