import os

files = [
    "main.py",
    "core/model.py",
    "core/features.py",
    "core/scorer.py",
    "core/pso.py",
    "gui/dashboard.py",
    "models/mlp_model.pkl",
    "models/scaler.pkl",
    "models/feature_names.json",
    "data/processed_dataset.csv",
]

print("="*55)
all_ok = True
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        status = "OK" if size > 50 else "EMPTY!"
    else:
        size = 0
        status = "MISSING!"
        all_ok = False
    print(f"  {status:10s} {size:8d} bytes   {f}")
print("="*55)
print("ALL GOOD!" if all_ok else "SOME FILES MISSING - see above")
print("="*55)

input("Press Enter to close...")