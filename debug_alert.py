import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scorer import RiskScorer, RiskLevel

print("="*50)
print("DEBUG: Alert System Check")
print("="*50)

# Check what directory app runs from
print(f"\nApp runs from: {os.getcwd()}")

# Check sim folder
sim_dir = "test_files/simulated"
print(f"Sim folder path: {os.path.abspath(sim_dir)}")
print(f"Sim folder exists: {os.path.exists(sim_dir)}")

if os.path.exists(sim_dir):
    files = os.listdir(sim_dir)
    print(f"Files in sim folder: {len(files)}")
    for f in files:
        print(f"  - {f}")
else:
    print("PROBLEM: Simulator folder not found!")
    print("Creating it now...")
    os.makedirs(sim_dir, exist_ok=True)

print("\n" + "="*50)
print("Testing scorer directly...")
print("="*50)

scorer = RiskScorer()

# Simulate CRITICAL conditions
result = scorer.calculate(
    model_confidence=0.95,
    entropy=8.5,
    file_mod_rate=15.0,
    registry_changes=9,
    process_anomaly=0.5
)
print(f"\nCRITICAL test:")
print(f"  Score: {result.score}")
print(f"  Level: {result.level.value}")
print(f"  Expected: HIGH or CRITICAL")
passed = result.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
print(f"  Result: {'PASS' if passed else 'FAIL'}")
print("="*50)
input("\nPress Enter to close...")