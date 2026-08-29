import os
import time
import random
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
sim_dir = os.path.join(base_dir, "test_files", "simulated")

print("=" * 55)
print("  DIRECT ALERT TRIGGER")
print(f"  Writing to: {sim_dir}")
print("=" * 55)

os.makedirs(sim_dir, exist_ok=True)

print()
print("MAKE SURE:")
print("  1. App is open (python main.py)")
print("  2. Live Monitor tab is active")
print("  3. Start Live Monitor button is clicked")
print()
input("Press Enter to start triggering alerts...")

print("Phase 1: Creating suspicious .bin files...")
for i in range(15):
    path = os.path.join(sim_dir, f"suspicious_{i:03d}.bin")
    with open(path, "wb") as f:
        f.write(bytes([random.randint(0, 255) for _ in range(8192)]))
    print(f"  Created {i+1}/15 files")
    time.sleep(0.1)

print()
print("Done. Waiting 4 seconds for app to detect...")
time.sleep(4)

print("Phase 2: Creating .encrypted files...")
for i in range(10):
    path = os.path.join(sim_dir, f"document_{i:03d}.encrypted")
    with open(path, "wb") as f:
        f.write(bytes([random.randint(0, 255) for _ in range(16384)]))
    print(f"  Encrypted {i+1}/10")
    time.sleep(0.1)

print()
print("Done. Waiting 4 seconds for HIGH alert...")
time.sleep(4)

print("Phase 3: Dropping ransom note - CRITICAL alert!")
note_path = os.path.join(sim_dir, "README_DECRYPT.txt")
with open(note_path, "w") as f:
    f.write("YOUR FILES HAVE BEEN ENCRYPTED\n")
    f.write("This is a SIMULATION - RansomGuard test\n")

print("  Ransom note created!")
print()
print("  WATCH THE APP - should show CRITICAL now!")

time.sleep(5)

print()
print("=" * 55)
print("  TEST COMPLETE")
print("=" * 55)
print("  Check Alert Log tab in the app.")

cleanup = input("Clean up test files? (y/n): ").strip().lower()
if cleanup == "y":
    shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)
    print("Cleaned up!")

input("Press Enter to close...")
