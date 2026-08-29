"""
SAFE Ransomware Behavior Simulator
===================================
This script simulates ransomware-like behavior safely.
- Does NOT encrypt real files
- Does NOT damage your system
- Only creates/modifies files inside test_files/simulated/ folder
- Can be stopped anytime with Ctrl+C
"""

import os
import sys
import time
import random
import shutil
import threading

# ── Safety check ─────────────────────────────────────────────
SAFE_DIR = "test_files/simulated"
os.makedirs(SAFE_DIR, exist_ok=True)

print("="*55)
print("  RANSOMWARE BEHAVIOR SIMULATOR")
print("  (100% Safe - only touches test_files/simulated/)")
print("="*55)
print()

running = True


def generate_high_entropy_bytes(size):
    """Generate random bytes - simulates encrypted file content."""
    return bytes([random.randint(0, 255) for _ in range(size)])


def simulate_file_encryption():
    """
    Simulates crypto-ransomware behavior:
    - Creates many files rapidly
    - Writes high-entropy (random) content
    - Renames files (simulates .encrypted extension)
    """
    print("[PHASE 1] Simulating rapid file creation...")
    files_created = []

    for i in range(20):
        if not running:
            break
        filename = os.path.join(SAFE_DIR, f"document_{i:03d}.txt")
        with open(filename, "w") as f:
            f.write(f"Normal document content {i}\n" * 50)
        files_created.append(filename)
        print(f"  Created: document_{i:03d}.txt", end="\r")
        time.sleep(0.05)

    print(f"\n  Created {len(files_created)} files")
    time.sleep(1)

    print("\n[PHASE 2] Simulating encryption (high entropy write)...")
    encrypted_files = []

    for i, original in enumerate(files_created):
        if not running:
            break
        # Write random bytes (simulates encryption)
        encrypted_name = original.replace(".txt", ".encrypted")
        with open(encrypted_name, "wb") as f:
            f.write(generate_high_entropy_bytes(random.randint(4096, 16384)))

        # Remove original (simulates ransomware deleting originals)
        os.remove(original)
        encrypted_files.append(encrypted_name)
        print(f"  Encrypted: {i+1}/{len(files_created)} files", end="\r")
        time.sleep(0.1)

    print(f"\n  Encrypted {len(encrypted_files)} files")
    time.sleep(1)

    print("\n[PHASE 3] Simulating ransom note drop...")
    note_path = os.path.join(SAFE_DIR, "README_DECRYPT.txt")
    with open(note_path, "w") as f:
        f.write("YOUR FILES HAVE BEEN ENCRYPTED (SIMULATION ONLY)\n")
        f.write("This is a test - no real encryption occurred.\n")
        f.write("Anomeryx should have detected this activity!\n")
    print(f"  Dropped: README_DECRYPT.txt")

    return encrypted_files


def simulate_mass_file_modification():
    """
    Simulates locker ransomware behavior:
    - Rapidly modifies many files
    - High file modification rate
    """
    print("\n[PHASE 4] Simulating rapid file modifications...")

    for round_num in range(3):
        if not running:
            break
        print(f"  Round {round_num+1}/3 - modifying files rapidly...")
        for i in range(10):
            if not running:
                break
            filename = os.path.join(
                SAFE_DIR, f"modified_{round_num}_{i}.bin")
            with open(filename, "wb") as f:
                f.write(generate_high_entropy_bytes(8192))
            time.sleep(0.02)
        print(f"  Round {round_num+1} done - 10 files modified")
        time.sleep(0.5)


def monitor_simulation():
    """
    Shows live stats during simulation.
    These are what Anomeryx's Live Monitor detects.
    """
    start_time = time.time()
    while running:
        elapsed   = time.time() - start_time
        files     = len([f for f in os.listdir(SAFE_DIR)
                         if os.path.isfile(os.path.join(SAFE_DIR, f))])
        enc_files = len([f for f in os.listdir(SAFE_DIR)
                         if f.endswith(".encrypted")])
        print(
            f"\r  [Monitor] Time: {elapsed:.0f}s | "
            f"Files: {files} | "
            f"Encrypted: {enc_files}    ",
            end=""
        )
        time.sleep(0.5)


def cleanup():
    """Remove all simulated files after test."""
    if os.path.exists(SAFE_DIR):
        shutil.rmtree(SAFE_DIR)
        os.makedirs(SAFE_DIR, exist_ok=True)
    print("\n  Cleaned up all simulated files.")


# ── Main simulation ───────────────────────────────────────────
def run_simulation(mode):
    global running

    if mode == "1":
        # Quick test - triggers MEDIUM alert
        print("\nRunning: QUICK TEST (30 seconds)")
        print("Expected alert level: MEDIUM")
        print("-"*40)
        monitor_thread = threading.Thread(
            target=monitor_simulation, daemon=True)
        monitor_thread.start()
        for i in range(5):
            if not running:
                break
            fname = os.path.join(SAFE_DIR, f"quick_test_{i}.bin")
            with open(fname, "wb") as f:
                f.write(generate_high_entropy_bytes(16384))
            time.sleep(0.3)
        time.sleep(2)

    elif mode == "2":
        # Full simulation - triggers HIGH/CRITICAL alert
        print("\nRunning: FULL SIMULATION (60 seconds)")
        print("Expected alert level: HIGH or CRITICAL")
        print("-"*40)
        monitor_thread = threading.Thread(
            target=monitor_simulation, daemon=True)
        monitor_thread.start()
        simulate_file_encryption()
        simulate_mass_file_modification()
        time.sleep(3)

    elif mode == "3":
        # Continuous simulation - keeps triggering alerts
        print("\nRunning: CONTINUOUS (press Ctrl+C to stop)")
        print("Expected alert level: HIGH")
        print("-"*40)
        monitor_thread = threading.Thread(
            target=monitor_simulation, daemon=True)
        monitor_thread.start()
        count = 0
        while running:
            fname = os.path.join(SAFE_DIR, f"continuous_{count}.bin")
            with open(fname, "wb") as f:
                f.write(generate_high_entropy_bytes(
                    random.randint(4096, 32768)))
            count += 1
            time.sleep(0.1)

    running = False


# ── Menu ─────────────────────────────────────────────────────
print("Choose simulation mode:")
print()
print("  [1] Quick Test       - 5 files, ~10 seconds")
print("      Triggers: MEDIUM alert")
print()
print("  [2] Full Simulation  - encrypts 20 files, ~60 seconds")
print("      Triggers: HIGH or CRITICAL alert")
print()
print("  [3] Continuous       - keeps running until Ctrl+C")
print("      Triggers: HIGH alert")
print()
print("  [4] Cleanup          - remove all test files")
print()

mode = input("Enter choice (1/2/3/4): ").strip()

if mode == "4":
    cleanup()
    print("Done!")
    sys.exit()

if mode not in ["1", "2", "3"]:
    print("Invalid choice!")
    sys.exit()

print()
print("INSTRUCTIONS:")
print("  1. Keep this window open")
print("  2. Open Anomeryx app (run.bat)")
print("  3. Click 'Start Live Monitor'")
print("  4. Watch the risk score go up!")
print("  5. Check Alert Log tab for alerts")
print()
input("Press Enter when ready to start simulation...")

try:
    run_simulation(mode)
except KeyboardInterrupt:
    running = False
    print("\n\nSimulation stopped by user.")

print()
print("="*55)
print("  SIMULATION COMPLETE")
print("="*55)
print("  Check Anomeryx app:")
print("  - Alert Log tab should show HIGH/CRITICAL alerts")
print("  - Risk score should have spiked during simulation")
print("="*55)

ask = input("\nClean up test files? (y/n): ").strip().lower()
if ask == "y":
    cleanup()