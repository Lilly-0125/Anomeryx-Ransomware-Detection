import os

print("="*55)
print("PATH DIAGNOSTIC")
print("="*55)
print(f"Current working directory: {os.getcwd()}")
print(f"Script location          : {os.path.dirname(os.path.abspath(__file__))}")

sim_dir_relative = "test_files/simulated"
sim_dir_absolute = os.path.abspath(sim_dir_relative)

print(f"\nRelative path used: {sim_dir_relative}")
print(f"Resolves to       : {sim_dir_absolute}")
print(f"Exists            : {os.path.exists(sim_dir_absolute)}")

if os.path.exists(sim_dir_absolute):
    files = os.listdir(sim_dir_absolute)
    print(f"Files inside      : {len(files)}")
    for f in files[:10]:
        print(f"  - {f}")

print("="*55)
input("Press Enter to close...")