import os
import random

os.makedirs("test_files", exist_ok=True)

# Normal text file - should be LOW risk
with open("test_files/normal_document.txt", "w") as f:
    f.write("This is a normal text document.\n" * 100)
print("Created: normal_document.txt")

# High entropy file - simulates encrypted content
with open("test_files/suspicious_file.bin", "wb") as f:
    f.write(bytes([random.randint(0, 255) for _ in range(50000)]))
print("Created: suspicious_file.bin")

# Executable-like file
with open("test_files/normal_image.jpg", "wb") as f:
    f.write(b'\xff\xd8\xff\xe0')  # JPEG header
    f.write(bytes([random.randint(100, 200) for _ in range(10000)]))
print("Created: normal_image.jpg")

print("\nTest files created in test_files/ folder!")
print("Now scan each one in the app under Scan File tab.")