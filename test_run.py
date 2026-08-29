import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Step 1: Python OK")

try:
    from PyQt5.QtWidgets import QApplication
    print("Step 2: PyQt5 OK")
except Exception as e:
    print(f"Step 2 FAILED: {e}")
    sys.exit()

try:
    from PyQt5.QtWidgets import QMainWindow
    app = QApplication(sys.argv)
    print("Step 3: QApplication created OK")
except Exception as e:
    print(f"Step 3 FAILED: {e}")
    sys.exit()

try:
    from gui.dashboard import MainDashboard
    print("Step 4: Dashboard import OK")
except Exception as e:
    print(f"Step 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit()

try:
    window = MainDashboard()
    print("Step 5: Window created OK")
except Exception as e:
    print(f"Step 5 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit()

try:
    window.show()
    print("Step 6: Window shown OK")
    print("="*40)
    print("App should be visible now!")
    print("="*40)
    sys.exit(app.exec_())
except Exception as e:
    print(f"Step 6 FAILED: {e}")
    import traceback
    traceback.print_exc()