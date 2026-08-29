import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from PyQt5.QtWidgets import QApplication, QMessageBox


def model_exists():
    return (
        os.path.exists("models/mlp_model.pkl") and
        os.path.exists("models/scaler.pkl") and
        os.path.exists("models/feature_names.json")
    )


def main():
    # --dev flag shows developer tab
    dev_mode = "--dev" in sys.argv

    app = QApplication(sys.argv)
    app.setApplicationName("RansomGuard")
    app.setStyle("Fusion")

    if not model_exists():
        QMessageBox.critical(
            None,
            "Model Not Found",
            "Trained model not found!\n\n"
            "Please run pretrain.py first:\n"
            "   python pretrain.py\n\n"
            "This only needs to be done once."
        )
        sys.exit(1)

    from gui.dashboard import MainDashboard
    window = MainDashboard(dev_mode=dev_mode)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()