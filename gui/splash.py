import os
import sys
import json
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QProgressBar, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


# ── Background training thread ────────────────────────────────────────────────
class AutoTrainThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def run(self):
        try:
            from sklearn.model_selection import train_test_split
            from core.model import RansomwareDetector
            from core.features import FeatureExtractor
            from core.pso import PSOFeatureSelector

            self.progress.emit(5, "Loading dataset...")
            dataset_path = "data/processed_dataset.csv"
            if not os.path.exists(dataset_path):
                self.error.emit(
                    f"Dataset not found at {dataset_path}\n"
                    "Please run prepare_dataset.py first.")
                return

            df = pd.read_csv(dataset_path)

            self.progress.emit(15, "Extracting features...")
            extractor = FeatureExtractor()
            X, y, feature_names = extractor.preprocess_dataset(df, "label")

            self.progress.emit(20, "Running PSO feature selection...")
            pso = PSOFeatureSelector(n_particles=10, n_iterations=15)

            def pso_cb(iteration, score):
                pct = 20 + int((iteration / 15) * 30)
                self.progress.emit(
                    pct,
                    f"Optimizing features... {iteration}/15")

            best_idx  = pso.run(X, y, callback=pso_cb)
            X_sel     = X[:, best_idx]
            sel_names = [feature_names[i] for i in best_idx]

            self.progress.emit(52, "Splitting dataset 80/20...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            self.progress.emit(55, "Training detection model...")
            detector = RansomwareDetector()

            def epoch_cb(chunk, logs):
                pct = 55 + int((chunk / 10) * 35)
                acc = logs.get("val_accuracy", 0) * 100
                self.progress.emit(
                    min(pct, 89),
                    f"Training... {chunk}/10 | Accuracy: {acc:.1f}%")

            detector.train(
                X_train, y_train,
                X_test,  y_test,
                callback=epoch_cb
            )

            self.progress.emit(93, "Saving model...")
            os.makedirs("models", exist_ok=True)
            np.save("models/selected_features.npy", best_idx)
            with open("models/feature_names.json", "w") as f:
                json.dump(sel_names, f)

            self.progress.emit(100, "Ready!")
            self.finished.emit()

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ── Splash screen widget ──────────────────────────────────────────────────────
class SplashScreen(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RansomGuard")
        self.setFixedSize(500, 300)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._setup_ui()
        self._center()

    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                border-radius: 12px;
            }
            QLabel {
                color: white;
            }
            QProgressBar {
                background: #0f3460;
                border-radius: 5px;
                text-align: center;
                color: white;
                height: 22px;
            }
            QProgressBar::chunk {
                background: #e94560;
                border-radius: 5px;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(15)

        # Logo / title
        title = QLabel("Anomeryx")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setStyleSheet("color: #e94560;")
        lay.addWidget(title)

        subtitle = QLabel("Behavior-Based Ransomware Detection")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #aaa; font-size: 13px;")
        lay.addWidget(subtitle)

        lay.addSpacing(10)

        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        lay.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        lay.addWidget(self.progress_bar)

        # Footer
        footer = QLabel("Setting up detection model on first launch...")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #555; font-size: 11px;")
        lay.addWidget(footer)

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width()  - self.width())  // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def update_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)
        QApplication.processEvents()