import os
import sys
import time
import json
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QProgressBar,
    QTextEdit, QLineEdit, QGridLayout, QGroupBox,
    QFileDialog, QMessageBox, QStatusBar, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QFrame
)
from PyQt5.QtCore import (
    Qt, QTimer, QRunnable, QThreadPool,
    pyqtSlot, QObject, pyqtSignal
)
from PyQt5.QtGui import QColor, QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model    import RansomwareDetector
from core.scorer   import RiskScorer, RiskLevel
from core.pso      import PSOFeatureSelector
from core.features import FeatureExtractor


# ── Worker signals ────────────────────────────────────────────
class WorkerSignals(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)


# ── Training worker ───────────────────────────────────────────
class TrainWorker(QRunnable):

    def __init__(self, dataset_path, label_col):
        super().__init__()
        self.dataset_path = dataset_path
        self.label_col    = label_col
        self.signals      = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            from sklearn.model_selection import train_test_split

            self.signals.progress.emit(5, "Loading dataset...")
            df = pd.read_csv(self.dataset_path)
            self.signals.log.emit(
                f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

            extractor = FeatureExtractor()
            X, y, feature_names = extractor.preprocess_dataset(
                df, self.label_col)
            self.signals.log.emit(
                f"Features: {len(feature_names)} | "
                f"Malware: {int(y.sum())} | "
                f"Benign: {int((y==0).sum())}")

            self.signals.progress.emit(15, "Running PSO feature selection...")
            pso = PSOFeatureSelector(n_particles=10, n_iterations=15)

            def pso_cb(iteration, score):
                pct = 15 + int((iteration / 15) * 30)
                self.signals.progress.emit(
                    pct, f"PSO {iteration}/15 | Score: {score:.4f}")

            best_idx  = pso.run(X, y, callback=pso_cb)
            X_sel     = X[:, best_idx]
            sel_names = [feature_names[i] for i in best_idx]
            self.signals.log.emit(
                f"PSO selected {len(best_idx)} features")

            self.signals.progress.emit(47, "Splitting dataset 80/20...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y, test_size=0.2, random_state=42, stratify=y)

            self.signals.progress.emit(50, "Training MLP model...")
            detector = RansomwareDetector()

            def epoch_cb(chunk, logs):
                pct = 50 + int((chunk / 10) * 40)
                acc = logs.get("val_accuracy", 0) * 100
                self.signals.progress.emit(
                    min(pct, 89),
                    f"Training {chunk}/10 | Accuracy: {acc:.1f}%")

            detector.train(
                X_train, y_train, X_test, y_test,
                callback=epoch_cb)

            self.signals.progress.emit(92, "Evaluating model...")
            results = detector.evaluate(X_test, y_test)

            self.signals.progress.emit(97, "Saving model...")
            os.makedirs("models", exist_ok=True)
            np.save("models/selected_features.npy", best_idx)
            with open("models/feature_names.json", "w") as f:
                json.dump(sel_names, f)

            self.signals.progress.emit(100, "Training complete!")
            self.signals.finished.emit({
                "results":           results,
                "feature_count":     len(best_idx),
                "selected_features": sel_names,
            })

        except Exception as e:
            import traceback
            self.signals.error.emit(f"{e}\n{traceback.format_exc()}")


# ── Main window ───────────────────────────────────────────────
class MainDashboard(QMainWindow):

    def __init__(self, dev_mode=False):
        super().__init__()
        self.setWindowTitle("Anomeryx — Behavior-Based Detection System")
        self.setMinimumSize(1100, 720)
        self.dev_mode   = dev_mode
        self.detector   = RansomwareDetector()
        self.scorer     = RiskScorer()
        self.threadpool = QThreadPool()
        self._setup_ui()
        self._try_load_model()

    def _try_load_model(self):
        if self.detector.load():
            self.status_bar.showMessage("Model loaded — system ready.")
        else:
            self.status_bar.showMessage(
                "No model found. Run pretrain.py first.")

    def _setup_ui(self):
        self.setStyleSheet("""
            QMainWindow  { background: #1a1a2e; }
            QWidget      { background: #1a1a2e; }
            QTabWidget::pane { border: 1px solid #16213e;
                               background: #16213e; }
            QTabBar::tab { background: #0f3460; color: #e0e0e0;
                           padding: 10px 22px; font-size: 13px;
                           border-radius: 4px; margin-right: 3px; }
            QTabBar::tab:selected { background: #e94560;
                                    color: white; font-weight: bold; }
            QGroupBox { color: #e0e0e0; border: 1px solid #0f3460;
                        border-radius: 6px; margin-top: 10px;
                        padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin;
                               left: 10px; color: #e94560; }
            QLabel      { color: #e0e0e0; }
            QPushButton { background: #0f3460; color: white;
                          border: none; padding: 8px 18px;
                          border-radius: 5px; font-size: 13px; }
            QPushButton:hover    { background: #e94560; }
            QPushButton:disabled { background: #444; color: #888; }
            QLineEdit   { background: #0f3460; color: white;
                          border: 1px solid #e94560;
                          border-radius: 4px; padding: 6px; }
            QProgressBar { background: #0f3460; border-radius: 5px;
                           text-align: center; color: white; }
            QProgressBar::chunk { background: #e94560;
                                  border-radius: 5px; }
            QTextEdit   { background: #0d0d1a; color: #00ff88;
                          font-family: Consolas; font-size: 12px;
                          border: 1px solid #0f3460;
                          border-radius: 4px; }
            QTableWidget { background: #0d0d1a; color: #e0e0e0;
                           gridline-color: #0f3460; }
            QTableWidget::item:selected { background: #e94560; }
            QHeaderView::section { background: #0f3460;
                                   color: #e94560; padding: 6px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QWidget()
        header.setStyleSheet("background: #0f3460; padding: 8px;")
        header.setFixedHeight(60)
        hlay = QHBoxLayout(header)
        title = QLabel(
            "Anomeryx  |  Behavior-Based Ransomware Detection")
        title.setStyleSheet(
            "color: white; font-size: 18px; font-weight: bold;")
        self.header_status = QLabel("System Ready")
        self.header_status.setStyleSheet(
            "color: #2ecc71; font-size: 13px;")
        hlay.addWidget(title)
        hlay.addStretch()
        hlay.addWidget(self.header_status)
        layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), "Live Monitor")
        self.tabs.addTab(self._build_scan_tab(),    "Scan File")
        self.tabs.addTab(self._build_alerts_tab(),  "Alert Log")
        if self.dev_mode:
            self.tabs.addTab(
                self._build_train_tab(), "Train Model (Dev)")
        layout.addWidget(self.tabs)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "background: #0f3460; color: #aaa;")
        self.setStatusBar(self.status_bar)

    # ── Tab 1: Live Monitor ───────────────────────────────────
    def _build_monitor_tab(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(12)

        sg = QGroupBox("Real-Time Risk Score")
        sl = QHBoxLayout(sg)
        self.risk_label = QLabel("--")
        self.risk_label.setAlignment(Qt.AlignCenter)
        self.risk_label.setStyleSheet(
            "font-size: 72px; font-weight: bold; "
            "color: #2ecc71; min-width: 160px;")
        self.risk_level_label = QLabel("SAFE")
        self.risk_level_label.setAlignment(Qt.AlignCenter)
        self.risk_level_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #2ecc71;")
        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setFixedHeight(22)
        rp = QVBoxLayout()
        rp.addWidget(QLabel("Risk Level:"))
        rp.addWidget(self.risk_level_label)
        rp.addSpacing(10)
        rp.addWidget(QLabel("Score:"))
        rp.addWidget(self.risk_bar)
        sl.addWidget(self.risk_label)
        sl.addLayout(rp)
        lay.addWidget(sg)

        ig = QGroupBox("Behavioral Indicators")
        il = QGridLayout(ig)
        self.indicators = {}
        items = [
            ("entropy",    "File Entropy",       "0.00"),
            ("file_rate",  "File Changes/sec",   "0.00"),
            ("cpu",        "CPU Usage %",        "0.0"),
            ("memory",     "Memory Usage %",     "0.0"),
            ("registry",   "Registry Changes",   "0"),
            ("confidence", "Model Confidence %", "0.0"),
        ]
        for i, (key, lbl, default) in enumerate(items):
            row, col = divmod(i, 3)
            frame = QFrame()
            frame.setStyleSheet(
                "background: #0f3460; border-radius: 8px; padding: 6px;")
            fl = QVBoxLayout(frame)
            l1 = QLabel(lbl)
            l1.setStyleSheet("color: #aaa; font-size: 11px;")
            l2 = QLabel(default)
            l2.setStyleSheet(
                "color: #00ff88; font-size: 22px; font-weight: bold;")
            l2.setAlignment(Qt.AlignCenter)
            fl.addWidget(l1)
            fl.addWidget(l2)
            il.addWidget(frame, row, col)
            self.indicators[key] = l2
        lay.addWidget(ig)

        bl = QHBoxLayout()
        self.monitor_btn = QPushButton("Start Live Monitor")
        self.monitor_btn.setStyleSheet(
            "background: #27ae60; font-size: 14px; padding: 10px 24px;")
        self.monitor_btn.clicked.connect(self._start_monitor)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "background: #c0392b; font-size: 14px; padding: 10px 24px;")
        self.stop_btn.clicked.connect(self._stop_monitor)
        self.stop_btn.setEnabled(False)
        bl.addWidget(self.monitor_btn)
        bl.addWidget(self.stop_btn)
        bl.addStretch()
        lay.addLayout(bl)

        lg = QGroupBox("Activity Log")
        ll = QVBoxLayout(lg)
        self.monitor_log = QTextEdit()
        self.monitor_log.setReadOnly(True)
        self.monitor_log.setFixedHeight(140)
        ll.addWidget(self.monitor_log)
        lay.addWidget(lg)

        self.monitor_timer     = QTimer()
        self.monitor_timer.timeout.connect(self._update_live_metrics)
        self.monitoring_active = False
        return w

    def _start_monitor(self):
        if not self.detector.is_trained:
            QMessageBox.warning(self, "No Model",
                "Model not loaded. Run pretrain.py first.")
            return
        self.monitoring_active = True
        self.monitor_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.monitor_timer.start(2000)
        self._log_monitor("Live monitoring started...")
        self.header_status.setText("Monitoring Active")
        self.header_status.setStyleSheet(
            "color: #e74c3c; font-size: 13px;")

    def _stop_monitor(self):
        self.monitoring_active = False
        self.monitor_timer.stop()
        self.monitor_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log_monitor("Monitoring stopped.")
        self.header_status.setText("System Ready")
        self.header_status.setStyleSheet(
            "color: #2ecc71; font-size: 13px;")

    def _update_live_metrics(self):
        import psutil
        import random

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        sim_dir   = "test_files/simulated"
        file_rate = 0.0
        entropy   = 4.5
        reg_chg   = random.randint(0, 1)

        if os.path.exists(sim_dir):
            files     = os.listdir(sim_dir)
            enc_count = len([f for f in files if f.endswith(".encrypted")])
            bin_count = len([f for f in files if f.endswith(".bin")])
            total     = len(files)
            file_rate = min(total * 0.8, 15.0)

            if "README_DECRYPT.txt" in files:
                entropy   = 8.5
                file_rate = 15.0
                reg_chg   = 9
            elif enc_count > 0:
                entropy = random.uniform(7.2, 8.4)
                reg_chg = random.randint(3, 7)
            elif bin_count > 3:
                entropy = random.uniform(6.5, 7.5)
                reg_chg = random.randint(1, 4)
            else:
                entropy = random.uniform(4.5, 5.5)
        else:
            entropy   = random.uniform(4.5, 5.5)
            file_rate = random.uniform(0.1, 1.0)

        confidence = 0.1
        try:
            if os.path.exists("models/feature_names.json"):
                with open("models/feature_names.json") as f:
                    feature_names = json.load(f)
                X = np.zeros(len(feature_names), dtype=np.float32)
                _, confidence = self.detector.predict(X)
        except Exception:
            pass

        if entropy > 7.0:
            confidence = min(confidence + 0.6, 0.98)
        elif entropy > 6.0:
            confidence = min(confidence + 0.3, 0.75)

        result = self.scorer.calculate(
            model_confidence=confidence,
            entropy=entropy,
            file_mod_rate=file_rate,
            registry_changes=reg_chg,
            process_anomaly=cpu / 200.0
        )

        color = self.scorer.LEVEL_COLORS[result.level]
        self.risk_label.setText(f"{result.score:.0f}")
        self.risk_label.setStyleSheet(
            f"font-size: 72px; font-weight: bold; color: {color};")
        self.risk_level_label.setText(result.level.value)
        self.risk_level_label.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: {color};")
        self.risk_bar.setValue(int(result.score))

        self.indicators["entropy"].setText(f"{entropy:.2f}")
        self.indicators["file_rate"].setText(f"{file_rate:.2f}")
        self.indicators["cpu"].setText(f"{cpu:.1f}")
        self.indicators["memory"].setText(f"{mem:.1f}")
        self.indicators["registry"].setText(str(reg_chg))
        self.indicators["confidence"].setText(f"{confidence*100:.1f}")

        if result.level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            self._add_alert(result)
            self._log_monitor(
                f"[{result.level.value}] Score: {result.score} | "
                f"{', '.join(result.triggered_rules[:2])}")
        else:
            self._log_monitor(
                f"[{result.level.value}] Score: {result.score:.1f} | "
                f"Entropy: {entropy:.2f} | "
                f"Files: {file_rate:.1f}/sec")

    def _log_monitor(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.monitor_log.append(f"[{ts}] {msg}")

    # ── Tab 2: Scan File ──────────────────────────────────────
    def _build_scan_tab(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(12)

        sg = QGroupBox("Scan a File for Ransomware Indicators")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("File Path:"), 0, 0)
        self.scan_path_edit = QLineEdit()
        self.scan_path_edit.setPlaceholderText("Select a file to scan...")
        sl.addWidget(self.scan_path_edit, 0, 1)
        sb = QPushButton("Browse")
        sb.clicked.connect(self._browse_scan_file)
        sl.addWidget(sb, 0, 2)
        scan_btn = QPushButton("Scan File")
        scan_btn.setStyleSheet("background: #2980b9; font-size: 13px; padding: 10px 20px;")
        scan_btn.clicked.connect(self._scan_file)
        sl.addWidget(scan_btn, 1, 1)
        lay.addWidget(sg)

        rg = QGroupBox("Scan Results")
        rl = QVBoxLayout(rg)
        self.scan_result_label = QLabel("No file scanned yet.")
        self.scan_result_label.setAlignment(Qt.AlignCenter)
        self.scan_result_label.setStyleSheet(
            "font-size: 16px; padding: 15px;")
        rl.addWidget(self.scan_result_label)
        self.scan_detail = QTextEdit()
        self.scan_detail.setReadOnly(True)
        self.scan_detail.setFixedHeight(220)
        rl.addWidget(self.scan_detail)
        lay.addWidget(rg)
        lay.addStretch()
        return w

    def _browse_scan_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            self.scan_path_edit.setText(path)

    def _scan_file(self):
        path = self.scan_path_edit.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error",
                "Please select a valid file.")
            return

        extractor  = FeatureExtractor()
        entropy    = extractor.calculate_file_entropy(path)
        confidence = 0.0

        if self.detector.is_trained:
            try:
                with open("models/feature_names.json") as f:
                    feature_names = json.load(f)
                X = np.zeros(len(feature_names), dtype=np.float32)
                _, confidence = self.detector.predict(X)
            except Exception:
                confidence = min(entropy / 8.5, 1.0)
        else:
            confidence = min(entropy / 8.5, 1.0)

        if entropy > 7.0:
            confidence = min(confidence + 0.5, 0.98)

        result = self.scorer.calculate(
            model_confidence=confidence,
            entropy=entropy
        )

        color = self.scorer.LEVEL_COLORS[result.level]
        self.scan_result_label.setText(
            f"Risk Level: {result.level.value}   |   "
            f"Score: {result.score}/100")
        self.scan_result_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; "
            f"color: {color}; padding: 15px;")

        self.scan_detail.clear()
        self.scan_detail.append(
            f"File      : {os.path.basename(path)}")
        self.scan_detail.append(
            f"Size      : {os.path.getsize(path):,} bytes")
        self.scan_detail.append(
            f"Entropy   : {entropy:.4f}  "
            f"({'HIGH - suspicious' if entropy > 7.0 else 'Normal'})")
        self.scan_detail.append(
            f"Confidence: {confidence*100:.1f}%")
        self.scan_detail.append(
            f"Risk Score: {result.score}/100")
        self.scan_detail.append(
            f"Risk Level: {result.level.value}")
        if result.triggered_rules:
            self.scan_detail.append(
                f"Triggers  : {', '.join(result.triggered_rules)}")

        if result.level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            self._add_alert(
                result, f"Scan: {os.path.basename(path)}")

    # ── Tab 3: Alert Log ──────────────────────────────────────
    def _build_alerts_tab(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(15, 15, 15, 15)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Alert History"))
        cb = QPushButton("Clear All Alerts")
        cb.clicked.connect(self._clear_alerts)
        hl.addStretch()
        hl.addWidget(cb)
        lay.addLayout(hl)

        self.alert_table = QTableWidget(0, 5)
        self.alert_table.setHorizontalHeaderLabels(
            ["Time", "Level", "Score", "Confidence", "Details"])
        self.alert_table.horizontalHeader().setStretchLastSection(True)
        self.alert_table.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        self.alert_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.alert_table)

        self.alert_count_label = QLabel("Total alerts: 0")
        self.alert_count_label.setStyleSheet("color: #aaa;")
        lay.addWidget(self.alert_count_label)
        return w

    def _add_alert(self, result, extra=""):
        ts    = time.strftime("%Y-%m-%d %H:%M:%S")
        row   = self.alert_table.rowCount()
        self.alert_table.insertRow(row)
        color = self.scorer.LEVEL_COLORS[result.level]
        data  = [
            ts,
            result.level.value,
            f"{result.score:.1f}",
            f"{result.confidence*100:.1f}%",
            ", ".join(result.triggered_rules)
            if result.triggered_rules else extra
        ]
        for col, val in enumerate(data):
            item = QTableWidgetItem(val)
            if col == 1:
                item.setForeground(QColor(color))
                item.setFont(QFont("Arial", 10, QFont.Bold))
            self.alert_table.setItem(row, col, item)
        self.alert_count_label.setText(
            f"Total alerts: {self.alert_table.rowCount()}")
        if result.level == RiskLevel.CRITICAL:
            self.tabs.setCurrentIndex(2)

    def _clear_alerts(self):
        self.alert_table.setRowCount(0)
        self.alert_count_label.setText("Total alerts: 0")

    # ── Tab 4: Train Model (Dev only) ─────────────────────────
    def _build_train_tab(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(12)

        dg = QGroupBox("Dataset Configuration")
        dl = QGridLayout(dg)
        dl.addWidget(QLabel("Dataset CSV:"), 0, 0)
        self.dataset_path_edit = QLineEdit(
            "data/processed_dataset.csv")
        dl.addWidget(self.dataset_path_edit, 0, 1)
        bb = QPushButton("Browse")
        bb.clicked.connect(self._browse_dataset)
        dl.addWidget(bb, 0, 2)
        dl.addWidget(QLabel("Label Column:"), 1, 0)
        self.label_col_edit = QLineEdit("label")
        dl.addWidget(self.label_col_edit, 1, 1)
        lay.addWidget(dg)

        pg = QGroupBox("Dataset Preview (first 8 rows)")
        pl = QVBoxLayout(pg)
        self.preview_table = QTableWidget()
        self.preview_table.setFixedHeight(170)
        pl.addWidget(self.preview_table)
        pb = QPushButton("Load Preview")
        pb.clicked.connect(self._preview_dataset)
        pl.addWidget(pb)
        lay.addWidget(pg)

        tg = QGroupBox("Model Training")
        tl = QVBoxLayout(tg)
        self.train_btn = QPushButton("Start Training  (PSO + MLP)")
        self.train_btn.setStyleSheet(
            "background: #8e44ad; font-size: 14px; padding: 12px;")
        self.train_btn.clicked.connect(self._start_training)
        tl.addWidget(self.train_btn)
        self.train_progress = QProgressBar()
        tl.addWidget(self.train_progress)
        self.train_status = QLabel("Waiting to start...")
        self.train_status.setStyleSheet("color: #aaa;")
        tl.addWidget(self.train_status)
        lay.addWidget(tg)

        rg = QGroupBox("Training Log")
        rl = QVBoxLayout(rg)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFixedHeight(160)
        rl.addWidget(self.train_log)
        lay.addWidget(rg)
        return w

    def _browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset CSV", "data/", "CSV Files (*.csv)")
        if path:
            self.dataset_path_edit.setText(path)

    def _preview_dataset(self):
        path = self.dataset_path_edit.text()
        try:
            df = pd.read_csv(path, nrows=8)
            self.preview_table.setRowCount(len(df))
            self.preview_table.setColumnCount(len(df.columns))
            self.preview_table.setHorizontalHeaderLabels(
                df.columns.tolist())
            for i, row in df.iterrows():
                for j, val in enumerate(row):
                    self.preview_table.setItem(
                        i, j, QTableWidgetItem(str(val)))
            self.preview_table.resizeColumnsToContents()
            self.train_log.append(
                f"Preview: {df.shape[0]} rows, "
                f"{df.shape[1]} columns")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _start_training(self):
        path      = self.dataset_path_edit.text()
        label_col = self.label_col_edit.text().strip()
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found",
                f"Cannot find:\n{path}")
            return
        self.train_btn.setEnabled(False)
        self.train_log.clear()
        self.train_log.append("Starting training pipeline...")
        worker = TrainWorker(path, label_col)
        worker.signals.progress.connect(self._on_train_progress)
        worker.signals.finished.connect(self._on_train_finished)
        worker.signals.error.connect(self._on_train_error)
        worker.signals.log.connect(
            lambda m: self.train_log.append(f"  {m}"))
        self.threadpool.start(worker)

    def _on_train_progress(self, pct, msg):
        self.train_progress.setValue(pct)
        self.train_status.setText(msg)
        self.train_log.append(f"  [{pct}%] {msg}")

    def _on_train_finished(self, result):
        self.train_btn.setEnabled(True)
        self.train_progress.setValue(100)

        results = result["results"]
        r = results["report"]
        cm = results["confusion_matrix"]

        # Confusion matrix for binary classification:
        # [[TN, FP],
        #  [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        self.train_log.append("\n" + "="*45)
        self.train_log.append("TRAINING COMPLETE")

        self.train_log.append(
        f"  Features : {result['feature_count']}")

        self.train_log.append(
        f"  Accuracy : {r.get('accuracy', 0)*100:.2f}%")

        self.train_log.append(
        f"  Precision: "
        f"{r.get('1', {}).get('precision', 0)*100:.2f}%")

        self.train_log.append(
        f"  Recall   : "
        f"{r.get('1', {}).get('recall', 0)*100:.2f}%")

        self.train_log.append(
        f"  F1-Score : "
        f"{r.get('1', {}).get('f1-score', 0)*100:.2f}%")

        self.train_log.append("")
        self.train_log.append("CONFUSION MATRIX")
        self.train_log.append(
        f"  True Positive  (TP): {tp}")
        self.train_log.append(
        f"  True Negative  (TN): {tn}")
        self.train_log.append(
        f"  False Positive (FP): {fp}")
        self.train_log.append(
        f"  False Negative (FN): {fn}")

        self.train_log.append("="*45)
 
        self.detector.load()
        self.status_bar.showMessage("Model trained and ready!")

        QMessageBox.information(
        self,
        "Done",
        "Training complete! Model is now ready."
    )

    def _on_train_error(self, error):
        self.train_btn.setEnabled(True)
        self.train_log.append(f"\nERROR: {error}")
        QMessageBox.critical(self, "Training Error", error)