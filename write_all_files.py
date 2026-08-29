import os

# ── core/scorer.py ────────────────────────────────────────────────────────────
scorer_content = '''import time
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    SAFE     = "SAFE"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskResult:
    score:           float
    level:           RiskLevel
    triggered_rules: list
    confidence:      float
    timestamp:       float


class RiskScorer:

    FEATURE_WEIGHTS = {
        "model_confidence":       0.40,
        "entropy_score":          0.25,
        "file_modification_rate": 0.20,
        "registry_changes":       0.10,
        "process_anomaly":        0.05,
    }

    THRESHOLDS = {
        RiskLevel.SAFE:     (0,   20),
        RiskLevel.LOW:      (20,  40),
        RiskLevel.MEDIUM:   (40,  65),
        RiskLevel.HIGH:     (65,  85),
        RiskLevel.CRITICAL: (85, 101),
    }

    LEVEL_COLORS = {
        RiskLevel.SAFE:     "#2ecc71",
        RiskLevel.LOW:      "#f1c40f",
        RiskLevel.MEDIUM:   "#e67e22",
        RiskLevel.HIGH:     "#e74c3c",
        RiskLevel.CRITICAL: "#8e44ad",
    }

    def calculate(self,
                  model_confidence: float = 0.0,
                  entropy:          float = 0.0,
                  file_mod_rate:    float = 0.0,
                  registry_changes: int   = 0,
                  process_anomaly:  float = 0.0) -> RiskResult:

        triggered = []

        m_score = model_confidence
        if model_confidence > 0.7:
            triggered.append(f"Model confidence: {model_confidence*100:.1f}%")

        e_score = min(max((entropy - 4.0) / 4.5, 0.0), 1.0)
        if entropy > 7.0:
            triggered.append(f"High entropy: {entropy:.2f}")

        f_score = min(file_mod_rate / 10.0, 1.0)
        if file_mod_rate > 5:
            triggered.append(f"Rapid file changes: {file_mod_rate:.1f}/sec")

        r_score = min(registry_changes / 10.0, 1.0)
        if registry_changes > 3:
            triggered.append(f"Registry modifications: {registry_changes}")

        p_score = min(process_anomaly, 1.0)
        if process_anomaly > 0.5:
            triggered.append("Suspicious process behavior")

        raw = (
            self.FEATURE_WEIGHTS["model_confidence"]       * m_score +
            self.FEATURE_WEIGHTS["entropy_score"]          * e_score +
            self.FEATURE_WEIGHTS["file_modification_rate"] * f_score +
            self.FEATURE_WEIGHTS["registry_changes"]       * r_score +
            self.FEATURE_WEIGHTS["process_anomaly"]        * p_score
        )

        score = round(raw * 100, 2)

        level = RiskLevel.SAFE
        for lvl, (lo, hi) in self.THRESHOLDS.items():
            if lo <= score < hi:
                level = lvl
                break

        return RiskResult(
            score=score,
            level=level,
            triggered_rules=triggered,
            confidence=model_confidence,
            timestamp=time.time()
        )
'''

# ── core/features.py ──────────────────────────────────────────────────────────
features_content = '''import numpy as np
import pandas as pd


class FeatureExtractor:

    def calculate_file_entropy(self, file_path: str) -> float:
        try:
            with open(file_path, "rb") as f:
                data = f.read(65536)
            if not data:
                return 0.0
            byte_counts   = np.bincount(
                np.frombuffer(data, dtype=np.uint8), minlength=256)
            probabilities = byte_counts / len(data)
            probabilities = probabilities[probabilities > 0]
            entropy       = -np.sum(probabilities * np.log2(probabilities))
            return round(float(entropy), 4)
        except Exception:
            return 0.0

    def preprocess_dataset(self, df: pd.DataFrame,
                           label_col: str) -> tuple:
        feature_cols = [
            c for c in df.columns
            if c != label_col and
            df[c].dtype in ["float64", "int64", "float32", "int32"]
        ]
        X = df[feature_cols].copy().fillna(0)
        y = df[label_col].copy()
        if y.dtype == object:
            y = (y.str.lower() != "benign").astype(int)
        return X.values, y.values, feature_cols
'''

# ── core/pso.py ───────────────────────────────────────────────────────────────
pso_content = '''import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class PSOFeatureSelector:

    def __init__(self, n_particles=20, n_iterations=30, n_features=None):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.n_features   = n_features
        self.best_features = None
        self.best_score    = 0.0

    def _fitness(self, particle, X, y):
        selected = np.where(particle > 0.5)[0]
        if len(selected) == 0:
            return 0.0
        X_sub = X[:, selected]
        clf   = RandomForestClassifier(
            n_estimators=20, random_state=42, n_jobs=-1)
        scores  = cross_val_score(clf, X_sub, y, cv=3, scoring="f1")
        penalty = len(selected) / self.n_features * 0.1
        return float(scores.mean() - penalty)

    def run(self, X, y, callback=None):
        self.n_features = X.shape[1]
        particles  = np.random.rand(self.n_particles, self.n_features)
        velocities = np.random.rand(
            self.n_particles, self.n_features) * 0.1
        personal_best        = particles.copy()
        personal_best_scores = np.array(
            [self._fitness(p, X, y) for p in particles])

        best_idx         = np.argmax(personal_best_scores)
        global_best      = personal_best[best_idx].copy()
        self.best_score  = personal_best_scores[best_idx]

        w, c1, c2 = 0.7, 1.5, 1.5

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                velocities[i] = (
                    w  * velocities[i]
                    + c1 * r1 * (personal_best[i] - particles[i])
                    + c2 * r2 * (global_best       - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], 0, 1)
                score = self._fitness(particles[i], X, y)
                if score > personal_best_scores[i]:
                    personal_best[i]        = particles[i].copy()
                    personal_best_scores[i] = score

            best_idx = np.argmax(personal_best_scores)
            if personal_best_scores[best_idx] > self.best_score:
                global_best     = personal_best[best_idx].copy()
                self.best_score = personal_best_scores[best_idx]

            if callback:
                callback(iteration + 1, round(self.best_score, 4))

        self.best_features = np.where(global_best > 0.5)[0]
        if len(self.best_features) == 0:
            self.best_features = np.argsort(global_best)[-10:]

        return self.best_features
'''

# ── core/__init__.py ──────────────────────────────────────────────────────────
init_content = ''

# ── Write all files ───────────────────────────────────────────────────────────
files = {
    "core/scorer.py":   scorer_content,
    "core/features.py": features_content,
    "core/pso.py":      pso_content,
    "core/__init__.py": init_content,
    "gui/__init__.py":  init_content,
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    size = os.path.getsize(path)
    print(f"  Written: {path}  ({size} bytes)")

print("\nAll core files written!")

# ── Verify imports work ───────────────────────────────────────────────────────
print("\nVerifying imports...")
try:
    from core.scorer import RiskScorer, RiskLevel
    print("  OK: core.scorer")
except Exception as e:
    print(f"  FAILED: core.scorer -> {e}")

try:
    from core.features import FeatureExtractor
    print("  OK: core.features")
except Exception as e:
    print(f"  FAILED: core.features -> {e}")

try:
    from core.pso import PSOFeatureSelector
    print("  OK: core.pso")
except Exception as e:
    print(f"  FAILED: core.pso -> {e}")

try:
    from core.model import RansomwareDetector
    print("  OK: core.model")
except Exception as e:
    print(f"  FAILED: core.model -> {e}")

print("\nDone! Now run: python main.py")