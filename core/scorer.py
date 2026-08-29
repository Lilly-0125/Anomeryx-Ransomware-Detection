import time
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
