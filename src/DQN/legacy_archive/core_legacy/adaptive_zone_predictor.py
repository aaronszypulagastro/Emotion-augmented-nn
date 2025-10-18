# adaptive_zone_predictor.py
import numpy as np

class AdaptiveZonePredictor:
    """
    Adaptive Vorhersage künftiger Zonenwechsel basierend auf
    Emotion-, TD-Error- und Reward-Dynamik.
    """

    def __init__(self, smoothing=0.9, sensitivity=1.3):
        self.prev = {"emotion": 0.0, "td": 0.0, "trend": 0.0}
        self.zone_pred = "stabilization"
        self.confidence = 0.5
        self.smoothing = smoothing
        self.sensitivity = sensitivity

    def step(self, emotion, td_error, reward_trend):
        # Numerisch robuste Deltas
        e = float(np.clip(emotion, 0.0, 1.0)) if np.isfinite(emotion) else 0.5
        td = float(td_error) if np.isfinite(td_error) else 0.0
        rt = float(reward_trend) if np.isfinite(reward_trend) else 0.0

        dE = e - self.prev["emotion"]
        dTD = td - self.prev["td"]
        dR = rt - self.prev["trend"]

        # Aktivitätsmaß
        raw = abs(dE)*0.6 + abs(dTD)*0.4 + dR*0.3
        if not np.isfinite(raw):
            raw = 0.0
        activity = np.tanh(self.sensitivity * raw)

        # Entscheidung mit hysterese
        if activity > 0.5:
            self.zone_pred = "exploration_soon"
        elif activity < -0.4:
            self.zone_pred = "stabilization_soon"
        else:
            self.zone_pred = "transition_zone"

        # Vertrauen & Intensität
        self.confidence = self.smoothing * self.confidence + (1 - self.smoothing) * abs(activity)
        if not np.isfinite(self.confidence):
            self.confidence = 0.0
        intensity = abs(activity)
        if not np.isfinite(intensity):
            intensity = 0.0

        # Zustand speichern
        self.prev.update({"emotion": e, "td": td, "trend": rt})

        return {
            "zone_pred": self.zone_pred,
            "confidence": round(self.confidence, 3),
            "intensity": round(float(intensity), 3)
        }
