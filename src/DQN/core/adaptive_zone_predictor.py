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
        # Änderungen berechnen
        dE = emotion - self.prev["emotion"]
        dTD = td_error - self.prev["td"]
        dR = reward_trend - self.prev["trend"]

        # Aktivitätsmaß
        activity = np.tanh(self.sensitivity * (abs(dE)*0.6 + abs(dTD)*0.4 + dR*0.3))

        # Entscheidung mit hysterese
        if activity > 0.5:
            self.zone_pred = "exploration_soon"
        elif activity < -0.4:
            self.zone_pred = "stabilization_soon"
        else:
            self.zone_pred = "transition_zone"

        # Vertrauen & Intensität
        self.confidence = self.smoothing * self.confidence + (1 - self.smoothing) * abs(activity)
        intensity = abs(activity)

        # Zustand speichern
        self.prev.update({"emotion": emotion, "td": td_error, "trend": reward_trend})

        return {
            "zone_pred": self.zone_pred,
            "confidence": round(self.confidence, 3),
            "intensity": round(float(intensity), 3)
        }
