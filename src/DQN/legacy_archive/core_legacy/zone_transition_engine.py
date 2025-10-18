import numpy as np 
from collections import deque 

class ZoneTransitionEngine: 
    """
    Meta-Regler, der vorausschauend emotionale Zonenwechsel steuert.
    Analysiert Trends von Emotion, TD-Error und η über Zeit.
    Gibt Vorsteuerungs-Signale (feedforward corrections) an die EmotionEngine.
    """
    def __init__(self, window=15, threshold=0.03):
        self.window = window
        self.threshold = threshold
        self.emotion_hist = deque(maxlen=window)
        self.td_hist = deque(maxlen=window)
        self.eta_hist = deque(maxlen=window)
        self.predicted_zone = None

    def update(self, emotion, td_error, eta):
        self.emotion_hist.append(emotion)
        self.td_hist.append(td_error)
        self.eta_hist.append(eta)
        
        if len(self.emotion_hist) == self.window:
            return None  # Nur wenn genug Daten vorliegen
        return self._predict_transition()

    def _predict_transition(self):
        emo_trend = np.mean(np.diff(self.emotion_hist))
        td_trend = np.mean(np.diff(self.td_hist))
        eta_trend = np.mean(np.diff(self.eta_hist))

        if emo_trend < -self.threshold and td_trend > self.threshold:
            zone = 'exploration_soon'
        elif emo_trend > self.threshold and eta_trend < -self.threshold:
            zone = 'stabilization_soon'
        else: 
            zone = 'neutral'

        self.predicted_zone = zone
        return zone

    def apply_to_emotion_engine(self, emotion_engine):
        """OPitonal: beeinflusst basierend auf Vorhersage die EmotionEngine."""
        
        if self.predicted_zone == 'exploration_soon':
            emotion_engine.gain *= 1.05
            emotion_engine.eta = min(getattr(emotion_engine, 'eta', 1e-3) * 1.2, 0.01)

        elif self.predicted_zone == 'stabilization_soon':
            emotion_engine.gain *= 0.95
            emotion_engine.eta = max(getattr(emotion_engine, 'eta', 1e-3) * 0.8, 1e-5)

        
        