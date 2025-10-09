import numpy as np
from collections import deque

class MetaOptimizerV2:
    """
    Erweiterte Version (Phase 5.6 – Self-Recovery-fähig)
    • erkennt Reward-Drift UND TD-Error-Volatilität
    • moduliert η und gain dynamischer und sicherer
    """

    def __init__(self, window=50, lr=0.12, min_gain=0.8, max_gain=1.5):
        self.window = window
        self.lr = lr
        self.rewards = deque(maxlen=window)
        self.etas = deque(maxlen=window)
        self.gains = deque(maxlen=window)
        self.memory = deque(maxlen=10)
        self.min_gain = min_gain
        self.max_gain = max_gain

    def update(self, reward, eta, gain, emotion):
        # Verlauf puffern
        self.rewards.append(reward)
        self.etas.append(eta)
        self.gains.append(gain)

        if len(self.rewards) < self.window:
            return None

        # Trend-Schätzung Reward
        recent = np.mean(list(self.rewards)[-int(self.window/2):])
        past   = np.mean(list(self.rewards)[:int(self.window/2)])
        trend  = (recent - past) / (abs(past) + 1e-6)
        self.memory.append(trend)

        smooth_trend = 2 / (1 + np.exp(-6 * trend)) - 1
        avg_trend = np.mean(self.memory)

        # --- Self-Recovery-Modulation ---
        # emotion_factor: wie stark Emotion η dämpft/erhöht
        emotion_factor = 1.0 + 0.5 * (emotion - 0.5)
        # volatility_factor: basierend auf Reward-Schwankungen
        volatility = np.std(self.rewards)
        volatility_factor = np.clip(1.0 + 0.3 * np.tanh(volatility / 100), 0.8, 1.3)

        # adaptives η-Update
        delta_eta = self.lr * (smooth_trend - 0.2 * emotion) * volatility_factor
        delta_gain = 0.5 * self.lr * smooth_trend

        new_eta  = np.clip(eta * (1 + delta_eta) * emotion_factor, 1e-5, 7e-3)
        new_gain = np.clip(gain * (1 + delta_gain), self.min_gain, self.max_gain)

        # Selbst-Korrektur bei NaN oder zu kleiner Dynamik
        if not np.isfinite(new_eta) or new_eta < 1e-6:
            new_eta = 1e-3
        if not np.isfinite(new_gain):
            new_gain = 1.0

        return {'eta': new_eta, 'gain': new_gain, 'trend': avg_trend}
