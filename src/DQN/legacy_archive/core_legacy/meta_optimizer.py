import numpy as np 
from collections import deque 

class MetaOptimizer: 
    """
    Analysiert den Reward-Trend über ein gleitendes Fenster
    und passt Meta-Parameter wie η und gain dynamisch an.
    """
    def __init__(self, window=50, lr=0.05, min_gain=0.8, max_gain=1.5):
        self.window = window
        self.lr = lr
        self.rewards = deque(maxlen=window)
        self.etas = deque(maxlen=window)
        self.gains = deque(maxlen=window)
        self.min_gain = min_gain
        self.max_gain = max_gain

    def update(self, reward, eta, gain, emotion): 
        # Aktualisiert Verlauf und gibt Meta-Korrelationen zurück
        self.rewards.append(reward)
        self.etas.append(eta)
        self.gains.append(gain)

        if len(self.rewards) < self.window: 
            return None # noch zu wenig Daten 
        
        # Trendanalyse 
        recent = np.mean(list(self.rewards)[-int(self.window/2):])
        past   = np.mean(list(self.rewards)[:int(self.window/2)])
        trend  = (recent - past) / (abs(past) + 1e-6)

        # Regelwerk 
        delta_eta, delta_gain = 0.0, 0.0 
        if trend < -0.05:   # Reward fällt 
            delta_eta  = +self.lr * (1.0 - emotion)   # mehr Lernen bei tiefer Emotion
            delta_gain = -self.lr * 0.5               # weniger Verstärkung

        elif trend > +0.05:  # Reward steigt
            delta_eta  = -self.lr * emotion           # konservativer bei Erfolg
            delta_gain = +self.lr * 0.5

        # Grenzen prüfen 
        new_gain = np.clip(gain * (1 + delta_gain), self.min_gain, self.max_gain)
        new_eta  = max(1e-5, eta * (1 + delta_eta))

        return {'eta': new_eta, 'gain': new_gain, 'trend': trend}