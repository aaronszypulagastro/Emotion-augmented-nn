# -*- coding: utf-8 -*- 
from __future__ import annotations
from typing import Optional
import random
import numpy as np


# === EMOTION: Engine ===
class EmotionEngine: 

    """
    Einfache 'Stimmungs'-Variable für den Agenten-
    - state in [0,1], init bei 0.5 (neutral)
    - Update nach Episodenrückgabe (EMA)
    """

    def __init__(self, init_state=0.5, alpha=0.5, target_return=300, floor=0.3, ceil=0.98):
        self.state = float(init_state)
        self.value = self.state  # Aktueller Emotionswert (für Ausgabe)
        self.alpha = float(alpha)
        self.target_return = float(target_return)
        self.floor = float(floor)
        self.ceil = float(ceil)

    def update_after_episode(self, episode_return: float):
        # Debug-Ausgabe
        if episode_return is None or np.isnan(episode_return):
            return # Ungültiger Rückgabewert, kein Update

        # 1. Normales EMA-Update basierend auf der Episoden-Rückgabe
        norm = max(0.0, min(1.0, episode_return / self.target_return))
        self.state = (1.0 - self.alpha) * self.state + self.alpha * norm
        
        #self.state = max(self.floor, min(self.ceil, self.state))

            # Zusätzliche Anpassungen (08.10.25):
        if episode_return > 0:
            delta = min(0.05, 0.1 * norm)
            self.state += delta
        else: 
            self.state -= 0.02 

            # Begrenzung auf [floor, ceil]
        self.state = max(self.floor, min(self.ceil, self.state))

        # 2. Momentum Effekt (WINNER MENTALITY)
        if episode_return > 400:
            self.win_streak = getattr(self, 'win_streak', 0) + 1
            self.state = min(1.0, self.state + 0.1 * self.win_streak)

        else: 
            self.win_streak = 0

        # 3. Frustration Effekt (LOSER MENTALITY)
        if episode_return < 50:
            self.loss_streak = getattr(self, 'loss_streak', 0) + 1
            self.state = max(0.0, self.state - 0.02 * self.loss_streak)

        else:
            self.loss_streak = 0

        # 4. SKalierung am REturn 
        boost = (episode_return / self.target_return) * 0.1
        self.state = min(1.0, self.state + boost) 

        # 5. Erholung nach schlechten Phasen 
        if self.loss_streak >= 5:
            self.state = min(1.0, self.state + 0.02)

        # 6. Noise / Zufallseinfluss
        noise = np.random.normal(-0.01, 0.01)
        self.state = max(self.floor, min(self.ceil, self.state + noise))

        # Ausagbe aktualisieren
        self.value = self.state 
        print(f'[DEBUG] Emotion updated: {self.state:.3f}')

 