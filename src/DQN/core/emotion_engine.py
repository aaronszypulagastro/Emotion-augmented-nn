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

    def __init__(self, init_state=0.5, alpha=0.15, target_return=300, floor=0.3, ceil=0.98, gain=1.07, noise_std=0.02):
        self.state = float(init_state)
        self.value = self.state  # Aktueller Emotionswert (für Ausgabe)
        self.alpha = float(alpha)
        self.target_return = float(target_return)
        self.floor = float(floor)
        self.ceil = float(ceil)
        self.gain = float(gain * 1.25)
        self.noise_std = float(noise_std * 1.3)
        self._momentum = 0.0  # Für glattere Übergänge

        # Initialisierung für feinere Kontrolle 
        self.win_streak = 0
        self.loss_streak = 0 


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
        noise = np.random.normal(0.0, self.noise_std * self.gain)
        self.state = max(self.floor, min(self.ceil, self.state + noise))

        # 7. Glättung mit Momentum 
        self.momentum = getattr(self, '_momentum', 0.0)
        target = np.clip(self.state, self.floor, self.ceil)
        self._momentum = 0.7 * self._momentum + 0.3 * (target - self.state)
        self.state += 0.5 * self._momentum
        self.state = np.clip(self.state, self.floor, self.ceil)

        # Ausagbe aktualisieren
        self.value = self.state 

        # Phase 5.5: Stabilisation Dämpfung 
        # Wenn Emotion dauerhaft zu hoch -> sanfte Rezentrierung
        if getattr(self, "avg_buffer", None) is None:
            self.avg_buffer = []
        self.avg_buffer.append(self.state)
        if len(self.avg_buffer) > 50:
            self.avg_buffer.pop(0)
        avg_val = np.mean(self.avg_buffer)

        if avg_val > 0.8:
            self.state -= 0.02  # dämpft Überoptimismus
        elif avg_val < 0.35:
            self.state += 0.02  # dämpft Dauerfrustration

        self.state = np.clip(self.state, self.floor, self.ceil)
        self.value = self.state

        # Optionaler Start-Snaity-Check (NaN-Prävention beim Initialisieren)
        if np.isnan(self.state) or np.isnan(self.value) or self.state < self.floor or self.state > self.ceil:
            print("[Init-Check] EmotionEngine start-reset auf 0.5 (NaN-Prävention)")
            self.state = 0.5
            self.value = 0.5

        # NaN - Schutz
        if not np.isfinite(self.state):
            print("[WARN] EmotionEngine State NaN → Reset auf 0.5")
            self.state = 0.5
            self.value = 0.5

        # Debug-Ausgabe
        print(f'[DEBUG] Emotion updated: {self.state:.3f}')

    def update(self, reward: float, td_error: float = 0.0): 
        """
        Schnelles Echtzeit-Update während des Trainings.
        reward: aktueller Reward
        td-error: optionaler TD Fehler für stärkere Anpassungen
        """
        # einfache gewichtete Kombination 
        delta = 0.08 * reward + 0.12 * td_error

        # kleine ZUfallskomponente (Noise) für lebendige Variation 
        delta += np.random.normal(0, 0.01)

        # kleine Dämpfung, um extremes Wachstum zu verhindern
        self.state = (1 - self.alpha) * self.state + self.alpha * (self.state + delta)
        
        # BEgrenzen auf sinnvollen Bereich  
        self.state = float(np.clip(self.state, self.floor, self.ceil))

        # Ausgabe akutualisieren
        self.value = self.state 

        # Debug-Ausgabe 
        if np.random.rand() < 0.01 :
            print(f'[EmotionEngine] state={self.state:.3f}, Δ={delta:.3f}')

    def apply_zone_response(self, emotion, td_error):
        """Steuert adaptive Emotion- & Lernratenregelung basierend auf Reward-Zonen."""
        # --- Heuristik: Zone basierend auf Emotion & TD-Error schätzen
        if emotion > 0.8 and td_error < 5:
            zone = 0  # Konsolidierung
        elif td_error < 50:
            zone = 1  # Übergang
        else:
            zone = 2  # Exploration

        # --- Regelparameter aus Zone-Response-Map
        if zone == 0:
            eta = max(getattr(self, 'eta', 0.001) * 0.9, 1e-5)
            emotion_target = 0.9
            mode = "stabil"
        elif zone == 1:
            eta = min(getattr(self, 'eta', 0.001) * 1.1, 0.005)
            emotion_target = 0.6
            mode = "neutral"
        else:  # Zone 2
            eta = min(getattr(self, 'eta', 0.001) * 1.5, 0.01)
            emotion_target = 0.4
            mode = "explore"

        # --- Emotion sanft anpassen
        self.state = 0.9 * self.state + 0.1 * emotion_target
        self.eta = eta

        if getattr(self, 'debug', False):
            print(f"[ZoneResponse] Zone={zone} | Mode={mode} | η={eta:.5f} | target_emotion={emotion_target:.2f}")