# -*- coding: utf-8 -*-
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from plot_utils import plot_comparison 

# --- Sicherstellen, dass Ergebnisdateien existieren ---
if not (os.path.exists("results/scores_emotion.npy") and os.path.exists("results/scores_baseline.npy")):
    raise FileNotFoundError("Ergebnisse nicht gefunden. Bitte zuerst beide Trainingsläufe ausführen.")
{ 

# --- Farbpalette für einheitliche Viusalisierung ---
COLORS = { 
    'emotion': 'orange'
    'baseline': 'steelblue'
}

# --- Daten laden ---
scores_emotion = np.load("results/scores_emotion.npy")
scores_baseline = np.load("results/scores_baseline.npy")

# --- Gleitenden Durchschnitt berechnen ---
def moving_avg(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

episodes = np.arange(len(scores_emotion))

# --- Plot erstellen ---
plt.figure(figsize=(10,6))
plt.plot(episodes[:len(moving_avg(scores_emotion))], moving_avg(scores_emotion, 50),
         label="Mit Emotion", color="orange")
plt.plot(episodes[:len(moving_avg(scores_baseline))], moving_avg(scores_baseline, 50),
         label="Ohne Emotion", color="steelblue")

plt.title(f"Vergleich: DQN mit vs. ohne Emotionseinfluss ({len(scores_emotion)} Episoden")
plt.xlabel("Episode")
plt.ylabel("Return (gleitender Durchschnitt 50)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Verzeichnis absichern ---
os.makedirs('results', exist_ok=True)

plt.savefig("results/comparison_plot.png", dpi=300)
plt.show()
