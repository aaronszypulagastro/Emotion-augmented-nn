# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np 
import os 

# --- Hilfsfunktion für gleitenden Durchschnitt ---
def smooth(data, window=50):
    """Berechnet gleitenden Durchschnitt (Moving Average)."""
    if len(data) < window or len(data) == 0: 
        return np.array(data)       # Kein Glötten möglich, einfach zurücggeben 
    return np.convolve(data, np.ones(window)/window, mode='valid')


# --- Einzelner Trainingslauf (z. B. Baseline oder Emotion) ---
def plot_single_run(scores, epsilon_values=None, save_path=None):
    """Zeigt Return-Verlauf (und optional Epsilon) für einen Run."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    episodes = np.arange(len(scores))

    ax1.plot(episodes[:len(smooth(scores))], smooth(scores), color="blue", label="Return (gl. Durchschnitt)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return", color="blue")

    # Falls Epsilon mitgegeben wurde
    if epsilon_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(episodes, epsilon_values, color="orange", alpha=0.5, label="Epsilon")
        ax2.set_ylabel("Epsilon", color="orange")

    plt.title("Trainingsverlauf")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else: 
        plt.savefig("results/training_curve.png", dpi=300)
    plt.pause(0.5)  # Kurze Pause, um das Plot-Fenster zu aktualisieren
    plt.show()
    finalize_plots()


# --- Vergleich zweier Läufe ---
def plot_comparison(scores_emotion, scores_baseline, save_path=None):
    """Vergleicht DQN mit vs. ohne Emotionseinfluss."""
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(len(scores_emotion))

    ax.plot(episodes[:len(smooth(scores_emotion))], smooth(scores_emotion), color="orange", label="Mit Emotion")
    ax.plot(episodes[:len(smooth(scores_baseline))], smooth(scores_baseline), color="steelblue", label="Ohne Emotion")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (gleitender Durchschnitt)")
    plt.title("Vergleich: DQN mit vs. ohne Emotionseinfluss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    else: 
        plt.savefig("results/comparison_plot.png", dpi=300)
    
    plt.pause(0.5)  # Kurze Pause, um das Plot-Fenster zu aktualisieren)
    plt.show(block=False)
    finalize_plots()

# Emotion & BDH-Dynamik Plot
def plot_emotion_bdh_dynamics(mods_history, sigma_history, save_path="results/emotion_bdh_dynamics.png"):
    """
    Zeigt, wie sich Emotion (mod) und BDH-Plastizität (|σ| Mittelwert) über das Training entwickeln.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(mods_history, label='Emotion-Modulator (mod)', color='orange')
    plt.plot(sigma_history, label='Σ-Aktivität (|σ|-Mittelwert)', color='cyan')
    plt.xlabel('Episode')
    plt.ylabel('Stärke / Aktivität')
    plt.title('Emotion & BDH-Plastizität über Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show(block=False)
    plt.pause(0.5)  # Kurze Pause, um das Plot-Fenster zu aktualisieren
    finalize_plots()

# Automatischer Vergleichsplot für mehrere Runs 
def finalize_plots(): 
    """ 
    Schließt alle offenen matplotlib-Fenster um Hängenbeliben zu verhindern.
    """
    try: 
        plt.close('all')
    except Exception: 
        pass






