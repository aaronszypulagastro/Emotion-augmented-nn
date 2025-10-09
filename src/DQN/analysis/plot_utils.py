# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import scipy.stats as stats 

# --- Hilfsfunktion für gleitenden Durchschnitt ---
def smooth(data, window=50):
    """Berechnet gleitenden Durchschnitt (Moving Average)."""
    if len(data) < window or len(data) == 0: 
        return np.array(data)       # Kein Glötten möglich, einfach zurücggeben 
    return np.convolve(data, np.ones(window)/window, mode='same')


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


def plot_emotion_bdh_dynamics(mods_history, sigma_history, sigma_activity_history=None, save_path="results/emotion_bdh_dynamics.png"):
    """
    Zeigt, wie sich Emotion (mod), BDH-Plastizität (|σ| Mittelwert)
    und σ-Amplitude (mittlere Stärke) über das Training entwickeln.
    """
    plt.figure(figsize=(10, 5))

    # Glätten für visuelle Stabilität
    mods_smooth = smooth(mods_history, window=10)
    sigmas_smooth = smooth(sigma_history, window=10)
    if sigma_activity_history is not None and len(sigma_activity_history) > 0:
        sigma_amp_smooth = smooth(sigma_activity_history, window=10)
    else:
        sigma_amp_smooth = None

    # --- Hauptkurven ---
    plt.plot(mods_smooth, label='Emotion-Modulator (mod)', color='orange', linewidth=2)
    plt.plot(sigmas_smooth, label='Σ-Aktivität (|σ|-Mittelwert)', color='cyan', linewidth=1.8)

    # --- Neue σ-Amplitude-Kurve ---
    if sigma_amp_smooth is not None:
        scale_factor = np.max(sigmas_smooth) / max(np.max(sigma_amp_smooth), 1e-6)
        plt.plot(np.array(sigma_amp_smooth) * scale_factor,
                 color='magenta', linestyle='--', linewidth=1.6,
                 label='σ-Amplitude (mittl. Stärke, skaliert)')

    # --- Formatierung ---
    plt.xlabel('Episode')
    plt.ylabel('Stärke / Aktivität')
    plt.title('Emotion & BDH-Plastizität über Training')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Speichern & Anzeigen ---
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.5)
    finalize_plots()

def plot_emotion_vs_reward(emotion_values, rewards, save_path="results/emotion_vs_reward.png"):
    """
    Zeigt Korrelation zwischen Emotion und durchschnittlichem Reward über alle Episoden.
    emotion_values: Liste oder np.array mit durchschnittlicher Emotion pro Episode
    rewards: Liste oder np.array mit Return pro Episode
    """
    if len(emotion_values) == 0 or len(rewards) == 0:
        print("[WARN] plot_emotion_vs_reward: Keine Daten zum Plotten vorhanden.")
        return
    # Sicherstellen, dass beide Listen gleich lang sind
    n = min(len(emotion_values), len(rewards))
    emotion_values = np.array(emotion_values[:n])
    rewards = np.array(rewards[:n])

    episodes = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Reward (blaue Linie, linke Achse)
    ax1.plot(episodes, smooth(rewards), color="steelblue", label="Return (smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return", color="steelblue")

    # Emotion (orange Linie, rechte Achse)
    ax2 = ax1.twinx()
    ax2.plot(episodes, smooth(emotion_values) * np.max(rewards), color="orange", alpha=0.7, label="Emotion (scaled)")
    ax2.set_ylabel("Emotion", color="orange")

    plt.title(f"Emotion vs Reward Verlauf\nØ Emotion={np.mean(emotion_values):.2f} | Ø Reward={np.mean(rewards):.1f}")
    fig.tight_layout()
    plt.grid(True)

    # Gemeinsame Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Einheitliches Format und saubere Achsen 
    fig.set_size_inches(8,5)                                # fixes Seitenverhältnis
    ax1.set_xlim(0, len(episodes))                          # x-Achse sauber auf Episoden begrenzen
    ax1.set_ylim(min(rewards) * 0.9, max(rewards) * 1.1)    # kleine Randpuffer
    ax2.set_ylim(0, 1.0)                                    # Emotion im Bereich [0, 1]

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.5)
    finalize_plots()

def plot_emotion_reward_correlation(emotions, rewards, save_path="results/emotion_reward_correlation.png"):
    """
    Erstellt ein Scatter-Plot (Emotion vs Reward) mit Regressionslinie und Korrelationskoeffizient.
    """
    if len(emotions) == 0 or len(rewards) == 0:
        print("[WARN] Keine Daten zum Plotten vorhanden.")
        return

    # Längen angleichen
    n = min(len(emotions), len(rewards))
    emotions = np.array(emotions[:n])
    rewards  = np.array(rewards[:n])

    # Korrelation berechnen
    r, p = stats.pearsonr(emotions, rewards)

    # Scatter Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(emotions, rewards, color="dodgerblue", alpha=0.6, label="Samples")

    # Regressionslinie
    slope, intercept, _, _, _ = stats.linregress(emotions, rewards)
    x_vals = np.linspace(min(emotions), max(emotions), 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="orange", linewidth=2, label="Regression")

    # Plot formatieren
    plt.title(f"Emotion ↔ Reward Korrelation\nr = {r:.3f}, p = {p:.3e}")
    plt.xlabel("Emotion")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Speichern
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Emotion-Reward-Korrelation gespeichert unter: {save_path}")

    if len(set(emotions)) <= 1:
        print("[WARN] Keine Variation in Emotion-Werten, Korrelation möglicherweise nicht aussagekräftig.")

def plot_zones(ax, zones):
    colors = {"exploration_soon":"#ffcccc","transition_zone":"#fff3cd","stabilization_soon":"#d4edda"}
    for i,z in enumerate(zones):
        ax.axvspan(i-0.5, i+0.5, color=colors.get(z,"#ffffff"), alpha=0.3)



# Automatischer Vergleichsplot für mehrere Runs 
def finalize_plots(): 
    """ 
    Schließt alle offenen matplotlib-Fenster um Hängenbeliben zu verhindern.
    """
    try: 
        plt.close('all')
    except Exception: 
        pass






