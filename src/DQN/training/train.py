# -*- coding: utf-8 -*-
# IMPORTS
from weakref import finalize
import torch
import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt
import random 
import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Pfad zu src hinzufügen
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Arbeitsverzeichnis setzen


from agent import DQNAgent, DQNConfig   # DQNAgent aus agent.py importieren
from tqdm import tqdm
from plot_utils import plot_emotion_reward_correlation, plot_single_run, plot_comparison, finalize_plots, plot_emotion_bdh_dynamics, plot_emotion_vs_reward, smooth
from emotion_engine import EmotionEngine

# Reproduzierbarkeit sicherstellen
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CONFIG = {
    'env_name': 'CartPole-v1',
    'seed': SEED,
    'episodes': 150,                # Max. Episoden
    'max_steps': 500,               # Max. Schritte pro Episode
    'batch_size': 64,               # Mini-Batch Größe
    'buffer_capacity': 200000,      # Replay-Puffer Kapazität
    'gamma': 0.99,                  # Diskontfaktor
    'lr': 5e-4,                     # Lernrate
    'target_update_freq': 50,       # Zielnetzwerk Update Frequenz (in Episoden)
    'epsilon_start': 1.0,           # Startwert für Epsilon (Exploration)
    'epsilon_end': 0.05,            # Endwert für Epsilon
    'epsilon_decay': 600         # Zerfallsrate für Epsilon
}

# Emotion Engine aktivieren/deaktivieren
CONFIG['emotion_enabled'] = True  # True = mit Emotion, False = ohne Emotion
# Evironment erstellen 
env = gym.make(CONFIG['env_name'])
state, _ = env.reset(seed=CONFIG['seed'])  # Umgebung zurücksetzen

state_dim = env.observation_space.shape[0]  # Zustandsdimension
action_dim = env.action_space.n             # Aktionsdimension

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=CONFIG)

scores = []             # Für die Episoden-Scores
scores_baseline = []    # Für Vergleich ohne Emotion
scores_emotion = []     # Für Vergleich mit Emotion

best_avg = -np.inf      # Beste durchschnittliche Belohnung
window = 100            # Fenstergröße für den gleitenden Durchschnitt

sigma_history = []      # Verlauf des EMotions Modulators (gain_Werte)
mods_history = []       # Verlauf der mittleren BDH Synapsenaktivität

# Trainingsschleife
for episode in tqdm(range(CONFIG['episodes']), desc="Training"):
    state, _ = env.reset()            # Umgebung zurücksetzen
    total_reward = 0.0                # Gesamtreward pro Episode
    

    for t in range(CONFIG['max_steps']):
        # Aktion vom Agenten auswählen
        action = agent.act(state)

        # Aktion in der Umgebung ausführen
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # === Reward Shaping mit Emotion ===
        if agent.emotion_enabled: 
            emotion_centered = agent.emotion.value - 0.05
        # Kleine Anpassung des Rewards basierend auf der Emotion
            reward += 0.005 * emotion_centered  
            
        # EMotionale Verstärkung des Rewards 
            reward *= (0.8 + 0.4 * agent.emotion.state)

        agent.push(state, action, reward, next_state, done)  # Erfahrung speichern
        agent.update()                                       # intern: multiple updates + soft target 

        # BDH / Plasticity Update 
        if hasattr(agent.q_network, 'plasticity_step'): 
            # Emotion modifizert den Lernverstärkunsfaktor (mod)
            mod = 1.0 
            if agent.emotion_enabled:
                mod = agent._emotion_gain() # nutzt emotion_gain Funktion 

            # Aufruf: Synapsen-Plastizität auf beiden LAyern
            agent.q_network.plasticity_step(
                mod=mod, 
                eta=1e-3,          # Lernrate auf beiden LAyern 
                decay=0.997,       # Leck-Term, verhindert Instabilitöt 
                clip=0.1           # Begrenzung gegen Explosion 
                ) 

        # Zustand aktualisieren
        state = next_state
        total_reward += reward

        # Abbruchbedingung
        if done:
            break

    # === EMOTION_UPDATE ===
        # Logging: EMotion + BDH Modulation 
        mod = 1.0 
        if hasattr(agent, '_emotion_gain'):
            mod = agent._emotion_gain()

        if getattr(agent, 'emotion', None) is not None: 
            print(f'Episode {episode+1:>4}: Return={total_reward:>6.1f} | '
            f'Emotion={agent.emotion.value:.3f} | mod={mod:.3f} | ' 
            f'e_eff={agent._last_eps:.3f}') 
    else:
        print(f'Episode {episode+1:>4}: Return={total_reward:>6.1f} | '
              f'(keine EmotionEngine aktiv) | e_eff={agent._last_eps:.3f}') 

    # Logging für BDH + Emotion-PLot 
    if agent.emotion_enabled: 
        mods_history.append(mod)
    else: 
        mods_history.append(1.0)

    # mittlere Sigma-Aktivität der BDH-Layer 
    if hasattr(agent.q_network, 'fc1') and hasattr(agent.q_network.fc1, 'sigma'):
        sigma_norm = torch.mean(agent.q_network.fc1.sigma.abs()).item()
        sigma_history.append(sigma_norm)
    else:
        sigma_history.append(0.0)

    # Nach Episode: Reset der BDH-Synapsenaktivität 
    if hasattr(agent.q_network, 'reset_plasticity'):
        agent.q_network.reset_plasticity()

    # Nach Episode: Emotion aktualisieren
    if agent.emotion_enabled:                              
        agent.emotion.update_after_episode(total_reward)

        # Debug: Emotion-Wert und Modulator ausgeben
        if agent.emotion_enabled and (episode +1) % 10 == 0: 
            print(f'[EmotionEngine] Emotion value = {agent.emotion.value:.3f}')

    # Episoden-Score speichern 
    try: 
        tr = float(total_reward)
    except:
        tr = float('nan')
    if not np.isfinite(tr):
        tr = float('nan')
    
    scores.append(tr)

    # === Logs für Vergleichsplot ===
    if episode == 0:
        emotion_history = []
        eps_eff_history = []

    if agent.emotion_enabled:
        emotion_history.append(agent.emotion.value)
    else:
        emotion_history.append(0.0)

    # _last_eps wird in act() gesetzt; Fallback auf agent.epsilon
    eps_eff_history.append(getattr(agent, '_last_eps', agent.epsilon))

    # Average Score der letzten 100 Episoden ausgeben
    if (episode + 1) % 10 == 0:
        if len(scores) > 0:  
            avg100 = np.nanmean(scores[-window:])  # NaN-sicherer Mittelwert
        else:
            avg100 = float('nan')

        if np.isnan(avg100):
            print(f'Episode {episode+1:4d} | avg100: ---     | eps: {agent.epsilon:.3f}')
        else:
            print(f'Episode {episode+1:4d} | avg100: {avg100:6.1f} | eps: {agent.epsilon:.3f}')

        # Bestes Modell speichern
        if not np.isnan(avg100) and (avg100 > best_avg):
            best_avg = avg100
            torch.save(agent.q_network.state_dict(), 'dqn_cartpole_best.pth')

        # Early stop (CartPole gelöst bei ~475)
        if not np.isnan(avg100) and avg100 >= 475.0 and len(scores) >= window:
            print(f"Lösung gefunden in Episode {episode+1} mit avg100: {avg100:0.1f}")
            break  # bleibt gültig, solange diese Zeile innerhalb der Episoden-Schleife eingerückt ist

env.close()

# NUR beim Emotion-Run:
#scores_emotion = scores.copy()  

# NUR beim Baseline-Run:
#scores_baseline = scores.copy()

# Automatische Erkennung des Run-Typs
if len(emotion_history) > 0:
    print('EmotionEngine aktiv erkannt - Plot mit Emotionen wird erstellt.')
else: 
    print('Keine Emotionseinfluss - Baseline-PLot wird erstellt.')

# Safe-Save Ergebnisse
os.makedirs('results', exist_ok=True)

# Debug-Ausgaben
print('DEBUG len(scores) =', len(scores))
print('DEBUG len(eps_eff_history) =', len(eps_eff_history))
print('DEBUG last 5 scores =', scores[-5:] if len(scores) >= 5 else scores)
print('DEBUG any NaN in scores? ->', any(not np.isfinite(x) for x in scores))

# Emotion & BDH-Dynamik Plot
if agent.emotion_enabled:
    from plot_utils import plot_emotion_bdh_dynamics
    plot_emotion_bdh_dynamics(
        mods_history, 
        sigma_history,
        save_path="results/emotion_bdh_dynamics.png"
    )

# Emotion-Reward-Korrelation
plot_emotion_reward_correlation(emotion_history, scores) 

# Emotion vs Reward Plot
if 'mods_history' in locals() and 'scores' in locals():
    plot_emotion_vs_reward(mods_history, scores)

# Statistik ausgeben
print(f"Durchschnittliche Emotion: {np.mean(emotion_history):.3f}")
print("Letzte 10 Emotion-Werte:", emotion_history[-10:])
print("Letzte 10 Epsilons:", eps_eff_history[-10:]) 

finalize_plots()
print('Training abgeschlossen und alle Plots gespeichert.')
