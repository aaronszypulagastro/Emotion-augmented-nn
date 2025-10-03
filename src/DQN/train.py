# -*- coding: utf-8 -*-
# Plan: 
# 1. Imports einfügen
# 2. Hyperparameter definieren
# 3. Trainingsschleife implementieren
# 4. Logging und Modell-Speicherung hinzufügen
# 5. Testen und Validieren
# 6. Dokumentation und Kommentare ergänzen


# IMPORTS
import torch
import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt
import random 

from agent import DQNAgent # DQNAgent aus agent.py importieren
from tqdm import tqdm

# Reproduzierbarkeit sicherstellen
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Evironment erstellen 
env = gym.make('CartPole-v1')
state, _ = env.reset(seed=SEED)  # Umgebung zurücksetzen

# Hyperparameter
num_episodes = 600  # Anzahl der Trainings-Episoden
max_steps = 500     # Maximale Schritte pro Episode
batch_size = 128    # Batch-Größe für das Training
state_dim = env.observation_space.shape[0]  # Zustandsdimension
action_dim = env.action_space.n             # Aktionsdimension

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size)

scores = []         # Liste zur Speicherung der Episoden-Scores
best_avg = -np.inf  # Beste durchschnittliche Belohnung
window = 100        # Fenstergröße für den gleitenden Durchschnitt



# Trainingsschleife
for episode in tqdm(range(num_episodes), desc="Training"):
    state, _ = env.reset()            # Umgebung zurücksetzen
    total_reward = 0.0                # Gesamtreward pro Episode
    

    for t in range(max_steps):
        # Aktion vom Agenten auswählen
        action = agent.act(state)

        # Aktion in der Umgebung ausführen
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.push(state, action, reward, next_state, done)  # Erfahrung speichern
        agent.update()                                       # intern: multiple updates + soft target 

        # Zustand aktualisieren
        state = next_state
        total_reward += reward

        # Abbruchbedingung
        if done:
            break

    # Episoden-Score speichern
    scores.append(total_reward)
    print(f"Episode {episode+1}/{num_episodes}, Score: {total_reward}")
        

    # Average Score der letzten 100 Episoden ausgeben
    if (episode + 1) % 10 == 0:
        avg100 = np.mean(scores[-window:])
        print(f'Episode {episode+1:4d} | avg100: {avg100:6.1f} | eps: {agent.epsilon:.3f}')
        
        # Das beste Modell speichern
        if avg100 > best_avg:
            best_avg = avg100
            torch.save(agent.q_network.state_dict(), 'dqn_cartpole_best.pth')
        # Early stop: CartPole gitl als gelöst bei ~475
        if avg100 >= 475.0 and len(scores) >= window:
            print(f"Lösung gefunden in Episode {episode+1} mit avg100: {avg100:.1f}")
            break


# Ergebnisse plotten
plt.plot(scores, label="Score pro Episode")
if len(scores) >= window:
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(scores)), moving_avg, label=f"Gleitender Durchschnitt ({window})", color='orange')

plt.xlabel("Episode"); 
plt.ylabel("Return"); 
plt.legend(); 
plt.tight_layout(); 
plt.show()

# End of Training: Modell speichern
env.close()
torch.save(agent.q_network.state_dict(), 'dqn_cartpole_final.pth')
print("Training beendet und Modell bei 'dqn_cartpole_final.pth' gespeichert.")
