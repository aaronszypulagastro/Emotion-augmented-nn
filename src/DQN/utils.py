# -*- coding: utf-8 -*-

from urllib.parse import _ResultMixinStr
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import numpy as np
from src.agent import DQNAgent  # Agenten importieren

def evaluate_and_plot(model_path='dqn_cartpole_final.pth', episodes=10, render=False):
    

    # Environment erstellen
    env = gym.make("CartPole-v1", render_mode="human")  # "human" = mit Animation

    # Gleiche Dimensionen wie im Training
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent erstellen
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, batch_size=128)

    # Trainiertes Modell laden (z. B. das finale Modell)
    agent.q_network.load_state_dict(torch.load("dqn_cartpole_final.pth", map_location=agent.device))
    agent.q_network.eval()
    agent.epsilon = 0.0  # Kein Epsilon-Greedy mehr

    scores = []
    # Evaluation starten
    num_episodes = 5
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
    

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        scores.append(total_reward)
        print(f"Episode {ep+1}: Score = {total_reward}")

    env.close()

    # Plot erstellen
    plt.figure(figsize=(8,5))
    plt.plot(scores, marker='o', label="Score")
    plt.axhline(y=np.mean(scores), color='r', linestyle='--', label=f"Durchschnitt {np.mean(scores):.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Ergebnisse (Greedy Policy)")
    plt.legend()
    plt.grid(True)
    plt.savefig("eval_results.png")
    plt.close()

    print(f"Durchschnitt über {episodes} Episoden: {np.mean(scores):.2f}")
    print("Ergebnisse als eval_results.png gespeichert.")

    return scores

def compare_models(model_paths, episodes=5, render=False):
    """
    Vergleicht mehrere trainierte Modelle über eine Anzahl von Episoden.
    model_paths: Liste von Pfaden zu den Modellen.
    episodes: Anzahl der Episoden pro Modell.
    """

    results = {}

    for path in model_paths:
        print(f"Evaluating model: {path}")
        scores = evaluate_and_plot(path, episodes=episodes, render=render)
        results[path] = np.mean(scores)

        # Dynamischer Dateiname für den Plot
        model_name = path.split('_')[-1].replace('.pth','')
        plt.figure(figsize=(8, 5))
        plt.plot(scores, marker='o', label=f"{model_name}")
        plt.axhline(np.mean(scores), color='r', linestyle='--', label=f"Durchschnitt {np.mean(scores):.2f}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Evaluation {model_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"eval_results_{model_name}.png")
        plt.close()


    # Vergleichsplot erstellen
    plt.figure(figsize=(8,5))
    plt.bar(range(len(results)), results.values(),
            tick_label=[p.split('_')[-1].replace('.pth','') for p in results.keys()])
    plt.ylabel("Durchschnittlicher Reward")
    plt.title("Modellvergleich")
    plt.savefig("compare_models.png")
    plt.close()

    print('\nVergleich abgeschlossen - Ergebnisse in compare_models.png gespeichert.')
    return results
   
