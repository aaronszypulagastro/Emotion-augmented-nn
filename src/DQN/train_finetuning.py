# -*- coding: utf-8 -*-

import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random, os, sys

# Pfad um eine Ebene erweitern (damit core/, analysis/, etc. sichtbar sind)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import csv  # für CSV-Logging

# Core-Module
from core.zone_transition_engine import ZoneTransitionEngine
from core.meta_optimizer_v2 import MetaOptimizerV2 as MetaOptimizer
from core.self_regulation_controller import SelfRegulationController, SRCConfig
from core.adaptive_zone_predictor import AdaptiveZonePredictor
from core.auto_tuner import AutoTuner
from core.emotion_engine import EmotionEngine

# Training + Analyse
from training.agent import DQNAgent
from analysis.plot_utils import (
    plot_emotion_bdh_dynamics,
    plot_emotion_reward_correlation,
    plot_emotion_vs_reward,
    finalize_plots,
    plot_zones,
    smooth,
)


# LOGS
log_path = "results/training_log.csv"
log_exists = os.path.exists(log_path)

with open(log_path, "a", newline='') as file:
    writer = csv.writer(file)
    if not log_exists:
        writer.writerow(["episode", "return", "td_error", "eta", "emotion",
                        "modulator", "sigma_mean", "momentum", "emotion_diff"
                        ])

# --- Reproduzierbarkeit ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Konfiguration ---
CONFIG = {
    'env_name': 'CartPole-v1',
    'seed': SEED,
    'episodes': 500,
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'lr': 5e-4,
    'target_update_freq': 25,  # <- engerer Update-Zyklus
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.99,     # <- leicht aggressiverer Zerfall
    'buffer_capacity': 200000,
    'emotion_enabled': True
}

# --- Umgebung & Agent ---
env = gym.make(CONFIG["env_name"])
state, _ = env.reset(seed=CONFIG['seed'])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim, config=CONFIG)

# --- Emotion-Reset am Episodenstart ---
if hasattr(agent, "emotion") and agent.emotion is not None:
    if not np.isfinite(agent.emotion.value):
        agent.emotion.state = 0.5
        agent.emotion.value = 0.5
        print("[Init-Check] Emotion reset auf neutral (0.5).")

zte = ZoneTransitionEngine(window=15, threshold=0.015)  # Meta-Regler
meta_opt = MetaOptimizer(window=30, lr=0.12)
auto_tuner = AutoTuner(target_corr=-0.3)
azp = AdaptiveZonePredictor(smoothing=0.9, sensitivity=1.4)
zone_history = []


src = SelfRegulationController(
    emotion_engine=agent.emotion,
    zte=zte,
    meta_opt=meta_opt,
    cfg=SRCConfig(
        w_emotion=1.0, w_zte=1.0, w_meta=1.0,
        eta_min=1e-5, eta_max=7e-3, gain_min=0.8, gain_max=1.6,
        blend=0.6, eta_smoothing=0.3
    )
)

# Sicherheitscheck
if agent.emotion is None: 
    from emotion_engine import EmotionEngine
    agent_emotion = EmotionEngine(
        init_state=0.4,
        alpha=0.85,
        target_return=60,
        noise_std=0.05,
        gain=1.2
        )
print(f"[INIT] Emotion Engine aktiviert: {agent.emotion is not None}")

# Logs
scores = []
mods_history, sigma_history = [], []
emotion_history, eps_eff_history, eta_history = [], [], []
td_error_history = []
sigma_activity_history = []

best_avg = -np.inf
window = 100
prev_emotion = agent.emotion.value


# Trainingsloop
for episode in tqdm(range(CONFIG['episodes']), desc="Fine-Tuning Training"):
    state, _ = env.reset()
    if not np.isfinite(state).all():
        print("[WARN] NaN im State → env reset.")
        state, _ = env.reset()
    total_reward = 0.0
    td_error_ep = []

    for t in range(CONFIG['max_steps']):
        # Aktion auswählen
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Emotion-abhängiges Reward-Shaping
        if agent.emotion_enabled:
            emotion_centered = agent.emotion.value - 0.05
            reward += 0.005 * emotion_centered
            reward *= (0.8 + 0.4 * agent.emotion.state)

        # Replay speichern
        agent.push(state, action, reward, next_state, done)
        agent.update()  # enthält Emotion.update()

        # Adaptive η
        eta = 1e-3 * (0.5 + getattr(agent.emotion, "value", 0.5))
        if not np.isfinite(eta):
            eta = 1e-3
        mod = agent._emotion_gain() if hasattr(agent, "_emotion_gain") else 1.0

        # BDH-Plastizität mit adaptivem η
        if hasattr(agent.q_network, "plasticity_step"):
            # --- Adaptive η-Regulierung (Emotion + TD-Error) ---
            base_eta = 2e-3

            # Sichere Berechnung von td_error_norm
            if len(td_error_history) > 0 and len(td_error_ep) > 0:
                td_error_norm = np.clip(
                np.mean(td_error_ep) / (np.max(td_error_history[-50:]) + 1e-6),
                0, 1
            )
            else:
                td_error_norm = 0.0
                if episode == 0:
                    print("[Info] Erste Episode – TD-Error-Normalisierung deaktiviert (noch keine History).")

            # v3: sanfter dämpfender η-Regler mit höherem Basislevel
            base_eta = 2.5e-3
            
           # --- Sicherer & NaN-resistenter TD-Error-Normalisierungs-Block ---
            if len(td_error_history) < 5 or not np.isfinite(np.mean(td_error_history[-5:])):
                # Zu wenig History oder NaN-Werte → neutraler Start
                td_smooth = np.mean(td_error_ep) if len(td_error_ep) > 0 else 0.0
                td_error_norm = 0.0
                if episode == 0:
                    print("[Init-Check] Erste Episode – TD-Error-Normalisierung deaktiviert (noch keine History).")
            else:
                # Nur berechnen, wenn genug Historie vorhanden und stabil
                td_smooth = np.mean(td_error_history[-10:])
                td_error_norm = np.clip(td_smooth / (np.mean(td_error_history[-50:]) + 1e-6), 0, 1)

            # Adaptive Dämpfung 
            emotion_factor = 0.7 + 0.6 * agent.emotion.value
            eta_raw = base_eta * emotion_factor * np.exp(-0.5 * td_error_norm)

            # Sanfte Exponential-GLättung 
            if len(eta_history) > 0:
                eta = 0.85 * eta_history[-1] + 0.15 * eta_raw
            else:
                eta = eta_raw

            eta = float(np.clip(eta, 1e-5, 7e-3)) # Lernrate reagiert träger auf kurzfristige Schwankungen


            # Emotion beeinflusst zusätzlich den Verstärkungsfaktor
            mod = 0.8 + 0.4 * agent.emotion.value   # mod  ∈ [0.8, 1.2]
            
            # σ-Homeostase
            with torch.no_grad():
                target_norm = 1.0 
                for layer in [agent.q_network.fc1, agent.q_network.fc2]:
                    sigma_norm = torch.norm(layer.sigma)
                    if sigma_norm > target_norm:
                        layer.sigma.mul_(target_norm / (sigma_norm + 1e-6))

            # BDH-Plastizität anwenden 
            decay_rate = 0.995 + 0.002 * (1 - agent.emotion.value)  # je höher Emotion, desto langsamer der Zerfall)
            agent.q_network.plasticity_step(
                mod=mod, 
                eta=eta, 
                decay=decay_rate,    
                clip=0.15       
                )

            # σ-Aktivität beleben: verhindert "eingeschlafene Synapsen"
            with torch.no_grad():
                for layer in [agent.q_network.fc1, agent.q_network.fc2]:
                    sigma_std = layer.sigma.std().item()
                    if sigma_std < 0.02:    # wenn σ zu klein wird
                        noise_boost = 0.07 + 0.06 * np.sin(episode / 45) # leichte Oszillation
                        layer.sigma.add_(torch.randn_like(layer.sigma) * noise_boost)


            # Kleine σ-Stimulation, proportional zur Emotion
            with torch.no_grad():
                noise_strength = 0.015 * agent.emotion.value
                agent.q_network.fc1.sigma.add_(
                    torch.randn_like(agent.q_network.fc1.sigma) * noise_strength
                )
        # Debug Ausgabe 
        if hasattr(agent.q_network, 'fc1') and hasattr(agent.q_network.fc2, 'sigma'): 
            sigma_mean = (
                agent.q_network.fc1.sigma.abs().mean().item()
                + agent.q_network.fc2.sigma.abs().mean().item()
            ) / 2
            if episode % 50 == 0: # nur alle 50 epeisoden 
                print(f"[σ-Debug] mean(|σ|)={sigma_mean:.3f} | "
                      f"gain={agent.emotion.gain:.2f} | "
                      f"emotion={agent.emotion.value:.3f} | "
                      f"η={eta:.5f}")    

        # Weicher σ-Reset, abhängig von Emotion
        if hasattr(agent.q_network, 'reset_plasticity'):
            decay_reset = 0.6 + 0.3 * agent.emotion.value   # je höher Emotion, desto mehr σ bleibt
            agent.q_network.fc1.sigma.mul_(decay_reset)
            agent.q_network.fc2.sigma.mul_(decay_reset)

        # TD-Error grob schätzen (für Logging)
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            s_next = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            q_t = agent.q_network(s_t)
            q_next = agent.target_network(s_next).max(dim=1, keepdim=True)[0]
            td_error = float((reward + CONFIG["gamma"] * q_next - q_t.max()).abs().item())
            td_error_ep.append(td_error)

        # NaN - Guradian 
        if not np.isfinite(td_error):
            print(f"[WARN] NaN/Inf in TD-Error erkannt → Reset.")
            td_error = 0.0

        if not np.isfinite(total_reward):
            print(f"[WARN] NaN/Inf im Reward → Reset.")
            total_reward = 0.0

        if hasattr(agent, "emotion") and agent.emotion is not None:
            if not np.isfinite(agent.emotion.value):
                print("[WARN] Emotion value NaN → Reset auf neutral (0.5)")
                agent.emotion.state = 0.5
                agent.emotion.value = 0.5

        state = next_state
        total_reward += reward
        if done:
            break

    # --- Episodenende ---
    agent.emotion.update_after_episode(total_reward)
    avg_td_error = np.mean(td_error_ep) if td_error_ep else 0.0

    # Zentrale Koordination (Phase 5)
    eta = src.step(reward_ep=total_reward, td_error_ep=avg_td_error, eta=eta)
    stat = src.status()  # {zone_pred, trend, gain}
    if episode % 10 == 0:
        print(f"[SRC] zone={stat['zone_pred']} | trend={stat['trend']:+.3f} | gain={stat['gain']:.3f} | η={eta:.5f}")

    # Zone Transition Engine Update
    zone_pred = zte.update(agent.emotion.value, avg_td_error, eta)
    if zone_pred is not None:
        zte.apply_to_emotion_engine(agent.emotion)
        print(f"[ZTE] -> zone_pred={zone_pred} | emotion={agent.emotion.value:.3f} | td_err={avg_td_error:.3f} | η={eta:.5f}")
    
    # Meta Optimizer 
    meta_feedback = meta_opt.update(total_reward, eta, agent.emotion.gain, agent.emotion.value)
    if meta_feedback is not None:
        eta = meta_feedback['eta']
        agent.emotion.gain = meta_feedback['gain']
        if episode % 10 == 0:
            print(f"[MetaOpt] trend={meta_feedback['trend']:+.3f} | η→{eta:.5f} | gain→{agent.emotion.gain:.3f}")
    
    # --- Phase 5.7: AutoTuner Routine ---
    auto_tuner.record(avg_td_error, eta)
    new_lr, corr = auto_tuner.tune(meta_opt)
    if new_lr != meta_opt.lr:
        print(f"[AutoTuner] lr angepasst: {meta_opt.lr:.3f} → {new_lr:.3f} (corr={corr:.2f})")
        meta_opt.lr = new_lr

    if not auto_tuner.active and episode % 20 == 0:
        print(f"[AutoTuner] abgeschlossen – stabile η-Korrelation erreicht.")


    # Reward-Trend über gleitendes Fenster
    reward_trend = np.mean(scores[-20:]) if len(scores) > 20 else np.mean(scores)
    azp_info = azp.step(agent.emotion.value, avg_td_error, reward_trend)

    if episode % 25 == 0:
        print(f"[AZP] zone={azp_info['zone_pred']} | conf={azp_info['confidence']:.3f} | int={azp_info['intensity']:.3f}")
    zone_history.append(azp_info['zone_pred'])

    # Adaptive Zonenreaktion 
    if hasattr(agent, "emotion") and hasattr(agent.emotion, "apply_zone_response"):
        agent.emotion.apply_zone_response(agent.emotion.value, avg_td_error)
    
    # --- Phase 5.6 : Self-Recovery Loop ---
    if np.isnan(agent.emotion.value) or np.isnan(td_error_norm):
        print("[Self-Recovery] NaN erkannt → Soft Reset")
        agent.emotion.value = 0.5
        eta = 1e-3
    elif td_error_norm > 2.0:
        eta *= 0.9  # starke Fehler → η leicht senken
    elif agent.emotion.value < 0.4:
        eta *= 1.1  # Emotion zu niedrig → etwas aggressiver lernen

    # Emotion-Differenz und Momentum für Logging
    emotion_diff = abs(agent.emotion.value - prev_emotion)
    prev_emotion = agent.emotion.value
    momentum = getattr(agent.emotion, "_momentum", 0.0)

    # Logging
    emotion_history.append(agent.emotion.value)
    eps_eff_history.append(agent._last_eps)
    td_error_history.append(avg_td_error)
    eta_history.append(eta)
    mods_history.append(mod)

    
    # σ-Statistik und Aktivität für Plot
    if hasattr(agent.q_network, "fc1") and hasattr(agent.q_network.fc2, "sigma"):
        sigma_mean = (
            agent.q_network.fc1.sigma.abs().mean().item()
            + agent.q_network.fc2.sigma.abs().mean().item()
        ) / 2
        sigma_history.append(sigma_mean)
        sigma_activity_history.append(sigma_mean)
    else:
        sigma_history.append(0.0)
        sigma_activity_history.append(0.0)


    # Weiches σ-Reset
    if hasattr(agent.q_network, "reset_plasticity"):
        agent.q_network.fc1.sigma.mul_(0.5)
        agent.q_network.fc2.sigma.mul_(0.5)

    scores.append(total_reward)

    # Debug-Prints
    print(f"[Ep {episode+1:03d}] Return={total_reward:6.1f} | "
          f"Emotion={agent.emotion.value:.3f} | η={eta:.5f} | "
          f"TD-Err={avg_td_error:.3f} | mod={mod:.3f} | eps={agent._last_eps:.3f} |  σ̄={sigma_mean:.4f}")

    # LOgging in CSV für spätere Analyse 
    if episode % 10 == 0: # nur alle 10 Episoden
        os.makedirs("results", exist_ok=True)
        csv_path = "results/training_log.csv"
        # falls Datei noch nicht existiert → Header hinzufügen
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("episode,reward,emotion,eta,td_error,sigma_mean\n")
        
        trend_val = meta_feedback['trend'] if meta_feedback is not None else 0.0
        zone_val  = stat['zone_pred'] if stat and stat['zone_pred'] is not None else "None"
        with open(csv_path, "a") as f:
            f.write(f"{episode+1},{total_reward:.2f},{agent.emotion.value:.3f},{eta:.5f},"
            f"{avg_td_error:.3f},{sigma_mean:.4f},{momentum:.4f},{emotion_diff:.4f},"
            f"{zone_val},{trend_val:.4f}\n")

    # Durchschnitt über 100 Episoden
    if (episode + 1) % 10 == 0:
        avg100 = np.nanmean(scores[-window:])
        print(f"→ Ø100: {avg100:6.2f}")
        if avg100 > best_avg:
            best_avg = avg100
            torch.save(agent.q_network.state_dict(), "dqn_finetune_best.pth")

env.close()

# --- Visualisierung ---
os.makedirs("results", exist_ok=True)

plot_emotion_bdh_dynamics(mods_history, sigma_history, sigma_activity_history)
plot_emotion_reward_correlation(emotion_history, scores)
plot_emotion_vs_reward(emotion_history, scores)

# TD-Error vs Emotion Plot
plt.figure(figsize=(7,5))
plt.plot(smooth(td_error_history, 10), label="TD-Error (smoothed)", color="red")
plt.plot(smooth(emotion_history, 10), label="Emotion", color="orange", alpha=0.7)
plt.title("Emotion ↔ TD-Error Verlauf")
plt.xlabel("Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/td_error_vs_emotion.png", dpi=200)

# η-Verlauf vs Emotion
plt.figure(figsize=(7,5))
plt.plot(smooth(eta_history, 10), label="η (adaptive)", color="blue")
plt.plot(smooth(emotion_history, 10), label="Emotion", color="orange", alpha=0.7)
plt.title("Adaptive η ↔ Emotion Verlauf")
plt.xlabel("Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/eta_vs_emotion.png", dpi=200)

# η vs TD-Error Plot
if len(eta_history) > 0 and len(td_error_history) > 0:
    plt.figure(figsize=(7,5))
    n = min(len(eta_history), len(td_error_history))
    etas = np.array(eta_history[:n])
    td_errors = np.array(td_error_history[:n])

    # Streuung + Regressionslinie
    plt.scatter(td_errors, etas, alpha=0.6, color="purple", label="Samples")
    if np.std(td_errors) > 1e-6:
        coeffs = np.polyfit(td_errors, etas, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(min(td_errors), max(td_errors), 100)
        plt.plot(x_fit, fit_line(x_fit), color="magenta", linewidth=2, label="Regression")

    plt.title("η (adaptive) ↔ TD-Error\nRegleranalyse: Lernrate vs Fehler")
    plt.xlabel("TD-Error (average per episode)")
    plt.ylabel("η (adaptive)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/eta_vs_td_error.png", dpi=200)
    print("[Plot] η vs TD-Error gespeichert unter results/eta_vs_td_error.png")


# Dashboard Plot: Überblick über Agent-Dynamik
if len(td_error_history) > 0 and len(emotion_history) > 0 and len(eta_history) > 0:
    plt.figure(figsize=(12, 8))

    episodes = np.arange(len(td_error_history))

    # TD-Error
    plt.plot(episodes, smooth(td_error_history), color="red", label="TD-Error (smoothed)", linewidth=1.8)

    # Emotion
    plt.plot(episodes, smooth(emotion_history), color="orange", label="Emotion", linewidth=1.6, alpha=0.8)

    # η (adaptive)
    plt.plot(episodes, smooth(eta_history), color="blue", label="η (adaptive)", linewidth=1.4, alpha=0.8)

    # Optional: Reward falls du reward_history mitloggst
    if "reward_history" in locals():
        plt.plot(episodes[:len(reward_history)], smooth(reward_history), color="green", label="Reward", linewidth=1.5, alpha=0.7)

    plt.title("Agent Dynamics Dashboard: Emotion • TD-Error • η • Reward")
    plt.xlabel("Episode")
    plt.ylabel("Intensity / Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/dashboard_agent_dynamics.png", dpi=200)
    print("[Plot] Dashboard gespeichert unter results/dashboard_agent_dynamics.png")

finalize_plots()
print("\nFine-Tuning abgeschlossen – Ergebnisse unter ./results gespeichert.")
print(f"Durchschnittliche Emotion: {np.mean(emotion_history):.3f}")
print(f"Durchschnittlicher TD-Error: {np.mean(td_error_history):.3f}")