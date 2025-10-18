# -*- coding: utf-8 -*-

import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random, os, sys
from tqdm import tqdm
import csv # für CSV-Logging
 
try:
    # Paketstart (aus src): DQN.training.train_finetuning
    from ..core.zone_transition_engine import ZoneTransitionEngine
    from ..core.meta_optimizer_v2 import MetaOptimizerV2 as MetaOptimizer
    from ..core.self_regulation_controller import SelfRegulationController, SRCConfig  # type: ignore
    from ..core.adaptive_zone_predictor import AdaptiveZonePredictor
    from ..core.adaptive_zone_predictor_v2 import AdaptiveZonePredictorV2
    from ..core.auto_tuner import AutoTuner
    from ..core.emotion_predictive_regulation_unit import EmotionPredictiveRegulationUnit
    from ..core.emotion_curriculum_learning import EmotionCurriculumLearning, ECLConfig
    from ..core.multi_objective_optimizer import MultiObjectiveOptimizer, MOOConfig
except ImportError:
    # Skriptstart (aus src\DQN): training/train_finetuning.py
    import os as _os, sys as _sys
    _CURR = _os.path.dirname(_os.path.abspath(__file__))
    _PKG_ROOT = _os.path.abspath(_os.path.join(_CURR, ".."))      # .../src/DQN
    _SRC_ROOT = _os.path.abspath(_os.path.join(_CURR, "..", ".."))  # .../src
    for _p in (_PKG_ROOT, _SRC_ROOT):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    from core.zone_transition_engine import ZoneTransitionEngine
    from core.meta_optimizer_v2 import MetaOptimizerV2 as MetaOptimizer
    from core.self_regulation_controller import SelfRegulationController, SRCConfig  # type: ignore
    from core.adaptive_zone_predictor import AdaptiveZonePredictor
    from core.adaptive_zone_predictor_v2 import AdaptiveZonePredictorV2
    from core.auto_tuner import AutoTuner
    from core.emotion_predictive_regulation_unit import EmotionPredictiveRegulationUnit
    from core.emotion_curriculum_learning import EmotionCurriculumLearning, ECLConfig
    from core.multi_objective_optimizer import MultiObjectiveOptimizer, MOOConfig
    from core.performance_stability_analyzer import PerformanceStabilityAnalyzer  # Phase 7.0 Option B

# Lokale Imports
try:
    from .agent import DQNAgent
    from ..analysis.plot_utils import (
        plot_emotion_bdh_dynamics,
        plot_emotion_reward_correlation,
        plot_emotion_vs_reward,
        finalize_plots,
        plot_zones,
        smooth,
        plot_zones
    )
except ImportError:
    from training.agent import DQNAgent
    from analysis.plot_utils import (
        plot_emotion_bdh_dynamics,
        plot_emotion_reward_correlation,
        plot_emotion_vs_reward,
        finalize_plots,
        plot_zones,
        smooth,
        plot_zones
    )

# LOGS
log_path = "results/training_log.csv"
log_exists = os.path.exists(log_path)

with open(log_path, "a", newline='') as file:
    writer = csv.writer(file)
    if not log_exists:
        writer.writerow(["episode", "return", "td_error", "eta", "emotion",
                        "modulator", "sigma_mean", "momentum", "emotion_diff", "zone", "trend", "gate", "eta_cap",
                        "n_step", "beta_current", "priority_weight", "epru_confidence", "epru_adjustment", "epru_intervention",
                        "azpv2_confidence", "azpv2_intensity", "azpv2_zone_pred",
                        "ecl_difficulty", "ecl_phase", "ecl_progress", "ecl_stability",
                        "moo_performance_score", "moo_eta_stability", "moo_sigma_health", "moo_predicted_perf"
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
    'episodes': int(os.environ.get('EPISODES', 500)),
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
meta_opt = MetaOptimizer(window=30, lr=0.102)
auto_tuner = AutoTuner(target_corr=-0.3)

# Phase 6.1: AdaptiveZonePredictor v2 (AZPv2)
azp = AdaptiveZonePredictorV2(
    sequence_length=10,
    hidden_size=32,
    num_layers=2,
    confidence_threshold=0.7,
    learning_rate=0.001,
    history_size=100
)

zone_history = []

# Phase 6.0: Emotion-Predictive Regulation Unit (EPRU)
epru = EmotionPredictiveRegulationUnit(
    emotion_horizon=5,
    td_horizon=3,
    confidence_threshold=0.7,
    intervention_strength=0.3,
    history_size=50
)

# Phase 6.2: Emotion-basiertes Curriculum Learning (ECL)
ecl = EmotionCurriculumLearning(
    config=ECLConfig(
        emotion_threshold_low=0.3,
        emotion_threshold_high=0.7,
        td_error_threshold_low=0.5,
        td_error_threshold_high=1.5,
        progress_window=20,
        progress_threshold=0.1,
        min_difficulty=0.1,
        max_difficulty=1.0,
        difficulty_step=0.05,
        stability_window=10,
        stability_threshold=0.8,
        reward_scaling_factor=0.1,
        action_noise_factor=0.05,
        state_noise_factor=0.02
    )
)

# Phase 6.3: Multi-Objective Optimization (MOO)
moo = MultiObjectiveOptimizer(
    config=MOOConfig(
        w_performance=0.4,
        w_eta_stability=0.3,
        w_sigma_health=0.3,
        performance_window=20,
        performance_target=50.0,
        eta_stability_weight=0.1,
        eta_efficiency_weight=0.2,
        sigma_health_weight=0.15,
        sigma_plasticity_weight=0.1,
        adaptation_rate=0.01,
        adaptation_window=50,
        pareto_alpha=0.1,
        pareto_beta=0.9,
        prediction_horizon=10,
        prediction_confidence=0.7
    )
)

src = SelfRegulationController(
    emotion_engine=agent.emotion,
    zte=zte,
    meta_opt=meta_opt,
    epru=epru,  # Phase 6.0: EPRU-Integration
    cfg=SRCConfig(
        w_emotion=1.0, w_zte=1.0, w_meta=1.0, w_epru=0.8,
        eta_min=1e-5, eta_max=7e-3, gain_min=0.8, gain_max=1.6,
        blend=0.6, eta_smoothing=0.3,
        epru_confidence_threshold=0.7,
        epru_intervention_strength=0.3
    )
)

# Sicherheitscheck
if agent.emotion is None: 
    from core.emotion_engine import EmotionEngine
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

# --- EMA-Zustände für effizientere, stabile Normalisierung/Glättung ---
td_fast = 0.0   # schnelle EMA des TD-Errors (≈ kurzes Fenster)
td_slow = 0.0   # langsame EMA des TD-Errors (≈ langes Fenster)
vol_ema = 0.0   # Volatilitäts-EMA von eta_raw
alpha_fast = 0.15
alpha_slow = 0.03
alpha_eta_min, alpha_eta_max = 0.75, 0.90
c_vol = 10.0
eps = 1e-6
eta_prev = 1e-3

best_avg = -np.inf
window = 100
prev_emotion = agent.emotion.value

# Phase 7.0 - Option B: Performance Stability Analyzer
psa = PerformanceStabilityAnalyzer(
    window_size=100,
    anomaly_threshold=3.0,
    trend_threshold=0.3
)
print("📊 [Phase 7.0 Option B] Performance Stability Analyzer aktiviert")

# Trainingsloop
for episode in tqdm(range(CONFIG['episodes']), desc="Fine-Tuning Training"):
    # Phase 6.0: Episode-Count für EPRU setzen
    src.set_episode_count(episode)
    
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

            # --- Robust-Quantile-Normalisierung (rauschresistent) ---
            if len(td_error_history) >= 32:
                p50 = float(np.quantile(td_error_history[-200:], 0.5))
                p90 = float(np.quantile(td_error_history[-200:], 0.9))
                td_error_norm = float(np.clip(p50 / (p90 + 1e-6), 0.0, 1.0))
            else:
                # Fallback bei geringer Historie
                td_error_norm = float(np.clip(
                    (np.mean(td_error_ep) / (np.max(td_error_history[-50:]) + 1e-6))
                    if len(td_error_history) > 0 and len(td_error_ep) > 0 else 0.0,
                    0.0, 1.0
                ))

            # v3: sanfter dämpfender eta-Regler mit höherem Basislevel
            # Phase 6.1 Original-Werte (funktionierten mit 40.05!)
            base_eta = 2.5e-3

            # Proportionaler η-Vorschlag
            emotion_factor = 0.7 + 0.6 * agent.emotion.value
            eta_prop = base_eta * emotion_factor * np.exp(-0.5 * td_error_norm)

            # --- Deadband-Hysterese + Volatilitäts-adaptive Glättung ---
            eta_prev = eta_history[-1] if len(eta_history) > 0 else eta_prop
            deadband = 0.15 * max(eta_prev, 1e-6)  # 15% um η_prev

            if abs(eta_prop - eta_prev) < deadband:
                eta_raw = eta_prev
            else:
                eta_raw = eta_prop

            # Volatilität aus letzter η-Differenz (einfaches Maß)
            vol = abs(eta_raw - eta_prev)
            alpha_low, alpha_high = 0.05, 0.20  # Trägheitsspanne
            alpha = alpha_high if vol < 0.25 * deadband else alpha_low

            eta = alpha * eta_raw + (1.0 - alpha) * eta_prev

            # Anti-Windup pro Schritt (begrenzt Änderungsrate)
            eta = float(np.clip(eta, 0.8 * eta_prev, 1.25 * eta_prev))
            # Phase 6.1 Original-Bounds
            eta = float(np.clip(eta, 1e-5, 7e-3))
            eta_prev = eta


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
            if episode % 10 == 0: # nur alle 10 epeisoden 
                print(f"[sigma-Debug] mean(|sigma|)={sigma_mean:.3f} | "
                      f"gain={agent.emotion.gain:.2f} | "
                      f"emotion={agent.emotion.value:.3f} | "
                      f"eta={eta:.5f}")    

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
    
    # Phase 6.2: ECL - Schwierigkeitsanpassung basierend auf Episode
    avg_td_error = np.mean(td_error_ep) if td_error_ep else 0.0
    ecl_difficulty = ecl.update_difficulty(
        emotion=agent.emotion.value,
        td_error=avg_td_error,
        reward=total_reward
    )
    
    # Phase 6.3: MOO - Multi-Objective Optimization
    sigma_mean = np.mean([layer.sigma.mean().item() for layer in [agent.q_network.fc1, agent.q_network.fc2] if hasattr(layer, 'sigma')])
    moo_optimization = moo.optimize_parameters(
        current_eta=eta,
        current_sigma_mean=sigma_mean,
        current_performance=total_reward
    )
    
    # MOO-Training (nur bei ausreichender Historie)
    moo.train_prediction_model()
    
    # N-Step Flush am Episodenende (für PRB + N-Step-Returns)
    if hasattr(agent, "on_episode_end"):
        agent.on_episode_end()

    # --- Nur für Regler: TD-Error-Clipping (Lernen bleibt unverändert) ---
    td_clip = 2.0
    td_error_ctrl = float(np.clip(avg_td_error, -td_clip, td_clip))

    # Zentrale Koordination (Phase 5) mit geclipptem Regler-Signal
    eta = src.step(reward_ep=total_reward, td_error_ep=td_error_ctrl, eta=eta)
    stat = src.status()  # {zone_pred, trend, gain}
    if episode % 10 == 0:
        print(f"[SRC] zone={stat['zone_pred']} | trend={stat['trend']:+.3f} | gain={stat['gain']:.3f} | eta={eta:.5f}")

    # Zone Transition Engine Update
    zone_pred = zte.update(agent.emotion.value, td_error_ctrl, eta)
    if zone_pred is not None:
        zte.apply_to_emotion_engine(agent.emotion)
        print(f"[ZTE] -> zone_pred={zone_pred} | emotion={agent.emotion.value:.3f} | td_err={avg_td_error:.3f} | eta={eta:.5f}")
    
    # Meta Optimizer 
    meta_feedback = meta_opt.update(total_reward, eta, agent.emotion.gain, agent.emotion.value)
    if meta_feedback is not None:
        eta = meta_feedback['eta']
        agent.emotion.gain = meta_feedback['gain']
        if episode % 10 == 0:
            react = meta_feedback.get('reactivity', None)
            pid_err = meta_feedback.get('pid_error', None)
            eta_emof = meta_feedback.get('eta_emotion_factor', None)
            if react is not None and pid_err is not None:
                print(
                    f"[MetaOpt] trend={meta_feedback['trend']:+.3f} | eta->{eta:.5f} | gain->{agent.emotion.gain:.3f} "
                    f"| reactivity={react:.3f} | pid_err={pid_err:+.3f}"
                )
                if eta_emof is not None:
                    print(f"[MetaOpt] Emotion-adaptive eta adjustment: factor={eta_emof:.3f}")
            else:
                print(f"[MetaOpt] trend={meta_feedback['trend']:+.3f} | eta->{eta:.5f} | gain->{agent.emotion.gain:.3f}")
    
    # --- Phase 5.7: AutoTuner Routine ---
    auto_tuner.record(avg_td_error, eta)
    new_lr, corr = auto_tuner.tune(meta_opt)
    if new_lr != meta_opt.lr:
        print(f"[AutoTuner] lr angepasst: {meta_opt.lr:.3f} -> {new_lr:.3f} (corr={corr:.2f})")
        meta_opt.lr = new_lr

    if not auto_tuner.active and episode % 20 == 0:
        print(f"[AutoTuner] abgeschlossen - stabile eta-Korrelation erreicht.")


    # Reward-Trend über gleitendes Fenster
    reward_trend = np.mean(scores[-20:]) if len(scores) > 20 else np.mean(scores)
    
    # Phase 6.1: AZPv2 mit erweiterten Features
    sigma_mean = np.mean([layer.sigma.mean().item() for layer in [agent.q_network.fc1, agent.q_network.fc2] if hasattr(layer, 'sigma')])
    azp_info = azp.step(agent.emotion.value, td_error_ctrl, reward_trend, eta, sigma_mean)
    
    # AZPv2-Training (nur bei ausreichender Historie)
    azp.train_step(episode)

    if episode % 25 == 0:
        print(f"[AZP] zone={azp_info['zone_pred']} | conf={azp_info['confidence']:.3f} | int={azp_info['intensity']:.3f}")
        # ECL-Status anzeigen
        ecl_info = ecl.get_curriculum_info()
        print(f"[ECL] difficulty={ecl_info['current_difficulty']:.3f} | phase={ecl_info['curriculum_phase']} | progress={ecl_info['learning_progress']:.3f}")
        # MOO-Status anzeigen
        moo_info = moo.get_optimization_info()
        print(f"[MOO] perf={moo_info['objective_scores']['performance']:.3f} | eta_stab={moo_info['objective_scores']['eta_stability']:.3f} | sigma_health={moo_info['objective_scores']['sigma_health']:.3f}")
    zone_history.append(azp_info['zone_pred'])

    # Confidence-Gating: Feedforward-Eingriff nur bei hoher Sicherheit
    if 'confidence' in azp_info and 'intensity' in azp_info:
        gate = float(np.clip(azp_info['confidence'] * azp_info['intensity'], 0.0, 1.0))
    else:
        gate = 0.0

    if hasattr(agent, "emotion"):
        scale = 0.03 if gate < 0.5 else 0.05
        if azp_info.get('zone_pred') == 'exploration_soon':
            agent.emotion.gain *= (1.0 + scale * gate)
        elif azp_info.get('zone_pred') == 'stabilization_soon':
            agent.emotion.gain *= (1.0 - scale * gate)

    # Reaktive Zonenreaktion mit geclipptem TD-Error
    # Emotion-EMA für Control-Pfad (glättet nur Controller, nicht Lernen)
    if hasattr(agent, "emotion"):
        if not hasattr(agent.emotion, "_ctrl_emotion_ema"):
            agent.emotion._ctrl_emotion_ema = float(agent.emotion.value)
        ctrl_alpha = 0.3  # sanftere Glättung
        agent.emotion._ctrl_emotion_ema = (
            ctrl_alpha * float(agent.emotion.value) + (1.0 - ctrl_alpha) * float(agent.emotion._ctrl_emotion_ema)
        )

    if hasattr(agent, "emotion") and hasattr(agent.emotion, "apply_zone_response"):
        ctrl_emotion_val = float(getattr(agent.emotion, "_ctrl_emotion_ema", agent.emotion.value))
        agent.emotion.apply_zone_response(ctrl_emotion_val, td_error_ctrl)
    
    # --- Phase 5.6 : Self-Recovery Loop ---
    # Sichere TD-Error-Normalisierung auf Episodenbasis (EMA-Ratio, verhaltensgleich)
    denom = td_slow if np.isfinite(td_slow) and td_slow > 0 else (np.mean(td_error_history[-50:]) if len(td_error_history) >= 50 else 1.0)
    td_error_norm_ep = float(np.clip((td_fast if td_fast > 0 else avg_td_error) / (denom + 1e-6), 0.0, 10.0))

    if np.isnan(agent.emotion.value) or np.isnan(td_error_norm_ep):
        print("[Self-Recovery] NaN erkannt → Soft Reset")
        agent.emotion.value = 0.5
        eta = 1e-3
    elif td_error_norm_ep > 2.0:
        eta *= 0.9  # starke Fehler → η leicht senken
    elif agent.emotion.value < 0.4:
        eta *= 1.1  # Emotion zu niedrig → etwas aggressiver lernen

    # --- Phase 5.8 : Self-Recovery Loop (Divergenz-Detektor) ---
    # Erkenne anhaltende Divergenz zwischen Emotion und TD-Error über 20 Episoden
    if len(emotion_history) >= 20 and len(td_error_history) >= 20:
        em_win = np.array(emotion_history[-20:], dtype=float)
        td_win = np.array(td_error_history[-20:], dtype=float)
        # Min-Max-Normalisierung mit stabilen Grenzfällen
        em_min, em_max = float(np.min(em_win)), float(np.max(em_win))
        td_min, td_max = float(np.min(td_win)), float(np.max(td_win))
        em_den = em_max - em_min if (em_max - em_min) > 1e-6 else 1.0
        td_den = td_max - td_min if (td_max - td_min) > 1e-6 else 1.0
        em_n = (em_win - em_min) / em_den
        td_n = (td_win - td_min) / td_den
        div_gap = float(np.mean(np.abs(em_n - td_n)))
        if div_gap > 0.3:
            # Kurzzeitig η senken (sanfte Dämpfung) und σ leicht erhöhen
            eta *= 0.9
            if hasattr(agent, 'q_network') and hasattr(agent.q_network, 'fc1') and hasattr(agent.q_network, 'fc2'):
                with torch.no_grad():
                    for layer in [agent.q_network.fc1, agent.q_network.fc2]:
                        if hasattr(layer, 'sigma'):
                            layer.sigma.mul_(1.05)
                            layer.sigma.add_(torch.randn_like(layer.sigma) * 0.01)

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

    # Driftgetriggerte σ-Rezentrierung (nur wenn Drift ansteigt)
    if len(sigma_history) >= 12 and np.mean(sigma_history[-12:-6]) < np.mean(sigma_history[-6:]):
        with torch.no_grad():
            for layer in [agent.q_network.fc1, agent.q_network.fc2]:
                if hasattr(layer, 'sigma'):
                    sigma_mean = layer.sigma.mean()
                    layer.sigma.add_(-0.03 * sigma_mean)

    # Adaptiver η-Deckel (mit Hysterese über Stabilitätsfenster)
    volatility = np.std(td_error_history[-10:]) if len(td_error_history) >= 10 else 0.0
    td_ctrl = locals().get('td_error_ctrl', 0.0)
    td_clip_val = locals().get('td_clip', 5.0)
    gate_val = locals().get('gate', 0.0)

    eta_cap_prev = locals().get('eta_cap', 5.5e-3)
    eta_cap_hi, eta_cap_lo = 5.5e-3, 2.0e-3

    # Zustands-Tracker für Stabilität
    stable = (volatility < 0.4) and (td_ctrl < 0.6 * td_clip_val)
    unstable = (volatility > 0.6) or (td_ctrl > 0.7 * td_clip_val) or (gate_val < 0.4)

    # Hysterese: erst nach >=5 stabilen Episoden wieder anheben
    if not hasattr(globals(), '_eta_stable_win'):
        _eta_stable_win = 0
    try:
        _eta_stable_win = _eta_stable_win + 1 if stable else 0
    except NameError:
        _eta_stable_win = 1 if stable else 0

    if unstable:
        eta_cap = eta_cap_lo
        _eta_stable_win = 0
    else:
        eta_cap = eta_cap_prev
        if _eta_stable_win >= 5:
            eta_cap = eta_cap_hi

    eta = float(np.clip(eta, 1e-5, eta_cap))

    scores.append(total_reward)
    
    # Phase 7.0 Option B: Update PSA
    psa.update(episode, total_reward)
    
    # Stability Report alle 50 Episoden
    if episode % 50 == 0 and episode > 0:
        metrics = psa.compute_stability_metrics()
        print(f"\n📊 [PSA] Stability Report (Episode {episode}):")
        print(f"   Stability Score: {metrics.stability_score:.3f}")
        print(f"   Trend: {metrics.trend} (strength: {metrics.trend_strength:.3f})")
        print(f"   Confidence: [{metrics.confidence_lower:.1f}, {metrics.confidence_upper:.1f}]")
        print(f"   Anomalies: {metrics.anomaly_count}")
        
        # Phase 6.1 + PSA: Nur Monitoring, keine Intervention
        # PSA beobachtet nur, greift nicht ein
        print()  # Leerzeile

    # Debug-Prints
    print(f"[Ep {episode+1:03d}] Return={total_reward:6.1f} | "
          f"Emotion={agent.emotion.value:.3f} | eta={eta:.5f} | "
          f"TD-Err={avg_td_error:.3f} | mod={mod:.3f} | eps={agent._last_eps:.3f} |  sigma_bar={sigma_mean:.4f}")

    # LOgging in CSV für spätere Analyse 
    if episode % 10 == 0: # nur alle 10 Episoden
        os.makedirs("results", exist_ok=True)
        csv_path = "results/training_log.csv"
        # falls Datei noch nicht existiert → Header hinzufügen
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("episode,reward,emotion,eta,td_error,sigma_mean,momentum,emotion_diff,zone,trend,gate,eta_cap,n_step,beta_current,priority_weight,epru_confidence,epru_adjustment,epru_intervention,azpv2_confidence,azpv2_intensity,azpv2_zone_pred,ecl_difficulty,ecl_phase,ecl_progress,ecl_stability,moo_performance_score,moo_eta_stability,moo_sigma_health,moo_predicted_perf,psa_stability_score,psa_trend,psa_confidence_lower,psa_confidence_upper,psa_anomaly_count\n")
        
        trend_val = meta_feedback['trend'] if meta_feedback is not None else 0.0
        zone_val  = stat['zone_pred'] if stat and stat['zone_pred'] is not None else "None"
        with open(csv_path, "a") as f:
            gate_val = float(gate) if 'gate' in locals() else 0.0
            eta_cap_val = float(eta_cap) if 'eta_cap' in locals() else 7e-3
            # N-Step und Priority-Metriken
            n_step_val = getattr(agent, 'n_step', 2)
            beta_current = getattr(agent, 'beta', 0.4)
            priority_weight = 1.0  # Placeholder - wird in agent.py berechnet
            
            # EPRU-Metriken (Phase 6.0)
            epru_debug = src.get_epru_debug_info()
            epru_confidence = epru_debug.get('confidence', 0.0) if epru_debug else 0.0
            epru_adjustment = epru_debug.get('eta_adjustment', 0.0) if epru_debug else 0.0
            epru_intervention = epru_debug.get('intervention', 'none') if epru_debug else 'none'
            
            # AZPv2-Metriken (Phase 6.1)
            azpv2_confidence = azp_info.get('confidence', 0.0)
            azpv2_intensity = azp_info.get('intensity', 0.0)
            azpv2_zone_pred = azp_info.get('zone_pred', 'neutral')
            
            # ECL-Metriken (Phase 6.2)
            ecl_info = ecl.get_curriculum_info()
            ecl_difficulty = ecl_info.get('current_difficulty', 0.5)
            ecl_phase = ecl_info.get('curriculum_phase', 'exploration')
            ecl_progress = ecl_info.get('learning_progress', 0.0)
            ecl_stability = ecl_info.get('emotional_stability', 0.0)
            
            # MOO-Metriken (Phase 6.3)
            moo_info = moo.get_optimization_info()
            moo_performance_score = moo_info.get('objective_scores', {}).get('performance', 0.0)
            moo_eta_stability = moo_info.get('objective_scores', {}).get('eta_stability', 0.0)
            moo_sigma_health = moo_info.get('objective_scores', {}).get('sigma_health', 0.0)
            moo_predicted_perf = moo_info.get('predicted_performance', 0.0)
            
            # PSA-Metriken (Phase 7.0 Option B)
            psa_metrics = psa.compute_stability_metrics()
            psa_stability_score = psa_metrics.stability_score
            psa_trend = psa_metrics.trend
            psa_confidence_lower = psa_metrics.confidence_lower
            psa_confidence_upper = psa_metrics.confidence_upper
            psa_anomaly_count = psa_metrics.anomaly_count
            
            f.write(
                f"{episode+1},{total_reward:.2f},{agent.emotion.value:.3f},{eta:.5f},"
                f"{avg_td_error:.3f},{sigma_mean:.4f},{momentum:.4f},{emotion_diff:.4f},"
                f"{zone_val},{trend_val:.4f},{gate_val:.3f},{eta_cap_val:.5f},"
                f"{n_step_val},{beta_current:.3f},{priority_weight:.3f},"
                f"{epru_confidence:.3f},{epru_adjustment:.5f},{epru_intervention},"
                f"{azpv2_confidence:.3f},{azpv2_intensity:.3f},{azpv2_zone_pred},"
                f"{ecl_difficulty:.3f},{ecl_phase},{ecl_progress:.3f},{ecl_stability:.3f},"
                f"{moo_performance_score:.3f},{moo_eta_stability:.3f},{moo_sigma_health:.3f},{moo_predicted_perf:.3f},"
                f"{psa_stability_score:.3f},{psa_trend},{psa_confidence_lower:.2f},{psa_confidence_upper:.2f},{psa_anomaly_count}\n"
            )

    # Durchschnitt über 100 Episoden
    if (episode + 1) % 10 == 0:
        avg100 = np.nanmean(scores[-window:])
        print(f"-> avg100: {avg100:6.2f}")
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
    # Hinweis: keine Unicode-Sonderzeichen im Print (Windows Console Encoding)
    print("[Plot] eta vs TD-Error gespeichert unter results/eta_vs_td_error.png")


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
    reward_history = locals().get("reward_history", None)
    if reward_history is not None:
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