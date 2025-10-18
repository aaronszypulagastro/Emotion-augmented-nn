"""
LunarLander Training mit Winner Mindset Framework
==================================================

Emotion-Augmented DQN für LunarLander-v2

Features:
---------
✅ Winner Mindset Regulator (Emotion → Exploration & Noise)
✅ Emotion-Engine (vereinfacht, nur für Meta-Signal)
✅ PSA (Performance Stability Analyzer)
✅ BDH-Plasticity (optional, mit Mindset-moduliertem Noise)
✅ Learning Efficiency Index Tracking

Keine LR-Modulation! (Lessons Learned aus CartPole Tests)

Author: Phase 8.0 - Winner Mindset on LunarLander
Date: 2025-10-16
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import os
import csv
from collections import deque
from tqdm import tqdm

# Imports
try:
    from ..core.winner_mindset_regulator import (
        WinnerMindsetRegulator, 
        create_winner_mindset_config,
        MindsetState
    )
    from ..core.emotion_engine_fixed import EmotionEngineFix
    from ..core.performance_stability_analyzer import PerformanceStabilityAnalyzer
    from ..analysis.plot_winner_mindset import (
        plot_winner_mindset_dashboard,
        plot_mindset_heatmap,
        print_mindset_summary
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.winner_mindset_regulator import (
        WinnerMindsetRegulator,
        create_winner_mindset_config,
        MindsetState
    )
    from core.emotion_engine_fixed import EmotionEngineFix
    from core.performance_stability_analyzer import PerformanceStabilityAnalyzer
    from analysis.plot_winner_mindset import (
        plot_winner_mindset_dashboard,
        plot_mindset_heatmap,
        print_mindset_summary
    )

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("╔══════════════════════════════════════════════════════════════╗")
print("║   LUNARLANDER + WINNER MINDSET FRAMEWORK                     ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# Configuration
CONFIG = {
    'env_name': 'LunarLander-v3',  # v3 statt v2 (v2 ist deprecated)
    'episodes': 2000,              # Mehr Episoden für komplexeren Task
    'max_steps': 1000,
    'batch_size': 128,            # Größerer Batch
    'gamma': 0.99,
    'lr': 5e-4,                   # KONSTANT (keine Modulation!)
    'target_update_freq': 10,
    'buffer_capacity': 100000,    # Größerer Buffer
    'use_bdh_plasticity': True,   # Optional: BDH mit Mindset-Noise
}

print("⚙️  Konfiguration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
print()


# Q-Network (Standard oder mit BDH-Plasticity)
class PlasticLinear(nn.Module):
    """Linear Layer mit BDH-Plasticity"""
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=bias)
        self.register_buffer('sigma', torch.zeros_like(self.base.weight))
        self._last_pre = None
        self._last_post = None
    
    def forward(self, x):
        self._last_pre = x.detach()
        W_eff = self.base.weight + self.sigma
        y = torch.nn.functional.linear(x, W_eff, self.base.bias)
        self._last_post = y.detach()
        return y
    
    @torch.no_grad()
    def plasticity_step(self, mod, eta, decay, clip):
        """Hebbian-Update mit Winner-Mindset moduliertem Noise"""
        if self._last_pre is None or self._last_post is None:
            return
        
        pre = self._last_pre.mean(0).unsqueeze(1)
        post = self._last_post.mean(0).unsqueeze(0)
        hebbian = torch.outer(post, pre)
        
        self.sigma.add_(hebbian * eta * mod)
        self.sigma.mul_(decay)
        self.sigma.clamp_(-clip, clip)


class QNetwork(nn.Module):
    """Q-Network für LunarLander"""
    def __init__(self, state_dim, action_dim, use_plasticity=False):
        super().__init__()
        self.use_plasticity = use_plasticity
        
        if use_plasticity:
            self.fc1 = PlasticLinear(state_dim, 256)
            self.fc2 = PlasticLinear(256, 256)
            self.fc3 = nn.Linear(256, action_dim)
        else:
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def plasticity_step(self, mod, eta, decay, clip):
        """BDH-Update (nur wenn Plasticity enabled)"""
        if self.use_plasticity and hasattr(self.fc1, 'plasticity_step'):
            self.fc1.plasticity_step(mod, eta, decay, clip)
            self.fc2.plasticity_step(mod, eta, decay, clip)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class WinnerMindsetDQNAgent:
    """DQN Agent mit Winner Mindset Framework"""
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, config['use_bdh_plasticity'])
        self.target_network = QNetwork(state_dim, action_dim, config['use_bdh_plasticity'])
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer - KONSTANTE LR!
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['lr'])
        
        # Replay Buffer
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        # Base Epsilon (wird von Winner Mindset moduliert)
        self.base_epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        
        # Emotion Engine (NUR für Meta-Signal, NICHT für LR!)
        self.emotion = EmotionEngineFix(
            init_state=0.5,
            alpha=0.05,                 # Sehr langsam für LunarLander
            target_return=200.0,        # LunarLander Target
            bounds=(0.2, 0.8),
            decay_rate=0.998,
            noise_std=0.005
        )
        
        # Winner Mindset Regulator
        wmr_config = create_winner_mindset_config("lunarlander")
        self.mindset = WinnerMindsetRegulator(**wmr_config)
        
        self.train_step = 0
    
    def select_action(self, state):
        """Epsilon-greedy mit Winner-Mindset moduliertem Epsilon"""
        # Get modulated epsilon from mindset
        epsilon = self.mindset.modulate_exploration(self.base_epsilon)
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self, episode_return, psa_metrics):
        """Train mit Winner Mindset Framework"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Update Emotion (Meta-Signal)
        self.emotion.update(episode_return)
        
        # Update Winner Mindset
        performance_metrics = {
            'avg_return': episode_return,
            'stability': psa_metrics.stability_score if psa_metrics else 0.5,
            'trend': psa_metrics.trend if psa_metrics else 'stable',
            'td_error': 0.0  # Wird unten aktualisiert
        }
        
        mindset_metrics = self.mindset.update(self.emotion.value, performance_metrics)
        
        # Standard DQN Training (KEINE LR-Modulation!)
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # BDH-Plasticity Update (mit Mindset-moduliertem Noise)
        if self.config['use_bdh_plasticity']:
            noise_scale = self.mindset.modulate_noise(base_noise=0.05)
            mod = 0.8 + 0.4 * self.emotion.value
            eta_plastic = 1e-3 * noise_scale  # Noise-moduliert!
            
            self.q_network.plasticity_step(
                mod=mod,
                eta=eta_plastic,
                decay=0.995,
                clip=0.1
            )
        
        # Target Network Update
        self.train_step += 1
        if self.train_step % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Base Epsilon Decay
        self.base_epsilon = max(self.epsilon_min, self.base_epsilon * self.epsilon_decay)
        
        td_error = (current_q - target_q).abs().mean().item()
        return td_error, mindset_metrics


# Main Training
print("🌍 Setting up LunarLander Environment...\n")

env = gym.make(CONFIG['env_name'])
env.reset(seed=SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"   State dim: {state_dim}")
print(f"   Action dim: {action_dim}\n")

# Agent & Analyzers
agent = WinnerMindsetDQNAgent(state_dim, action_dim, CONFIG)
psa = PerformanceStabilityAnalyzer(window_size=100)

print("✅ Winner Mindset Regulator aktiviert")
print("✅ Emotion-Engine (Meta-Signal only)")
print("✅ PSA Monitoring")
if CONFIG['use_bdh_plasticity']:
    print("✅ BDH-Plasticity mit Mindset-Noise\n")

# CSV Logging
csv_path = "results/lunarlander_winner_mindset_log.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'episode', 'return', 'td_error', 'emotion', 'mindset_state',
        'exploration_factor', 'noise_scale', 'focus', 'learning_efficiency',
        'psa_stability', 'psa_trend', 'epsilon'
    ])

# Training
episode_returns = []
print("🚀 Winner Mindset Training auf LunarLander startet...\n")
print("="*60)

for episode in tqdm(range(CONFIG['episodes']), desc="LunarLander Winner Mindset"):
    state, _ = env.reset()
    episode_return = 0.0
    done = False
    
    for step in range(CONFIG['max_steps']):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        
        episode_return += reward
        state = next_state
        
        if done:
            break
    
    episode_returns.append(episode_return)
    
    # Update PSA
    psa.update(episode, episode_return)
    psa_metrics = psa.compute_stability_metrics() if episode > 10 else None
    
    # Train (mit Winner Mindset!)
    result = agent.train(episode_return, psa_metrics)
    
    if result:
        td_error, mindset_metrics = result
    else:
        td_error = 0.0
        mindset_metrics = None
    
    # Logging
    if episode % 10 == 0:
        avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if mindset_metrics:
                writer.writerow([
                    episode, episode_return, td_error, agent.emotion.value,
                    mindset_metrics.state.value,
                    mindset_metrics.exploration_factor,
                    mindset_metrics.noise_scale,
                    mindset_metrics.focus_intensity,
                    mindset_metrics.learning_efficiency,
                    psa_metrics.stability_score if psa_metrics else 0.0,
                    psa_metrics.trend if psa_metrics else 'stable',
                    agent.mindset.modulate_exploration(agent.base_epsilon)
                ])
    
    # Progress Report
    if episode % 50 == 0 and episode > 0:
        avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
        
        mindset = agent.mindset.get_current_mindset()
        
        print(f"\n📊 [Episode {episode}]")
        print(f"   Return: {episode_return:7.2f} | avg100: {avg100:7.2f}")
        print(f"   Emotion: {agent.emotion.value:.3f} | Mindset: {mindset.state.value}")
        print(f"   Exploration: {mindset.exploration_factor:.3f} | Focus: {mindset.focus_intensity:.3f}")
        
        if psa_metrics:
            print(f"   PSA: Stability={psa_metrics.stability_score:.3f}, Trend={psa_metrics.trend}")
        print()

env.close()

# Final Analysis
print("\n" + "="*60)
print("LUNARLANDER WINNER MINDSET - FINALE ERGEBNISSE")
print("="*60)

avg100_final = np.mean(episode_returns[-100:])
best_return = max(episode_returns)
avg_return = np.mean(episode_returns)

print(f"\n📊 Performance:")
print(f"   avg100:      {avg100_final:.2f}")
print(f"   Best Return: {best_return:.2f}")
print(f"   Mean Return: {avg_return:.2f}")

# Mindset Summary
mindset_dynamics = agent.mindset.log_mindset_dynamics()
print_mindset_summary(mindset_dynamics)

# State Statistics
state_stats = agent.mindset.get_state_statistics()
print("📈 Mindset State Verteilung:")
for state, count in state_stats.items():
    pct = 100 * count / len(agent.mindset.mindset_log)
    print(f"   {state.capitalize():12s}: {count:4d} ({pct:5.1f}%)")

# Final Mindset
final_mindset = agent.mindset.get_current_mindset()
print(f"\n🎯 Final Mindset:")
print(f"   State: {final_mindset.state.value}")
print(f"   Emotion: {final_mindset.emotion_value:.3f}")
print(f"   Learning Efficiency: {final_mindset.learning_efficiency:.3f}")

# Visualizations
print("\n📊 Erstelle Visualisierungen...")
plot_winner_mindset_dashboard(mindset_dynamics, "results/lunarlander_winner_mindset_dashboard.png")
plot_mindset_heatmap(mindset_dynamics, "results/lunarlander_mindset_heatmap.png")

print(f"\n💾 Log: {csv_path}")
print("\n" + "="*60)
print("✅ WINNER MINDSET TRAINING ABGESCHLOSSEN!")
print("="*60 + "\n")

