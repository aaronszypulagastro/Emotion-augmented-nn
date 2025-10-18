"""
Acrobot Training mit Winner Mindset Framework
==============================================

Alternative zu LunarLander (keine extra Dependencies!)

Acrobot-v1:
- Komplexer als CartPole
- Keine Installation benötigt
- Guter Test für Winner Mindset
- Continuous state space (6D)
- Sparse rewards

Author: Phase 8.0 - Winner Mindset on Acrobot
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("╔══════════════════════════════════════════════════════════════╗")
print("║   ACROBOT + WINNER MINDSET FRAMEWORK                         ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# Configuration
CONFIG = {
    'env_name': 'Acrobot-v1',
    'episodes': 1500,
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'lr': 5e-4,                   # KONSTANT
    'target_update_freq': 10,
    'buffer_capacity': 50000,
    'use_bdh_plasticity': False,  # Erstmal ohne
}

print("⚙️  Konfiguration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
print()

# Standard Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer - KONSTANTE LR!
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['lr'])
        
        # Replay Buffer
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        # Base Epsilon
        self.base_epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        
        # Emotion Engine
        self.emotion = EmotionEngineFix(
            init_state=0.5,
            alpha=0.08,
            target_return=-100.0,       # Acrobot: -500 schlecht, -100 OK, < -80 gut
            bounds=(0.2, 0.8),
            decay_rate=0.997,
            noise_std=0.01
        )
        
        # Winner Mindset Regulator
        wmr_config = {
            'epsilon_min': 0.01,
            'epsilon_max': 0.3,
            'noise_min': 0.001,
            'noise_max': 0.05,
            'frustration_threshold': 0.35,
            'pride_threshold': 0.65,
            'focus_decay': 0.96,
            'history_window': 50
        }
        self.mindset = WinnerMindsetRegulator(**wmr_config)
        
        self.train_step = 0
    
    def select_action(self, state):
        """Epsilon-greedy mit Winner-Mindset moduliertem Epsilon"""
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
        
        # Update Emotion
        self.emotion.update(episode_return)
        
        # Update Winner Mindset
        performance_metrics = {
            'avg_return': episode_return,
            'stability': psa_metrics.stability_score if psa_metrics else 0.5,
            'trend': psa_metrics.trend if psa_metrics else 'stable',
            'td_error': 0.0
        }
        
        mindset_metrics = self.mindset.update(self.emotion.value, performance_metrics)
        
        # Standard DQN Training
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
        
        # Target Network Update
        self.train_step += 1
        if self.train_step % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Base Epsilon Decay
        self.base_epsilon = max(self.epsilon_min, self.base_epsilon * self.epsilon_decay)
        
        td_error = (current_q - target_q).abs().mean().item()
        return td_error, mindset_metrics

# Setup
env = gym.make(CONFIG['env_name'])
env.reset(seed=SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"🌍 Environment: {CONFIG['env_name']}")
print(f"   State dim: {state_dim}")
print(f"   Action dim: {action_dim}\n")

# Agent & Analyzers
agent = WinnerMindsetDQNAgent(state_dim, action_dim, CONFIG)
psa = PerformanceStabilityAnalyzer(window_size=100)

print("✅ Winner Mindset Regulator aktiviert")
print("✅ Emotion-Engine (Meta-Signal only)")
print("✅ PSA Monitoring\n")

# CSV Logging
csv_path = "results/acrobot_winner_mindset_log.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'episode', 'return', 'td_error', 'emotion', 'mindset_state',
        'exploration_factor', 'focus', 'learning_efficiency',
        'psa_stability', 'psa_trend', 'epsilon'
    ])

# Training
episode_returns = []
print("🚀 Winner Mindset Training auf Acrobot startet...\n")
print("="*60)

for episode in tqdm(range(CONFIG['episodes']), desc="Acrobot Winner Mindset"):
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
    
    # Train
    result = agent.train(episode_return, psa_metrics)
    
    if result:
        td_error, mindset_metrics = result
    else:
        td_error = 0.0
        mindset_metrics = None
    
    # Logging
    if episode % 10 == 0:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if mindset_metrics:
                writer.writerow([
                    episode, episode_return, td_error, agent.emotion.value,
                    mindset_metrics.state.value,
                    mindset_metrics.exploration_factor,
                    mindset_metrics.focus_intensity,
                    mindset_metrics.learning_efficiency,
                    psa_metrics.stability_score if psa_metrics else 0.0,
                    psa_metrics.trend if psa_metrics else 'stable',
                    agent.mindset.modulate_exploration(agent.base_epsilon)
                ])
    
    # Progress Report
    if episode % 100 == 0 and episode > 0:
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
print("ACROBOT WINNER MINDSET - FINALE ERGEBNISSE")
print("="*60)

avg100_final = np.mean(episode_returns[-100:])
best_return = max(episode_returns)
avg_return = np.mean(episode_returns)

print(f"\n📊 Performance:")
print(f"   avg100:      {avg100_final:.2f}")
print(f"   Best Return: {best_return:.2f}")
print(f"   Mean Return: {avg_return:.2f}")

# Acrobot Benchmark (gelöst wenn avg100 > -100)
if avg100_final > -100:
    print(f"   ✅ GELÖST! (avg100 > -100)")
elif avg100_final > -150:
    print(f"   ⚠️  GUT (avg100 > -150)")
else:
    print(f"   ❌ Mehr Training nötig")

# Mindset Summary
state_stats = agent.mindset.get_state_statistics()
print(f"\n📈 Mindset State Verteilung:")
total = sum(state_stats.values())
for state, count in state_stats.items():
    pct = 100 * count / total if total > 0 else 0
    print(f"   {state.capitalize():12s}: {count:4d} ({pct:5.1f}%)")

# Final Mindset
final_mindset = agent.mindset.get_current_mindset()
print(f"\n🎯 Final Mindset:")
print(f"   State: {final_mindset.state.value}")
print(f"   Emotion: {final_mindset.emotion_value:.3f}")
print(f"   Learning Efficiency: {final_mindset.learning_efficiency:.3f}")

print(f"\n💾 Log: {csv_path}")
print(f"\n✅ WINNER MINDSET TRAINING ABGESCHLOSSEN!")
print("="*60 + "\n")

# Erstelle Quick Summary für Visualisierung
print("📊 Für Visualisierung nutzen Sie:")
print("   from analysis.plot_winner_mindset import plot_winner_mindset_dashboard")
print(f"   mindset_dynamics = agent.mindset.log_mindset_dynamics()")
print(f"   plot_winner_mindset_dashboard(mindset_dynamics)")


