"""
TEST 1: Vanilla DQN + Emotion-Engine (EINFACH)
===============================================

Test ob Emotion-Engine ALLEINE das Problem verursacht

Features:
✅ Vanilla DQN (wissen dass es funktioniert)
✅ Emotion-Engine (einfache Version)
❌ KEIN BDH-Plasticity
❌ KEIN SRC, EPRU, AZPv2, ECL, MOO
✅ PSA für Monitoring

Erwartung:
- Falls stabil (avg100 > 200): Emotion ist OK, Problem ist woanders
- Falls Collapse: Emotion-Engine ist das Problem

Author: Test 1 - Schrittweises Debugging
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
    from ..core.emotion_engine import EmotionEngine
    from ..core.performance_stability_analyzer import PerformanceStabilityAnalyzer
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.emotion_engine import EmotionEngine
    from core.performance_stability_analyzer import PerformanceStabilityAnalyzer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("╔══════════════════════════════════════════════════════════════╗")
print("║   TEST 1: VANILLA DQN + EMOTION-ENGINE                       ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# Config
CONFIG = {
    'env_name': 'CartPole-v1',
    'episodes': 500,
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'lr': 1e-3,
    'target_update_freq': 10,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_capacity': 10000,
}

# Standard Q-Network (KEIN BDH!)
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

class DQNPlusEmotionAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks (Standard, KEIN BDH)
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer - Learning Rate wird von Emotion moduliert
        self.base_lr = config['lr']
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.base_lr)
        
        # Replay Buffer
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        # Epsilon
        self.epsilon = config['epsilon_start']
        
        # Emotion-Engine (EINFACH)
        self.emotion = EmotionEngine(
            init_state=0.5,
            alpha=0.85,
            target_return=100,  # Für CartPole
            noise_std=0.05,
            gain=1.0
        )
        
        self.train_step = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self, episode_return):
        """Train mit Emotion-modulierter Learning Rate"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Update Emotion basierend auf Episode Return
        self.emotion.update(episode_return)
        
        # Moduliere Learning Rate mit Emotion
        # Höhere Emotion = höhere LR (mehr Lernen)
        emotion_factor = 0.5 + 0.5 * self.emotion.value  # [0.5, 1.0]
        new_lr = self.base_lr * emotion_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
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
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.config['epsilon_end'], 
                          self.epsilon * self.config['epsilon_decay'])
        
        td_error = (current_q - target_q).abs().mean().item()
        return td_error

# Setup
env = gym.make(CONFIG['env_name'])
env.reset(seed=SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNPlusEmotionAgent(state_dim, action_dim, CONFIG)
psa = PerformanceStabilityAnalyzer(window_size=100)

print(f"🌍 Environment: {CONFIG['env_name']}")
print(f"   ✅ Emotion-Engine AKTIV (einfache Version)")
print(f"   ❌ KEIN BDH-Plasticity")
print(f"   📊 PSA Monitoring AKTIV\n")

# CSV
csv_path = "results/test1_vanilla_plus_emotion.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'return', 'td_error', 'emotion', 'lr_factor',
                     'psa_stability', 'psa_trend', 'psa_anomaly_count'])

# Training
episode_returns = []
td_errors = []

print("🚀 TEST 1 Training startet...\n")

for episode in tqdm(range(CONFIG['episodes']), desc="Test 1 Training"):
    state, _ = env.reset()
    episode_return = 0.0
    done = False
    
    for step in range(CONFIG['max_steps']):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        
        episode_return += reward
        state = next_state
        
        if done:
            break
    
    # Train am Ende der Episode
    td_error = agent.train(episode_return)
    if td_error is None:
        td_error = 0.0
    
    episode_returns.append(episode_return)
    td_errors.append(td_error)
    
    # Update PSA
    psa.update(episode, episode_return)
    
    # PSA Report
    if episode % 50 == 0 and episode > 0:
        metrics = psa.compute_stability_metrics()
        print(f"\n📊 [PSA] Episode {episode}:")
        print(f"   Stability: {metrics.stability_score:.3f} | Trend: {metrics.trend} | Anomalies: {metrics.anomaly_count}\n")
    
    # Log
    if episode % 10 == 0:
        psa_metrics = psa.compute_stability_metrics()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            emotion_factor = 0.5 + 0.5 * agent.emotion.value
            writer.writerow([
                episode, episode_return, td_error, agent.emotion.value,
                emotion_factor, psa_metrics.stability_score,
                psa_metrics.trend, psa_metrics.anomaly_count
            ])
    
    if episode % 10 == 0:
        avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
        print(f"[Ep {episode:3d}] Return={episode_return:6.1f} | avg100={avg100:6.2f} | "
              f"Emotion={agent.emotion.value:.3f} | TD-Err={td_error:.3f}")

env.close()

# Ergebnisse
print("\n" + "="*60)
print("TEST 1 ERGEBNISSE - VANILLA DQN + EMOTION")
print("="*60)

avg100 = np.mean(episode_returns[-100:])
print(f"\n📊 avg100: {avg100:.2f}")
print(f"   Best: {max(episode_returns):.2f}")
print(f"   Final Emotion: {agent.emotion.value:.3f}")
print(f"   Final TD-Error: {np.mean(td_errors[-10:]):.3f}")

print(f"\n🎯 BEWERTUNG:")
if avg100 > 200:
    print(f"   ✅ ERFOLGREICH - Emotion-Engine ist OK!")
    print(f"   → Problem liegt NICHT in Emotion-Engine")
    print(f"   → Teste jetzt BDH-Plasticity (Test 2)")
else:
    print(f"   ❌ PROBLEM - Emotion-Engine verursacht Collapse!")
    print(f"   → Emotion-Implementierung überarbeiten")

print(f"\n💾 Log: {csv_path}\n")


