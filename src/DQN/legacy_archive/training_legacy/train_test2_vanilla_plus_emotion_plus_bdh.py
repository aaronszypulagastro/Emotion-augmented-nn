"""
TEST 2: Vanilla DQN + Emotion + BDH-Plasticity
===============================================

Test ob BDH-Plasticity das Problem verursacht

Features:
✅ Vanilla DQN
✅ Emotion-Engine
✅ BDH-Plasticity (σ-Modulation)
❌ KEIN SRC, EPRU, AZPv2, ECL, MOO
✅ PSA für Monitoring

Erwartung:
- Falls stabil: BDH ist OK, Problem in höheren Ebenen (SRC/EPRU)
- Falls Collapse: BDH-Plasticity ist der Täter!

Author: Test 2 - Schrittweises Debugging
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
print("║   TEST 2: VANILLA + EMOTION + BDH-PLASTICITY                 ║")
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

# BDH-Plastic Linear Layer
class PlasticLinear(nn.Module):
    """Linear mit plastischer Komponente σ"""
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
    def plasticity_step(self, mod, eta=2e-3, decay=0.996, clip=0.2):
        """Hebbian-ähnliches σ-Update"""
        if self._last_pre is None or self._last_post is None:
            return
        
        # Hebbian Regel: Δσ ∝ pre × post
        pre = self._last_pre.mean(0).unsqueeze(1)
        post = self._last_post.mean(0).unsqueeze(0)
        hebbian = torch.outer(post, pre)
        
        # Update mit Modulation
        self.sigma.add_(hebbian * eta * mod)
        self.sigma.mul_(decay)
        self.sigma.clamp_(-clip, clip)

# Q-Network mit BDH
class PlasticQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = PlasticLinear(state_dim, 128)
        self.fc2 = PlasticLinear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)  # Output ohne Plasticity
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def plasticity_step(self, mod, eta, decay, clip):
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

class DQNPlusEmotionPlusBDH:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Plastic Networks
        self.q_network = PlasticQNetwork(state_dim, action_dim)
        self.target_network = PlasticQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.base_lr = config['lr']
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.base_lr)
        
        self.memory = ReplayBuffer(config['buffer_capacity'])
        self.epsilon = config['epsilon_start']
        
        # Emotion
        self.emotion = EmotionEngine(
            init_state=0.5,
            alpha=0.85,
            target_return=100,
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
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Update Emotion
        self.emotion.update(episode_return)
        
        # LR Modulation
        emotion_factor = 0.5 + 0.5 * self.emotion.value
        new_lr = self.base_lr * emotion_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Standard Training
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
        
        # BDH-Plasticity Update
        mod = 0.8 + 0.4 * self.emotion.value
        eta_plastic = 2e-3 * (0.5 + 0.5 * self.emotion.value)
        self.q_network.plasticity_step(mod=mod, eta=eta_plastic, decay=0.995, clip=0.15)
        
        # Target Update
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

agent = DQNPlusEmotionPlusBDH(state_dim, action_dim, CONFIG)
psa = PerformanceStabilityAnalyzer(window_size=100)

print(f"🌍 Environment: {CONFIG['env_name']}")
print(f"   ✅ Emotion-Engine AKTIV")
print(f"   ✅ BDH-Plasticity AKTIV")
print(f"   📊 PSA Monitoring AKTIV\n")

# CSV
csv_path = "results/test2_vanilla_plus_emotion_plus_bdh.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'return', 'td_error', 'emotion',
                     'psa_stability', 'psa_trend', 'psa_anomaly_count'])

# Training
episode_returns = []
td_errors = []

print("🚀 TEST 2 Training startet...\n")

for episode in tqdm(range(CONFIG['episodes']), desc="Test 2 Training"):
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
    
    td_error = agent.train(episode_return)
    if td_error is None:
        td_error = 0.0
    
    episode_returns.append(episode_return)
    td_errors.append(td_error)
    
    psa.update(episode, episode_return)
    
    if episode % 50 == 0 and episode > 0:
        metrics = psa.compute_stability_metrics()
        print(f"\n📊 [PSA] Episode {episode}:")
        print(f"   Stability: {metrics.stability_score:.3f} | Trend: {metrics.trend} | Anomalies: {metrics.anomaly_count}\n")
    
    if episode % 10 == 0:
        psa_metrics = psa.compute_stability_metrics()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, episode_return, td_error, agent.emotion.value,
                psa_metrics.stability_score, psa_metrics.trend, psa_metrics.anomaly_count
            ])
    
    if episode % 10 == 0:
        avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
        print(f"[Ep {episode:3d}] Return={episode_return:6.1f} | avg100={avg100:6.2f} | "
              f"Emotion={agent.emotion.value:.3f} | TD-Err={td_error:.3f}")

env.close()

# Ergebnisse
print("\n" + "="*60)
print("TEST 2 ERGEBNISSE - VANILLA + EMOTION + BDH")
print("="*60)

avg100 = np.mean(episode_returns[-100:])
print(f"\n📊 avg100: {avg100:.2f}")
print(f"   Best: {max(episode_returns):.2f}")
print(f"   Final Emotion: {agent.emotion.value:.3f}")
print(f"   Final TD-Error: {np.mean(td_errors[-10:]):.3f}")

print(f"\n🎯 BEWERTUNG:")
if avg100 > 200:
    print(f"   ✅ ERFOLGREICH - BDH-Plasticity ist OK!")
    print(f"   → Problem liegt in SRC/EPRU/höheren Ebenen")
else:
    print(f"   ❌ PROBLEM GEFUNDEN - BDH-Plasticity verursacht Collapse!")
    print(f"   → BDH-Implementierung überarbeiten")
    print(f"   → Oder: BDH-Parameter (eta, decay, clip) anpassen")

print(f"\n💾 Log: {csv_path}\n")


