"""
TEST 3: Vanilla DQN + FIXED Emotion-Engine
===========================================

Teste die REPARIERTE Emotion-Engine:
- NUR EMA Update (keine 7 Mechanismen)
- Alpha 0.1 (langsamer)
- Bounds [0.3, 0.7] (statt [0.3, 0.98])
- Sanfter Decay zu 0.5
- Minimaler Noise

Erwartung:
- Falls avg100 > 200: ✅ Fix funktioniert!
- Falls avg100 < 100: ❌ Immer noch Probleme

Author: Test 3 - Fixed Emotion
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
    from ..core.emotion_engine_fixed import EmotionEngineFix
    from ..core.performance_stability_analyzer import PerformanceStabilityAnalyzer
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.emotion_engine_fixed import EmotionEngineFix
    from core.performance_stability_analyzer import PerformanceStabilityAnalyzer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("╔══════════════════════════════════════════════════════════════╗")
print("║   TEST 3: VANILLA DQN + FIXED EMOTION-ENGINE                 ║")
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

class DQNPlusFixedEmotionAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.base_lr = config['lr']
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.base_lr)
        
        # Replay Buffer
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        # Epsilon
        self.epsilon = config['epsilon_start']
        
        # FIXED Emotion-Engine
        self.emotion = EmotionEngineFix(
            init_state=0.5,
            alpha=0.1,                  # Langsamer
            target_return=300.0,        # CartPole Ziel
            bounds=(0.3, 0.7),          # Engere Bounds
            decay_rate=0.995,           # Sanfter Decay
            noise_std=0.01              # Minimaler Noise
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
        """Train mit FIXED Emotion-modulierter Learning Rate"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Update FIXED Emotion
        self.emotion.update(episode_return)
        
        # Moduliere Learning Rate
        # Emotion in [0.3, 0.7] -> LR factor in [0.65, 1.0]
        emotion_factor = 0.5 + 0.5 * self.emotion.value
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

agent = DQNPlusFixedEmotionAgent(state_dim, action_dim, CONFIG)
psa = PerformanceStabilityAnalyzer(window_size=100)

print(f"🌍 Environment: {CONFIG['env_name']}")
print(f"   ✅ FIXED Emotion-Engine AKTIV")
print(f"      - Alpha: 0.1 (langsamer)")
print(f"      - Bounds: [0.3, 0.7]")
print(f"      - Decay: 0.995 zu 0.5")
print(f"   ❌ KEIN BDH-Plasticity")
print(f"   📊 PSA Monitoring AKTIV\n")

# CSV
csv_path = "results/test3_vanilla_plus_fixed_emotion.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'return', 'td_error', 'emotion', 'lr_factor',
                     'psa_stability', 'psa_trend', 'psa_anomaly_count'])

# Training
episode_returns = []
td_errors = []
emotions = []

print("🚀 TEST 3 Training startet...\n")

for episode in tqdm(range(CONFIG['episodes']), desc="Test 3 Training"):
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
    emotions.append(agent.emotion.value)
    
    # Update PSA
    psa.update(episode, episode_return)
    
    # PSA Report
    if episode % 50 == 0 and episode > 0:
        metrics = psa.compute_stability_metrics()
        print(f"\n📊 [PSA] Episode {episode}:")
        print(f"   Stability: {metrics.stability_score:.3f} | Trend: {metrics.trend}")
        print(f"   Emotion: {agent.emotion.value:.3f} (min={min(emotions[-50:]):.3f}, max={max(emotions[-50:]):.3f})\n")
    
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
print("TEST 3 ERGEBNISSE - VANILLA DQN + FIXED EMOTION")
print("="*60)

avg100 = np.mean(episode_returns[-100:])
emotion_range = (min(emotions), max(emotions))
print(f"\n📊 avg100: {avg100:.2f}")
print(f"   Best: {max(episode_returns):.2f}")
print(f"   Emotion Range: [{emotion_range[0]:.3f}, {emotion_range[1]:.3f}]")
print(f"   Final Emotion: {agent.emotion.value:.3f}")
print(f"   Final TD-Error: {np.mean(td_errors[-10:]):.3f}")

print(f"\n📈 VERGLEICH:")
print(f"   Vanilla DQN:     268.75 ✅")
print(f"   + OLD Emotion:    95.03 ❌")
print(f"   + FIXED Emotion: {avg100:6.2f} {'✅' if avg100 > 200 else '⚠️' if avg100 > 150 else '❌'}")

print(f"\n🎯 BEWERTUNG:")
if avg100 > 250:
    print(f"   ✅ EXZELLENT - Fixed Emotion funktioniert perfekt!")
    print(f"   → Emotion-Engine ist jetzt stabil")
    print(f"   → Nächster Schritt: BDH hinzufügen (Test 2)")
elif avg100 > 200:
    print(f"   ✅ GUT - Fixed Emotion ist stabil!")
    print(f"   → Kleine Verbesserungen möglich")
elif avg100 > 150:
    print(f"   ⚠️  OK - Besser, aber nicht optimal")
    print(f"   → Parameter weiter anpassen")
else:
    print(f"   ❌ PROBLEM - Immer noch Collapse")
    print(f"   → Emotion-Konzept überdenken")

print(f"\n💾 Log: {csv_path}\n")


