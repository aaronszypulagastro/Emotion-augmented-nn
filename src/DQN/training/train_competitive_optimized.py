"""
Optimized Competitive Self-Play - Level 1 Improvements
=======================================================

IMPROVEMENTS from baseline competitive_selfplay.py:
1. Competition frequency: 5 → 20 (less disruption)
2. LR modulation: [0.7, 1.3] → [0.9, 1.1] (more stable)
3. Episodes: 500 → 1000 (better convergence)
4. Soft target updates (smoother learning)

Expected Impact: +50-80% performance improvement

Author: Level 1 Optimization
Date: 2025-10-17
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.competitive_emotion_engine import (
    CompetitiveEmotionEngine,
    SelfPlayCompetitor,
    create_competitive_config
)
from core.performance_stability_analyzer import PerformanceStabilityAnalyzer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("   OPTIMIZED COMPETITIVE SELF-PLAY (LEVEL 1)")
print("   Expected: +50-80% performance improvement")
print("=" * 70 + "\n")

CONFIG = {
    'env_name': 'CartPole-v1',
    'episodes': 1000,  # DOUBLED for better convergence
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'base_lr': 5e-4,
    'target_update_freq': 10,
    'buffer_capacity': 50000,
    
    # OPTIMIZED Competitive Settings
    'competition_freq': 20,  # Was: 5 → Less disruption!
    'competitor_strategy': 'past_self',
    'competitor_history_depth': 50,
    'save_checkpoint_freq': 25,
    
    # NEW: Soft target updates
    'use_soft_updates': True,
    'tau': 0.005,  # Soft update rate
}

print("[CONFIG] Optimized Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
print()

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

class OptimizedCompetitiveDQNAgent:
    """Optimized Competitive DQN Agent with Level 1 improvements"""
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['base_lr'])
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        self.base_epsilon = 1.0
        self.epsilon_decay = 0.996  # Slightly slower for 1000 episodes
        self.epsilon_min = 0.01
        
        comp_config = create_competitive_config("balanced")
        self.emotion = CompetitiveEmotionEngine(init_emotion=0.5, **comp_config)
        
        self.competitor = SelfPlayCompetitor(
            strategy=config['competitor_strategy'],
            history_depth=config['competitor_history_depth']
        )
        
        self.train_step_count = 0
    
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            # IMPROVED: Inverse emotion-exploration relationship
            # Low emotion (frustrated) → MORE exploration
            emotion_factor = 1.3 - 0.6 * self.emotion.value  # [0.7, 1.3]
            epsilon = self.base_epsilon * emotion_factor
            epsilon = max(self.epsilon_min, epsilon)
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def play_episode(self, env, deterministic=False):
        state, _ = env.reset()
        total_reward = 0.0
        
        for _ in range(self.config['max_steps']):
            if deterministic:
                action = self.select_action(state, epsilon=0.0)
            else:
                action = self.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        return total_reward
    
    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return None
        
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
        
        # IMPROVED: Reduced LR modulation range
        lr = self.config['base_lr']
        emotion_lr_factor = 0.9 + 0.2 * self.emotion.value  # [0.9, 1.1] instead of [0.7, 1.3]
        lr = lr * emotion_lr_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_step_count += 1
        
        return loss.item()
    
    def compete(self, env, episode):
        score_main = self.play_episode(env, deterministic=True)
        
        competitor_state = self.competitor.get_competitor_model_state(episode)
        if competitor_state is None:
            return None
        
        competitor_network = QNetwork(self.state_dim, self.action_dim)
        competitor_network.load_state_dict(competitor_state)
        competitor_network.eval()
        
        state, _ = env.reset()
        score_competitor = 0.0
        
        for _ in range(self.config['max_steps']):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = competitor_network(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            score_competitor += reward
            
            if done:
                break
            
            state = next_state
        
        result = self.emotion.compete(score_main, score_competitor, episode)
        return result
    
    def save_checkpoint(self, episode, avg_score):
        self.competitor.save_checkpoint(episode, self.q_network.state_dict(), avg_score)
    
    def update_target_network(self, soft=False):
        """
        NEW: Support soft (Polyak) updates
        """
        if soft and self.config.get('use_soft_updates', False):
            tau = self.config.get('tau', 0.005)
            for target_param, param in zip(self.target_network.parameters(), 
                                          self.q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.base_epsilon = max(self.epsilon_min, self.base_epsilon * self.epsilon_decay)

# ==================== TRAINING ====================

def train_optimized():
    env = gym.make(CONFIG['env_name'])
    state, _ = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = OptimizedCompetitiveDQNAgent(state_dim, action_dim, CONFIG)
    psa = PerformanceStabilityAnalyzer(window_size=100)
    
    log_path = "results/competitive_optimized_log.csv"
    os.makedirs("results", exist_ok=True)
    
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "return", "epsilon", "emotion", "competitive_mindset",
            "win_rate", "had_competition", "lr_actual"
        ])
    
    scores = []
    
    print("[START] Optimized training...\n")
    
    for episode in tqdm(range(CONFIG['episodes']), desc="Optimized Competitive"):
        state, _ = env.reset()
        total_reward = 0.0
        
        for t in range(CONFIG['max_steps']):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()
            
            # NEW: Soft target updates every step
            if CONFIG.get('use_soft_updates', False):
                agent.update_target_network(soft=True)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Hard target update (less frequent with soft updates)
        if not CONFIG.get('use_soft_updates', False):
            if episode % CONFIG['target_update_freq'] == 0:
                agent.update_target_network(soft=False)
        
        agent.decay_epsilon()
        
        # Competition (LESS FREQUENT!)
        had_competition = False
        if episode > 0 and episode % CONFIG['competition_freq'] == 0:
            result = agent.compete(env, episode)
            if result is not None:
                had_competition = True
                
                if episode % 100 == 0:
                    print(f"\n[COMPETITION] Ep {episode}: "
                          f"Score {result.score_self:.0f} vs {result.score_competitor:.0f} | "
                          f"Outcome: {result.outcome.value} | "
                          f"Emotion: {result.new_emotion:.3f}")
        
        if episode % CONFIG['save_checkpoint_freq'] == 0:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            agent.save_checkpoint(episode, avg_score)
        
        psa.update(episode, total_reward)
        
        # Logging
        if episode % 10 == 0:
            stats = agent.emotion.get_stats()
            lr_actual = agent.optimizer.param_groups[0]['lr']
            
            with open(log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, total_reward, agent.base_epsilon, agent.emotion.value,
                    agent.emotion.get_competitive_mindset(),
                    stats.get('win_rate', 0.0),
                    had_competition, lr_actual
                ])
        
        if episode % 100 == 0 and episode > 0:
            avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"\n[STATUS] Ep {episode}: Avg100={avg_100:.1f} | Emo={agent.emotion.value:.3f} | ε={agent.base_epsilon:.3f}")
    
    env.close()
    
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    
    print("\n" + "="*70)
    print("OPTIMIZED RESULTS")
    print("="*70)
    print(f"\nFinal avg100: {final_avg:.1f}")
    print(f"Best episode: {max(scores):.1f}")
    print(f"Final emotion: {agent.emotion.value:.3f}")
    
    stats = agent.emotion.get_stats()
    print(f"\nCompetition Stats:")
    print(f"  Win Rate: {stats.get('win_rate', 0.0):.1%}")
    print(f"  Total Competitions: {stats.get('total_competitions', 0):.0f}")
    
    # Compare to baseline
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE:")
    print("="*70)
    baseline_avg = 131.7  # From previous run
    improvement = ((final_avg - baseline_avg) / baseline_avg) * 100
    
    print(f"  Baseline (original):  {baseline_avg:.1f}")
    print(f"  Optimized (Level 1):  {final_avg:.1f}")
    print(f"  Improvement:          {improvement:+.1f}%")
    
    if final_avg > baseline_avg * 1.5:
        print("\n  ✅ EXCELLENT: >50% improvement achieved!")
    elif final_avg > baseline_avg * 1.3:
        print("\n  ✅ GOOD: >30% improvement achieved!")
    elif final_avg > baseline_avg * 1.1:
        print("\n  ⚠️  MODERATE: >10% improvement")
    else:
        print("\n  ❌ MINIMAL: Need Level 2 improvements (Rainbow DQN)")
    
    torch.save(agent.q_network.state_dict(), "results/competitive_optimized_final.pth")
    print(f"\n[OK] Saved: results/competitive_optimized_log.csv")
    print(f"[OK] Saved: results/competitive_optimized_final.pth")

if __name__ == "__main__":
    train_optimized()





