"""
Acrobot Regional Infrastructure Training - Phase 8.2.1
======================================================

EXTENDED VALIDATION: Acrobot-v1 across regions
------------------------------------------------

Acrobot is MORE CHALLENGING than CartPole:
- 6D continuous state space (vs 4D)
- 3 discrete actions (vs 2)
- Sparse rewards (only -1 per step)
- Success: reach top position

This tests if Regional Infrastructure generalizes!

Author: Phase 8.2.1 - Multi-Environment Validation
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

# Imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.competitive_emotion_engine import (
    CompetitiveEmotionEngine,
    SelfPlayCompetitor,
    create_competitive_config
)
from core.infrastructure_profile import InfrastructureProfile
from core.performance_stability_analyzer import PerformanceStabilityAnalyzer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("   ACROBOT REGIONAL INFRASTRUCTURE TRAINING")
print("   Phase 8.2.1: Multi-Environment Validation")
print("=" * 70 + "\n")

# Configuration
CONFIG = {
    'env_name': 'Acrobot-v1',
    'episodes_per_region': 500,  # Mehr Episodes (schwieriger Task)
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'base_lr': 5e-4,
    'target_update_freq': 10,
    'buffer_capacity': 50000,
    
    # Competitive Settings
    'competition_freq': 10,  # Weniger frequent (stabilerer)
    'competitor_strategy': 'past_self',
    'competitor_history_depth': 50,
    'save_checkpoint_freq': 25,
    
    # Regional Settings
    'regions': ['China', 'Germany', 'USA'],
}

print("[CONFIG] Acrobot Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
print()

# Q-Network (angepasst für Acrobot's 6D state)
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

class AcrobotRegionalAgent:
    """Acrobot Agent mit Regional Infrastructure"""
    
    def __init__(self, state_dim, action_dim, config, infrastructure=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.infrastructure = infrastructure
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['base_lr'])
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        # Epsilon
        self.base_epsilon = 1.0
        self.epsilon_decay = 0.997  # Langsamer für Acrobot
        self.epsilon_min = 0.01
        
        # Competitive Emotion
        comp_config = create_competitive_config("balanced")
        self.emotion = CompetitiveEmotionEngine(init_emotion=0.5, **comp_config)
        
        # Self-Play
        self.competitor = SelfPlayCompetitor(
            strategy=config['competitor_strategy'],
            history_depth=config['competitor_history_depth']
        )
        
        self.train_step_count = 0
    
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            emotion_factor = 1.0 - 0.3 * (self.emotion.value - 0.5)
            epsilon = self.base_epsilon * emotion_factor
            
            if self.infrastructure:
                epsilon = self.infrastructure.modulate_exploration(epsilon)
            
            epsilon = max(self.epsilon_min, epsilon)
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def play_episode(self, env, deterministic=False):
        state, _ = env.reset()
        
        if self.infrastructure:
            self.infrastructure.reset()
        
        total_reward = 0.0
        step = 0
        
        for step in range(self.config['max_steps']):
            if self.infrastructure:
                state = self.infrastructure.modulate_observation(state)
            
            if deterministic:
                action = self.select_action(state, epsilon=0.0)
            else:
                action = self.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if self.infrastructure:
                reward = self.infrastructure.modulate_reward(reward, step)
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        if self.infrastructure:
            final_reward = self.infrastructure.modulate_reward(0.0, step, flush=True)
            total_reward += final_reward
        
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
        
        # Infrastructure + Emotion modulated LR
        lr = self.config['base_lr']
        
        if self.infrastructure:
            lr = self.infrastructure.modulate_learning_rate(lr)
        
        emotion_lr_factor = 0.7 + 0.6 * self.emotion.value
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
        if self.infrastructure:
            self.infrastructure.reset()
        
        score_competitor = 0.0
        step = 0
        
        for step in range(self.config['max_steps']):
            if self.infrastructure:
                state = self.infrastructure.modulate_observation(state)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = competitor_network(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if self.infrastructure:
                reward = self.infrastructure.modulate_reward(reward, step)
            
            score_competitor += reward
            
            if done:
                break
            
            state = next_state
        
        if self.infrastructure:
            final_reward = self.infrastructure.modulate_reward(0.0, step, flush=True)
            score_competitor += final_reward
        
        result = self.emotion.compete(score_main, score_competitor, episode)
        return result
    
    def save_checkpoint(self, episode, avg_score):
        self.competitor.save_checkpoint(episode, self.q_network.state_dict(), avg_score)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.base_epsilon = max(self.epsilon_min, self.base_epsilon * self.epsilon_decay)

# ==================== TRAINING LOOP ====================

def train_acrobot_multi_region():
    """Multi-Region Acrobot Training"""
    
    env = gym.make(CONFIG['env_name'])
    state, _ = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"[ENV] Acrobot-v1: state_dim={state_dim}, action_dim={action_dim}\n")
    
    # Load regional profiles
    profiles = {}
    for region_name in CONFIG['regions']:
        profiles[region_name] = InfrastructureProfile(region_name)
        print(f"[REGION] {region_name}:")
        print(f"  Loop: {profiles[region_name].loop_speed:.2f} | "
              f"Auto: {profiles[region_name].automation:.2f} | "
              f"Tol: {profiles[region_name].error_tolerance:.2f}")
    print()
    
    # Results
    results = {region: {'scores': [], 'emotions': []} for region in CONFIG['regions']}
    
    os.makedirs("results/regional_acrobot", exist_ok=True)
    
    # Train each region
    for region_name in CONFIG['regions']:
        print("\n" + "="*70)
        print(f"TRAINING: {region_name} on Acrobot")
        print("="*70 + "\n")
        
        infrastructure = profiles[region_name]
        agent = AcrobotRegionalAgent(state_dim, action_dim, CONFIG, infrastructure)
        psa = PerformanceStabilityAnalyzer(window_size=100, anomaly_threshold=3.0, trend_threshold=0.3)
        
        log_path = f"results/regional_acrobot/{region_name.lower()}_acrobot.csv"
        
        with open(log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "return", "epsilon", "emotion", "competitive_mindset",
                "win_rate", "had_competition", "competition_outcome", "psa_stability",
                "infrastructure_loop_speed", "infrastructure_automation", "lr_actual"
            ])
        
        scores = []
        
        print(f"[START] {region_name} training...\n")
        
        for episode in tqdm(range(CONFIG['episodes_per_region']), desc=f"{region_name} Acrobot"):
            state, _ = env.reset()
            total_reward = 0.0
            infrastructure.reset()
            
            for t in range(CONFIG['max_steps']):
                obs = infrastructure.modulate_observation(state)
                action = agent.select_action(obs)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                reward_delayed = infrastructure.modulate_reward(reward, t)
                agent.memory.push(obs, action, reward_delayed, next_state, done)
                agent.train()
                
                total_reward += reward_delayed
                state = next_state
                
                if done:
                    break
            
            final_reward = infrastructure.modulate_reward(0.0, t, flush=True)
            total_reward += final_reward
            
            scores.append(total_reward)
            results[region_name]['scores'].append(total_reward)
            results[region_name]['emotions'].append(agent.emotion.value)
            
            if episode % CONFIG['target_update_freq'] == 0:
                agent.update_target_network()
            
            agent.decay_epsilon()
            
            # Competition
            had_competition = False
            competition_outcome = "none"
            
            if episode > 0 and episode % CONFIG['competition_freq'] == 0:
                result = agent.compete(env, episode)
                if result is not None:
                    had_competition = True
                    competition_outcome = result.outcome.value
            
            if episode % CONFIG['save_checkpoint_freq'] == 0:
                avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
                agent.save_checkpoint(episode, avg_score)
            
            psa.update(episode, total_reward)
            psa_metrics = psa.compute_stability_metrics()
            
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
                        had_competition, competition_outcome,
                        psa_metrics.stability_score,
                        infrastructure.loop_speed, infrastructure.automation, lr_actual
                    ])
            
            # Progress
            if episode % 50 == 0 and episode > 0:
                avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"\n[STATUS] {region_name} Episode {episode}/{CONFIG['episodes_per_region']}:")
                print(f"   Avg100: {avg_100:.1f}")
                print(f"   Emotion: {agent.emotion.value:.3f}")
                print(f"   Epsilon: {agent.base_epsilon:.3f}")
        
        # Save final model
        model_path = f"results/regional_acrobot/{region_name.lower()}_acrobot_final.pth"
        torch.save(agent.q_network.state_dict(), model_path)
        
        final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        stats = agent.emotion.get_stats()
        
        print(f"\n[FINAL] {region_name} Acrobot Results:")
        print(f"   Last 100 avg: {final_avg:.1f}")
        print(f"   Best Episode: {max(scores):.1f}")
        print(f"   Win Rate: {stats.get('win_rate', 0.0):.1%}")
        print(f"   Final Emotion: {agent.emotion.value:.3f}")
    
    env.close()
    
    # Final Comparison
    print("\n" + "="*70)
    print("ACROBOT REGIONAL COMPARISON")
    print("="*70 + "\n")
    
    for region_name in CONFIG['regions']:
        scores = results[region_name]['scores']
        final_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        best = max(scores)
        
        print(f"{region_name:12s}: avg100={final_100:7.1f} | best={best:7.1f}")
    
    print("\n[OK] Acrobot Multi-Region Training Complete!")

if __name__ == "__main__":
    train_acrobot_multi_region()





