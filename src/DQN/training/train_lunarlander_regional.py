"""
LunarLander Regional Infrastructure Training - Phase 8.2.1
==========================================================

ADVANCED VALIDATION: LunarLander-v2 across regions
---------------------------------------------------

LunarLander is MOST CHALLENGING:
- 8D continuous state space
- 4 discrete actions (including engine control)
- Complex reward structure (landing, fuel, angle)
- Success: avg100 > 200

This is the ULTIMATE test for Regional Infrastructure!

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
print("   LUNARLANDER REGIONAL INFRASTRUCTURE TRAINING")
print("   Phase 8.2.1: Advanced Multi-Environment Validation")
print("=" * 70 + "\n")

CONFIG = {
    'env_name': 'LunarLander-v2',
    'episodes_per_region': 800,  # Noch mehr für komplexen Task
    'max_steps': 1000,  # Längere Episodes
    'batch_size': 64,
    'gamma': 0.99,
    'base_lr': 5e-4,
    'target_update_freq': 10,
    'buffer_capacity': 100000,  # Größerer Buffer
    
    'competition_freq': 15,  # Noch seltener (mehr Stabilität)
    'competitor_strategy': 'past_self',
    'competitor_history_depth': 80,  # Weiter zurück
    'save_checkpoint_freq': 40,
    
    'regions': ['China', 'Germany', 'USA'],
}

print("[CONFIG] LunarLander Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
print()

# Larger network for LunarLander (8D state)
class LunarLanderQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # Größer!
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    
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

class LunarLanderRegionalAgent:
    """LunarLander Agent with Regional Infrastructure"""
    
    def __init__(self, state_dim, action_dim, config, infrastructure=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.infrastructure = infrastructure
        
        # Larger networks for complex task
        self.q_network = LunarLanderQNetwork(state_dim, action_dim)
        self.target_network = LunarLanderQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['base_lr'])
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        self.base_epsilon = 1.0
        self.epsilon_decay = 0.998  # Sehr langsam für LunarLander
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
            emotion_factor = 1.0 - 0.2 * (self.emotion.value - 0.5)  # Weniger Modulation
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
            
            action = self.select_action(state, epsilon=0.0 if deterministic else None)
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
        
        lr = self.config['base_lr']
        if self.infrastructure:
            lr = self.infrastructure.modulate_learning_rate(lr)
        
        emotion_lr_factor = 0.8 + 0.4 * self.emotion.value  # Weniger aggressive Modulation
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
        
        competitor_network = LunarLanderQNetwork(self.state_dim, self.action_dim)
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

def train_lunarlander_multi_region():
    """Multi-Region LunarLander Training"""
    
    env = gym.make(CONFIG['env_name'])
    state, _ = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"[ENV] LunarLander-v2: state_dim={state_dim}, action_dim={action_dim}\n")
    
    profiles = {}
    for region_name in CONFIG['regions']:
        profiles[region_name] = InfrastructureProfile(region_name)
        print(f"[REGION] {region_name}: Loop={profiles[region_name].loop_speed:.2f}")
    print()
    
    results = {region: {'scores': [], 'emotions': []} for region in CONFIG['regions']}
    os.makedirs("results/regional_lunarlander", exist_ok=True)
    
    for region_name in CONFIG['regions']:
        print(f"\n{'='*70}\nTRAINING: {region_name} on LunarLander\n{'='*70}\n")
        
        infrastructure = profiles[region_name]
        agent = LunarLanderRegionalAgent(state_dim, action_dim, CONFIG, infrastructure)
        psa = PerformanceStabilityAnalyzer(window_size=100)
        
        log_path = f"results/regional_lunarlander/{region_name.lower()}_lunarlander.csv"
        
        with open(log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "return", "epsilon", "emotion", "competitive_mindset",
                "win_rate", "had_competition", "psa_stability", "lr_actual"
            ])
        
        scores = []
        
        for episode in tqdm(range(CONFIG['episodes_per_region']), desc=f"{region_name} LunarLander"):
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
            
            if episode % CONFIG['target_update_freq'] == 0:
                agent.update_target_network()
            agent.decay_epsilon()
            
            had_competition = False
            if episode > 0 and episode % CONFIG['competition_freq'] == 0:
                result = agent.compete(env, episode)
                if result:
                    had_competition = True
            
            if episode % CONFIG['save_checkpoint_freq'] == 0:
                avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
                agent.save_checkpoint(episode, avg_score)
            
            psa.update(episode, total_reward)
            psa_metrics = psa.compute_stability_metrics()
            
            if episode % 20 == 0:
                stats = agent.emotion.get_stats()
                lr = agent.optimizer.param_groups[0]['lr']
                
                with open(log_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode, total_reward, agent.base_epsilon, agent.emotion.value,
                        agent.emotion.get_competitive_mindset(),
                        stats.get('win_rate', 0.0), had_competition,
                        psa_metrics.stability_score, lr
                    ])
            
            if episode % 100 == 0 and episode > 0:
                avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"\n[STATUS] {region_name} Ep {episode}: Avg100={avg_100:.1f} | Emo={agent.emotion.value:.3f}")
        
        model_path = f"results/regional_lunarlander/{region_name.lower()}_lunarlander_final.pth"
        torch.save(agent.q_network.state_dict(), model_path)
        
        final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        print(f"\n[FINAL] {region_name}: avg100={final_avg:.1f} | best={max(scores):.1f}")
    
    env.close()
    print(f"\n[OK] LunarLander Multi-Region Complete!")

if __name__ == "__main__":
    train_lunarlander_multi_region()





