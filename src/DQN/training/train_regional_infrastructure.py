"""
Regional Infrastructure Meta-Learning - Phase 8.2
=================================================

COMPETITIVE SELF-PLAY + REGIONAL CONDITIONS
--------------------------------------------

Training unter verschiedenen regionalen Produktionsbedingungen:
- China: Schnelle Feedbackschleifen, hohe Automation
- Germany: Qualitätsfokus, moderate Geschwindigkeit  
- USA: High-Tech aber geografisch verteilt
- Brazil/India: Emerging Markets, flexible Prozesse

Das ist BAHNBRECHEND:
1. Erste Integration von Real-World Infrastructure in RL
2. Systematic Benchmark über Regionen
3. Praktischer Wert für Robotik-Deployment
4. Publikationswürdige Forschung

Author: Phase 8.2 - Regional Infrastructure Meta-Learning
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
from copy import deepcopy

# Imports
try:
    from ..core.competitive_emotion_engine import (
        CompetitiveEmotionEngine,
        SelfPlayCompetitor,
        create_competitive_config
    )
    from ..core.infrastructure_profile import (
        InfrastructureProfile,
        create_all_profiles
    )
    from ..core.performance_stability_analyzer import PerformanceStabilityAnalyzer
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.competitive_emotion_engine import (
        CompetitiveEmotionEngine,
        SelfPlayCompetitor,
        create_competitive_config
    )
    from core.infrastructure_profile import (
        InfrastructureProfile,
        create_all_profiles
    )
    from core.performance_stability_analyzer import PerformanceStabilityAnalyzer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("   REGIONAL INFRASTRUCTURE META-LEARNING")
print("   Phase 8.2: Infrastructure-Aware Competitive Learning")
print("=" * 70 + "\n")

# Configuration
CONFIG = {
    'env_name': 'CartPole-v1',
    'episodes_per_region': 300,  # Pro Region
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'base_lr': 5e-4,
    'target_update_freq': 10,
    'buffer_capacity': 50000,
    
    # Competitive Settings
    'competition_freq': 5,
    'competitor_strategy': 'past_self',
    'competitor_history_depth': 50,
    'save_checkpoint_freq': 20,
    
    # Regional Settings
    'regions': ['China', 'Germany', 'USA'],  # Start mit 3 Regionen
    'region_switch_mode': 'sequential',  # oder 'round_robin'
}

print("[CONFIG] Konfiguration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
print()

# Standard Q-Network (same as before)
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

class RegionalCompetitiveDQNAgent:
    """
    DQN Agent mit Regional Infrastructure Modulation
    
    Neu: Infrastructure-Profile beeinflussen:
    - Reward Propagation (delay basierend auf loop_speed)
    - Observation Noise (basierend auf automation)
    - Learning Rate (basierend auf automation efficiency)
    - Exploration (basierend auf error_tolerance)
    """
    
    def __init__(self, state_dim, action_dim, config, infrastructure=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Infrastructure Profile
        self.infrastructure = infrastructure
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['base_lr'])
        
        # Replay Buffer
        self.memory = ReplayBuffer(config['buffer_capacity'])
        
        # Epsilon
        self.base_epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Competitive Emotion Engine
        comp_config = create_competitive_config("balanced")
        self.emotion = CompetitiveEmotionEngine(
            init_emotion=0.5,
            **comp_config
        )
        
        # Self-Play Competitor
        self.competitor = SelfPlayCompetitor(
            strategy=config['competitor_strategy'],
            history_depth=config['competitor_history_depth']
        )
        
        self.train_step_count = 0
    
    def select_action(self, state, epsilon=None):
        """Action selection mit Infrastructure-moduliertem Epsilon"""
        if epsilon is None:
            # Emotion moduliert Exploration
            emotion_factor = 1.0 - 0.3 * (self.emotion.value - 0.5)
            epsilon = self.base_epsilon * emotion_factor
            
            # Infrastructure moduliert auch!
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
        """
        Spiele Episode MIT Infrastructure-Modulation
        
        Infrastructure beeinflusst:
        - Observations (Noise)
        - Rewards (Delay)
        """
        state, _ = env.reset()
        
        # Reset Infrastructure reward buffer
        if self.infrastructure:
            self.infrastructure.reset()
        
        total_reward = 0.0
        step = 0
        
        for step in range(self.config['max_steps']):
            # Infrastructure: Modulate Observation
            if self.infrastructure:
                state = self.infrastructure.modulate_observation(state)
            
            # Select Action
            if deterministic:
                action = self.select_action(state, epsilon=0.0)
            else:
                action = self.select_action(state)
            
            # Environment Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Infrastructure: Modulate Reward (delay)
            if self.infrastructure:
                reward = self.infrastructure.modulate_reward(reward, step)
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        # Flush any remaining delayed rewards
        if self.infrastructure:
            final_reward = self.infrastructure.modulate_reward(0.0, step, flush=True)
            total_reward += final_reward
        
        return total_reward
    
    def train(self):
        """Training Step mit Infrastructure-modulierter LR"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample Batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config['batch_size']
        )
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Infrastructure + Emotion modulated LR
        lr = self.config['base_lr']
        
        # Infrastructure modulation
        if self.infrastructure:
            lr = self.infrastructure.modulate_learning_rate(lr)
        
        # Emotion modulation
        emotion_lr_factor = 0.7 + 0.6 * self.emotion.value
        lr = lr * emotion_lr_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_step_count += 1
        
        return loss.item()
    
    def compete(self, env, episode):
        """Competition gegen Past-Self"""
        # Main Agent
        score_main = self.play_episode(env, deterministic=True)
        
        # Competitor
        competitor_state = self.competitor.get_competitor_model_state(episode)
        
        if competitor_state is None:
            return None
        
        # Load Competitor
        competitor_network = QNetwork(self.state_dim, self.action_dim)
        competitor_network.load_state_dict(competitor_state)
        competitor_network.eval()
        
        # Competitor plays (with same infrastructure!)
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
        
        # Update Emotion
        result = self.emotion.compete(score_main, score_competitor, episode)
        
        return result
    
    def save_checkpoint(self, episode, avg_score):
        """Save checkpoint"""
        self.competitor.save_checkpoint(
            episode=episode,
            model_state_dict=self.q_network.state_dict(),
            avg_score=avg_score
        )
    
    def update_target_network(self):
        """Update Target Network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay Epsilon"""
        self.base_epsilon = max(
            self.epsilon_min,
            self.base_epsilon * self.epsilon_decay
        )
    
    def set_infrastructure(self, infrastructure):
        """Wechsle Infrastructure Profile"""
        self.infrastructure = infrastructure

# ==================== MULTI-REGION TRAINING ====================

def train_multi_region():
    """
    Main Training Loop - trainiert über mehrere Regionen
    """
    
    # Environment
    env = gym.make(CONFIG['env_name'])
    state, _ = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Regional Profiles
    profiles = {}
    for region_name in CONFIG['regions']:
        profiles[region_name] = InfrastructureProfile(region_name)
        print(f"\n[REGION] {region_name} Profile:")
        print(profiles[region_name])
        print()
    
    # Results Storage
    results = {region: {'scores': [], 'emotions': [], 'competitions': []} 
               for region in CONFIG['regions']}
    
    # Main Output Directory
    os.makedirs("results/regional", exist_ok=True)
    
    # Train each region
    for region_name in CONFIG['regions']:
        print("\n" + "="*70)
        print(f"TRAINING: {region_name}")
        print("="*70 + "\n")
        
        infrastructure = profiles[region_name]
        
        # Create Agent for this region
        agent = RegionalCompetitiveDQNAgent(
            state_dim, 
            action_dim, 
            CONFIG, 
            infrastructure=infrastructure
        )
        
        # PSA
        psa = PerformanceStabilityAnalyzer(
            window_size=100,
            anomaly_threshold=3.0,
            trend_threshold=0.3
        )
        
        # Logging
        log_path = f"results/regional/{region_name.lower()}_training.csv"
        
        with open(log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "return", "epsilon", "emotion", "competitive_mindset",
                "win_rate", "loss_rate", "draw_rate", "win_loss_momentum",
                "had_competition", "competition_outcome", "score_diff",
                "psa_stability", "psa_trend", "lr_actual",
                "infrastructure_loop_speed", "infrastructure_automation", "infrastructure_error_tolerance"
            ])
        
        scores = []
        
        print(f"[START] Training in {region_name}...\n")
        
        for episode in tqdm(range(CONFIG['episodes_per_region']), 
                           desc=f"{region_name} Training"):
            
            # Normal Episode
            state, _ = env.reset()
            total_reward = 0.0
            
            infrastructure.reset()
            
            for t in range(CONFIG['max_steps']):
                # Infrastructure: Modulate Observation
                obs = infrastructure.modulate_observation(state)
                
                action = agent.select_action(obs)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Infrastructure: Modulate Reward
                reward_delayed = infrastructure.modulate_reward(reward, t)
                
                agent.memory.push(obs, action, reward_delayed, next_state, done)
                
                # Train
                loss = agent.train()
                
                total_reward += reward_delayed
                state = next_state
                
                if done:
                    break
            
            # Flush remaining rewards
            final_reward = infrastructure.modulate_reward(0.0, t, flush=True)
            total_reward += final_reward
            
            scores.append(total_reward)
            results[region_name]['scores'].append(total_reward)
            results[region_name]['emotions'].append(agent.emotion.value)
            
            # Update Target Network
            if episode % CONFIG['target_update_freq'] == 0:
                agent.update_target_network()
            
            # Decay Epsilon
            agent.decay_epsilon()
            
            # Competition
            had_competition = False
            competition_outcome = "none"
            score_diff = 0.0
            
            if episode > 0 and episode % CONFIG['competition_freq'] == 0:
                result = agent.compete(env, episode)
                
                if result is not None:
                    had_competition = True
                    competition_outcome = result.outcome.value
                    score_diff = result.score_diff
                    results[region_name]['competitions'].append(result)
            
            # Save Checkpoint
            if episode % CONFIG['save_checkpoint_freq'] == 0:
                avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
                agent.save_checkpoint(episode, avg_score)
            
            # Update PSA
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
                        stats['win_rate'], stats['loss_rate'], stats['draw_rate'],
                        stats['win_loss_momentum'],
                        had_competition, competition_outcome, score_diff,
                        psa_metrics.stability_score, psa_metrics.trend, lr_actual,
                        infrastructure.loop_speed, infrastructure.automation, infrastructure.error_tolerance
                    ])
            
            # Progress Print
            if episode % 50 == 0 and episode > 0:
                avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"\n[STATUS] {region_name} Episode {episode}:")
                print(f"   Avg100: {avg_100:.1f}")
                print(f"   Emotion: {agent.emotion.value:.3f}")
                print(f"   LR: {lr_actual:.6f}")
        
        # Save Final Model
        model_path = f"results/regional/{region_name.lower()}_final.pth"
        torch.save(agent.q_network.state_dict(), model_path)
        print(f"\n[OK] {region_name} model saved: {model_path}")
        
        # Final Stats for Region
        final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        stats = agent.emotion.get_stats()
        
        print(f"\n[FINAL] {region_name} Performance:")
        print(f"   Last 100 avg: {final_avg:.1f}")
        print(f"   Best Episode: {max(scores):.1f}")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        print(f"   Final Emotion: {agent.emotion.value:.3f}")
    
    env.close()
    
    # ===== COMPARISON SUMMARY =====
    print("\n" + "="*70)
    print("REGIONAL COMPARISON")
    print("="*70 + "\n")
    
    for region_name in CONFIG['regions']:
        scores = results[region_name]['scores']
        final_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        best = max(scores)
        
        print(f"{region_name:12s}: avg100={final_100:6.1f} | best={best:6.1f}")
    
    print("\n[OK] Multi-Region Training Complete!")
    print(f"[OK] Results saved in: results/regional/")
    
    return results

if __name__ == "__main__":
    results = train_multi_region()





