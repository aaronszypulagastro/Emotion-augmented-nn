"""
Competitive Self-Play Training - Phase 8.1
==========================================

KONZEPT: Emotion durch direkten Wettbewerb
-------------------------------------------

Statt:
- Target-Returns kalibrieren
- Alpha, Bounds, Decay tunen
- Emotion manuell steuern

→ Agent konkurriert gegen vergangene Versionen
→ Emotion = Win/Loss Signal (100% klar!)
→ Meta-Learning durch Competition

Workflow:
---------
1. Agent spielt Episode → Score A
2. Past-Self spielt Episode → Score B
3. Compare: A vs B
4. Update Emotion basierend auf Outcome
5. Exploration & Learning-Rate werden von Emotion moduliert

Author: Phase 8.1 - Competitive Meta-Learning
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
    from ..core.performance_stability_analyzer import PerformanceStabilityAnalyzer
except ImportError:
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

print("=" * 67)
print("   COMPETITIVE SELF-PLAY META-LEARNING")
print("   Phase 8.1: Emotion durch Wettbewerb")
print("=" * 67 + "\n")

# Configuration
CONFIG = {
    'env_name': 'CartPole-v1',
    'episodes': 500,
    'max_steps': 500,
    'batch_size': 64,
    'gamma': 0.99,
    'base_lr': 5e-4,              # Wird durch Emotion moduliert
    'target_update_freq': 10,
    'buffer_capacity': 50000,
    
    # Competitive Settings
    'competition_freq': 5,        # Alle N Episodes ein Competition
    'competitor_strategy': 'past_self',  # 'past_self', 'best_self'
    'competitor_history_depth': 50,      # Wie weit zurück für Competitor
    'save_checkpoint_freq': 20,          # Wie oft Checkpoints speichern
}

print("[CONFIG] Konfiguration:")
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

class CompetitiveDQNAgent:
    """DQN Agent mit Competitive Emotion"""
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer - Base LR wird durch Emotion moduliert
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
        """Epsilon-greedy action selection"""
        if epsilon is None:
            # Emotion moduliert Exploration
            # Hohe Emotion (Confidence) → weniger Exploration
            # Niedrige Emotion (Frustration) → mehr Exploration (neue Strategien suchen!)
            emotion_factor = 1.0 - 0.3 * (self.emotion.value - 0.5)  # [0.85, 1.15]
            epsilon = self.base_epsilon * emotion_factor
            epsilon = max(self.epsilon_min, epsilon)
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def play_episode(self, env, deterministic=False):
        """Spiele eine Episode und gib Return zurück"""
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
        """Standard DQN Training Step"""
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
        
        # Emotion-Modulated Learning Rate
        # Hohe Emotion (Pride) → normale LR
        # Niedrige Emotion (Frustration) → erhöhte LR (aggressiver lernen!)
        emotion_lr_factor = 0.7 + 0.6 * self.emotion.value  # [0.7, 1.3]
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['base_lr'] * emotion_lr_factor
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_step_count += 1
        
        return loss.item()
    
    def compete(self, env, episode):
        """
        Competition gegen Past-Self
        
        Returns:
            CompetitionResult
        """
        # 1. Main Agent spielt
        score_main = self.play_episode(env, deterministic=True)
        
        # 2. Hole Competitor
        competitor_state = self.competitor.get_competitor_model_state(episode)
        
        if competitor_state is None:
            # Kein Competitor vorhanden → kein Competition
            return None
        
        # 3. Lade Competitor Network
        competitor_network = QNetwork(self.state_dim, self.action_dim)
        competitor_network.load_state_dict(competitor_state)
        competitor_network.eval()
        
        # 4. Competitor spielt
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
        
        # 5. Update Emotion basierend auf Competition
        result = self.emotion.compete(score_main, score_competitor, episode)
        
        return result
    
    def save_checkpoint(self, episode, avg_score):
        """Speichere Checkpoint für späteren Wettbewerb"""
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

# ==================== TRAINING LOOP ====================

def train_competitive_selfplay():
    """Main Training Loop"""
    
    # Environment
    env = gym.make(CONFIG['env_name'])
    state, _ = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Agent
    agent = CompetitiveDQNAgent(state_dim, action_dim, CONFIG)
    
    # Performance Stability Analyzer
    psa = PerformanceStabilityAnalyzer(
        window_size=100,
        anomaly_threshold=3.0,
        trend_threshold=0.3
    )
    
    # Logging
    log_path = "results/competitive_selfplay_log.csv"
    os.makedirs("results", exist_ok=True)
    
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "return", "epsilon", "emotion", "competitive_mindset",
            "win_rate", "loss_rate", "draw_rate", "win_loss_momentum",
            "had_competition", "competition_outcome", "score_diff",
            "psa_stability", "psa_trend", "lr_actual"
        ])
    
    # Training History
    scores = []
    competition_history = []
    
    print("[START] Training startet...\n")
    
    for episode in tqdm(range(CONFIG['episodes']), desc="Competitive Training"):
        
        # ===== NORMAL EPISODE =====
        state, _ = env.reset()
        total_reward = 0.0
        
        for t in range(CONFIG['max_steps']):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Update Target Network
        if episode % CONFIG['target_update_freq'] == 0:
            agent.update_target_network()
        
        # Decay Epsilon
        agent.decay_epsilon()
        
        # ===== COMPETITION =====
        had_competition = False
        competition_outcome = "none"
        score_diff = 0.0
        
        if episode > 0 and episode % CONFIG['competition_freq'] == 0:
            result = agent.compete(env, episode)
            
            if result is not None:
                had_competition = True
                competition_outcome = result.outcome.value
                score_diff = result.score_diff
                
                competition_history.append(result)
                
                if episode % 20 == 0:
                    print(f"\n[COMPETITION] Episode {episode}:")
                    print(f"   Main: {result.score_self:.1f} vs Competitor: {result.score_competitor:.1f}")
                    print(f"   Outcome: {result.outcome.value}")
                    print(f"   Emotion: {result.new_emotion:.3f} (Delta{result.emotion_delta:+.3f})")
                    print(f"   Mindset: {agent.emotion.get_competitive_mindset()}")
        
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
            
            # Get actual LR
            lr_actual = agent.optimizer.param_groups[0]['lr']
            
            with open(log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode,
                    total_reward,
                    agent.base_epsilon,
                    agent.emotion.value,
                    agent.emotion.get_competitive_mindset(),
                    stats['win_rate'],
                    stats['loss_rate'],
                    stats['draw_rate'],
                    stats['win_loss_momentum'],
                    had_competition,
                    competition_outcome,
                    score_diff,
                    psa_metrics.stability_score,
                    psa_metrics.trend,
                    lr_actual
                ])
        
        # Progress Print
        if episode % 50 == 0 and episode > 0:
            avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            stats = agent.emotion.get_stats()
            
            print(f"\n[STATUS] Episode {episode}:")
            print(f"   Avg100: {avg_100:.1f}")
            print(f"   Emotion: {agent.emotion.value:.3f}")
            print(f"   Mindset: {agent.emotion.get_competitive_mindset()}")
            print(f"   Win Rate: {stats['win_rate']:.1%}")
            print(f"   Epsilon: {agent.base_epsilon:.3f}")
            print(f"   LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
    
    env.close()
    
    # Final Stats
    print("\n" + "="*60)
    print("TRAINING ABGESCHLOSSEN")
    print("="*60)
    
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    print(f"\n[FINAL] Performance:")
    print(f"   Last 100 avg: {final_avg:.1f}")
    print(f"   Best Episode: {max(scores):.1f}")
    
    stats = agent.emotion.get_stats()
    print(f"\n[COMPETITION] Stats:")
    print(f"   Total Competitions: {stats['total_competitions']:.0f}")
    print(f"   Win Rate: {stats['win_rate']:.1%}")
    print(f"   Loss Rate: {stats['loss_rate']:.1%}")
    print(f"   Draw Rate: {stats['draw_rate']:.1%}")
    
    print(f"\n[EMOTION] Final State:")
    print(f"   Emotion: {agent.emotion.value:.3f}")
    print(f"   Mindset: {agent.emotion.get_competitive_mindset()}")
    
    # Save Final Model
    torch.save(agent.q_network.state_dict(), "results/competitive_selfplay_final.pth")
    print(f"\n[OK] Model saved to: results/competitive_selfplay_final.pth")
    print(f"[OK] Log saved to: {log_path}")

if __name__ == "__main__":
    train_competitive_selfplay()

