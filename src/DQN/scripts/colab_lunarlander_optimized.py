"""
Google Colab - OPTIMIZED Rainbow DQN + Emotion Engine
====================================================

Phase 1: LunarLander Optimization
- 2000 Episodes for better convergence
- Optimized hyperparameters
- Larger network architecture
- Enhanced training strategies

Author: Phase 8.2.1 - Multi-Environment Validation
Date: 2025-10-17
"""

# =============================================================================
# DEPENDENCIES (Run this cell first)
# =============================================================================

!pip install torch torchvision torchaudio
!pip install gymnasium[box2d]
!pip install pandas matplotlib seaborn tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple
import random
from tqdm import tqdm
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Test LunarLander
print("🧪 Testing LunarLander-v3...")
try:
    env = gym.make('LunarLander-v3')
    print("✅ LunarLander-v3 works!")
    env.close()
except Exception as e:
    print(f"❌ Error: {e}")

# =============================================================================
# ENHANCED PRIORITIZED EXPERIENCE REPLAY BUFFER
# =============================================================================

class EnhancedPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        if self.size < batch_size:
            return None, None, None
            
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights
        
    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) + 1e-6
            
    def __len__(self):
        return self.size

# =============================================================================
# ENHANCED DUELING NETWORK ARCHITECTURE
# =============================================================================

class EnhancedDuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=1024):
        super(EnhancedDuelingNetwork, self).__init__()
        
        # Larger shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# =============================================================================
# ENHANCED COMPETITIVE EMOTION ENGINE
# =============================================================================

class EnhancedCompetitiveEmotionEngine:
    def __init__(self, alpha=0.1, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.emotion = 0.5
        self.momentum = 0.0
        self.past_scores = deque(maxlen=100)  # Larger history
        
        # Adaptive thresholds based on environment
        self.thresholds = {
            'decisive_win': 0.25,   # 25% improvement
            'win': 0.08,            # 8% improvement
            'draw': 0.03,           # 3% improvement
            'loss': -0.08,          # 8% worse
            'decisive_loss': -0.25  # 25% worse
        }
        
        # Adaptive learning
        self.alpha_decay = 0.9999
        self.min_alpha = 0.01
        
    def update(self, current_score, episode):
        self.past_scores.append(current_score)
        
        if len(self.past_scores) < 20:  # Need more history
            return self.emotion, 'insufficient_data'
            
        # Compare with past performance
        past_avg = np.mean(list(self.past_scores)[:-1])
        
        if past_avg == 0:
            relative_improvement = 0
        else:
            relative_improvement = (current_score - past_avg) / abs(past_avg)
            
        # Determine outcome
        outcome = self._determine_outcome(relative_improvement)
        
        # Update emotion with adaptive learning rate
        emotion_delta = self._outcome_to_emotion_delta(outcome)
        self.emotion = np.clip(self.emotion + self.alpha * emotion_delta, 0.2, 0.8)
        
        # Update momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * np.sign(emotion_delta)
        
        # Decay learning rate
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        
        return self.emotion, outcome
        
    def _determine_outcome(self, relative_improvement):
        if relative_improvement >= self.thresholds['decisive_win']:
            return 'decisive_win'
        elif relative_improvement >= self.thresholds['win']:
            return 'win'
        elif relative_improvement >= self.thresholds['draw']:
            return 'draw'
        elif relative_improvement <= self.thresholds['decisive_loss']:
            return 'decisive_loss'
        else:
            return 'loss'
            
    def _outcome_to_emotion_delta(self, outcome):
        deltas = {
            'decisive_win': 0.25,
            'win': 0.08,
            'draw': 0.0,
            'loss': -0.08,
            'decisive_loss': -0.25
        }
        return deltas.get(outcome, 0.0)
        
    def get_mindset(self):
        if self.emotion > 0.7:
            return 'CONFIDENT'
        elif self.emotion > 0.6:
            return 'BALANCED'
        elif self.emotion > 0.5:
            return 'DETERMINED'
        else:
            return 'FRUSTRATED'

# =============================================================================
# ENHANCED RAINBOW DQN AGENT
# =============================================================================

class EnhancedRainbowDQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, 
                 epsilon_start=0.99, epsilon_end=0.01, epsilon_decay=0.999,
                 buffer_size=200000, batch_size=128, target_update=500):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Enhanced networks
        self.q_network = EnhancedDuelingNetwork(state_size, action_size).to(device)
        self.target_network = EnhancedDuelingNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Enhanced replay buffer
        self.memory = EnhancedPrioritizedReplayBuffer(buffer_size)
        
        # Enhanced emotion engine
        self.emotion_engine = EnhancedCompetitiveEmotionEngine()
        
        # Training tracking
        self.training_step = 0
        self.episode_rewards = []
        self.emotion_history = []
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
        
    def step(self, state, action, reward, next_state, done):
        # Store experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn
        if len(self.memory) > self.batch_size:
            self.learn()
            
    def learn(self):
        # Sample batch
        batch, indices, weights = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # TD errors
        td_errors = (current_q_values - target_q_values).squeeze()
        
        # Weighted loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none').squeeze()).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
    def update_emotion(self, episode_reward, episode):
        emotion, outcome = self.emotion_engine.update(episode_reward, episode)
        self.emotion_history.append(emotion)
        return emotion, outcome

# =============================================================================
# ENHANCED TRAINING FUNCTION
# =============================================================================

def train_enhanced_rainbow_lunarlander(episodes=2000, max_steps=1000):
    """Train Enhanced Rainbow DQN on LunarLander-v3"""
    
    # Environment
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"🎯 Environment: LunarLander-v3")
    print(f"📊 State size: {state_size}, Action size: {action_size}")
    print(f"🚀 Enhanced Training: {episodes} episodes")
    
    # Enhanced Agent
    agent = EnhancedRainbowDQNAgent(state_size, action_size)
    
    # Training tracking
    scores = []
    avg_scores = []
    emotion_history = []
    outcome_history = []
    solved_episode = None
    
    print(f"🚀 Starting ENHANCED training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in tqdm(range(episodes), desc="Enhanced Training"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Update emotion
        emotion, outcome = agent.update_emotion(episode_reward, episode)
        
        # Track results
        scores.append(episode_reward)
        emotion_history.append(emotion)
        outcome_history.append(outcome)
        
        # Calculate average
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            if episode % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                      f"Emotion: {emotion:.3f}, Mindset: {agent.emotion_engine.get_mindset()}, "
                      f"Time: {elapsed_time/60:.1f}min")
                
                # Check if solved
                if avg_score >= 200 and solved_episode is None:
                    solved_episode = episode
                    print(f"🎉 SOLVED in {episode} episodes! (avg100: {avg_score:.2f})")
                    
    env.close()
    
    total_time = time.time() - start_time
    print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
    
    return {
        'scores': scores,
        'avg_scores': avg_scores,
        'emotion_history': emotion_history,
        'outcome_history': outcome_history,
        'agent': agent,
        'solved_episode': solved_episode,
        'training_time': total_time
    }

# =============================================================================
# ENHANCED VISUALIZATION
# =============================================================================

def plot_enhanced_results(results):
    """Plot enhanced training results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scores
    axes[0, 0].plot(results['scores'], alpha=0.6, color='blue')
    if len(results['scores']) > 100:
        # Moving average
        window = 50
        moving_avg = pd.Series(results['scores']).rolling(window=window).mean()
        axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average scores
    if results['avg_scores']:
        axes[0, 1].plot(results['avg_scores'], color='green', linewidth=2)
        axes[0, 1].axhline(y=200, color='r', linestyle='--', label='Solved (200)')
        if results['solved_episode']:
            axes[0, 1].axvline(x=results['solved_episode'], color='orange', 
                              linestyle=':', label=f'Solved at {results["solved_episode"]}')
        axes[0, 1].set_title('Average Scores (100 episodes)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Emotion history
    axes[1, 0].plot(results['emotion_history'], color='purple', alpha=0.7)
    axes[1, 0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='BALANCED')
    axes[1, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='CONFIDENT')
    axes[1, 0].set_title('Emotion Evolution')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Emotion Level')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Outcome distribution
    outcome_counts = pd.Series(results['outcome_history']).value_counts()
    colors = ['green', 'lightgreen', 'orange', 'red', 'darkred', 'gray']
    axes[1, 1].pie(outcome_counts.values, labels=outcome_counts.index, 
                   autopct='%1.1f%%', colors=colors[:len(outcome_counts)])
    axes[1, 1].set_title('Competition Outcomes')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎮 ENHANCED Rainbow DQN + Emotion Engine - LunarLander Optimization")
    print("=" * 70)
    
    # Train with enhanced settings
    results = train_enhanced_rainbow_lunarlander(episodes=2000)
    
    # Plot results
    plot_enhanced_results(results)
    
    # Final statistics
    final_avg = np.mean(results['scores'][-100:]) if len(results['scores']) >= 100 else np.mean(results['scores'])
    best_episode = max(results['scores'])
    
    print(f"\n🏆 ENHANCED Results:")
    print(f"   Final Average (100 episodes): {final_avg:.2f}")
    print(f"   Best Episode: {best_episode:.2f}")
    print(f"   Final Emotion: {results['emotion_history'][-1]:.3f}")
    print(f"   Final Mindset: {results['agent'].emotion_engine.get_mindset()}")
    print(f"   Training Time: {results['training_time']/60:.1f} minutes")
    
    if results['solved_episode']:
        print(f"🎉 SUCCESS: LunarLander solved in {results['solved_episode']} episodes!")
    elif final_avg >= 200:
        print("🎉 SUCCESS: LunarLander solved!")
    else:
        print(f"📈 Progress: {final_avg:.2f}/200 ({(final_avg/200)*100:.1f}% of target)")
        
    # Performance comparison
    print(f"\n📊 Performance Analysis:")
    print(f"   Improvement over baseline: {((final_avg - 100) / 100) * 100:.1f}%")
    print(f"   Emotion stability: {np.std(results['emotion_history'][-100:]):.3f}")
    print(f"   Win rate: {(pd.Series(results['outcome_history']).isin(['win', 'decisive_win']).sum() / len(results['outcome_history'])) * 100:.1f}%")
