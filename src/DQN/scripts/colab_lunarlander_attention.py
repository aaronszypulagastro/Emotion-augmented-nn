"""
Google Colab - ATTENTION MECHANISMS + Enhanced Emotion Engine + LunarLander
===========================================================================

Phase 5.1: Attention Mechanisms Implementation
- Multi-Head Attention für State-Representation
- Enhanced Emotion Engine Integration
- LunarLander-v3 Optimization

Author: Enhanced Meta-Learning Project
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
# ATTENTION STATE ENCODER
# =============================================================================

class AttentionStateEncoder(nn.Module):
    def __init__(self, state_size, hidden_size=512, num_heads=8):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # State Feature Extraction
        self.state_features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Output Projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, state):
        # Extract state features
        state_features = self.state_features(state)
        
        # Add sequence dimension for attention
        state_seq = state_features.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Self-Attention
        attended_features, attention_weights = self.attention(
            state_seq, state_seq, state_seq
        )
        
        # Residual connection + Layer Norm
        attended_features = self.layer_norm1(attended_features + state_seq)
        
        # Feed Forward
        ff_output = self.feed_forward(attended_features)
        
        # Residual connection + Layer Norm
        output = self.layer_norm2(ff_output + attended_features)
        
        # Remove sequence dimension
        output = output.squeeze(1)
        
        # Final projection
        output = self.output_projection(output)
        
        return output, attention_weights

# =============================================================================
# ATTENTION DUELING NETWORK
# =============================================================================

class AttentionDuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512, num_heads=8):
        super().__init__()
        
        # Attention State Encoder
        self.attention_encoder = AttentionStateEncoder(
            state_size, hidden_size, num_heads
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        # Get attention-encoded features
        features, attention_weights = self.attention_encoder(x)
        
        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, attention_weights

# =============================================================================
# ENHANCED EMOTION ENGINE (from previous implementation)
# =============================================================================

class EnhancedEmotionEngine:
    def __init__(self, alpha=0.15, beta=0.85, initial_emotion=0.5, threshold=0.1, momentum=0.3, sensitivity=1.2):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.emotion = float(initial_emotion)
        self.threshold = float(threshold)
        self.momentum = float(momentum)
        self.sensitivity = float(sensitivity)
        
        self.past_scores = deque(maxlen=20)
        self.emotion_history = deque(maxlen=10)
        self.performance_trend = deque(maxlen=5)
        
    def update_parameters(self, new_params):
        self.alpha = float(new_params['alpha'].detach())
        self.beta = float(new_params['beta'].detach())
        self.emotion = float(new_params['initial_emotion'].detach())
        self.threshold = float(new_params['threshold'].detach())
        self.momentum = float(new_params['momentum'].detach())
        self.sensitivity = float(new_params['sensitivity'].detach())
        
    def update(self, current_score):
        self.past_scores.append(current_score)
        
        if len(self.past_scores) < 5:
            return self.emotion
            
        # Calculate performance metrics
        recent_avg = np.mean(list(self.past_scores)[-5:])
        older_avg = np.mean(list(self.past_scores)[-10:-5]) if len(self.past_scores) >= 10 else recent_avg
        
        # Performance trend
        trend = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)
        self.performance_trend.append(trend)
        
        # Adaptive emotion update based on trend and sensitivity
        if abs(trend) > self.threshold:
            # Strong trend detected
            if trend > 0:
                # Positive trend - increase emotion
                emotion_delta = self.alpha * self.sensitivity * min(trend, 1.0)
            else:
                # Negative trend - decrease emotion
                emotion_delta = -self.alpha * self.sensitivity * min(abs(trend), 1.0)
        else:
            # No strong trend - maintain current emotion
            emotion_delta = 0
        
        # Apply momentum
        if len(self.emotion_history) > 0:
            momentum_factor = self.momentum * (self.emotion - self.emotion_history[-1])
            emotion_delta += momentum_factor
        
        # Update emotion with bounds
        self.emotion = np.clip(self.emotion + emotion_delta, 0.1, 0.9)
        self.emotion_history.append(self.emotion)
        
        return self.emotion
    
    def get_emotion_state(self):
        """Returns detailed emotion state"""
        if len(self.emotion_history) < 3:
            return 'INITIALIZING'
        
        recent_emotions = list(self.emotion_history)[-3:]
        emotion_trend = np.mean(np.diff(recent_emotions))
        
        if self.emotion > 0.7:
            return 'CONFIDENT'
        elif self.emotion > 0.6:
            return 'BALANCED'
        elif self.emotion > 0.5:
            return 'DETERMINED'
        elif self.emotion > 0.3:
            return 'CAUTIOUS'
        else:
            return 'FRUSTRATED'

# =============================================================================
# PRIORITIZED REPLAY BUFFER (with NaN protection)
# =============================================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], [], [], []
        
        # Calculate sampling probabilities with NaN protection
        priorities = self.priorities[:len(self.buffer)]
        # Replace any NaN or zero priorities
        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1.0, neginf=1e-6)
        priorities = np.maximum(priorities, 1e-6)  # Ensure all positive
        
        probabilities = priorities ** self.alpha
        probabilities = np.nan_to_num(probabilities, nan=1e-6)
        probabilities = np.maximum(probabilities, 1e-6)
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (torch.FloatTensor(states).to(device),
                torch.LongTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(next_states).to(device),
                torch.BoolTensor(dones).to(device),
                torch.FloatTensor(weights).to(device))
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# ATTENTION RAINBOW DQN AGENT
# =============================================================================

class AttentionRainbowDQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.999, epsilon_min=0.01, batch_size=128, target_update=500,
                 memory_size=100000, alpha=0.6, beta=0.4, num_heads=8):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Attention Networks
        self.q_network = AttentionDuelingNetwork(
            state_size, action_size, num_heads=num_heads
        ).to(device)
        self.target_network = AttentionDuelingNetwork(
            state_size, action_size, num_heads=num_heads
        ).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(memory_size, alpha, beta)
        
        # Enhanced Emotion Engine
        self.emotion_engine = EnhancedEmotionEngine()
        
        # Training tracking
        self.step_count = 0
        self.episode_rewards = []
        self.emotion_history = []
        self.mindset_history = []
        self.attention_history = []
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values, attention_weights = self.q_network(state)
        return q_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done):
        # Store experience
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn
        if len(self.memory) > self.batch_size:
            self.learn()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn(self):
        # Sample from memory
        states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        
        if len(states) == 0:
            return
        
        # Current Q values
        current_q_values, attention_weights = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate TD error
        td_errors = current_q_values.squeeze() - target_q_values
        
        # Weighted loss
        loss = (weights * td_errors ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities with NaN protection
        td_errors_np = td_errors.detach().cpu().numpy()
        # Replace NaN and inf values with small positive number
        td_errors_np = np.nan_to_num(td_errors_np, nan=1e-6, posinf=1.0, neginf=-1.0)
        priorities = (np.abs(td_errors_np) + 1e-6) ** self.memory.alpha
        self.memory.update_priorities(range(len(priorities)), priorities)
        
        # Store attention weights for analysis
        if len(self.attention_history) < 1000:  # Keep last 1000 attention weights
            self.attention_history.append(attention_weights.detach().cpu().numpy())
    
    def update_emotion(self, episode_reward):
        """Update emotion based on episode reward"""
        emotion = self.emotion_engine.update(episode_reward)
        mindset = self.emotion_engine.get_emotion_state()
        
        self.emotion_history.append(emotion)
        self.mindset_history.append(mindset)
        
        return emotion, mindset

# =============================================================================
# ATTENTION TRAINING FUNCTION
# =============================================================================

def train_attention_rainbow_lunarlander(env_name='LunarLander-v3', episodes=2000, lr=1e-4, 
                                      batch_size=128, target_update=500, epsilon_decay=0.999,
                                      gamma=0.99, memory_size=100000, num_heads=8):
    """Train Attention Rainbow DQN with Enhanced Emotion Engine on LunarLander"""
    
    print(f"🎮 ATTENTION Rainbow DQN + Enhanced Emotion Engine LunarLander")
    print("=" * 70)
    
    # Environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Attention heads: {num_heads}")
    
    # Agent
    agent = AttentionRainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        batch_size=batch_size,
        target_update=target_update,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        memory_size=memory_size,
        num_heads=num_heads
    )
    
    # Training tracking
    scores = []
    avg_scores = []
    best_score = -float('inf')
    
    print(f"Attention Training: {episodes} episodes")
    print("Starting ATTENTION training...")
    
    start_time = time.time()
    
    for episode in tqdm(range(episodes), desc="Attention Training"):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 1000:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update emotion
        emotion, mindset = agent.update_emotion(episode_reward)
        
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode_reward > best_score:
            best_score = episode_reward
        
        # Logging
        if episode % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode+1}/{episodes}: "
                  f"Avg Score: {avg_score:.2f}, "
                  f"Emotion: {emotion:.3f}, "
                  f"Mindset: {mindset}, "
                  f"Time: {elapsed_time/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.1f} minutes")
    
    env.close()
    
    return agent, scores, avg_scores, agent.emotion_history, agent.mindset_history, agent.attention_history

# =============================================================================
# ATTENTION VISUALIZATION
# =============================================================================

def plot_attention_results(scores, avg_scores, emotion_history, mindset_history, attention_history):
    """Plot attention training results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Episode rewards
    axes[0, 0].plot(scores, alpha=0.3, color='blue', label='Episode Rewards')
    axes[0, 0].plot(avg_scores, color='red', linewidth=2, label='Moving Avg (100)')
    axes[0, 0].axhline(y=200, color='green', linestyle='--', label='Solved (200)')
    axes[0, 0].set_title('Attention Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Average scores
    axes[0, 1].plot(avg_scores, color='green', linewidth=2)
    axes[0, 1].axhline(y=200, color='red', linestyle='--', label='Solved (200)')
    axes[0, 1].set_title('Attention Average Scores (100 episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Emotion evolution
    axes[0, 2].plot(emotion_history, color='purple', linewidth=2)
    axes[0, 2].axhline(y=0.6, color='orange', linestyle='--', label='BALANCED')
    axes[0, 2].axhline(y=0.7, color='green', linestyle='--', label='CONFIDENT')
    axes[0, 2].set_title('Attention Emotion Evolution')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Emotion Level')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Mindset distribution
    mindset_counts = {}
    for mindset in mindset_history:
        mindset_counts[mindset] = mindset_counts.get(mindset, 0) + 1
    
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    axes[1, 0].pie(mindset_counts.values(), labels=mindset_counts.keys(), 
                   autopct='%1.1f%%', colors=colors[:len(mindset_counts)])
    axes[1, 0].set_title('Attention Mindset Distribution')
    
    # Attention weights visualization (if available)
    if len(attention_history) > 0:
        # Take the last attention weights
        last_attention = attention_history[-1]
        if last_attention.ndim == 3:  # [batch, heads, seq_len]
            # Average across heads and batch
            avg_attention = np.mean(last_attention, axis=(0, 1))
            axes[1, 1].bar(range(len(avg_attention)), avg_attention)
            axes[1, 1].set_title('Average Attention Weights')
            axes[1, 1].set_xlabel('State Features')
            axes[1, 1].set_ylabel('Attention Weight')
        else:
            axes[1, 1].text(0.5, 0.5, 'Attention Weights\n(Processing...)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, 'Attention Weights\n(Not Available)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Performance comparison placeholder
    axes[1, 2].text(0.5, 0.5, 'Performance Comparison\n(Will be added)', 
                   ha='center', va='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 ATTENTION MECHANISMS + ENHANCED EMOTION ENGINE + LUNARLANDER")
    print("=" * 70)
    
    # Attention training with optimized parameters
    results = train_attention_rainbow_lunarlander(
        episodes=2000,
        lr=1e-4,
        epsilon_decay=0.999,
        batch_size=128,
        target_update=500,
        num_heads=8
    )
    
    agent, scores, avg_scores, emotion_history, mindset_history, attention_history = results
    
    # Plot results
    plot_attention_results(scores, avg_scores, emotion_history, mindset_history, attention_history)
    
    # Final results
    final_avg = np.mean(scores[-100:])
    best_episode = max(scores)
    final_emotion = emotion_history[-1] if emotion_history else 0.5
    final_mindset = mindset_history[-1] if mindset_history else 'UNKNOWN'
    
    print(f"\n🏆 ATTENTION Results:")
    print(f"   Final Average (100 episodes): {final_avg:.2f}")
    print(f"   Best Episode: {best_episode:.2f}")
    print(f"   Final Emotion: {final_emotion:.3f}")
    print(f"   Final Mindset: {final_mindset}")
    print(f"   Progress: {final_avg}/200 ({final_avg/200*100:.1f}% of target)")
    
    # Performance analysis
    emotion_stability = 1.0 - np.std(emotion_history) if len(emotion_history) > 1 else 0
    
    print(f"\n📊 Attention Performance Analysis:")
    print(f"   Emotion stability: {emotion_stability:.3f}")
    
    # Win rate analysis
    wins = sum(1 for score in scores if score > 0)
    win_rate = (wins / len(scores)) * 100
    print(f"   Win rate: {win_rate:.1f}%")
    
    # Attention analysis
    if len(attention_history) > 0:
        print(f"   Attention weights collected: {len(attention_history)}")
        print(f"   Attention heads: 8")
    
    print("\n🎉 Attention Mechanisms + Enhanced Emotion Engine testing complete!")
    print("🚀 Ready for Phase 5.2: Self-Correction Mechanism!")
