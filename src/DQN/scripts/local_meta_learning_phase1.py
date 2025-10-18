"""
Local Meta-Learning Phase 1 - Task Encoder + Emotion Adaptor
============================================================

Lokale Implementation für Meta-Learning
- Funktioniert ohne Box2D (nur CartPole, Acrobot, MountainCar)
- Task Encoder + Emotion Adaptor
- Meta-Learning Foundation

Author: Emotional Meta-Learning Agent Project
Date: 2025-10-17
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Test environments (ohne Box2D)
test_envs = {
    'CartPole-v1': 4,
    'Acrobot-v1': 6, 
    'MountainCar-v0': 2
}

print("🧪 Testing environments...")
for env_name, obs_dim in test_envs.items():
    try:
        env = gym.make(env_name)
        print(f"✅ {env_name}: {obs_dim}D state, {env.action_space.n} actions")
        env.close()
    except Exception as e:
        print(f"❌ {env_name}: {e}")

print("\n🎯 Ready for Local Meta-Learning Phase 1!")

# =============================================================================
# ADAPTIVE TASK ENCODER
# =============================================================================

class AdaptiveTaskEncoder(nn.Module):
    def __init__(self, max_obs_dim=8, action_dim=4, embedding_dim=64):
        super().__init__()
        
        self.max_obs_dim = max_obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Adaptive observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(max_obs_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Reward encoder
        self.reward_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, observations, rewards):
        # Pad observations to max_obs_dim if needed
        if observations.shape[1] < self.max_obs_dim:
            padding = torch.zeros(observations.shape[0], self.max_obs_dim - observations.shape[1]).to(observations.device)
            observations = torch.cat([observations, padding], dim=1)
        elif observations.shape[1] > self.max_obs_dim:
            observations = observations[:, :self.max_obs_dim]
        
        obs_emb = self.obs_encoder(observations).mean(dim=0)
        rew_emb = self.reward_encoder(rewards.unsqueeze(-1)).mean(dim=0)
        combined = torch.cat([obs_emb, rew_emb], dim=-1)
        return self.fusion(combined)

# =============================================================================
# EMOTION ADAPTOR
# =============================================================================

class EmotionAdaptor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [alpha, beta, initial_emotion, threshold]
        )
        
    def forward(self, task_embedding):
        params = self.predictor(task_embedding)
        return {
            'alpha': torch.sigmoid(params[0]) * 0.2,  # 0-0.2
            'beta': torch.sigmoid(params[1]) * 0.5 + 0.5,  # 0.5-1.0
            'initial_emotion': torch.sigmoid(params[2]) * 0.6 + 0.2,  # 0.2-0.8
            'threshold': torch.sigmoid(params[3]) * 0.3 + 0.1  # 0.1-0.4
        }

# =============================================================================
# ADAPTIVE EMOTION ENGINE
# =============================================================================

class AdaptiveEmotionEngine:
    def __init__(self, alpha=0.1, beta=0.9, initial_emotion=0.5, threshold=0.1):
        self.alpha = alpha
        self.beta = beta
        self.emotion = initial_emotion
        self.threshold = threshold
        self.momentum = 0.0
        self.past_scores = deque(maxlen=100)
        
    def update_parameters(self, new_params):
        self.alpha = new_params['alpha'].item() if hasattr(new_params['alpha'], 'item') else new_params['alpha']
        self.beta = new_params['beta'].item() if hasattr(new_params['beta'], 'item') else new_params['beta']
        self.emotion = new_params['initial_emotion'].item() if hasattr(new_params['initial_emotion'], 'item') else new_params['initial_emotion']
        self.threshold = new_params['threshold'].item() if hasattr(new_params['threshold'], 'item') else new_params['threshold']
        
    def update(self, current_score, episode):
        self.past_scores.append(current_score)
        
        if len(self.past_scores) < 10:
            return self.emotion, 'insufficient_data'
            
        past_avg = np.mean(list(self.past_scores)[:-1])
        
        if past_avg == 0:
            relative_improvement = 0
        else:
            relative_improvement = (current_score - past_avg) / abs(past_avg)
            
        outcome = self._determine_outcome(relative_improvement)
        emotion_delta = self._outcome_to_emotion_delta(outcome)
        self.emotion = np.clip(self.emotion + self.alpha * emotion_delta, 0.2, 0.8)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * np.sign(emotion_delta)
        
        return self.emotion, outcome
        
    def _determine_outcome(self, relative_improvement):
        if relative_improvement >= self.threshold * 2:
            return 'decisive_win'
        elif relative_improvement >= self.threshold:
            return 'win'
        elif relative_improvement >= self.threshold * 0.5:
            return 'draw'
        elif relative_improvement <= -self.threshold:
            return 'loss'
        else:
            return 'decisive_loss'
            
    def _outcome_to_emotion_delta(self, outcome):
        deltas = {
            'decisive_win': 0.3,
            'win': 0.1,
            'draw': 0.0,
            'loss': -0.1,
            'decisive_loss': -0.3
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
# DATA COLLECTOR
# =============================================================================

def collect_task_data(env_name, episodes=50):
    """Sammelt Daten von einem Environment"""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    
    observations = []
    rewards = []
    scores = []
    
    print(f"📊 Collecting data from {env_name}...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_obs = []
        episode_rewards = []
        episode_score = 0
        
        for step in range(1000):  # Max steps
            action = env.action_space.sample()  # Random actions for data collection
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_obs.append(state)
            episode_rewards.append(reward)
            episode_score += reward
            
            state = next_state
            if done:
                break
        
        observations.extend(episode_obs)
        rewards.extend(episode_rewards)
        scores.append(episode_score)
    
    env.close()
    
    # Convert to tensors
    obs_tensor = torch.FloatTensor(observations).to(device)
    rew_tensor = torch.FloatTensor(rewards).to(device)
    
    print(f"   Episodes: {len(scores)}, Avg Score: {np.mean(scores):.2f}, Obs Dim: {obs_dim}")
    
    return {
        'observations': obs_tensor,
        'rewards': rew_tensor,
        'scores': scores,
        'env_name': env_name,
        'obs_dim': obs_dim
    }

# =============================================================================
# META-LEARNING TRAINER
# =============================================================================

class MetaLearningTrainer:
    def __init__(self, task_encoder, emotion_adaptor):
        self.task_encoder = task_encoder.to(device)
        self.emotion_adaptor = emotion_adaptor.to(device)
        self.meta_optimizer = optim.Adam(
            list(task_encoder.parameters()) + list(emotion_adaptor.parameters()),
            lr=1e-4
        )
        
    def meta_train_step(self, task_data):
        """Single meta-learning training step"""
        # Get task embedding
        task_embedding = self.task_encoder(
            task_data['observations'],
            task_data['rewards']
        )
        
        # Predict emotion parameters
        emotion_params = self.emotion_adaptor(task_embedding)
        
        # Create emotion engine with predicted parameters
        emotion_engine = AdaptiveEmotionEngine()
        emotion_engine.update_parameters(emotion_params)
        
        # Simulate emotion updates on task data
        emotion_history = []
        for score in task_data['scores']:
            emotion, outcome = emotion_engine.update(score, 0)
            emotion_history.append(emotion)
        
        # Calculate meta-loss
        emotion_stability = 1.0 - np.std(emotion_history)
        performance_correlation = np.corrcoef(emotion_history, task_data['scores'])[0, 1]
        
        # Meta-loss: maximize stability and correlation (keep as tensor)
        meta_loss = torch.tensor(-emotion_stability - abs(performance_correlation), requires_grad=True, device=device)
        
        return meta_loss, emotion_params, emotion_history
    
    def train(self, task_names, epochs=100, episodes_per_task=50):
        """Meta-learning training loop"""
        print(f"🚀 Starting Meta-Learning Training...")
        print(f"📊 Tasks: {task_names}")
        print(f"🎯 Epochs: {epochs}, Episodes per task: {episodes_per_task}")
        
        # Collect data from all tasks
        task_data = {}
        for task_name in task_names:
            task_data[task_name] = collect_task_data(task_name, episodes_per_task)
        
        training_history = {
            'epoch': [],
            'meta_loss': [],
            'emotion_stability': [],
            'performance_correlation': []
        }
        
        for epoch in tqdm(range(epochs), desc="Meta-Learning"):
            epoch_meta_loss = 0
            epoch_stability = 0
            epoch_correlation = 0
            total_meta_loss = None
            
            for task_name in task_names:
                data = task_data[task_name]
                
                # Meta-learning step
                meta_loss, emotion_params, emotion_history = self.meta_train_step(data)
                
                # Accumulate loss for backward pass
                if total_meta_loss is None:
                    total_meta_loss = meta_loss
                else:
                    total_meta_loss += meta_loss
                
                epoch_meta_loss += meta_loss.item()
                epoch_stability += 1.0 - np.std(emotion_history)
                epoch_correlation += abs(np.corrcoef(emotion_history, data['scores'])[0, 1])
            
            # Average over tasks
            epoch_meta_loss /= len(task_names)
            epoch_stability /= len(task_names)
            epoch_correlation /= len(task_names)
            total_meta_loss /= len(task_names)
            
            # Meta-update
            self.meta_optimizer.zero_grad()
            total_meta_loss.backward()
            self.meta_optimizer.step()
            
            # Track history
            training_history['epoch'].append(epoch)
            training_history['meta_loss'].append(epoch_meta_loss)
            training_history['emotion_stability'].append(epoch_stability)
            training_history['performance_correlation'].append(epoch_correlation)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Meta-Loss: {epoch_meta_loss:.4f}, "
                      f"Stability: {epoch_stability:.4f}, Correlation: {epoch_correlation:.4f}")
        
        return training_history, task_data

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_emotion_adaptation(task_encoder, emotion_adaptor, task_data):
    """Evaluates emotion adaptation on different tasks"""
    print(f"🧪 Evaluating Emotion Adaptation...")
    
    results = {}
    
    for task_name, data in task_data.items():
        print(f"Testing {task_name}...")
        
        # Get task embedding
        with torch.no_grad():
            task_embedding = task_encoder(data['observations'], data['rewards'])
            emotion_params = emotion_adaptor(task_embedding)
        
        # Create adapted emotion engine
        emotion_engine = AdaptiveEmotionEngine()
        emotion_engine.update_parameters(emotion_params)
        
        # Test emotion adaptation
        emotion_history = []
        for score in data['scores']:
            emotion, outcome = emotion_engine.update(score, 0)
            emotion_history.append(emotion)
        
        # Calculate metrics
        stability = 1.0 - np.std(emotion_history)
        correlation = abs(np.corrcoef(emotion_history, data['scores'])[0, 1])
        
        results[task_name] = {
            'emotion_params': {k: v.item() for k, v in emotion_params.items()},
            'stability': stability,
            'correlation': correlation,
            'emotion_history': emotion_history,
            'scores': data['scores'],
            'obs_dim': data['obs_dim']
        }
        
        print(f"  Stability: {stability:.3f}, Correlation: {correlation:.3f}")
        print(f"  Alpha: {emotion_params['alpha'].item():.3f}, "
              f"Initial Emotion: {emotion_params['initial_emotion'].item():.3f}")
    
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_meta_learning_results(training_history, evaluation_results):
    """Plots meta-learning results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(training_history['epoch'], training_history['meta_loss'])
    axes[0, 0].set_title('Meta-Learning Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Meta-Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(training_history['epoch'], training_history['emotion_stability'])
    axes[0, 1].set_title('Emotion Stability')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Stability')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(training_history['epoch'], training_history['performance_correlation'])
    axes[0, 2].set_title('Performance Correlation')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].grid(True)
    
    # Evaluation results
    task_names = list(evaluation_results.keys())
    stabilities = [evaluation_results[task]['stability'] for task in task_names]
    correlations = [evaluation_results[task]['correlation'] for task in task_names]
    obs_dims = [evaluation_results[task]['obs_dim'] for task in task_names]
    
    bars1 = axes[1, 0].bar(task_names, stabilities, color='blue', alpha=0.7)
    axes[1, 0].set_title('Emotion Stability by Task')
    axes[1, 0].set_ylabel('Stability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add obs dim labels
    for i, (bar, dim) in enumerate(zip(bars1, obs_dims)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{dim}D', ha='center', va='bottom', fontsize=8)
    
    bars2 = axes[1, 1].bar(task_names, correlations, color='green', alpha=0.7)
    axes[1, 1].set_title('Performance Correlation by Task')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add obs dim labels
    for i, (bar, dim) in enumerate(zip(bars2, obs_dims)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{dim}D', ha='center', va='bottom', fontsize=8)
    
    # Emotion parameters heatmap
    param_names = ['alpha', 'beta', 'initial_emotion', 'threshold']
    param_matrix = np.array([
        [evaluation_results[task]['emotion_params'][param] for param in param_names]
        for task in task_names
    ])
    
    im = axes[1, 2].imshow(param_matrix, cmap='viridis', aspect='auto')
    axes[1, 2].set_title('Emotion Parameters by Task')
    axes[1, 2].set_xticks(range(len(param_names)))
    axes[1, 2].set_xticklabels(param_names)
    axes[1, 2].set_yticks(range(len(task_names)))
    axes[1, 2].set_yticklabels([f'{name}\n({obs_dims[i]}D)' for i, name in enumerate(task_names)])
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎮 LOCAL META-LEARNING PHASE 1")
    print("=" * 50)
    
    # Initialize models
    task_encoder = AdaptiveTaskEncoder(max_obs_dim=8, action_dim=4, embedding_dim=64)
    emotion_adaptor = EmotionAdaptor(embedding_dim=64)
    
    print(f"📊 Task Encoder: {sum(p.numel() for p in task_encoder.parameters())} parameters")
    print(f"📊 Emotion Adaptor: {sum(p.numel() for p in emotion_adaptor.parameters())} parameters")
    
    # Training tasks
    training_tasks = list(test_envs.keys())
    
    # Meta-learning training
    trainer = MetaLearningTrainer(task_encoder, emotion_adaptor)
    training_history, task_data = trainer.train(
        task_names=training_tasks,
        epochs=50,
        episodes_per_task=30
    )
    
    # Evaluation
    evaluation_results = evaluate_emotion_adaptation(
        task_encoder, emotion_adaptor, task_data
    )
    
    # Plot results
    plot_meta_learning_results(training_history, evaluation_results)
    
    # Final statistics
    print(f"\n🏆 LOCAL META-LEARNING RESULTS:")
    print(f"   Average Stability: {np.mean([r['stability'] for r in evaluation_results.values()]):.3f}")
    print(f"   Average Correlation: {np.mean([r['correlation'] for r in evaluation_results.values()]):.3f}")
    print(f"   Tasks Tested: {len(training_tasks)}")
    
    # Save models
    torch.save(task_encoder.state_dict(), 'local_task_encoder.pth')
    torch.save(emotion_adaptor.state_dict(), 'local_emotion_adaptor.pth')
    print("💾 Models saved!")
    
    print("\n🎉 PHASE 1 COMPLETE: Local Meta-Learning Foundation Ready!")
    print("🚀 Ready for Phase 2: Few-Shot Learning!")
