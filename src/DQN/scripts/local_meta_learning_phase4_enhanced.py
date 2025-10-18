"""
Local Meta-Learning Phase 4 - Enhanced Emotion Engine
====================================================

Enhanced Emotion Engine Implementation
- Bessere Performance-Emotion Mapping
- Task-spezifische Parameter-Anpassung
- Numerische Stabilität
- Adaptive Learning Rates

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
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Test environments
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

print("\n🎯 Ready for Meta-Learning Phase 4: Enhanced Emotion Engine!")

# =============================================================================
# ENHANCED TASK ENCODER
# =============================================================================

class EnhancedTaskEncoder(nn.Module):
    def __init__(self, max_obs_dim=8, embedding_dim=128):
        super().__init__()
        
        self.max_obs_dim = max_obs_dim
        self.embedding_dim = embedding_dim
        
        # Enhanced encoder with attention mechanism
        self.obs_encoder = nn.Sequential(
            nn.Linear(max_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.reward_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Attention mechanism for task features
        self.attention = nn.MultiheadAttention(embed_dim=40, num_heads=4, batch_first=True)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, obs_mean, reward_mean):
        # Convert to numpy first
        if isinstance(obs_mean, torch.Tensor):
            obs_mean = obs_mean.detach().cpu().numpy()
        if isinstance(reward_mean, torch.Tensor):
            reward_mean = reward_mean.detach().cpu().numpy()
        
        # Ensure proper dimensions
        if obs_mean.ndim > 1:
            obs_mean = obs_mean.flatten()
        if reward_mean.ndim > 0:
            reward_mean = reward_mean.item() if reward_mean.size == 1 else reward_mean[0]
        
        # Create input vector
        input_vector = np.zeros(self.max_obs_dim + 1)
        obs_len = min(len(obs_mean), self.max_obs_dim)
        input_vector[:obs_len] = obs_mean[:obs_len]
        input_vector[self.max_obs_dim] = reward_mean
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_vector).to(device)
        
        # Encode observations and rewards separately
        obs_emb = self.obs_encoder(input_tensor[:self.max_obs_dim])
        rew_emb = self.reward_encoder(input_tensor[self.max_obs_dim:].unsqueeze(0))
        
        # Combine features
        combined = torch.cat([obs_emb, rew_emb.squeeze(0)], dim=0)
        
        # Apply attention
        attended, _ = self.attention(
            combined.unsqueeze(0).unsqueeze(0), 
            combined.unsqueeze(0).unsqueeze(0), 
            combined.unsqueeze(0).unsqueeze(0)
        )
        
        # Final fusion
        output = self.fusion(attended.squeeze())
        
        return output

# =============================================================================
# ENHANCED EMOTION ADAPTOR
# =============================================================================

class EnhancedEmotionAdaptor(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # More sophisticated emotion predictor
        self.emotion_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # [alpha, beta, initial_emotion, threshold, momentum, sensitivity]
        )
        
    def forward(self, task_embedding):
        params = self.emotion_predictor(task_embedding)
        
        return {
            'alpha': torch.sigmoid(params[0]) * 0.3,  # 0-0.3 (increased range)
            'beta': torch.sigmoid(params[1]) * 0.8 + 0.2,  # 0.2-1.0
            'initial_emotion': torch.sigmoid(params[2]) * 0.6 + 0.2,  # 0.2-0.8
            'threshold': torch.sigmoid(params[3]) * 0.4 + 0.05,  # 0.05-0.45
            'momentum': torch.sigmoid(params[4]) * 0.9 + 0.1,  # 0.1-1.0
            'sensitivity': torch.sigmoid(params[5]) * 2.0 + 0.5  # 0.5-2.5
        }

# =============================================================================
# ENHANCED EMOTION ENGINE
# =============================================================================

class EnhancedEmotionEngine:
    def __init__(self, alpha=0.1, beta=0.9, initial_emotion=0.5, threshold=0.1, momentum=0.5, sensitivity=1.0):
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
# ENHANCED META-LEARNING TRAINER
# =============================================================================

class EnhancedMetaLearningTrainer:
    def __init__(self, task_encoder, emotion_adaptor):
        self.task_encoder = task_encoder.to(device)
        self.emotion_adaptor = emotion_adaptor.to(device)
        
        # Load previous models if available
        try:
            self.task_encoder.load_state_dict(torch.load('final_task_encoder.pth', map_location=device))
            self.emotion_adaptor.load_state_dict(torch.load('final_emotion_adaptor.pth', map_location=device))
            print("✅ Loaded previous models!")
        except (FileNotFoundError, RuntimeError):
            print("⚠️ Previous models not found or incompatible, using random initialization")
        
        # Enhanced optimizer with different learning rates
        self.optimizer = optim.AdamW([
            {'params': self.task_encoder.parameters(), 'lr': 1e-4},
            {'params': self.emotion_adaptor.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def collect_enhanced_data(self, env_name, episodes=30):
        """Collects enhanced data with more episodes"""
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        
        all_observations = []
        all_rewards = []
        scores = []
        
        print(f"📊 Collecting enhanced data from {env_name}...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_obs = []
            episode_rewards = []
            episode_score = 0
            
            for step in range(500):
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_obs.append(state)
                episode_rewards.append(reward)
                episode_score += reward
                
                state = next_state
                if done:
                    break
            
            all_observations.extend(episode_obs)
            all_rewards.extend(episode_rewards)
            scores.append(episode_score)
        
        env.close()
        
        # Calculate enhanced statistics
        obs_mean = np.mean(all_observations, axis=0)
        reward_mean = np.mean(all_rewards)
        score_std = np.std(scores)
        score_range = np.max(scores) - np.min(scores)
        
        print(f"   Episodes: {len(scores)}, Avg Score: {np.mean(scores):.2f}, "
              f"Std: {score_std:.2f}, Range: {score_range:.2f}")
        
        return {
            'obs_mean': obs_mean,
            'reward_mean': reward_mean,
            'scores': scores,
            'score_std': score_std,
            'score_range': score_range,
            'env_name': env_name,
            'obs_dim': obs_dim
        }
    
    def enhanced_meta_train_step(self, task_data):
        """Enhanced meta-learning training step"""
        # Get task embedding
        task_embedding = self.task_encoder(task_data['obs_mean'], task_data['reward_mean'])
        
        # Predict emotion parameters
        emotion_params = self.emotion_adaptor(task_embedding)
        
        # Create enhanced emotion engine
        emotion_engine = EnhancedEmotionEngine()
        emotion_engine.update_parameters(emotion_params)
        
        # Simulate emotion updates
        emotion_history = []
        for score in task_data['scores']:
            emotion = emotion_engine.update(score)
            emotion_history.append(emotion)
        
        # Enhanced metrics
        stability = 1.0 - np.std(emotion_history)
        
        # Better correlation calculation
        if len(emotion_history) > 1 and np.std(task_data['scores']) > 0:
            correlation = abs(np.corrcoef(emotion_history, task_data['scores'])[0, 1])
        else:
            correlation = 0
        
        # Emotion range (diversity)
        emotion_range = np.max(emotion_history) - np.min(emotion_history)
        
        # Performance-emotion alignment
        score_trend = np.mean(np.diff(task_data['scores'][-5:])) if len(task_data['scores']) >= 5 else 0
        emotion_trend = np.mean(np.diff(emotion_history[-5:])) if len(emotion_history) >= 5 else 0
        alignment = 1.0 - abs(score_trend - emotion_trend) / (abs(score_trend) + 1e-8)
        
        # Enhanced loss function
        stability_loss = -stability
        correlation_loss = -correlation
        diversity_loss = -emotion_range * 0.1  # Encourage emotion diversity
        alignment_loss = -alignment * 0.2
        
        total_loss = stability_loss + correlation_loss + diversity_loss + alignment_loss
        
        return total_loss, emotion_params, {
            'stability': stability,
            'correlation': correlation,
            'emotion_range': emotion_range,
            'alignment': alignment,
            'emotion_history': emotion_history
        }
    
    def train_enhanced(self, task_names, epochs=100, episodes_per_task=30):
        """Enhanced meta-learning training"""
        print(f"🚀 Starting Enhanced Meta-Learning Training...")
        print(f"📊 Tasks: {task_names}")
        print(f"🎯 Epochs: {epochs}, Episodes per task: {episodes_per_task}")
        
        # Collect enhanced data
        task_data = {}
        for task_name in task_names:
            task_data[task_name] = self.collect_enhanced_data(task_name, episodes_per_task)
        
        training_history = {
            'epoch': [],
            'loss': [],
            'stability': [],
            'correlation': [],
            'emotion_range': [],
            'alignment': []
        }
        
        best_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Enhanced Meta-Learning"):
            epoch_loss = 0
            epoch_stability = 0
            epoch_correlation = 0
            epoch_emotion_range = 0
            epoch_alignment = 0
            
            for task_name in task_names:
                data = task_data[task_name]
                
                # Enhanced meta-learning step
                loss, emotion_params, metrics = self.enhanced_meta_train_step(data)
                
                epoch_loss += loss
                epoch_stability += metrics['stability']
                epoch_correlation += metrics['correlation']
                epoch_emotion_range += metrics['emotion_range']
                epoch_alignment += metrics['alignment']
            
            # Average over tasks
            epoch_loss /= len(task_names)
            epoch_stability /= len(task_names)
            epoch_correlation /= len(task_names)
            epoch_emotion_range /= len(task_names)
            epoch_alignment /= len(task_names)
            
            # Convert to tensor for backward pass
            loss_tensor = torch.tensor(epoch_loss, requires_grad=True, device=device)
            
            # Update
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()
            
            # Learning rate scheduling
            self.scheduler.step(epoch_loss)
            
            # Track history
            training_history['epoch'].append(epoch)
            training_history['loss'].append(epoch_loss)
            training_history['stability'].append(epoch_stability)
            training_history['correlation'].append(epoch_correlation)
            training_history['emotion_range'].append(epoch_emotion_range)
            training_history['alignment'].append(epoch_alignment)
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.task_encoder.state_dict(), 'enhanced_task_encoder.pth')
                torch.save(self.emotion_adaptor.state_dict(), 'enhanced_emotion_adaptor.pth')
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss: {epoch_loss:.4f}, "
                      f"Stability: {epoch_stability:.4f}, Correlation: {epoch_correlation:.4f}, "
                      f"Range: {epoch_emotion_range:.4f}, Alignment: {epoch_alignment:.4f}")
        
        return training_history, task_data

# =============================================================================
# ENHANCED EVALUATION
# =============================================================================

def evaluate_enhanced_adaptation(task_encoder, emotion_adaptor, task_data):
    """Evaluates enhanced emotion adaptation"""
    print(f"🧪 Evaluating Enhanced Emotion Adaptation...")
    
    results = {}
    
    for task_name, data in task_data.items():
        print(f"Testing {task_name}...")
        
        with torch.no_grad():
            task_emb = task_encoder(data['obs_mean'], data['reward_mean'])
            emotion_params = emotion_adaptor(task_emb)
        
        # Create enhanced emotion engine
        emotion_engine = EnhancedEmotionEngine()
        emotion_engine.update_parameters(emotion_params)
        
        # Test emotion adaptation
        emotion_history = []
        emotion_states = []
        
        for score in data['scores']:
            emotion = emotion_engine.update(score)
            emotion_history.append(emotion)
            emotion_states.append(emotion_engine.get_emotion_state())
        
        # Enhanced metrics
        stability = 1.0 - np.std(emotion_history)
        
        if len(emotion_history) > 1 and np.std(data['scores']) > 0:
            correlation = abs(np.corrcoef(emotion_history, data['scores'])[0, 1])
        else:
            correlation = 0
        
        emotion_range = np.max(emotion_history) - np.min(emotion_history)
        
        # Emotion state diversity
        unique_states = len(set(emotion_states))
        state_diversity = unique_states / len(emotion_states)
        
        results[task_name] = {
            'emotion_params': {k: float(v) for k, v in emotion_params.items()},
            'stability': stability,
            'correlation': correlation,
            'emotion_range': emotion_range,
            'state_diversity': state_diversity,
            'emotion_history': emotion_history,
            'emotion_states': emotion_states,
            'scores': data['scores']
        }
        
        print(f"  Alpha: {emotion_params['alpha']:.3f}, "
              f"Initial Emotion: {emotion_params['initial_emotion']:.3f}")
        print(f"  Stability: {stability:.3f}, Correlation: {correlation:.3f}")
        print(f"  Range: {emotion_range:.3f}, State Diversity: {state_diversity:.3f}")
    
    return results

# =============================================================================
# ENHANCED VISUALIZATION
# =============================================================================

def plot_enhanced_results(training_history, evaluation_results):
    """Plots enhanced results"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Training history
    axes[0, 0].plot(training_history['epoch'], training_history['loss'])
    axes[0, 0].set_title('Enhanced Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(training_history['epoch'], training_history['stability'])
    axes[0, 1].set_title('Emotion Stability')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Stability')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(training_history['epoch'], training_history['correlation'])
    axes[0, 2].set_title('Performance-Emotion Correlation')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].grid(True)
    
    # Evaluation results
    task_names = list(evaluation_results.keys())
    stabilities = [evaluation_results[task]['stability'] for task in task_names]
    correlations = [evaluation_results[task]['correlation'] for task in task_names]
    emotion_ranges = [evaluation_results[task]['emotion_range'] for task in task_names]
    state_diversities = [evaluation_results[task]['state_diversity'] for task in task_names]
    
    axes[1, 0].bar(task_names, stabilities, color='blue', alpha=0.7)
    axes[1, 0].set_title('Enhanced Stability by Task')
    axes[1, 0].set_ylabel('Stability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(task_names, correlations, color='green', alpha=0.7)
    axes[1, 1].set_title('Enhanced Correlation by Task')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 2].bar(task_names, emotion_ranges, color='orange', alpha=0.7)
    axes[1, 2].set_title('Emotion Range by Task')
    axes[1, 2].set_ylabel('Emotion Range')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # Emotion parameter heatmap
    param_names = ['alpha', 'beta', 'initial_emotion', 'threshold', 'momentum', 'sensitivity']
    param_matrix = np.array([
        [evaluation_results[task]['emotion_params'][param] for param in param_names]
        for task in task_names
    ])
    
    im = axes[2, 0].imshow(param_matrix, cmap='viridis', aspect='auto')
    axes[2, 0].set_title('Enhanced Emotion Parameters')
    axes[2, 0].set_xticks(range(len(param_names)))
    axes[2, 0].set_xticklabels(param_names, rotation=45)
    axes[2, 0].set_yticks(range(len(task_names)))
    axes[2, 0].set_yticklabels(task_names)
    plt.colorbar(im, ax=axes[2, 0])
    
    # State diversity
    axes[2, 1].bar(task_names, state_diversities, color='purple', alpha=0.7)
    axes[2, 1].set_title('Emotion State Diversity')
    axes[2, 1].set_ylabel('State Diversity')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    # Correlation vs Range scatter
    scatter = axes[2, 2].scatter(correlations, emotion_ranges, s=100, alpha=0.7)
    for i, task in enumerate(task_names):
        axes[2, 2].annotate(task, (correlations[i], emotion_ranges[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[2, 2].set_title('Correlation vs Emotion Range')
    axes[2, 2].set_xlabel('Correlation')
    axes[2, 2].set_ylabel('Emotion Range')
    axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎮 META-LEARNING PHASE 4: ENHANCED EMOTION ENGINE")
    print("=" * 60)
    
    # Initialize enhanced models
    task_encoder = EnhancedTaskEncoder(max_obs_dim=8, embedding_dim=128)
    emotion_adaptor = EnhancedEmotionAdaptor(embedding_dim=128)
    
    print(f"📊 Enhanced Task Encoder: {sum(p.numel() for p in task_encoder.parameters())} parameters")
    print(f"📊 Enhanced Emotion Adaptor: {sum(p.numel() for p in emotion_adaptor.parameters())} parameters")
    
    # Initialize Enhanced Meta-Learning Trainer
    enhanced_trainer = EnhancedMetaLearningTrainer(task_encoder, emotion_adaptor)
    
    # Training tasks
    training_tasks = list(test_envs.keys())
    
    # Enhanced meta-learning training
    training_history, task_data = enhanced_trainer.train_enhanced(
        task_names=training_tasks,
        epochs=50,
        episodes_per_task=20
    )
    
    # Enhanced evaluation
    evaluation_results = evaluate_enhanced_adaptation(
        task_encoder, emotion_adaptor, task_data
    )
    
    # Plot enhanced results
    plot_enhanced_results(training_history, evaluation_results)
    
    # Final statistics
    print(f"\n🏆 ENHANCED EMOTION ENGINE RESULTS:")
    print(f"   Average Stability: {np.mean([r['stability'] for r in evaluation_results.values()]):.3f}")
    print(f"   Average Correlation: {np.mean([r['correlation'] for r in evaluation_results.values()]):.3f}")
    print(f"   Average Emotion Range: {np.mean([r['emotion_range'] for r in evaluation_results.values()]):.3f}")
    print(f"   Average State Diversity: {np.mean([r['state_diversity'] for r in evaluation_results.values()]):.3f}")
    print(f"   Tasks Tested: {len(training_tasks)}")
    
    # Save enhanced results
    with open('enhanced_emotion_results.json', 'w') as f:
        json_results = {}
        for task, result in evaluation_results.items():
            json_results[task] = {
                'emotion_params': result['emotion_params'],
                'stability': result['stability'],
                'correlation': result['correlation'],
                'emotion_range': result['emotion_range'],
                'state_diversity': result['state_diversity']
            }
        json.dump(json_results, f, indent=2)
    
    print("💾 Enhanced results saved to enhanced_emotion_results.json!")
    
    print("\n🎉 PHASE 4 COMPLETE: Enhanced Emotion Engine Ready!")
    print("🚀 Ready for Phase 5: Advanced Meta-Learning!")
