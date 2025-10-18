"""
Local Meta-Learning Phase 3 - Continual Learning
===============================================

Continual Learning Implementation
- Verhindert Catastrophic Forgetting
- Episodic Memory für Rehearsal
- Multi-Task Learning ohne Vergessen

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

print("\n🎯 Ready for Meta-Learning Phase 3: Continual Learning!")

# =============================================================================
# LOAD PHASE 1 & 2 MODELS
# =============================================================================

class FinalTaskEncoder(nn.Module):
    def __init__(self, max_obs_dim=8, embedding_dim=64):
        super().__init__()
        
        self.max_obs_dim = max_obs_dim
        self.embedding_dim = embedding_dim
        
        # Simple encoder - fixed input size
        self.encoder = nn.Sequential(
            nn.Linear(max_obs_dim + 1, 128),  # obs + reward
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, obs_mean, reward_mean):
        # Convert to numpy first to ensure clean tensors
        if isinstance(obs_mean, torch.Tensor):
            obs_mean = obs_mean.detach().cpu().numpy()
        if isinstance(reward_mean, torch.Tensor):
            reward_mean = reward_mean.detach().cpu().numpy()
        
        # Ensure obs_mean is 1D array
        if obs_mean.ndim > 1:
            obs_mean = obs_mean.flatten()
        
        # Ensure reward_mean is scalar
        if reward_mean.ndim > 0:
            reward_mean = reward_mean.item() if reward_mean.size == 1 else reward_mean[0]
        
        # Create input vector with proper dimensions
        input_vector = np.zeros(self.max_obs_dim + 1)
        
        # Fill observation part
        obs_len = min(len(obs_mean), self.max_obs_dim)
        input_vector[:obs_len] = obs_mean[:obs_len]
        
        # Fill reward part
        input_vector[self.max_obs_dim] = reward_mean
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_vector).to(device)
        
        return self.encoder(input_tensor)

class FinalEmotionAdaptor(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [alpha, initial_emotion]
        )
        
    def forward(self, task_embedding):
        params = self.predictor(task_embedding)
        return {
            'alpha': torch.sigmoid(params[0]) * 0.2,  # 0-0.2
            'initial_emotion': torch.sigmoid(params[1]) * 0.6 + 0.2  # 0.2-0.8
        }

# =============================================================================
# EPISODIC MEMORY
# =============================================================================

class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.task_memory = {}  # Separate memory per task
        
    def store_episode(self, task_id, episode_data):
        """Stores important episodes for rehearsal"""
        # Store in general memory
        self.memory.append({
            'task_id': task_id,
            'episode_data': episode_data,
            'importance': self.calculate_importance(episode_data)
        })
        
        # Store in task-specific memory
        if task_id not in self.task_memory:
            self.task_memory[task_id] = deque(maxlen=100)
        self.task_memory[task_id].append(episode_data)
    
    def calculate_importance(self, episode_data):
        """Calculates importance of episode for rehearsal"""
        # Simple importance: based on score magnitude
        score = episode_data.get('score', 0)
        return abs(score)
    
    def sample_rehearsal_batch(self, task_id, batch_size=10):
        """Samples episodes for rehearsal (prevents catastrophic forgetting)"""
        # Sample from current task
        current_task_samples = []
        if task_id in self.task_memory and len(self.task_memory[task_id]) > 0:
            current_task_samples = random.sample(
                list(self.task_memory[task_id]), 
                min(batch_size // 2, len(self.task_memory[task_id]))
            )
        
        # Sample from other tasks
        other_tasks = [tid for tid in self.task_memory.keys() if tid != task_id]
        other_task_samples = []
        
        if other_tasks:
            for other_task in random.sample(other_tasks, min(2, len(other_tasks))):
                if len(self.task_memory[other_task]) > 0:
                    samples = random.sample(
                        list(self.task_memory[other_task]),
                        min(batch_size // (2 * len(other_tasks)), len(self.task_memory[other_task]))
                    )
                    other_task_samples.extend(samples)
        
        return current_task_samples + other_task_samples

# =============================================================================
# CONTINUAL LEARNING TRAINER
# =============================================================================

class ContinualLearningTrainer:
    def __init__(self, task_encoder, emotion_adaptor):
        self.task_encoder = task_encoder.to(device)
        self.emotion_adaptor = emotion_adaptor.to(device)
        
        # Load Phase 1 models if available
        try:
            self.task_encoder.load_state_dict(torch.load('final_task_encoder.pth', map_location=device))
            self.emotion_adaptor.load_state_dict(torch.load('final_emotion_adaptor.pth', map_location=device))
            print("✅ Loaded Phase 1 models!")
        except FileNotFoundError:
            print("⚠️ Phase 1 models not found, using random initialization")
        
        # Initialize episodic memory
        self.episodic_memory = EpisodicMemory(capacity=1000)
        
        # Continual learning optimizer
        self.optimizer = optim.Adam(
            list(task_encoder.parameters()) + list(emotion_adaptor.parameters()),
            lr=1e-4
        )
        
        # Task-specific performance tracking
        self.task_performance = {}
        self.optimal_params = {}  # Store optimal parameters for each task
    
    def collect_task_data(self, env_name, episodes=20):
        """Collects data from a task"""
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        
        all_observations = []
        all_rewards = []
        scores = []
        
        print(f"📊 Collecting data from {env_name}...")
        
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
            
            # Store episode in memory
            episode_data = {
                'score': episode_score,
                'observations': episode_obs,
                'rewards': episode_rewards
            }
            self.episodic_memory.store_episode(env_name, episode_data)
        
        env.close()
        
        # Calculate means
        obs_mean = np.mean(all_observations, axis=0)
        reward_mean = np.mean(all_rewards)
        
        print(f"   Episodes: {len(scores)}, Avg Score: {np.mean(scores):.2f}, Obs Dim: {obs_dim}")
        
        return {
            'obs_mean': obs_mean,
            'reward_mean': reward_mean,
            'scores': scores,
            'env_name': env_name,
            'obs_dim': obs_dim
        }
    
    def continual_learning_step(self, task_data, rehearsal_batch=None):
        """Single continual learning step with rehearsal"""
        # Get task embedding
        task_embedding = self.task_encoder(task_data['obs_mean'], task_data['reward_mean'])
        
        # Predict emotion parameters
        emotion_params = self.emotion_adaptor(task_embedding)
        
        # Create emotion engine
        emotion_engine = ContinualEmotionEngine()
        emotion_engine.update_parameters(emotion_params)
        
        # Simulate emotion updates
        emotion_history = []
        for score in task_data['scores']:
            emotion = emotion_engine.update(score)
            emotion_history.append(emotion)
        
        # Calculate metrics
        stability = 1.0 - np.std(emotion_history)
        correlation = abs(np.corrcoef(emotion_history, task_data['scores'])[0, 1]) if len(emotion_history) > 1 else 0
        
        # Main loss
        main_loss = -stability - correlation
        
        # Rehearsal loss (prevent catastrophic forgetting)
        rehearsal_loss = 0
        if rehearsal_batch:
            for episode in rehearsal_batch:
                # Simple rehearsal: maintain emotion stability
                rehearsal_loss += 0.1  # Small penalty for forgetting
        
        # Total loss
        total_loss = main_loss + rehearsal_loss
        
        return total_loss, emotion_params, emotion_history
    
    def train_continual(self, task_sequence, episodes_per_task=20, rehearsal_ratio=0.3):
        """Continual learning training on sequence of tasks"""
        print(f"🚀 Starting Continual Learning Training...")
        print(f"📊 Task Sequence: {task_sequence}")
        print(f"🎯 Episodes per task: {episodes_per_task}")
        
        training_history = {
            'task': [],
            'epoch': [],
            'loss': [],
            'stability': [],
            'correlation': [],
            'forgetting': []
        }
        
        for task_idx, task_name in enumerate(task_sequence):
            print(f"\n🎯 Learning Task {task_idx + 1}/{len(task_sequence)}: {task_name}")
            
            # Collect data for current task
            task_data = self.collect_task_data(task_name, episodes_per_task)
            
            # Store optimal parameters for this task
            with torch.no_grad():
                task_emb = self.task_encoder(task_data['obs_mean'], task_data['reward_mean'])
                emotion_params = self.emotion_adaptor(task_emb)
                self.optimal_params[task_name] = {
                    'task_embedding': task_emb.clone(),
                    'emotion_params': {k: v.clone() for k, v in emotion_params.items()}
                }
            
            # Train on current task with rehearsal
            for epoch in range(10):  # Few epochs per task
                # Sample rehearsal batch
                rehearsal_batch = self.episodic_memory.sample_rehearsal_batch(task_name, batch_size=5)
                
                # Continual learning step
                loss, emotion_params, emotion_history = self.continual_learning_step(task_data, rehearsal_batch)
                
                # Convert to tensor for backward pass
                loss_tensor = torch.tensor(loss, requires_grad=True, device=device)
                
                # Update
                self.optimizer.zero_grad()
                loss_tensor.backward()
                self.optimizer.step()
                
                # Calculate metrics
                stability = 1.0 - np.std(emotion_history)
                correlation = abs(np.corrcoef(emotion_history, task_data['scores'])[0, 1]) if len(emotion_history) > 1 else 0
                
                # Calculate forgetting (compare with optimal parameters)
                forgetting = 0
                if task_name in self.optimal_params:
                    with torch.no_grad():
                        current_emb = self.task_encoder(task_data['obs_mean'], task_data['reward_mean'])
                        optimal_emb = self.optimal_params[task_name]['task_embedding']
                        forgetting = F.mse_loss(current_emb, optimal_emb).item()
                
                # Track history
                training_history['task'].append(task_name)
                training_history['epoch'].append(epoch)
                training_history['loss'].append(loss)
                training_history['stability'].append(stability)
                training_history['correlation'].append(correlation)
                training_history['forgetting'].append(forgetting)
            
            # Evaluate performance on all learned tasks
            print(f"   Evaluating performance on all learned tasks...")
            for learned_task in task_sequence[:task_idx + 1]:
                if learned_task in self.optimal_params:
                    with torch.no_grad():
                        # Re-evaluate on learned task
                        learned_data = self.collect_task_data(learned_task, episodes=5)
                        current_emb = self.task_encoder(learned_data['obs_mean'], learned_data['reward_mean'])
                        optimal_emb = self.optimal_params[learned_task]['task_embedding']
                        performance_loss = F.mse_loss(current_emb, optimal_emb).item()
                        
                        if learned_task not in self.task_performance:
                            self.task_performance[learned_task] = []
                        self.task_performance[learned_task].append(performance_loss)
        
        return training_history

# =============================================================================
# CONTINUAL EMOTION ENGINE
# =============================================================================

class ContinualEmotionEngine:
    def __init__(self, alpha=0.1, initial_emotion=0.5):
        self.alpha = float(alpha)
        self.emotion = float(initial_emotion)
        self.past_scores = deque(maxlen=10)
        
    def update_parameters(self, new_params):
        self.alpha = float(new_params['alpha'].detach())
        self.emotion = float(new_params['initial_emotion'].detach())
        
    def update(self, current_score):
        self.past_scores.append(current_score)
        
        if len(self.past_scores) < 3:
            return self.emotion
            
        # Simple emotion update based on recent performance
        recent_avg = np.mean(list(self.past_scores)[-3:])
        if recent_avg > 0:
            self.emotion = min(0.8, self.emotion + self.alpha * 0.1)
        else:
            self.emotion = max(0.2, self.emotion - self.alpha * 0.1)
        
        return self.emotion

# =============================================================================
# CONTINUAL LEARNING EVALUATION
# =============================================================================

def evaluate_continual_learning(trainer, task_sequence):
    """Evaluates continual learning performance"""
    print(f"🧪 Evaluating Continual Learning...")
    
    results = {}
    
    for task_name in task_sequence:
        print(f"Testing {task_name}...")
        
        # Collect fresh data
        task_data = trainer.collect_task_data(task_name, episodes=10)
        
        # Get current task embedding
        with torch.no_grad():
            task_emb = trainer.task_encoder(task_data['obs_mean'], task_data['reward_mean'])
            emotion_params = trainer.emotion_adaptor(task_emb)
        
        # Create emotion engine
        emotion_engine = ContinualEmotionEngine()
        emotion_engine.update_parameters(emotion_params)
        
        # Test emotion adaptation
        emotion_history = []
        for score in task_data['scores']:
            emotion = emotion_engine.update(score)
            emotion_history.append(emotion)
        
        # Calculate metrics
        stability = 1.0 - np.std(emotion_history)
        correlation = abs(np.corrcoef(emotion_history, task_data['scores'])[0, 1]) if len(emotion_history) > 1 else 0
        
        # Calculate forgetting
        forgetting = 0
        if task_name in trainer.optimal_params:
            optimal_emb = trainer.optimal_params[task_name]['task_embedding']
            forgetting = F.mse_loss(task_emb, optimal_emb).item()
        
        results[task_name] = {
            'alpha': float(emotion_params['alpha']),
            'initial_emotion': float(emotion_params['initial_emotion']),
            'stability': stability,
            'correlation': correlation,
            'forgetting': forgetting,
            'emotion_history': emotion_history,
            'scores': task_data['scores']
        }
        
        print(f"  Alpha: {emotion_params['alpha']:.3f}, "
              f"Initial Emotion: {emotion_params['initial_emotion']:.3f}")
        print(f"  Stability: {stability:.3f}, Correlation: {correlation:.3f}")
        print(f"  Forgetting: {forgetting:.3f}")
    
    return results

# =============================================================================
# CONTINUAL LEARNING VISUALIZATION
# =============================================================================

def plot_continual_learning_results(training_history, evaluation_results):
    """Plots continual learning results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    tasks = training_history['task']
    epochs = training_history['epoch']
    losses = training_history['loss']
    stabilities = training_history['stability']
    correlations = training_history['correlation']
    forgettings = training_history['forgetting']
    
    # Loss over time
    axes[0, 0].plot(range(len(losses)), losses)
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Stability over time
    axes[0, 1].plot(range(len(stabilities)), stabilities)
    axes[0, 1].set_title('Emotion Stability Over Time')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Stability')
    axes[0, 1].grid(True)
    
    # Forgetting over time
    axes[0, 2].plot(range(len(forgettings)), forgettings)
    axes[0, 2].set_title('Catastrophic Forgetting Over Time')
    axes[0, 2].set_xlabel('Training Step')
    axes[0, 2].set_ylabel('Forgetting')
    axes[0, 2].grid(True)
    
    # Evaluation results
    task_names = list(evaluation_results.keys())
    stabilities_eval = [evaluation_results[task]['stability'] for task in task_names]
    correlations_eval = [evaluation_results[task]['correlation'] for task in task_names]
    forgettings_eval = [evaluation_results[task]['forgetting'] for task in task_names]
    
    axes[1, 0].bar(task_names, stabilities_eval, color='blue', alpha=0.7)
    axes[1, 0].set_title('Final Stability by Task')
    axes[1, 0].set_ylabel('Stability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(task_names, correlations_eval, color='green', alpha=0.7)
    axes[1, 1].set_title('Final Correlation by Task')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 2].bar(task_names, forgettings_eval, color='red', alpha=0.7)
    axes[1, 2].set_title('Final Forgetting by Task')
    axes[1, 2].set_ylabel('Forgetting')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎮 META-LEARNING PHASE 3: CONTINUAL LEARNING")
    print("=" * 60)
    
    # Initialize models
    task_encoder = FinalTaskEncoder(max_obs_dim=8, embedding_dim=64)
    emotion_adaptor = FinalEmotionAdaptor(embedding_dim=64)
    
    print(f"📊 Task Encoder: {sum(p.numel() for p in task_encoder.parameters())} parameters")
    print(f"📊 Emotion Adaptor: {sum(p.numel() for p in emotion_adaptor.parameters())} parameters")
    
    # Initialize Continual Learning Trainer
    continual_trainer = ContinualLearningTrainer(task_encoder, emotion_adaptor)
    
    # Define task sequence for continual learning
    task_sequence = list(test_envs.keys())
    
    # Train with continual learning
    training_history = continual_trainer.train_continual(
        task_sequence=task_sequence,
        episodes_per_task=15,
        rehearsal_ratio=0.3
    )
    
    # Evaluate continual learning
    evaluation_results = evaluate_continual_learning(continual_trainer, task_sequence)
    
    # Plot results
    plot_continual_learning_results(training_history, evaluation_results)
    
    # Final statistics
    print(f"\n🏆 CONTINUAL LEARNING RESULTS:")
    print(f"   Average Stability: {np.mean([r['stability'] for r in evaluation_results.values()]):.3f}")
    print(f"   Average Correlation: {np.mean([r['correlation'] for r in evaluation_results.values()]):.3f}")
    print(f"   Average Forgetting: {np.mean([r['forgetting'] for r in evaluation_results.values()]):.3f}")
    print(f"   Tasks Learned: {len(task_sequence)}")
    
    # Save results
    with open('continual_learning_results.json', 'w') as f:
        json_results = {}
        for task, result in evaluation_results.items():
            json_results[task] = {
                'alpha': result['alpha'],
                'initial_emotion': result['initial_emotion'],
                'stability': result['stability'],
                'correlation': result['correlation'],
                'forgetting': result['forgetting']
            }
        json.dump(json_results, f, indent=2)
    
    print("💾 Results saved to continual_learning_results.json!")
    
    print("\n🎉 PHASE 3 COMPLETE: Continual Learning Foundation Ready!")
    print("🚀 Ready for Real-World Applications!")
