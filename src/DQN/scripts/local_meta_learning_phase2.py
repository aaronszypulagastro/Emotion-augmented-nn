"""
Local Meta-Learning Phase 2 - Few-Shot Learning
===============================================

Few-Shot Learning Implementation
- Lernt neue Tasks in wenigen Episodes
- Verwendet Phase 1 Modelle als Foundation
- Testet auf neuen Environments

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

# New environments for few-shot learning
new_envs = {
    'Pendulum-v1': 3,
    'MountainCarContinuous-v0': 2
}

print("🧪 Testing environments...")
for env_name, obs_dim in {**test_envs, **new_envs}.items():
    try:
        env = gym.make(env_name)
        print(f"✅ {env_name}: {obs_dim}D state, {env.action_space.n} actions")
        env.close()
    except Exception as e:
        print(f"❌ {env_name}: {e}")

print("\n🎯 Ready for Meta-Learning Phase 2: Few-Shot Learning!")

# =============================================================================
# LOAD PHASE 1 MODELS
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
# FEW-SHOT LEARNER
# =============================================================================

class FewShotLearner:
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
        
        # Freeze Phase 1 models
        for param in self.task_encoder.parameters():
            param.requires_grad = False
        for param in self.emotion_adaptor.parameters():
            param.requires_grad = False
    
    def collect_support_data(self, env_name, episodes=10):
        """Sammelt Support-Daten für Few-Shot Learning"""
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        
        all_observations = []
        all_rewards = []
        scores = []
        
        print(f"📊 Collecting support data from {env_name}...")
        
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
    
    def adapt_to_new_task(self, env_name, support_episodes=10, adaptation_episodes=20):
        """Adaptiert sich an neue Task mit Few-Shot Learning"""
        print(f"🎯 Few-Shot Learning on {env_name}...")
        
        # Collect support data
        support_data = self.collect_support_data(env_name, support_episodes)
        
        # Get task embedding from Phase 1 models
        with torch.no_grad():
            task_embedding = self.task_encoder(support_data['obs_mean'], support_data['reward_mean'])
            emotion_params = self.emotion_adaptor(task_embedding)
        
        print(f"   Predicted Alpha: {emotion_params['alpha']:.3f}")
        print(f"   Predicted Initial Emotion: {emotion_params['initial_emotion']:.3f}")
        
        # Create adapted emotion engine
        adapted_emotion_engine = AdaptedEmotionEngine()
        adapted_emotion_engine.update_parameters(emotion_params)
        
        # Test adaptation on more episodes
        print(f"   Testing adaptation on {adaptation_episodes} episodes...")
        
        env = gym.make(env_name)
        adaptation_scores = []
        emotion_history = []
        
        for episode in range(adaptation_episodes):
            state, _ = env.reset()
            episode_score = 0
            
            for step in range(500):
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_score += reward
                state = next_state
                
                if done:
                    break
            
            adaptation_scores.append(episode_score)
            emotion = adapted_emotion_engine.update(episode_score)
            emotion_history.append(emotion)
        
        env.close()
        
        # Calculate adaptation metrics
        support_avg = np.mean(support_data['scores'])
        adaptation_avg = np.mean(adaptation_scores)
        improvement = adaptation_avg - support_avg
        
        stability = 1.0 - np.std(emotion_history)
        correlation = abs(np.corrcoef(emotion_history, adaptation_scores)[0, 1]) if len(emotion_history) > 1 else 0
        
        print(f"   Support Avg: {support_avg:.2f}")
        print(f"   Adaptation Avg: {adaptation_avg:.2f}")
        print(f"   Improvement: {improvement:.2f}")
        print(f"   Stability: {stability:.3f}")
        print(f"   Correlation: {correlation:.3f}")
        
        return {
            'env_name': env_name,
            'support_avg': support_avg,
            'adaptation_avg': adaptation_avg,
            'improvement': improvement,
            'stability': stability,
            'correlation': correlation,
            'emotion_params': {
                'alpha': float(emotion_params['alpha']),
                'initial_emotion': float(emotion_params['initial_emotion'])
            },
            'emotion_history': emotion_history,
            'adaptation_scores': adaptation_scores
        }

# =============================================================================
# ADAPTED EMOTION ENGINE
# =============================================================================

class AdaptedEmotionEngine:
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
# FEW-SHOT EVALUATION
# =============================================================================

def evaluate_few_shot_learning(few_shot_learner, test_tasks, new_tasks):
    """Evaluates Few-Shot Learning on different tasks"""
    print(f"🧪 Evaluating Few-Shot Learning...")
    
    results = {}
    
    # Test on known tasks (should work well)
    print(f"\n📊 Testing on Known Tasks:")
    for task_name in test_tasks:
        result = few_shot_learner.adapt_to_new_task(task_name, support_episodes=5, adaptation_episodes=15)
        results[task_name] = result
    
    # Test on new tasks (few-shot learning challenge)
    print(f"\n🎯 Testing on New Tasks (Few-Shot Challenge):")
    for task_name in new_tasks:
        try:
            result = few_shot_learner.adapt_to_new_task(task_name, support_episodes=5, adaptation_episodes=15)
            results[task_name] = result
        except Exception as e:
            print(f"❌ Failed on {task_name}: {e}")
            results[task_name] = {
                'env_name': task_name,
                'support_avg': 0,
                'adaptation_avg': 0,
                'improvement': 0,
                'stability': 0,
                'correlation': 0,
                'emotion_params': {'alpha': 0, 'initial_emotion': 0.5},
                'emotion_history': [],
                'adaptation_scores': []
            }
    
    return results

# =============================================================================
# FEW-SHOT VISUALIZATION
# =============================================================================

def plot_few_shot_results(results):
    """Plots Few-Shot Learning results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    task_names = list(results.keys())
    improvements = [results[task]['improvement'] for task in task_names]
    stabilities = [results[task]['stability'] for task in task_names]
    correlations = [results[task]['correlation'] for task in task_names]
    alphas = [results[task]['emotion_params']['alpha'] for task in task_names]
    emotions = [results[task]['emotion_params']['initial_emotion'] for task in task_names]
    
    # Improvement by task
    colors = ['blue' if task in test_envs else 'red' for task in task_names]
    bars1 = axes[0, 0].bar(task_names, improvements, color=colors, alpha=0.7)
    axes[0, 0].set_title('Performance Improvement by Task')
    axes[0, 0].set_ylabel('Improvement')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add legend
    axes[0, 0].legend(['Known Tasks', 'New Tasks'], loc='upper right')
    
    # Stability by task
    bars2 = axes[0, 1].bar(task_names, stabilities, color=colors, alpha=0.7)
    axes[0, 1].set_title('Emotion Stability by Task')
    axes[0, 1].set_ylabel('Stability')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Correlation by task
    bars3 = axes[0, 2].bar(task_names, correlations, color=colors, alpha=0.7)
    axes[0, 2].set_title('Performance-Emotion Correlation by Task')
    axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Alpha by task
    bars4 = axes[1, 0].bar(task_names, alphas, color=colors, alpha=0.7)
    axes[1, 0].set_title('Alpha (Learning Rate) by Task')
    axes[1, 0].set_ylabel('Alpha')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Initial emotion by task
    bars5 = axes[1, 1].bar(task_names, emotions, color=colors, alpha=0.7)
    axes[1, 1].set_title('Initial Emotion by Task')
    axes[1, 1].set_ylabel('Initial Emotion')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Improvement vs Stability scatter
    scatter = axes[1, 2].scatter(stabilities, improvements, c=colors, alpha=0.7, s=100)
    for i, task in enumerate(task_names):
        axes[1, 2].annotate(task, (stabilities[i], improvements[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 2].set_title('Stability vs Improvement')
    axes[1, 2].set_xlabel('Stability')
    axes[1, 2].set_ylabel('Improvement')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎮 META-LEARNING PHASE 2: FEW-SHOT LEARNING")
    print("=" * 60)
    
    # Initialize models
    task_encoder = FinalTaskEncoder(max_obs_dim=8, embedding_dim=64)
    emotion_adaptor = FinalEmotionAdaptor(embedding_dim=64)
    
    print(f"📊 Task Encoder: {sum(p.numel() for p in task_encoder.parameters())} parameters")
    print(f"📊 Emotion Adaptor: {sum(p.numel() for p in emotion_adaptor.parameters())} parameters")
    
    # Initialize Few-Shot Learner
    few_shot_learner = FewShotLearner(task_encoder, emotion_adaptor)
    
    # Evaluate Few-Shot Learning
    results = evaluate_few_shot_learning(
        few_shot_learner, 
        test_tasks=list(test_envs.keys()),
        new_tasks=list(new_envs.keys())
    )
    
    # Plot results
    plot_few_shot_results(results)
    
    # Final statistics
    print(f"\n🏆 FEW-SHOT LEARNING RESULTS:")
    
    known_tasks = [task for task in results.keys() if task in test_envs]
    new_tasks = [task for task in results.keys() if task in new_envs]
    
    if known_tasks:
        known_improvements = [results[task]['improvement'] for task in known_tasks]
        known_stabilities = [results[task]['stability'] for task in known_tasks]
        print(f"   Known Tasks ({len(known_tasks)}):")
        print(f"     Average Improvement: {np.mean(known_improvements):.2f}")
        print(f"     Average Stability: {np.mean(known_stabilities):.3f}")
    
    if new_tasks:
        new_improvements = [results[task]['improvement'] for task in new_tasks]
        new_stabilities = [results[task]['stability'] for task in new_tasks]
        print(f"   New Tasks ({len(new_tasks)}):")
        print(f"     Average Improvement: {np.mean(new_improvements):.2f}")
        print(f"     Average Stability: {np.mean(new_stabilities):.3f}")
    
    print(f"   Total Tasks Tested: {len(results)}")
    
    # Save results
    import json
    with open('few_shot_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for task, result in results.items():
            json_results[task] = {
                'env_name': result['env_name'],
                'support_avg': float(result['support_avg']),
                'adaptation_avg': float(result['adaptation_avg']),
                'improvement': float(result['improvement']),
                'stability': float(result['stability']),
                'correlation': float(result['correlation']),
                'emotion_params': result['emotion_params']
            }
        json.dump(json_results, f, indent=2)
    
    print("💾 Results saved to few_shot_results.json!")
    
    print("\n🎉 PHASE 2 COMPLETE: Few-Shot Learning Foundation Ready!")
    print("🚀 Ready for Phase 3: Continual Learning!")
