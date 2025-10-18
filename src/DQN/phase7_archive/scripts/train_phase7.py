"""
Phase 7.0 Training Script
=========================

Vereinfachtes Training-Skript mit Phase 7.0 Features:
- Automatische Hyperparameter-Optimierung
- Performance-Stabilitäts-Tracking
- Adaptive Konfiguration
- Meta-Performance-Vorhersage

Quick Start für Phase 7.0 Evaluation.

Author: Phase 7.0 Implementation
Date: 2025-10-16
"""

import torch
import gymnasium as gym
import numpy as np
import random
import os
import csv
from tqdm import tqdm

# Phase 7 Integration
from core.phase7_integration_manager import Phase7IntegrationManager

# Existing components
try:
    from training.agent import DQNAgent
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from training.agent import DQNAgent


def setup_environment(seed=42):
    """Setup reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🎲 Random seed set to {seed}")


def train_phase7(
    episodes: int = 200,
    env_name: str = 'CartPole-v1',
    seed: int = 42,
    enable_phase7_full: bool = True,
    save_dir: str = 'phase7_training'
):
    """
    Train agent with Phase 7.0 features.
    
    Args:
        episodes: Number of training episodes
        env_name: Gym environment name
        seed: Random seed
        enable_phase7_full: Enable all Phase 7 features
        save_dir: Directory for saving results
    """
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║           Phase 7.0 Training                         ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    
    # Setup
    setup_environment(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"🌍 Environment: {env_name}")
    print(f"   State dim: {state_dim}, Action dim: {action_dim}")
    print(f"   Episodes: {episodes}\n")
    
    # Create Phase 7 Manager
    if enable_phase7_full:
        print("⚙️  Initializing Phase 7.0 components...")
        phase7_manager = Phase7IntegrationManager(
            enable_bayesian_optimization=True,
            enable_stability_analysis=True,
            enable_adaptive_config=True,
            enable_performance_prediction=True,
            save_dir=os.path.join(save_dir, 'checkpoints')
        )
        
        # Get initial hyperparameters
        hyperparams = phase7_manager.start_new_run()
        print(f"   ✓ Bayesian Hyperparameter Optimizer")
        print(f"   ✓ Performance Stability Analyzer")
        print(f"   ✓ Adaptive Configuration Manager")
        print(f"   ✓ Meta-Performance-Predictor")
        print(f"\n🎯 Initial Hyperparameters:")
        for key, value in list(hyperparams.items())[:5]:
            print(f"   {key}: {value:.4f}")
        print(f"   ... (showing first 5 of {len(hyperparams)})\n")
    else:
        print("⚙️  Phase 7.0 features disabled (baseline mode)\n")
        phase7_manager = None
        hyperparams = {}
    
    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.05,
        buffer_capacity=50000,
        emotion_enabled=True
    )
    
    # Apply hyperparameters if available
    if hyperparams and hasattr(agent, 'emotion_engine'):
        if 'eta_min' in hyperparams:
            agent.emotion_engine.eta_min = hyperparams['eta_min']
        if 'eta_max' in hyperparams:
            agent.emotion_engine.eta_max = hyperparams['eta_max']
        if 'eta_decay_rate' in hyperparams:
            agent.emotion_engine.eta_decay = hyperparams['eta_decay_rate']
    
    # CSV Logging Setup
    log_path = os.path.join(save_dir, 'training_log.csv')
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # CSV Header
    header = ['episode', 'return', 'td_error', 'eta', 'emotion']
    if phase7_manager:
        header.extend([
            'bho_iteration', 'bho_best_performance',
            'psa_stability_score', 'psa_trend',
            'acm_weight_reactivity', 'acm_weight_anticipation', 'acm_weight_reflection', 'acm_weight_prediction',
            'acm_system_state',
            'mpp_predicted_performance', 'mpp_confidence'
        ])
    csv_writer.writerow(header)
    
    # Training Loop
    print("🚀 Starting Training...\n")
    
    episode_returns = []
    td_errors = []
    
    progress_bar = tqdm(range(episodes), desc="Training")
    
    for episode in progress_bar:
        state, _ = env.reset()
        episode_return = 0.0
        episode_td_errors = []
        done = False
        step_count = 0
        
        while not done and step_count < 500:
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) >= agent.batch_size:
                td_error = agent.train()
                if td_error is not None:
                    episode_td_errors.append(td_error)
            
            episode_return += reward
            state = next_state
            step_count += 1
        
        episode_returns.append(episode_return)
        avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0.0
        td_errors.append(avg_td_error)
        
        # Get agent metrics
        emotion = agent.emotion_engine.emotion if hasattr(agent, 'emotion_engine') else 0.4
        eta = agent.emotion_engine.eta if hasattr(agent, 'emotion_engine') else 0.01
        
        # Update Phase 7 components
        phase7_metrics = {}
        if phase7_manager:
            # Compute layer activities (simplified)
            layer_activities = {
                'reactivity': 1.0,
                'anticipation': 1.0,
                'reflection': 1.0,
                'prediction': 1.0
            }
            
            phase7_manager.update_episode(
                episode,
                episode_return,
                avg_td_error,
                emotion,
                layer_activities
            )
            
            # Get Phase 7 metrics
            phase7_metrics = phase7_manager.get_csv_metrics()
        
        # Log to CSV
        row = [episode, episode_return, avg_td_error, eta, emotion]
        if phase7_manager:
            row.extend([
                phase7_metrics.get('bho_iteration', 0),
                phase7_metrics.get('bho_best_performance', 0.0),
                phase7_metrics.get('psa_stability_score', 0.0),
                phase7_metrics.get('psa_trend', 'unknown'),
                phase7_metrics.get('acm_weight_reactivity', 1.0),
                phase7_metrics.get('acm_weight_anticipation', 1.0),
                phase7_metrics.get('acm_weight_reflection', 1.0),
                phase7_metrics.get('acm_weight_prediction', 1.0),
                phase7_metrics.get('acm_system_state', 'unknown'),
                phase7_metrics.get('mpp_predicted_performance', 0.0),
                phase7_metrics.get('mpp_confidence', 0.0)
            ])
        csv_writer.writerow(row)
        
        # Update progress bar
        if episode % 10 == 0:
            avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            progress_bar.set_postfix({
                'avg100': f'{avg100:.2f}',
                'emotion': f'{emotion:.3f}',
                'td_error': f'{avg_td_error:.3f}'
            })
        
        # Periodic reports
        if episode > 0 and episode % 50 == 0:
            print(f"\n{'='*60}")
            print(f"Episode {episode} Checkpoint")
            print(f"{'='*60}")
            print(f"📊 Performance:")
            print(f"   avg100: {np.mean(episode_returns[-100:]):.2f}")
            print(f"   Best: {max(episode_returns):.2f}")
            print(f"   TD Error: {avg_td_error:.3f}")
            
            if phase7_manager:
                stability = phase7_manager.get_stability_metrics()
                weights = phase7_manager.get_layer_weights()
                print(f"\n📈 Phase 7 Status:")
                print(f"   Stability Score: {stability.get('stability_score', 0.0):.3f}")
                print(f"   Trend: {stability.get('trend', 'unknown')}")
                print(f"   Layer Weights: R={weights['reactivity']:.2f}, A={weights['anticipation']:.2f}, "
                      f"Ref={weights['reflection']:.2f}, P={weights['prediction']:.2f}")
            print()
    
    # Close CSV
    csv_file.close()
    env.close()
    
    # Final Statistics
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
    print(f"\n📊 Final Performance:")
    print(f"   avg100: {avg100:.2f}")
    print(f"   Best Episode: {max(episode_returns):.2f}")
    print(f"   Final TD Error: {np.mean(td_errors[-10:]):.3f}")
    
    if phase7_manager:
        print(f"\n🎯 Phase 7 Summary:")
        print(phase7_manager.get_comprehensive_report())
        
        # Save checkpoint
        phase7_manager.save_checkpoint(prefix="final")
        
        # Best hyperparameters
        best_params = phase7_manager.get_best_hyperparams()
        if best_params:
            print(f"\n🏆 Best Hyperparameters Found:")
            for key, value in list(best_params.items())[:5]:
                print(f"   {key}: {value:.4f}")
            print(f"   ... (showing first 5 of {len(best_params)})")
    
    print(f"\n💾 Results saved to: {save_dir}")
    print(f"   Training log: {log_path}")
    
    return episode_returns, td_errors, phase7_manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 7.0 Training')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-phase7', action='store_true', help='Disable Phase 7 features (baseline)')
    parser.add_argument('--save-dir', type=str, default='phase7_training', help='Save directory')
    
    args = parser.parse_args()
    
    train_phase7(
        episodes=args.episodes,
        env_name=args.env,
        seed=args.seed,
        enable_phase7_full=not args.no_phase7,
        save_dir=args.save_dir
    )

