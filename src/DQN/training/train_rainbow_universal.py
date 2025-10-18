"""
Universal Rainbow DQN Training - Multi-Environment Support
===========================================================

PHILOSOPHY: "Test on MULTIPLE environments before optimizing"
-------------------------------------------------------------

Instead of:
❌ Optimize on CartPole → Hope it generalizes
   
We do:
✅ Test on CartPole, Acrobot, LunarLander
✅ Find what works UNIVERSALLY
✅ Then optimize based on multi-env data

SUPPORTED ENVIRONMENTS:
-----------------------
- CartPole-v1 (baseline, 4D state, 2 actions)
- Acrobot-v1 (harder, 6D state, 3 actions, sparse rewards)
- LunarLander-v2 (hardest, 8D state, 4 actions, complex rewards)

FEATURES:
---------
✅ Rainbow DQN (PER + Dueling + Double + N-Step)
✅ Competitive Emotion (no saturation)
✅ Infrastructure Modulation (regional conditions)
✅ Auto-configured per environment
✅ Systematic logging for comparison

Author: Multi-Environment Universal Training
Date: 2025-10-17
"""

import torch
import gymnasium as gym
import numpy as np
import random
import os
import csv
import argparse
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.rainbow_dqn_agent import RainbowDQNAgent
from core.infrastructure_profile import InfrastructureProfile

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==================== AUTO-CONFIGURATION ====================

def get_environment_config(env_name: str) -> dict:
    """
    Auto-configure hyperparameters based on environment
    
    Based on best practices from literature + our experiments
    """
    
    base_config = {
        'base_lr': 5e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_capacity': 50000,
        
        # Rainbow DQN
        'n_step': 3,
        'per_alpha': 0.6,
        'per_beta_start': 0.4,
        'per_beta_frames': 100000,
        'tau': 0.005,
        'use_soft_updates': True,
        
        # Competition (LESS frequent for stability)
        'competition_freq': 20,
        'competitor_strategy': 'past_self',
        'competitor_history_depth': 50,
        'save_checkpoint_freq': 25,
        
        # Exploration
        'epsilon_min': 0.01,
        'epsilon_decay': 0.996,
    }
    
    # Environment-specific adjustments
    if env_name == 'CartPole-v1':
        config = base_config.copy()
        config.update({
            'episodes': 800,
            'max_steps': 500,
            'use_large_network': False,
        })
        
    elif env_name == 'Acrobot-v1':
        config = base_config.copy()
        config.update({
            'episodes': 1000,  # More episodes (harder)
            'max_steps': 500,
            'epsilon_decay': 0.997,  # Slower exploration decay
            'competition_freq': 30,  # Even less frequent
            'use_large_network': False,
        })
        
    elif env_name == 'LunarLander-v2':
        config = base_config.copy()
        config.update({
            'episodes': 1200,  # Most episodes
            'max_steps': 1000,
            'buffer_capacity': 100000,  # Larger buffer
            'epsilon_decay': 0.998,  # Very slow decay
            'competition_freq': 40,  # Least frequent
            'use_large_network': True,  # Larger network
        })
    
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    return config

# ==================== TRAINING FUNCTION ====================

def train_rainbow_universal(
    env_name: str = 'CartPole-v1',
    region: str = None,
    config_override: dict = None
):
    """
    Universal training function
    
    Args:
        env_name: Gymnasium environment name
        region: Optional infrastructure region
        config_override: Optional config overrides
    """
    
    # Auto-configure
    config = get_environment_config(env_name)
    
    if config_override:
        config.update(config_override)
    
    print("=" * 70)
    print(f"   RAINBOW DQN UNIVERSAL TRAINING")
    print(f"   Environment: {env_name}")
    if region:
        print(f"   Region: {region}")
    print("=" * 70 + "\n")
    
    # Environment
    env = gym.make(env_name)
    state, _ = env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"[ENV] {env_name}:")
    print(f"  State dim:  {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Episodes:   {config['episodes']}")
    print(f"  Max steps:  {config['max_steps']}\n")
    
    # Infrastructure (optional)
    infrastructure = None
    if region:
        infrastructure = InfrastructureProfile(region)
        print(f"[INFRASTRUCTURE] {region}:")
        print(f"  Loop Speed: {infrastructure.loop_speed:.2f}")
        print(f"  Automation: {infrastructure.automation:.2f}\n")
    
    # Create Rainbow Agent
    agent = RainbowDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        infrastructure=infrastructure,
        use_large_network=config.get('use_large_network', False)
    )
    
    # Logging
    env_short = env_name.replace('-v1', '').replace('-v2', '').lower()
    region_short = region.lower() if region else 'noregion'
    log_path = f"results/rainbow_{env_short}_{region_short}.csv"
    
    os.makedirs("results", exist_ok=True)
    
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "return", "epsilon", "emotion", "competitive_mindset",
            "win_rate", "had_competition", "lr_actual",
            "buffer_size", "train_steps"
        ])
    
    scores = []
    
    print("[START] Training...\n")
    
    # ==================== TRAINING LOOP ====================
    
    for episode in tqdm(range(config['episodes']), desc=f"Rainbow {env_short}"):
        
        # Train episode
        score = agent.train_episode(env, deterministic=False)
        scores.append(score)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Competition
        had_competition = False
        if episode > 0 and episode % config['competition_freq'] == 0:
            result = agent.compete(env, episode)
            if result is not None:
                had_competition = True
                
                if episode % 100 == 0:
                    print(f"\n[COMPETITION] Ep {episode}:")
                    print(f"  Main: {result.score_self:.1f} vs Past: {result.score_competitor:.1f}")
                    print(f"  Outcome: {result.outcome.value}")
                    print(f"  Emotion: {result.new_emotion:.3f} (Delta{result.emotion_delta:+.3f})")
        
        # Save checkpoint
        if episode % config['save_checkpoint_freq'] == 0:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            agent.save_checkpoint(episode, avg_score)
        
        # Logging
        if episode % 10 == 0:
            stats = agent.emotion.get_stats()
            metrics = agent.get_metrics()
            
            with open(log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, score, metrics['epsilon'], metrics['emotion'],
                    metrics['mindset'], stats.get('win_rate', 0.0),
                    had_competition,
                    metrics['lr'], metrics['buffer_size'], metrics['train_steps']
                ])
        
        # Progress
        if episode % 100 == 0 and episode > 0:
            avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            metrics = agent.get_metrics()
            
            print(f"\n[STATUS] Episode {episode}:")
            print(f"  Avg100: {avg_100:.1f}")
            print(f"  Emotion: {metrics['emotion']:.3f}")
            print(f"  Mindset: {metrics['mindset']}")
            print(f"  Buffer: {metrics['buffer_size']}")
    
    env.close()
    
    # ==================== FINAL RESULTS ====================
    
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    best = max(scores)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nEnvironment: {env_name}")
    if region:
        print(f"Region:      {region}")
    print(f"\nPerformance:")
    print(f"  Final avg100: {final_avg:.1f}")
    print(f"  Best episode: {best:.1f}")
    
    stats = agent.emotion.get_stats()
    print(f"\nCompetition:")
    print(f"  Win Rate: {stats.get('win_rate', 0.0):.1%}")
    print(f"  Total:    {stats.get('total_competitions', 0):.0f}")
    
    print(f"\nEmotion:")
    print(f"  Final:  {agent.emotion.value:.3f}")
    print(f"  Mindset: {agent.emotion.get_competitive_mindset()}")
    
    # Save model
    model_path = f"results/rainbow_{env_short}_{region_short}_final.pth"
    agent.save_model(model_path)
    
    print(f"\n[OK] Model saved: {model_path}")
    print(f"[OK] Log saved:   {log_path}")
    
    return final_avg, best


# ==================== MULTI-ENVIRONMENT TESTING ====================

def test_all_environments(region: str = None):
    """
    Test Rainbow Agent on ALL environments
    
    This is the RIGHT way: Test generalization BEFORE optimizing!
    """
    
    print("\n" + "="*70)
    print("   MULTI-ENVIRONMENT GENERALIZATION TEST")
    print("   Testing Rainbow DQN across 3 environments")
    print("="*70 + "\n")
    
    environments = ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    results = {}
    
    for env_name in environments:
        print(f"\n{'#'*70}")
        print(f"# TESTING: {env_name}")
        print(f"{'#'*70}\n")
        
        try:
            avg100, best = train_rainbow_universal(env_name, region=region)
            results[env_name] = {
                'avg100': avg100,
                'best': best,
                'status': 'SUCCESS'
            }
        except Exception as e:
            print(f"\n[ERROR] {env_name} failed: {e}")
            results[env_name] = {
                'avg100': None,
                'best': None,
                'status': 'FAILED',
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*70)
    print("MULTI-ENVIRONMENT SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Environment':<20} {'Avg100':<12} {'Best':<12} {'Status'}")
    print("-"*70)
    
    for env_name, result in results.items():
        env_short = env_name.replace('-v1', '').replace('-v2', '')
        if result['status'] == 'SUCCESS':
            print(f"{env_short:<20} {result['avg100']:<12.1f} {result['best']:<12.1f} {result['status']}")
        else:
            print(f"{env_short:<20} {'N/A':<12} {'N/A':<12} {result['status']}")
    
    print("\n" + "="*70)
    
    # Check generalization
    successes = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    
    if successes == len(environments):
        print("\n✅ EXCELLENT: Rainbow agent works on ALL environments!")
        print("   → System generalizes well")
        print("   → Ready for optimization")
    elif successes >= 2:
        print("\n⚠️  GOOD: Rainbow agent works on most environments")
        print("   → Identify why one failed")
        print("   → Fix specific issue before optimizing")
    else:
        print("\n❌ PROBLEM: Rainbow agent fails on multiple environments")
        print("   → Debug integration before proceeding")
        print("   → Don't optimize yet!")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universal Rainbow DQN Training')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       choices=['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2'],
                       help='Environment to train on')
    parser.add_argument('--region', type=str, default=None,
                       help='Infrastructure region (optional)')
    parser.add_argument('--test-all', action='store_true',
                       help='Test on all environments sequentially')
    
    args = parser.parse_args()
    
    if args.test_all:
        # Test on ALL environments
        test_all_environments(region=args.region)
    else:
        # Train on single environment
        train_rainbow_universal(args.env, region=args.region)

