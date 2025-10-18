"""
Phase 7.0 Benchmark Suite
==========================

Vergleichende Tests mit verschiedenen Konfigurationen:
- Baseline (ohne Phase 7 Features)
- Phase 7 mit BHO only
- Phase 7 mit allen Features
- Phase 6.3 (beste bisherige Konfiguration)

Führt mehrere Runs durch und vergleicht Performance und Stabilität.

Author: Phase 7.0 Implementation
Date: 2025-10-16
"""

import torch
import gymnasium as gym
import numpy as np
import random
import os
import csv
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Phase 7 components
from core.phase7_integration_manager import Phase7IntegrationManager

# Import existing training components
try:
    from training.agent import DQNAgent
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from training.agent import DQNAgent


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    enable_bho: bool = False
    enable_psa: bool = False
    enable_acm: bool = False
    enable_mpp: bool = False
    episodes: int = 200
    n_runs: int = 3
    seed_start: int = 42


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config_name: str
    run_id: int
    seed: int
    avg_performance: float
    final_td_error: float
    stability_score: float
    best_episode_return: float
    convergence_episode: int  # Episode where performance stabilized


class Phase7Benchmark:
    """
    Benchmark suite for Phase 7.0 evaluation.
    
    Compares different configurations and measures:
    - Average performance (avg100)
    - Stability
    - Convergence speed
    - Robustness across seeds
    """
    
    def __init__(
        self,
        env_name: str = 'CartPole-v1',
        save_dir: str = 'phase7_benchmark_results'
    ):
        """
        Initialize benchmark suite.
        
        Args:
            env_name: Gym environment name
            save_dir: Directory for saving results
        """
        self.env_name = env_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        
        # Define benchmark configurations
        self.configs = [
            BenchmarkConfig(
                name="Baseline_No_Phase7",
                enable_bho=False,
                enable_psa=False,
                enable_acm=False,
                enable_mpp=False
            ),
            BenchmarkConfig(
                name="Phase7_BHO_Only",
                enable_bho=True,
                enable_psa=False,
                enable_acm=False,
                enable_mpp=False
            ),
            BenchmarkConfig(
                name="Phase7_Full",
                enable_bho=True,
                enable_psa=True,
                enable_acm=True,
                enable_mpp=True
            )
        ]
    
    def run_single_configuration(
        self,
        config: BenchmarkConfig,
        run_id: int,
        seed: int,
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Run a single configuration with given seed.
        
        Args:
            config: Benchmark configuration
            run_id: Run identifier
            seed: Random seed
            verbose: Print progress
            
        Returns:
            BenchmarkResult object
        """
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        env = gym.make(self.env_name)
        env.reset(seed=seed)
        
        # Create agent (simplified configuration)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=5e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.05,
            buffer_capacity=10000,
            emotion_enabled=True
        )
        
        # Create Phase 7 manager
        phase7_manager = None
        if any([config.enable_bho, config.enable_psa, config.enable_acm, config.enable_mpp]):
            phase7_manager = Phase7IntegrationManager(
                enable_bayesian_optimization=config.enable_bho,
                enable_stability_analysis=config.enable_psa,
                enable_adaptive_config=config.enable_acm,
                enable_performance_prediction=config.enable_mpp,
                save_dir=os.path.join(self.save_dir, f"{config.name}_run{run_id}")
            )
            
            if config.enable_bho:
                hyperparams = phase7_manager.start_new_run()
                # Apply hyperparameters (simplified - would need full integration)
                if 'eta_min' in hyperparams:
                    agent.emotion_engine.eta_min = hyperparams['eta_min']
                if 'eta_max' in hyperparams:
                    agent.emotion_engine.eta_max = hyperparams['eta_max']
        
        # Training loop
        episode_returns = []
        td_errors = []
        
        if verbose:
            progress_bar = tqdm(range(config.episodes), desc=f"{config.name} Run {run_id}")
        else:
            progress_bar = range(config.episodes)
        
        for episode in progress_bar:
            state, _ = env.reset()
            episode_return = 0.0
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
                        td_errors.append(td_error)
                
                episode_return += reward
                state = next_state
                step_count += 1
            
            episode_returns.append(episode_return)
            
            # Update Phase 7 components
            if phase7_manager is not None:
                layer_activities = {
                    'reactivity': 1.0,
                    'anticipation': 1.0,
                    'reflection': 1.0,
                    'prediction': 1.0
                }
                
                emotion = agent.emotion_engine.emotion if hasattr(agent, 'emotion_engine') else 0.4
                td_error_val = td_errors[-1] if td_errors else 0.0
                
                phase7_manager.update_episode(
                    episode,
                    episode_return,
                    td_error_val,
                    emotion,
                    layer_activities
                )
            
            # Update progress bar
            if verbose and episode % 10 == 0:
                avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
                progress_bar.set_postfix({'avg100': f'{avg100:.2f}'})
        
        env.close()
        
        # Compute metrics
        avg_performance = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
        final_td_error = np.mean(td_errors[-100:]) if len(td_errors) >= 100 else 0.0
        best_episode_return = max(episode_returns)
        
        # Compute stability score
        if len(episode_returns) >= 100:
            stability_score = 1.0 / (1.0 + np.std(episode_returns[-100:]) / (abs(np.mean(episode_returns[-100:])) + 1e-8))
        else:
            stability_score = 0.0
        
        # Find convergence episode (when avg10 first exceeds 90% of final avg100)
        convergence_episode = config.episodes
        if len(episode_returns) >= 10:
            target = avg_performance * 0.9
            for i in range(10, len(episode_returns)):
                if np.mean(episode_returns[i-10:i]) >= target:
                    convergence_episode = i
                    break
        
        result = BenchmarkResult(
            config_name=config.name,
            run_id=run_id,
            seed=seed,
            avg_performance=avg_performance,
            final_td_error=final_td_error,
            stability_score=stability_score,
            best_episode_return=best_episode_return,
            convergence_episode=convergence_episode
        )
        
        return result
    
    def run_all_benchmarks(self, verbose: bool = True):
        """Run all benchmark configurations"""
        print("╔══════════════════════════════════════════════════════╗")
        print("║         Phase 7.0 Benchmark Suite                   ║")
        print("╚══════════════════════════════════════════════════════╝\n")
        
        for config in self.configs:
            print(f"\n{'='*60}")
            print(f"Running Configuration: {config.name}")
            print(f"{'='*60}\n")
            
            for run_id in range(config.n_runs):
                seed = config.seed_start + run_id
                print(f"  Run {run_id + 1}/{config.n_runs} (seed={seed})")
                
                result = self.run_single_configuration(
                    config,
                    run_id,
                    seed,
                    verbose=verbose
                )
                
                self.results.append(result)
                
                print(f"    → avg100: {result.avg_performance:.2f}, "
                      f"stability: {result.stability_score:.3f}, "
                      f"convergence: ep{result.convergence_episode}")
        
        print(f"\n{'='*60}")
        print("Benchmark Complete!")
        print(f"{'='*60}\n")
        
        # Save results
        self.save_results()
        
        # Generate report
        self.generate_report()
        
        # Generate plots
        self.generate_plots()
    
    def save_results(self):
        """Save benchmark results to CSV and JSON"""
        # CSV
        csv_path = os.path.join(self.save_dir, 'benchmark_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'config_name', 'run_id', 'seed',
                'avg_performance', 'final_td_error', 'stability_score',
                'best_episode_return', 'convergence_episode'
            ])
            
            for result in self.results:
                writer.writerow([
                    result.config_name, result.run_id, result.seed,
                    result.avg_performance, result.final_td_error, result.stability_score,
                    result.best_episode_return, result.convergence_episode
                ])
        
        print(f"💾 Results saved to {csv_path}")
        
        # JSON
        json_path = os.path.join(self.save_dir, 'benchmark_results.json')
        results_dict = [vars(r) for r in self.results]
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def generate_report(self):
        """Generate summary report"""
        report_path = os.path.join(self.save_dir, 'benchmark_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Phase 7.0 Benchmark Report\n\n")
            f.write("## Summary Statistics\n\n")
            
            # Group by configuration
            config_names = list(set(r.config_name for r in self.results))
            
            f.write("| Configuration | Avg Performance | Stability | Convergence (ep) |\n")
            f.write("|---------------|-----------------|-----------|------------------|\n")
            
            for config_name in config_names:
                config_results = [r for r in self.results if r.config_name == config_name]
                
                avg_perf = np.mean([r.avg_performance for r in config_results])
                std_perf = np.std([r.avg_performance for r in config_results])
                
                avg_stab = np.mean([r.stability_score for r in config_results])
                avg_conv = np.mean([r.convergence_episode for r in config_results])
                
                f.write(f"| {config_name} | {avg_perf:.2f} ± {std_perf:.2f} | "
                       f"{avg_stab:.3f} | {avg_conv:.0f} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for config_name in config_names:
                f.write(f"### {config_name}\n\n")
                config_results = [r for r in self.results if r.config_name == config_name]
                
                for result in config_results:
                    f.write(f"- **Run {result.run_id}** (seed={result.seed}):\n")
                    f.write(f"  - Avg100: {result.avg_performance:.2f}\n")
                    f.write(f"  - Stability: {result.stability_score:.3f}\n")
                    f.write(f"  - Best Episode: {result.best_episode_return:.2f}\n")
                    f.write(f"  - Convergence: Episode {result.convergence_episode}\n\n")
        
        print(f"📄 Report generated at {report_path}")
    
    def generate_plots(self):
        """Generate comparison plots"""
        config_names = list(set(r.config_name for r in self.results))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 7.0 Benchmark Comparison', fontsize=16)
        
        # Plot 1: Average Performance
        ax = axes[0, 0]
        perfs = []
        labels = []
        for config_name in config_names:
            config_results = [r for r in self.results if r.config_name == config_name]
            perfs.append([r.avg_performance for r in config_results])
            labels.append(config_name.replace('_', '\n'))
        
        ax.boxplot(perfs, labels=labels)
        ax.set_title('Average Performance (avg100)')
        ax.set_ylabel('Performance')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Stability Score
        ax = axes[0, 1]
        stabs = []
        for config_name in config_names:
            config_results = [r for r in self.results if r.config_name == config_name]
            stabs.append([r.stability_score for r in config_results])
        
        ax.boxplot(stabs, labels=labels)
        ax.set_title('Stability Score')
        ax.set_ylabel('Stability')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Convergence Episode
        ax = axes[1, 0]
        convs = []
        for config_name in config_names:
            config_results = [r for r in self.results if r.config_name == config_name]
            convs.append([r.convergence_episode for r in config_results])
        
        ax.boxplot(convs, labels=labels)
        ax.set_title('Convergence Speed (episodes)')
        ax.set_ylabel('Episodes')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary Bar Chart
        ax = axes[1, 1]
        x = np.arange(len(config_names))
        width = 0.25
        
        avg_perfs = [np.mean([r.avg_performance for r in self.results if r.config_name == cn]) for cn in config_names]
        avg_stabs = [np.mean([r.stability_score for r in self.results if r.config_name == cn]) * 100 for cn in config_names]
        
        ax.bar(x - width/2, avg_perfs, width, label='Avg100')
        ax.bar(x + width/2, avg_stabs, width, label='Stability x100')
        
        ax.set_title('Summary Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'benchmark_comparison.png')
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Plots saved to {plot_path}")
        
        plt.close()


if __name__ == "__main__":
    # Create and run benchmark
    benchmark = Phase7Benchmark(
        env_name='CartPole-v1',
        save_dir='phase7_benchmark_results'
    )
    
    benchmark.run_all_benchmarks(verbose=True)
    
    print("\n✅ Benchmark suite complete!")
    print(f"   Results directory: {benchmark.save_dir}")

