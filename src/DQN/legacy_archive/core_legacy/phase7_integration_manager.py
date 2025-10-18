"""
Phase 7 Integration Manager
============================

Koordiniert alle Phase 7.0 Komponenten:
- Bayesian Hyperparameter Optimizer (BHO)
- Performance Stability Analyzer (PSA)
- Adaptive Configuration Manager (ACM)
- Meta-Performance-Predictor (MPP)

Bietet einfache API für Integration in bestehenden Training-Loop.

Author: Phase 7.0 Implementation
Date: 2025-10-16
"""

import numpy as np
from typing import Dict, Optional, Tuple
import os

from .bayesian_hyperparameter_optimizer import BayesianHyperparameterOptimizer, HyperparameterSpace
from .performance_stability_analyzer import PerformanceStabilityAnalyzer
from .adaptive_configuration_manager import AdaptiveConfigurationManager
from .meta_performance_predictor import MetaPerformancePredictor


class Phase7IntegrationManager:
    """
    Hauptmanager für Phase 7.0 Komponenten.
    
    Koordiniert automatische Hyperparameter-Optimierung,
    Stabilitäts-Analyse und adaptive Konfiguration.
    """
    
    def __init__(
        self,
        enable_bayesian_optimization: bool = True,
        enable_stability_analysis: bool = True,
        enable_adaptive_config: bool = True,
        enable_performance_prediction: bool = True,
        hyperparameter_space: Optional[HyperparameterSpace] = None,
        optimization_interval: int = 100,  # Episodes between optimization
        save_dir: str = "phase7_checkpoints"
    ):
        """
        Initialize Phase 7 manager.
        
        Args:
            enable_bayesian_optimization: Enable BHO
            enable_stability_analysis: Enable PSA
            enable_adaptive_config: Enable ACM
            enable_performance_prediction: Enable MPP
            hyperparameter_space: Custom hyperparameter space
            optimization_interval: Episodes between optimization iterations
            save_dir: Directory for saving checkpoints
        """
        self.enable_bho = enable_bayesian_optimization
        self.enable_psa = enable_stability_analysis
        self.enable_acm = enable_adaptive_config
        self.enable_mpp = enable_performance_prediction
        
        self.optimization_interval = optimization_interval
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        if self.enable_bho:
            self.bho = BayesianHyperparameterOptimizer(
                search_space=hyperparameter_space,
                n_initial_points=5,
                random_state=42
            )
            self.current_hyperparams = self.bho.suggest_next_params()
        else:
            self.bho = None
            self.current_hyperparams = {}
        
        if self.enable_psa:
            self.psa = PerformanceStabilityAnalyzer(
                window_size=100,
                anomaly_threshold=3.0
            )
        else:
            self.psa = None
        
        if self.enable_acm:
            self.acm = AdaptiveConfigurationManager(
                adaptation_rate=0.05,
                stability_window=50
            )
        else:
            self.acm = None
        
        if self.enable_mpp:
            # Input dimension = number of hyperparameters
            input_dim = len(self.current_hyperparams) if self.current_hyperparams else 13
            self.mpp = MetaPerformancePredictor(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                ensemble_size=3
            )
        else:
            self.mpp = None
        
        # Training state
        self.episode_count = 0
        self.current_run_performance = []
        self.optimization_history = []
        
    def start_new_run(self):
        """Start a new training run with suggested hyperparameters"""
        if self.enable_bho:
            self.current_hyperparams = self.bho.suggest_next_params()
        self.current_run_performance = []
        return self.current_hyperparams
    
    def update_episode(
        self,
        episode: int,
        episode_return: float,
        td_error: float,
        emotion: float,
        layer_activities: Optional[Dict[str, float]] = None
    ):
        """
        Update all components with episode results.
        
        Args:
            episode: Episode number
            episode_return: Episode return (reward)
            td_error: TD error
            emotion: Emotion value
            layer_activities: Activity levels of different layers
        """
        self.episode_count = episode
        self.current_run_performance.append(episode_return)
        
        # Update PSA
        if self.enable_psa:
            self.psa.update(episode, episode_return)
        
        # Update ACM
        if self.enable_acm:
            self.acm.update(
                episode,
                episode_return,
                td_error,
                emotion,
                layer_activities
            )
    
    def finish_run(self) -> Dict[str, any]:
        """
        Finish current run and register results with optimization.
        
        Returns:
            Dictionary with run statistics
        """
        if len(self.current_run_performance) == 0:
            return {}
        
        # Compute metrics
        avg_performance = np.mean(self.current_run_performance[-100:])  # avg100
        final_td_error = 0.0  # Will be updated externally
        
        # Get stability metrics
        if self.enable_psa:
            stability_metrics = self.psa.compute_stability_metrics()
            emotion_stability = stability_metrics.stability_score
        else:
            emotion_stability = 0.5
        
        # Register with BHO
        if self.enable_bho:
            self.bho.register_result(
                params=self.current_hyperparams,
                performance=avg_performance,
                td_error=final_td_error,
                emotion_stability=emotion_stability
            )
        
        # Add experience to MPP
        if self.enable_mpp and self.current_hyperparams:
            hyperparam_array = np.array(list(self.current_hyperparams.values()))
            self.mpp.add_experience(
                hyperparam_array,
                avg_performance,
                final_td_error,
                emotion_stability
            )
            
            # Train MPP
            if len(self.mpp.buffer) >= 32:
                self.mpp.train_step(n_epochs=5)
        
        # Save statistics
        run_stats = {
            'avg_performance': avg_performance,
            'stability_score': emotion_stability,
            'hyperparams': self.current_hyperparams.copy(),
            'system_state': self.acm.get_system_state() if self.enable_acm else 'unknown'
        }
        
        self.optimization_history.append(run_stats)
        
        return run_stats
    
    def get_current_hyperparams(self) -> Dict[str, float]:
        """Get current hyperparameters"""
        return self.current_hyperparams.copy()
    
    def get_layer_weights(self) -> Dict[str, float]:
        """Get current adaptive layer weights from ACM"""
        if self.enable_acm:
            return self.acm.get_current_weights()
        return {
            'reactivity': 1.0,
            'anticipation': 1.0,
            'reflection': 1.0,
            'prediction': 1.0
        }
    
    def get_stability_metrics(self) -> Dict[str, any]:
        """Get current stability metrics from PSA"""
        if self.enable_psa:
            metrics = self.psa.compute_stability_metrics()
            return {
                'stability_score': metrics.stability_score,
                'trend': metrics.trend,
                'confidence_lower': metrics.confidence_lower,
                'confidence_upper': metrics.confidence_upper,
                'anomaly_count': metrics.anomaly_count
            }
        return {}
    
    def predict_performance(
        self,
        hyperparams: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Predict performance for given hyperparameters.
        
        Args:
            hyperparams: Hyperparameters to evaluate (None = current)
            
        Returns:
            (predicted_performance, confidence)
        """
        if not self.enable_mpp:
            return 0.0, 0.0
        
        if hyperparams is None:
            hyperparams = self.current_hyperparams
        
        hyperparam_array = np.array(list(hyperparams.values()))
        prediction = self.mpp.predict(hyperparam_array)
        
        return prediction.avg_performance, prediction.confidence
    
    def get_best_hyperparams(self) -> Optional[Dict[str, float]]:
        """Get best hyperparameters found so far"""
        if self.enable_bho:
            return self.bho.get_best_params()
        return None
    
    def save_checkpoint(self, prefix: str = "phase7"):
        """Save all component states"""
        filepath_base = os.path.join(self.save_dir, prefix)
        
        if self.enable_bho:
            self.bho.save_state(f"{filepath_base}_bho.json")
        
        if self.enable_acm:
            self.acm.save_configuration(f"{filepath_base}_acm.json")
        
        if self.enable_mpp:
            self.mpp.save_models(f"{filepath_base}_mpp")
        
        print(f"💾 Phase 7 checkpoint saved to {self.save_dir}")
    
    def load_checkpoint(self, prefix: str = "phase7"):
        """Load all component states"""
        filepath_base = os.path.join(self.save_dir, prefix)
        
        if self.enable_bho and os.path.exists(f"{filepath_base}_bho.json"):
            self.bho.load_state(f"{filepath_base}_bho.json")
        
        if self.enable_acm and os.path.exists(f"{filepath_base}_acm.json"):
            self.acm.load_configuration(f"{filepath_base}_acm.json")
        
        if self.enable_mpp:
            try:
                self.mpp.load_models(f"{filepath_base}_mpp")
            except:
                pass  # Models might not exist yet
        
        print(f"📂 Phase 7 checkpoint loaded from {self.save_dir}")
    
    def get_comprehensive_report(self) -> str:
        """Generate comprehensive report of all components"""
        report = """
╔══════════════════════════════════════════════════════════════╗
║            Phase 7.0 Integration Manager Report              ║
╚══════════════════════════════════════════════════════════════╝

"""
        
        # BHO Report
        if self.enable_bho:
            report += "🎯 Bayesian Hyperparameter Optimization:\n"
            best_params = self.bho.get_best_params()
            if best_params:
                report += f"   Best Performance: {self.bho.best_performance:.2f}\n"
                report += f"   Optimization Iterations: {self.bho.iteration}\n"
                report += f"   Best Config: {list(best_params.keys())[:3]}... (showing first 3)\n"
            else:
                report += "   No optimization data yet.\n"
            report += "\n"
        
        # PSA Report
        if self.enable_psa:
            report += "📊 Performance Stability Analysis:\n"
            metrics = self.psa.compute_stability_metrics()
            report += f"   Stability Score: {metrics.stability_score:.3f}\n"
            report += f"   Trend: {metrics.trend}\n"
            report += f"   Anomaly Count: {metrics.anomaly_count}\n"
            report += "\n"
        
        # ACM Report
        if self.enable_acm:
            report += "⚙️  Adaptive Configuration:\n"
            stats = self.acm.get_statistics()
            report += f"   System State: {stats['system_state']}\n"
            weights = stats['layer_weights']
            report += f"   Layer Weights: R={weights['reactivity']:.2f}, "
            report += f"A={weights['anticipation']:.2f}, "
            report += f"Ref={weights['reflection']:.2f}, "
            report += f"P={weights['prediction']:.2f}\n"
            report += f"   Conflicts Detected: {stats['total_conflicts']}\n"
            report += "\n"
        
        # MPP Report
        if self.enable_mpp:
            report += "🔮 Meta-Performance Prediction:\n"
            stats = self.mpp.get_statistics()
            report += f"   Buffer Size: {stats['buffer_size']}\n"
            report += f"   Training Iterations: {stats['training_iterations']}\n"
            report += f"   Recent Loss: {stats['recent_loss']:.4f}\n"
            report += "\n"
        
        report += "═" * 62 + "\n"
        
        return report
    
    def get_csv_metrics(self) -> Dict[str, any]:
        """
        Get metrics for CSV logging.
        
        Returns:
            Dictionary with all Phase 7 metrics
        """
        metrics = {}
        
        # BHO metrics
        if self.enable_bho:
            metrics['bho_iteration'] = self.bho.iteration
            # Get acquisition value for current params (approximation)
            metrics['bho_best_performance'] = self.bho.best_performance
        else:
            metrics['bho_iteration'] = 0
            metrics['bho_best_performance'] = 0.0
        
        # PSA metrics
        if self.enable_psa:
            stability = self.psa.compute_stability_metrics()
            metrics['psa_stability_score'] = stability.stability_score
            metrics['psa_trend'] = stability.trend
            metrics['psa_anomaly_count'] = stability.anomaly_count
        else:
            metrics['psa_stability_score'] = 0.0
            metrics['psa_trend'] = 'unknown'
            metrics['psa_anomaly_count'] = 0
        
        # ACM metrics
        if self.enable_acm:
            weights = self.acm.get_current_weights()
            metrics['acm_weight_reactivity'] = weights['reactivity']
            metrics['acm_weight_anticipation'] = weights['anticipation']
            metrics['acm_weight_reflection'] = weights['reflection']
            metrics['acm_weight_prediction'] = weights['prediction']
            metrics['acm_system_state'] = self.acm.get_system_state()
        else:
            metrics['acm_weight_reactivity'] = 1.0
            metrics['acm_weight_anticipation'] = 1.0
            metrics['acm_weight_reflection'] = 1.0
            metrics['acm_weight_prediction'] = 1.0
            metrics['acm_system_state'] = 'unknown'
        
        # MPP metrics
        if self.enable_mpp:
            pred_perf, pred_conf = self.predict_performance()
            metrics['mpp_predicted_performance'] = pred_perf
            metrics['mpp_confidence'] = pred_conf
        else:
            metrics['mpp_predicted_performance'] = 0.0
            metrics['mpp_confidence'] = 0.0
        
        return metrics


# Example usage
if __name__ == "__main__":
    print("=== Phase 7 Integration Manager Demo ===\n")
    
    # Create manager
    manager = Phase7IntegrationManager(
        enable_bayesian_optimization=True,
        enable_stability_analysis=True,
        enable_adaptive_config=True,
        enable_performance_prediction=True
    )
    
    # Simulate training runs
    for run in range(3):
        print(f"\n🏃 Run {run + 1}")
        
        # Start new run with suggested hyperparameters
        hyperparams = manager.start_new_run()
        print(f"   Using hyperparameters: {list(hyperparams.keys())[:3]}... (showing first 3)")
        
        # Simulate episodes
        for episode in range(100):
            episode_return = 20.0 + run * 10.0 + episode * 0.1 + np.random.normal(0, 5.0)
            td_error = 1.0 - episode * 0.005 + np.random.normal(0, 0.1)
            emotion = 0.4 + np.random.normal(0, 0.05)
            
            layer_activities = {
                'reactivity': np.random.uniform(0.5, 1.5),
                'anticipation': np.random.uniform(0.5, 1.5),
                'reflection': np.random.uniform(0.5, 1.5),
                'prediction': np.random.uniform(0.5, 1.5)
            }
            
            manager.update_episode(episode, episode_return, td_error, emotion, layer_activities)
        
        # Finish run
        stats = manager.finish_run()
        print(f"   Average Performance: {stats['avg_performance']:.2f}")
        print(f"   System State: {stats['system_state']}")
    
    # Print comprehensive report
    print(manager.get_comprehensive_report())
    
    # Best hyperparameters
    best = manager.get_best_hyperparams()
    if best:
        print(f"\n🏆 Best Hyperparameters Found:")
        for key, value in list(best.items())[:5]:
            print(f"   {key}: {value:.4f}")


