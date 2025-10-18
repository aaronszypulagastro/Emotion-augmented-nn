"""
Bayesian Hyperparameter Optimizer (BHO)
========================================

Automatische Hyperparameter-Optimierung für das Emotion-Augmented DQN System
mittels Bayesian Optimization.

Features:
- Gaussian Process-basierte Surrogate-Modelle
- Expected Improvement Acquisition Function
- Multi-dimensional Parameter-Optimierung
- Integration mit Performance-Tracking

Author: Phase 7.0 Implementation
Date: 2025-10-16
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass, asdict
from scipy.stats import norm
from scipy.optimize import minimize


@dataclass
class HyperparameterSpace:
    """Definition des Hyperparameter-Suchraums"""
    # η-Steuerung
    eta_min: Tuple[float, float] = (0.0001, 0.01)
    eta_max: Tuple[float, float] = (0.1, 1.0)
    eta_decay_rate: Tuple[float, float] = (0.9, 0.9999)
    
    # EPRU-Parameter
    epru_confidence_threshold: Tuple[float, float] = (0.5, 0.9)
    epru_intervention_strength: Tuple[float, float] = (0.01, 0.1)
    
    # Gain-Faktoren
    gain_reactivity: Tuple[float, float] = (0.1, 1.0)
    gain_anticipation: Tuple[float, float] = (0.1, 1.0)
    gain_reflection: Tuple[float, float] = (0.01, 0.5)
    
    # AZPv2-Parameter
    azpv2_zone_intensity_scaling: Tuple[float, float] = (0.5, 2.0)
    
    # ECL-Parameter
    ecl_difficulty_adaptation_rate: Tuple[float, float] = (0.01, 0.1)
    
    # MOO-Gewichtungen (müssen zu 1.0 summieren)
    moo_performance_weight: Tuple[float, float] = (0.2, 0.6)
    moo_stability_weight: Tuple[float, float] = (0.1, 0.4)
    moo_health_weight: Tuple[float, float] = (0.1, 0.4)
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Returns list of (min, max) bounds for all parameters"""
        return [
            self.eta_min, self.eta_max, self.eta_decay_rate,
            self.epru_confidence_threshold, self.epru_intervention_strength,
            self.gain_reactivity, self.gain_anticipation, self.gain_reflection,
            self.azpv2_zone_intensity_scaling,
            self.ecl_difficulty_adaptation_rate,
            self.moo_performance_weight, self.moo_stability_weight, self.moo_health_weight
        ]
    
    def get_param_names(self) -> List[str]:
        """Returns list of parameter names in order"""
        return [
            'eta_min', 'eta_max', 'eta_decay_rate',
            'epru_confidence_threshold', 'epru_intervention_strength',
            'gain_reactivity', 'gain_anticipation', 'gain_reflection',
            'azpv2_zone_intensity_scaling',
            'ecl_difficulty_adaptation_rate',
            'moo_performance_weight', 'moo_stability_weight', 'moo_health_weight'
        ]
    
    def normalize_weights(self, params: np.ndarray) -> np.ndarray:
        """Normalize MOO weights to sum to 1.0"""
        params = params.copy()
        # Last 3 params are MOO weights
        weight_sum = params[-3:].sum()
        if weight_sum > 0:
            params[-3:] /= weight_sum
        return params


@dataclass
class OptimizationResult:
    """Result of a single optimization iteration"""
    iteration: int
    parameters: Dict[str, float]
    performance: float
    td_error: float
    emotion_stability: float
    acquisition_value: float
    timestamp: float


class GaussianProcessSurrogate:
    """
    Simplified Gaussian Process for Bayesian Optimization.
    Uses RBF kernel for modeling the objective function.
    """
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = []
        self.y_train = []
        self.K_inv = None
        
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Radial Basis Function (RBF) kernel"""
        # Compute squared Euclidean distances
        dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                np.sum(X2**2, axis=1) - \
                2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * dists / self.length_scale**2)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP model to training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Compute kernel matrix with noise
        K = self.rbf_kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(self.X_train))
        
        # Compute inverse for predictions
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Add more noise if singular
            K += 1e-4 * np.eye(len(self.X_train))
            self.K_inv = np.linalg.inv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at new points.
        Returns: (mean, std)
        """
        if len(self.X_train) == 0:
            return np.zeros(len(X)), np.ones(len(X))
        
        X = np.array(X)
        
        # Compute kernel between test and training points
        K_star = self.rbf_kernel(X, self.X_train)
        
        # Compute mean
        mean = K_star @ self.K_inv @ self.y_train
        
        # Compute variance
        K_star_star = self.rbf_kernel(X, X)
        var = K_star_star - K_star @ self.K_inv @ K_star.T
        
        # Extract diagonal and ensure positive
        std = np.sqrt(np.maximum(np.diag(var), 1e-6))
        
        return mean, std


class BayesianHyperparameterOptimizer:
    """
    Main Bayesian Optimization class for hyperparameter tuning.
    
    Uses Gaussian Process as surrogate model and Expected Improvement
    as acquisition function.
    """
    
    def __init__(
        self,
        search_space: Optional[HyperparameterSpace] = None,
        n_initial_points: int = 5,
        xi: float = 0.01,  # Exploration parameter
        random_state: Optional[int] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            search_space: Hyperparameter space definition
            n_initial_points: Number of random initial samples
            xi: Exploration parameter for EI acquisition
            random_state: Random seed for reproducibility
        """
        self.search_space = search_space or HyperparameterSpace()
        self.n_initial_points = n_initial_points
        self.xi = xi
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Optimization state
        self.iteration = 0
        self.results: List[OptimizationResult] = []
        self.gp = GaussianProcessSurrogate()
        
        # Extract bounds and names
        self.bounds = self.search_space.get_bounds()
        self.param_names = self.search_space.get_param_names()
        self.dim = len(self.bounds)
        
        # Best result tracking
        self.best_params = None
        self.best_performance = -np.inf
    
    def _sample_random_params(self) -> np.ndarray:
        """Sample random parameters from search space"""
        params = np.array([
            np.random.uniform(low, high)
            for low, high in self.bounds
        ])
        return self.search_space.normalize_weights(params)
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary"""
        return {name: float(val) for name, val in zip(self.param_names, params)}
    
    def _dict_to_params(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array"""
        return np.array([param_dict[name] for name in self.param_names])
    
    def expected_improvement(
        self,
        X: np.ndarray,
        best_y: float
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        Args:
            X: Points to evaluate (n_points, n_dims)
            best_y: Best observed value so far
            
        Returns:
            EI values for each point
        """
        mean, std = self.gp.predict(X)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-8)
        
        # Compute EI
        z = (mean - best_y - self.xi) / std
        ei = (mean - best_y - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
    
    def suggest_next_params(self) -> Dict[str, float]:
        """
        Suggest next hyperparameters to evaluate.
        
        Returns:
            Dictionary of parameter names to values
        """
        self.iteration += 1
        
        # Initial random exploration
        if len(self.results) < self.n_initial_points:
            params = self._sample_random_params()
            return self._params_to_dict(params)
        
        # Use Bayesian optimization
        best_y = max(r.performance for r in self.results)
        
        # Optimize acquisition function
        best_ei = -np.inf
        best_params = None
        
        # Multi-start optimization
        n_starts = 10
        for _ in range(n_starts):
            # Random starting point
            x0 = self._sample_random_params()
            
            # Minimize negative EI
            def neg_ei(x):
                x_norm = self.search_space.normalize_weights(x)
                return -self.expected_improvement(x_norm.reshape(1, -1), best_y)[0]
            
            # Optimize
            result = minimize(
                neg_ei,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds
            )
            
            if -result.fun > best_ei:
                best_ei = -result.fun
                best_params = result.x
        
        # Normalize and return
        if best_params is not None:
            best_params = self.search_space.normalize_weights(best_params)
            return self._params_to_dict(best_params)
        else:
            # Fallback to random if optimization fails
            params = self._sample_random_params()
            return self._params_to_dict(params)
    
    def register_result(
        self,
        params: Dict[str, float],
        performance: float,
        td_error: float = 0.0,
        emotion_stability: float = 0.0
    ):
        """
        Register the result of a training run.
        
        Args:
            params: Parameter configuration used
            performance: avg100 performance achieved
            td_error: Final TD error
            emotion_stability: Emotion stability metric
        """
        import time
        
        # Convert params to array
        param_array = self._dict_to_params(params)
        
        # Compute acquisition value (0 for initial points)
        if len(self.results) >= self.n_initial_points:
            best_y = max(r.performance for r in self.results)
            acq_val = self.expected_improvement(param_array.reshape(1, -1), best_y)[0]
        else:
            acq_val = 0.0
        
        # Create result
        result = OptimizationResult(
            iteration=self.iteration,
            parameters=params,
            performance=performance,
            td_error=td_error,
            emotion_stability=emotion_stability,
            acquisition_value=float(acq_val),
            timestamp=time.time()
        )
        
        self.results.append(result)
        
        # Update best
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_params = params.copy()
        
        # Update GP model
        X = np.array([self._dict_to_params(r.parameters) for r in self.results])
        y = np.array([r.performance for r in self.results])
        self.gp.fit(X, y)
    
    def get_best_params(self) -> Optional[Dict[str, float]]:
        """Get the best parameters found so far"""
        return self.best_params
    
    def get_optimization_history(self) -> List[Dict]:
        """Get full optimization history as list of dicts"""
        return [asdict(r) for r in self.results]
    
    def save_state(self, filepath: str):
        """Save optimization state to file"""
        state = {
            'iteration': self.iteration,
            'best_params': self.best_params,
            'best_performance': self.best_performance,
            'results': self.get_optimization_history()
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load optimization state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.iteration = state['iteration']
        self.best_params = state['best_params']
        self.best_performance = state['best_performance']
        
        # Reconstruct results
        self.results = [
            OptimizationResult(**r) for r in state['results']
        ]
        
        # Rebuild GP
        if self.results:
            X = np.array([self._dict_to_params(r.parameters) for r in self.results])
            y = np.array([r.performance for r in self.results])
            self.gp.fit(X, y)


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = BayesianHyperparameterOptimizer(
        n_initial_points=5,
        random_state=42
    )
    
    # Simulate optimization loop
    print("=== Bayesian Hyperparameter Optimization ===\n")
    
    for i in range(10):
        # Get next parameters to try
        params = optimizer.suggest_next_params()
        
        print(f"Iteration {i+1}:")
        print(f"  Suggested params: {params}")
        
        # Simulate training (in reality, this would run actual training)
        # For demo, use a noisy quadratic function
        performance = np.random.normal(
            -sum((params[k] - 0.5)**2 for k in params),
            0.1
        )
        
        # Register result
        optimizer.register_result(
            params=params,
            performance=performance,
            td_error=np.random.uniform(0.8, 1.0),
            emotion_stability=np.random.uniform(0.3, 0.5)
        )
        
        print(f"  Performance: {performance:.3f}")
        print(f"  Best so far: {optimizer.best_performance:.3f}\n")
    
    # Print best configuration
    print("\n=== Best Configuration ===")
    print(f"Performance: {optimizer.best_performance:.3f}")
    print(f"Parameters: {optimizer.get_best_params()}")


