"""
Adaptive Configuration Manager (ACM)
======================================

Dynamische Anpassung von System-Gewichtungen und Koordination zwischen
den 4 Ebenen des Emotion-Systems (Reaktiv, Vorausschauend, Reflektierend, Prädiktiv).

Features:
- Automatische Gewichtungs-Anpassung basierend auf Performance
- Konflikt-Erkennung zwischen Ebenen
- Load-Balancing zwischen reaktiven und prädiktiven Ebenen
- Emergency-Fallback bei instabilen Konfigurationen
- Adaptive Exploration/Exploitation-Balance

Author: Phase 7.0 Implementation
Date: 2025-10-16
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum


class SystemState(Enum):
    """Current system state classification"""
    STABLE = "stable"
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    UNSTABLE = "unstable"
    EMERGENCY = "emergency"


@dataclass
class LayerWeights:
    """Weights for the 4-layer emotion system"""
    reactivity: float = 1.0      # Layer 1: EmotionEngine
    anticipation: float = 1.0    # Layer 2: ZoneTransitionEngine + AZPv2
    reflection: float = 1.0      # Layer 3: MetaOptimizer
    prediction: float = 1.0      # Layer 4: EPRU + MOO
    
    def normalize(self) -> 'LayerWeights':
        """Normalize weights to sum to 4.0 (average = 1.0)"""
        total = self.reactivity + self.anticipation + self.reflection + self.prediction
        if total > 0:
            factor = 4.0 / total
            return LayerWeights(
                reactivity=self.reactivity * factor,
                anticipation=self.anticipation * factor,
                reflection=self.reflection * factor,
                prediction=self.prediction * factor
            )
        return self
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'reactivity': self.reactivity,
            'anticipation': self.anticipation,
            'reflection': self.reflection,
            'prediction': self.prediction
        }


@dataclass
class ConflictDetection:
    """Information about detected conflicts between layers"""
    conflict_detected: bool
    conflicting_layers: List[str]
    conflict_severity: float  # 0-1
    resolution_action: str


class AdaptiveConfigurationManager:
    """
    Manages adaptive configuration of the 4-layer emotion system.
    
    Monitors performance and automatically adjusts layer weights,
    detects conflicts, and provides emergency fallback mechanisms.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.05,
        conflict_threshold: float = 0.7,
        stability_window: int = 50,
        emergency_threshold: float = 0.3
    ):
        """
        Initialize manager.
        
        Args:
            adaptation_rate: Rate of weight adaptation (0-1)
            conflict_threshold: Threshold for conflict detection
            stability_window: Episodes to consider for stability
            emergency_threshold: Performance drop threshold for emergency mode
        """
        self.adaptation_rate = adaptation_rate
        self.conflict_threshold = conflict_threshold
        self.stability_window = stability_window
        self.emergency_threshold = emergency_threshold
        
        # Current configuration
        self.layer_weights = LayerWeights()
        self.system_state = SystemState.STABLE
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=stability_window)
        self.td_error_history: deque = deque(maxlen=stability_window)
        self.emotion_history: deque = deque(maxlen=stability_window)
        
        # Layer contribution tracking
        self.layer_contributions = {
            'reactivity': deque(maxlen=stability_window),
            'anticipation': deque(maxlen=stability_window),
            'reflection': deque(maxlen=stability_window),
            'prediction': deque(maxlen=stability_window)
        }
        
        # Conflict tracking
        self.conflict_history: List[ConflictDetection] = []
        self.emergency_activations = 0
        
        # Baseline for comparison
        self.baseline_performance = None
        self.best_weights = LayerWeights()
        self.best_performance = -np.inf
    
    def update(
        self,
        episode: int,
        performance: float,
        td_error: float,
        emotion: float,
        layer_activities: Optional[Dict[str, float]] = None
    ):
        """
        Update manager with new episode data.
        
        Args:
            episode: Episode number
            performance: Episode performance (reward)
            td_error: TD error
            emotion: Emotion value
            layer_activities: Dict of layer activity metrics
        """
        # Update histories
        self.performance_history.append(performance)
        self.td_error_history.append(td_error)
        self.emotion_history.append(emotion)
        
        # Update layer contributions
        if layer_activities:
            for layer, activity in layer_activities.items():
                if layer in self.layer_contributions:
                    self.layer_contributions[layer].append(activity)
        
        # Set baseline if not set
        if self.baseline_performance is None and len(self.performance_history) >= 10:
            self.baseline_performance = np.mean(self.performance_history)
        
        # Update best
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_weights = LayerWeights(**self.layer_weights.to_dict())
        
        # Analyze and adapt
        if len(self.performance_history) >= 20:  # Need minimum data
            self._analyze_system_state()
            self._detect_conflicts()
            self._adapt_weights()
    
    def _analyze_system_state(self):
        """Analyze current system state and update classification"""
        recent_perf = list(self.performance_history)[-20:]
        recent_td = list(self.td_error_history)[-20:]
        
        # Performance metrics
        mean_perf = np.mean(recent_perf)
        std_perf = np.std(recent_perf)
        cv_perf = std_perf / (abs(mean_perf) + 1e-8)
        
        # TD error metrics
        mean_td = np.mean(recent_td)
        
        # Determine state
        if self.baseline_performance is not None:
            perf_ratio = mean_perf / (self.baseline_performance + 1e-8)
            
            # Emergency: severe performance drop
            if perf_ratio < self.emergency_threshold:
                self.system_state = SystemState.EMERGENCY
                self.emergency_activations += 1
                return
        
        # Unstable: high variance
        if cv_perf > 0.5 or mean_td > 2.0:
            self.system_state = SystemState.UNSTABLE
            return
        
        # Exploring: increasing TD error, moderate performance
        if len(recent_td) >= 10:
            td_trend = np.polyfit(range(len(recent_td)), recent_td, 1)[0]
            if td_trend > 0.01:
                self.system_state = SystemState.EXPLORING
                return
        
        # Exploiting: low TD error, stable performance
        if mean_td < 0.5 and cv_perf < 0.2:
            self.system_state = SystemState.EXPLOITING
            return
        
        # Default: stable
        self.system_state = SystemState.STABLE
    
    def _detect_conflicts(self) -> ConflictDetection:
        """
        Detect conflicts between layers.
        
        Returns:
            ConflictDetection object
        """
        conflicts = []
        max_severity = 0.0
        
        # Check if we have enough data
        min_data = min(len(contrib) for contrib in self.layer_contributions.values())
        if min_data < 10:
            return ConflictDetection(
                conflict_detected=False,
                conflicting_layers=[],
                conflict_severity=0.0,
                resolution_action="none"
            )
        
        # Check for opposing trends between layers
        layers = list(self.layer_contributions.keys())
        
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                layer1 = layers[i]
                layer2 = layers[j]
                
                contrib1 = list(self.layer_contributions[layer1])[-10:]
                contrib2 = list(self.layer_contributions[layer2])[-10:]
                
                # Compute trends
                trend1 = np.polyfit(range(len(contrib1)), contrib1, 1)[0]
                trend2 = np.polyfit(range(len(contrib2)), contrib2, 1)[0]
                
                # Check for opposite trends
                if trend1 * trend2 < 0:  # Opposite signs
                    # Compute correlation (negative = conflict)
                    correlation = np.corrcoef(contrib1, contrib2)[0, 1]
                    
                    if correlation < -self.conflict_threshold:
                        conflicts.append(f"{layer1}-{layer2}")
                        severity = abs(correlation)
                        max_severity = max(max_severity, severity)
        
        # Determine resolution action
        if conflicts:
            if max_severity > 0.9:
                resolution = "emergency_rebalance"
            elif max_severity > 0.8:
                resolution = "strong_adaptation"
            else:
                resolution = "gentle_adaptation"
        else:
            resolution = "none"
        
        conflict_detection = ConflictDetection(
            conflict_detected=len(conflicts) > 0,
            conflicting_layers=conflicts,
            conflict_severity=max_severity,
            resolution_action=resolution
        )
        
        # Store in history
        if len(conflicts) > 0:
            self.conflict_history.append(conflict_detection)
        
        return conflict_detection
    
    def _adapt_weights(self):
        """Adapt layer weights based on current system state"""
        conflict = self._detect_conflicts()
        
        # Emergency mode: revert to best known weights
        if self.system_state == SystemState.EMERGENCY:
            self.layer_weights = LayerWeights(**self.best_weights.to_dict())
            return
        
        # Conflict resolution
        if conflict.conflict_detected:
            if conflict.resolution_action == "emergency_rebalance":
                # Reset to balanced weights
                self.layer_weights = LayerWeights()
                return
            elif conflict.resolution_action == "strong_adaptation":
                adapt_rate = self.adaptation_rate * 2.0
            else:
                adapt_rate = self.adaptation_rate
        else:
            adapt_rate = self.adaptation_rate
        
        # State-based adaptation
        if self.system_state == SystemState.EXPLORING:
            # Boost reactive and anticipation layers
            self.layer_weights.reactivity += adapt_rate * 0.5
            self.layer_weights.anticipation += adapt_rate * 0.5
            self.layer_weights.reflection -= adapt_rate * 0.3
            self.layer_weights.prediction -= adapt_rate * 0.2
            
        elif self.system_state == SystemState.EXPLOITING:
            # Boost reflection and prediction layers
            self.layer_weights.reactivity -= adapt_rate * 0.2
            self.layer_weights.anticipation -= adapt_rate * 0.2
            self.layer_weights.reflection += adapt_rate * 0.4
            self.layer_weights.prediction += adapt_rate * 0.4
            
        elif self.system_state == SystemState.UNSTABLE:
            # Reduce all weights slightly, boost stability
            self.layer_weights.reactivity *= (1.0 - adapt_rate * 0.5)
            self.layer_weights.anticipation *= (1.0 - adapt_rate * 0.3)
            self.layer_weights.reflection += adapt_rate * 0.3
            self.layer_weights.prediction -= adapt_rate * 0.2
        
        # Ensure weights stay in valid range [0.2, 2.0]
        for attr in ['reactivity', 'anticipation', 'reflection', 'prediction']:
            val = getattr(self.layer_weights, attr)
            setattr(self.layer_weights, attr, np.clip(val, 0.2, 2.0))
        
        # Normalize
        self.layer_weights = self.layer_weights.normalize()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current layer weights"""
        return self.layer_weights.to_dict()
    
    def get_system_state(self) -> str:
        """Get current system state"""
        return self.system_state.value
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics"""
        return {
            'system_state': self.system_state.value,
            'layer_weights': self.layer_weights.to_dict(),
            'best_weights': self.best_weights.to_dict(),
            'best_performance': self.best_performance,
            'emergency_activations': self.emergency_activations,
            'total_conflicts': len(self.conflict_history),
            'recent_performance_mean': np.mean(self.performance_history) if self.performance_history else 0.0,
            'recent_performance_std': np.std(self.performance_history) if self.performance_history else 0.0
        }
    
    def get_report(self) -> str:
        """Generate human-readable report"""
        stats = self.get_statistics()
        weights = self.layer_weights.to_dict()
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║      Adaptive Configuration Manager Report          ║
╚══════════════════════════════════════════════════════╝

🎛️  System State: {stats['system_state'].upper()}

📊 Current Layer Weights:
   - Reactivity     (L1): {weights['reactivity']:.3f}
   - Anticipation   (L2): {weights['anticipation']:.3f}
   - Reflection     (L3): {weights['reflection']:.3f}
   - Prediction     (L4): {weights['prediction']:.3f}

🏆 Best Configuration:
   - Best Performance:    {stats['best_performance']:.2f}
   - Best Weights:        R={self.best_weights.reactivity:.2f}, A={self.best_weights.anticipation:.2f}, 
                          Ref={self.best_weights.reflection:.2f}, P={self.best_weights.prediction:.2f}

⚠️  System Health:
   - Emergency Activations: {stats['emergency_activations']}
   - Total Conflicts:       {stats['total_conflicts']}
   - Recent Perf Mean:      {stats['recent_performance_mean']:.2f}
   - Recent Perf Std:       {stats['recent_performance_std']:.2f}

{'═' * 54}
"""
        return report
    
    def save_configuration(self, filepath: str):
        """Save current configuration to file"""
        import json
        config = {
            'layer_weights': self.layer_weights.to_dict(),
            'best_weights': self.best_weights.to_dict(),
            'best_performance': self.best_performance,
            'statistics': self.get_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_configuration(self, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        weights_dict = config['layer_weights']
        self.layer_weights = LayerWeights(**weights_dict)
        
        best_dict = config['best_weights']
        self.best_weights = LayerWeights(**best_dict)
        
        self.best_performance = config['best_performance']


# Example usage
if __name__ == "__main__":
    # Create manager
    manager = AdaptiveConfigurationManager(
        adaptation_rate=0.05,
        stability_window=50
    )
    
    print("=== Adaptive Configuration Manager Demo ===\n")
    
    # Simulate training episodes
    np.random.seed(42)
    
    for episode in range(200):
        # Simulate metrics
        performance = 20.0 + episode * 0.1 + np.random.normal(0, 5.0)
        td_error = 1.0 - episode * 0.003 + np.random.normal(0, 0.1)
        emotion = 0.4 + np.random.normal(0, 0.05)
        
        # Simulate layer activities
        layer_activities = {
            'reactivity': np.random.uniform(0.5, 1.5),
            'anticipation': np.random.uniform(0.5, 1.5),
            'reflection': np.random.uniform(0.5, 1.5),
            'prediction': np.random.uniform(0.5, 1.5)
        }
        
        # Update manager
        manager.update(episode, performance, td_error, emotion, layer_activities)
        
        # Print status every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"\n📍 Episode {episode + 1}")
            print(f"   State: {manager.get_system_state()}")
            weights = manager.get_current_weights()
            print(f"   Weights: R={weights['reactivity']:.2f}, A={weights['anticipation']:.2f}, "
                  f"Ref={weights['reflection']:.2f}, P={weights['prediction']:.2f}")
    
    # Final report
    print(manager.get_report())


