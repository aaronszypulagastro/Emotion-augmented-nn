"""
Performance Stability Analyzer (PSA)
=====================================

Analysiert und tracked die Stabilität und Variabilität der Performance
über Training-Runs hinweg.

Features:
- Varianz- und Stabilitäts-Metriken
- Trend-Erkennung (aufsteigend/absteigend/stabil)
- Konfidenzintervalle für Performance-Vorhersagen
- Anomalie-Erkennung für instabile Episoden
- Rolling-Window-Statistiken

Author: Phase 7.0 Implementation
Date: 2025-10-16
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from scipy import stats


@dataclass
class StabilityMetrics:
    """Container for stability analysis results"""
    stability_score: float  # 0-1, higher is more stable
    performance_variance: float
    trend: str  # 'ascending', 'descending', 'stable'
    trend_strength: float  # 0-1, how strong the trend is
    confidence_lower: float  # Lower bound of 95% CI
    confidence_upper: float  # Upper bound of 95% CI
    anomaly_count: int  # Number of anomalous episodes
    coefficient_of_variation: float  # Normalized variability


class PerformanceStabilityAnalyzer:
    """
    Analyzes performance stability across training episodes.
    
    Provides metrics for:
    - Overall stability (low variance = high stability)
    - Trend detection (learning progress)
    - Confidence intervals
    - Anomaly detection
    """
    
    def __init__(
        self,
        window_size: int = 100,
        anomaly_threshold: float = 3.0,  # Z-score threshold
        trend_threshold: float = 0.3,  # Minimum slope for trend detection
        stability_decay: float = 0.99  # EMA decay for online stability
    ):
        """
        Initialize analyzer.
        
        Args:
            window_size: Window size for rolling statistics
            anomaly_threshold: Z-score threshold for anomaly detection
            trend_threshold: Minimum normalized slope for trend detection
            stability_decay: Decay factor for exponential moving average
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.trend_threshold = trend_threshold
        self.stability_decay = stability_decay
        
        # Storage for performance history
        self.performance_history: deque = deque(maxlen=10000)
        self.episode_rewards: deque = deque(maxlen=window_size)
        
        # Online statistics
        self.ema_performance = None
        self.ema_variance = None
        
        # Anomaly tracking
        self.anomaly_episodes: List[int] = []
    
    def update(self, episode: int, performance: float):
        """
        Update analyzer with new episode performance.
        
        Args:
            episode: Episode number
            performance: Performance metric (e.g., episode reward)
        """
        self.performance_history.append((episode, performance))
        self.episode_rewards.append(performance)
        
        # Update EMA
        if self.ema_performance is None:
            self.ema_performance = performance
            self.ema_variance = 0.0
        else:
            # Update EMA of performance
            delta = performance - self.ema_performance
            self.ema_performance += (1 - self.stability_decay) * delta
            
            # Update EMA of variance (Welford's algorithm)
            self.ema_variance = self.stability_decay * self.ema_variance + \
                               (1 - self.stability_decay) * delta**2
        
        # Check for anomaly
        if len(self.episode_rewards) >= 10:  # Need minimum data
            mean = np.mean(self.episode_rewards)
            std = np.std(self.episode_rewards)
            if std > 0:
                z_score = abs((performance - mean) / std)
                if z_score > self.anomaly_threshold:
                    self.anomaly_episodes.append(episode)
    
    def compute_stability_metrics(
        self,
        window: Optional[int] = None
    ) -> StabilityMetrics:
        """
        Compute comprehensive stability metrics.
        
        Args:
            window: Number of recent episodes to analyze (None = all)
            
        Returns:
            StabilityMetrics object with analysis results
        """
        if len(self.performance_history) < 2:
            # Not enough data
            return StabilityMetrics(
                stability_score=0.0,
                performance_variance=0.0,
                trend='stable',
                trend_strength=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                anomaly_count=0,
                coefficient_of_variation=0.0
            )
        
        # Get data window
        if window is None:
            data = list(self.performance_history)
        else:
            data = list(self.performance_history)[-window:]
        
        episodes = np.array([x[0] for x in data])
        performances = np.array([x[1] for x in data])
        
        # Basic statistics
        mean_perf = np.mean(performances)
        var_perf = np.var(performances)
        std_perf = np.std(performances)
        
        # Stability score (inverse of coefficient of variation)
        # CV = std / mean, stability = 1 / (1 + CV)
        if mean_perf != 0:
            cv = std_perf / abs(mean_perf)
            stability_score = 1.0 / (1.0 + cv)
        else:
            cv = 0.0
            stability_score = 0.0
        
        # Trend analysis using linear regression
        if len(performances) >= 3:
            # Normalize episode numbers to [0, 1]
            x_norm = (episodes - episodes.min()) / (episodes.max() - episodes.min() + 1e-8)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_norm, performances)
            
            # Normalize slope by mean performance
            normalized_slope = slope / (abs(mean_perf) + 1e-8)
            
            # Determine trend
            if abs(normalized_slope) < self.trend_threshold:
                trend = 'stable'
                trend_strength = 0.0
            elif normalized_slope > 0:
                trend = 'ascending'
                trend_strength = min(abs(r_value), 1.0)  # Use R² as strength
            else:
                trend = 'descending'
                trend_strength = min(abs(r_value), 1.0)
        else:
            trend = 'stable'
            trend_strength = 0.0
        
        # Confidence intervals (95%)
        if len(performances) >= 2:
            confidence_interval = stats.t.interval(
                0.95,
                len(performances) - 1,
                loc=mean_perf,
                scale=stats.sem(performances)
            )
            confidence_lower = confidence_interval[0]
            confidence_upper = confidence_interval[1]
        else:
            confidence_lower = mean_perf
            confidence_upper = mean_perf
        
        # Anomaly count (recent window)
        recent_anomalies = sum(
            1 for ep in self.anomaly_episodes
            if ep >= episodes.min()
        )
        
        return StabilityMetrics(
            stability_score=float(stability_score),
            performance_variance=float(var_perf),
            trend=trend,
            trend_strength=float(trend_strength),
            confidence_lower=float(confidence_lower),
            confidence_upper=float(confidence_upper),
            anomaly_count=recent_anomalies,
            coefficient_of_variation=float(cv)
        )
    
    def get_rolling_statistics(
        self,
        window: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute rolling statistics over the performance history.
        
        Args:
            window: Window size for rolling computation
            
        Returns:
            Dictionary with arrays of rolling mean, std, min, max
        """
        if len(self.performance_history) < window:
            return {}
        
        performances = np.array([x[1] for x in self.performance_history])
        
        # Compute rolling statistics
        rolling_mean = np.convolve(
            performances,
            np.ones(window) / window,
            mode='valid'
        )
        
        # Rolling std (more complex)
        rolling_std = []
        for i in range(len(performances) - window + 1):
            rolling_std.append(np.std(performances[i:i+window]))
        rolling_std = np.array(rolling_std)
        
        # Rolling min/max
        rolling_min = []
        rolling_max = []
        for i in range(len(performances) - window + 1):
            rolling_min.append(np.min(performances[i:i+window]))
            rolling_max.append(np.max(performances[i:i+window]))
        rolling_min = np.array(rolling_min)
        rolling_max = np.array(rolling_max)
        
        return {
            'mean': rolling_mean,
            'std': rolling_std,
            'min': rolling_min,
            'max': rolling_max
        }
    
    def detect_regime_changes(
        self,
        window: int = 50,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Detect episodes where performance regime changed significantly.
        
        Args:
            window: Window size for regime comparison
            threshold: Relative change threshold for detection
            
        Returns:
            List of episode numbers where regime change detected
        """
        if len(self.performance_history) < 2 * window:
            return []
        
        performances = np.array([x[1] for x in self.performance_history])
        episodes = np.array([x[0] for x in self.performance_history])
        
        regime_changes = []
        
        for i in range(window, len(performances) - window):
            # Compare mean before and after
            mean_before = np.mean(performances[i-window:i])
            mean_after = np.mean(performances[i:i+window])
            
            # Relative change
            if mean_before != 0:
                rel_change = abs(mean_after - mean_before) / abs(mean_before)
                if rel_change > threshold:
                    regime_changes.append(int(episodes[i]))
        
        return regime_changes
    
    def get_stability_report(self) -> str:
        """
        Generate a human-readable stability report.
        
        Returns:
            Formatted string with stability analysis
        """
        metrics = self.compute_stability_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║        Performance Stability Analysis Report        ║
╚══════════════════════════════════════════════════════╝

📊 Overall Stability:
   - Stability Score:  {metrics.stability_score:.3f} / 1.0
   - Performance Var:  {metrics.performance_variance:.3f}
   - Coeff. of Var:    {metrics.coefficient_of_variation:.3f}

📈 Trend Analysis:
   - Trend:            {metrics.trend.upper()}
   - Trend Strength:   {metrics.trend_strength:.3f}

🎯 Confidence Interval (95%):
   - Lower Bound:      {metrics.confidence_lower:.2f}
   - Upper Bound:      {metrics.confidence_upper:.2f}
   - Range:            {metrics.confidence_upper - metrics.confidence_lower:.2f}

⚠️  Anomalies:
   - Anomaly Count:    {metrics.anomaly_count}
   - Anomaly Rate:     {metrics.anomaly_count / max(len(self.performance_history), 1) * 100:.1f}%

📉 Historical Data:
   - Total Episodes:   {len(self.performance_history)}
   - Recent Mean:      {np.mean(list(self.episode_rewards)) if self.episode_rewards else 0:.2f}
   - Recent Std:       {np.std(list(self.episode_rewards)) if self.episode_rewards else 0:.2f}

{'═' * 54}
"""
        return report
    
    def compare_runs(
        self,
        other_analyzer: 'PerformanceStabilityAnalyzer',
        run_name_1: str = "Run 1",
        run_name_2: str = "Run 2"
    ) -> str:
        """
        Compare stability between two runs.
        
        Args:
            other_analyzer: Another PSA instance to compare with
            run_name_1: Name of first run (this one)
            run_name_2: Name of second run (other)
            
        Returns:
            Formatted comparison report
        """
        metrics1 = self.compute_stability_metrics()
        metrics2 = other_analyzer.compute_stability_metrics()
        
        def winner(val1, val2, higher_better=True):
            if higher_better:
                return "✓" if val1 > val2 else " "
            else:
                return "✓" if val1 < val2 else " "
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║          Performance Stability Comparison            ║
╚══════════════════════════════════════════════════════╝

Metric                    {run_name_1:>12}  {run_name_2:>12}  Winner
─────────────────────────────────────────────────────────
Stability Score           {metrics1.stability_score:>12.3f}  {metrics2.stability_score:>12.3f}  {winner(metrics1.stability_score, metrics2.stability_score)}
Performance Variance      {metrics1.performance_variance:>12.3f}  {metrics2.performance_variance:>12.3f}  {winner(metrics1.performance_variance, metrics2.performance_variance, False)}
Coeff. of Variation       {metrics1.coefficient_of_variation:>12.3f}  {metrics2.coefficient_of_variation:>12.3f}  {winner(metrics1.coefficient_of_variation, metrics2.coefficient_of_variation, False)}
Trend Strength            {metrics1.trend_strength:>12.3f}  {metrics2.trend_strength:>12.3f}  {winner(metrics1.trend_strength, metrics2.trend_strength)}
Anomaly Count             {metrics1.anomaly_count:>12}  {metrics2.anomaly_count:>12}  {winner(metrics1.anomaly_count, metrics2.anomaly_count, False)}
─────────────────────────────────────────────────────────

Trend: {run_name_1} = {metrics1.trend}, {run_name_2} = {metrics2.trend}

{'═' * 57}
"""
        return report


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = PerformanceStabilityAnalyzer(window_size=100)
    
    # Simulate training episodes
    print("=== Simulating Training ===\n")
    
    np.random.seed(42)
    base_performance = 10.0
    
    for episode in range(200):
        # Simulate improving performance with noise
        trend = episode * 0.1
        noise = np.random.normal(0, 2.0)
        
        # Add some anomalies
        if episode in [50, 100, 150]:
            noise += np.random.choice([-10, 10])
        
        performance = base_performance + trend + noise
        
        analyzer.update(episode, performance)
    
    # Print stability report
    print(analyzer.get_stability_report())
    
    # Get metrics
    metrics = analyzer.compute_stability_metrics()
    print(f"\n📊 Detailed Metrics:")
    print(f"   Stability Score: {metrics.stability_score:.3f}")
    print(f"   Trend: {metrics.trend} (strength: {metrics.trend_strength:.3f})")
    print(f"   Confidence Interval: [{metrics.confidence_lower:.2f}, {metrics.confidence_upper:.2f}]")
    
    # Detect regime changes
    changes = analyzer.detect_regime_changes()
    print(f"\n🔄 Regime Changes detected at episodes: {changes}")


