"""
Multi-Objective Optimization (MOO) - Phase 6.3
==============================================

Simultane Optimierung von η (Lernrate), σ (Plastizität) und Performance
für maximale Trainingseffizienz. Das System nutzt Pareto-Optimierung
und adaptive Gewichtung für ausgewogene Zielerreichung.

Key Features:
- Pareto-Optimierung für η, σ und Performance
- Adaptive Zielgewichtung basierend auf Training-Phase
- Multi-Objective Loss-Funktion
- Dynamische Trade-off-Anpassung
- Performance-Prediction für zukünftige Episoden
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MOOConfig:
    """Konfiguration für Multi-Objective Optimization."""
    
    # Zielgewichte (werden adaptiv angepasst)
    w_performance: float = 0.4      # Performance-Gewicht
    w_eta_stability: float = 0.3    # η-Stabilität-Gewicht
    w_sigma_health: float = 0.3     # σ-Gesundheit-Gewicht
    
    # Performance-Metriken
    performance_window: int = 20     # Fenster für Performance-Bewertung
    performance_target: float = 50.0 # Ziel-Performance
    
    # η-Optimierung
    eta_stability_weight: float = 0.1  # Gewicht für η-Stabilität
    eta_efficiency_weight: float = 0.2 # Gewicht für η-Effizienz
    
    # σ-Optimierung
    sigma_health_weight: float = 0.15  # Gewicht für σ-Gesundheit
    sigma_plasticity_weight: float = 0.1 # Gewicht für σ-Plastizität
    
    # Adaptive Gewichtung
    adaptation_rate: float = 0.01   # Rate für Gewichtsanpassung
    adaptation_window: int = 50     # Fenster für Gewichtsanpassung
    
    # Pareto-Optimierung
    pareto_alpha: float = 0.1       # Pareto-Dominanz-Parameter
    pareto_beta: float = 0.9        # Pareto-Diversität-Parameter
    
    # Performance-Prediction
    prediction_horizon: int = 10    # Vorhersagehorizont
    prediction_confidence: float = 0.7 # Mindest-Konfidenz für Vorhersage


class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimization System.
    
    Optimiert gleichzeitig η, σ und Performance für maximale
    Trainingseffizienz unter Berücksichtigung von Trade-offs.
    """
    
    def __init__(self, config: MOOConfig = MOOConfig()):
        """
        Initialisiert das MOO-System.
        
        Args:
            config: MOO-Konfiguration
        """
        self.config = config
        
        # Historie für Multi-Objective-Bewertung
        self.performance_history = deque(maxlen=config.performance_window)
        self.eta_history = deque(maxlen=config.performance_window)
        self.sigma_history = deque(maxlen=config.performance_window)
        self.td_error_history = deque(maxlen=config.performance_window)
        self.emotion_history = deque(maxlen=config.performance_window)
        
        # Pareto-Front für Multi-Objective-Optimierung
        self.pareto_front = []
        self.pareto_solutions = []
        
        # Adaptive Gewichte
        self.current_weights = {
            'performance': config.w_performance,
            'eta_stability': config.w_eta_stability,
            'sigma_health': config.w_sigma_health
        }
        
        # Performance-Prediction-Modell
        self.prediction_model = self._build_prediction_model()
        self.prediction_optimizer = torch.optim.Adam(
            self.prediction_model.parameters(), lr=0.001
        )
        
        # MOO-Status
        self.optimization_phase = "exploration"  # exploration, exploitation, balance
        self.trade_off_ratio = 0.5  # Trade-off zwischen Zielen
        
        # Metriken
        self.objective_scores = {
            'performance': 0.0,
            'eta_stability': 0.0,
            'sigma_health': 0.0
        }
        
        logger.info(f"MOO initialisiert: weights={self.current_weights}, "
                   f"phase={self.optimization_phase}")
    
    def _build_prediction_model(self) -> nn.Module:
        """Baut das Performance-Prediction-Modell."""
        class PerformancePredictor(nn.Module):
            def __init__(self, input_dim=5, hidden_dim=32, output_dim=1):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return PerformancePredictor()
    
    def update_history(self, performance: float, eta: float, sigma_mean: float, 
                      td_error: float, emotion: float):
        """Aktualisiert die Historie mit neuen Werten."""
        self.performance_history.append(performance)
        self.eta_history.append(eta)
        self.sigma_history.append(sigma_mean)
        self.td_error_history.append(td_error)
        self.emotion_history.append(emotion)
    
    def calculate_performance_score(self) -> float:
        """
        Berechnet den Performance-Score.
        
        Returns:
            float: Performance-Score (0.0 = schlecht, 1.0 = exzellent)
        """
        if len(self.performance_history) < 5:
            return 0.0
        
        # Aktuelle Performance vs. Ziel
        recent_performance = np.mean(list(self.performance_history)[-5:])
        performance_score = min(recent_performance / self.config.performance_target, 1.0)
        
        # Trend-Bonus
        if len(self.performance_history) >= 10:
            recent_trend = np.mean(list(self.performance_history)[-5:]) - \
                          np.mean(list(self.performance_history)[-10:-5])
            trend_bonus = max(0.0, recent_trend / self.config.performance_target)
            performance_score = min(performance_score + trend_bonus, 1.0)
        
        return np.clip(performance_score, 0.0, 1.0)
    
    def calculate_eta_stability_score(self) -> float:
        """
        Berechnet den η-Stabilität-Score.
        
        Returns:
            float: η-Stabilität-Score (0.0 = instabil, 1.0 = sehr stabil)
        """
        if len(self.eta_history) < 5:
            return 0.0
        
        # Varianz der η-Werte als Stabilitätsmaß
        eta_variance = np.var(list(self.eta_history))
        stability_score = 1.0 / (1.0 + eta_variance * 1000)  # Normalisierung
        
        # Effizienz-Bonus: η sollte im optimalen Bereich sein
        eta_mean = np.mean(list(self.eta_history))
        optimal_eta_range = (1e-4, 5e-3)  # Optimaler η-Bereich
        if optimal_eta_range[0] <= eta_mean <= optimal_eta_range[1]:
            efficiency_bonus = 0.2
        else:
            efficiency_bonus = 0.0
        
        return np.clip(stability_score + efficiency_bonus, 0.0, 1.0)
    
    def calculate_sigma_health_score(self) -> float:
        """
        Berechnet den σ-Gesundheit-Score.
        
        Returns:
            float: σ-Gesundheit-Score (0.0 = ungesund, 1.0 = sehr gesund)
        """
        if len(self.sigma_history) < 5:
            return 0.0
        
        # σ-Magnitude als Gesundheitsmaß
        sigma_mean = np.mean(list(self.sigma_history))
        sigma_variance = np.var(list(self.sigma_history))
        
        # Gesundheits-Score basierend auf Magnitude und Stabilität
        magnitude_score = 1.0 - abs(sigma_mean - 0.05) / 0.1  # Optimal um 0.05
        stability_score = 1.0 / (1.0 + sigma_variance * 100)
        
        # Plastizität-Bonus: σ sollte aktiv sein
        plasticity_bonus = 0.1 if sigma_mean > 0.01 else 0.0
        
        health_score = (magnitude_score + stability_score) / 2 + plasticity_bonus
        return np.clip(health_score, 0.0, 1.0)
    
    def predict_future_performance(self) -> Tuple[float, float]:
        """
        Vorhersage der zukünftigen Performance.
        
        Returns:
            Tuple[float, float]: (vorhergesagte_performance, konfidenz)
        """
        if len(self.performance_history) < self.config.prediction_horizon:
            return 0.0, 0.0
        
        # Input-Features für Vorhersage
        features = np.array([
            np.mean(list(self.performance_history)[-5:]),
            np.mean(list(self.eta_history)[-5:]),
            np.mean(list(self.sigma_history)[-5:]),
            np.mean(list(self.td_error_history)[-5:]),
            np.mean(list(self.emotion_history)[-5:])
        ])
        
        # Normalisierung
        features = features / np.array([100.0, 1e-3, 0.1, 1.0, 1.0])
        
        # Vorhersage
        self.prediction_model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            prediction = self.prediction_model(input_tensor).item()
        
        # Konfidenz basierend auf Historie-Stabilität
        confidence = min(len(self.performance_history) / 50.0, 1.0)
        
        return prediction, confidence
    
    def update_adaptive_weights(self):
        """Aktualisiert die adaptiven Gewichte basierend auf aktueller Performance."""
        if len(self.performance_history) < self.config.adaptation_window:
            return
        
        # Aktuelle Ziel-Scores
        perf_score = self.calculate_performance_score()
        eta_score = self.calculate_eta_stability_score()
        sigma_score = self.calculate_sigma_health_score()
        
        # Adaptive Gewichtsanpassung
        if perf_score < 0.3:  # Performance ist schlecht
            self.current_weights['performance'] += self.config.adaptation_rate
            self.current_weights['eta_stability'] -= self.config.adaptation_rate * 0.5
            self.current_weights['sigma_health'] -= self.config.adaptation_rate * 0.5
        elif eta_score < 0.3:  # η ist instabil
            self.current_weights['eta_stability'] += self.config.adaptation_rate
            self.current_weights['performance'] -= self.config.adaptation_rate * 0.5
        elif sigma_score < 0.3:  # σ ist ungesund
            self.current_weights['sigma_health'] += self.config.adaptation_rate
            self.current_weights['performance'] -= self.config.adaptation_rate * 0.5
        
        # Gewichte normalisieren
        total_weight = sum(self.current_weights.values())
        for key in self.current_weights:
            self.current_weights[key] = max(0.1, self.current_weights[key] / total_weight)
        
        # Gewichte normalisieren auf 1.0
        total_weight = sum(self.current_weights.values())
        for key in self.current_weights:
            self.current_weights[key] /= total_weight
    
    def calculate_multi_objective_loss(self, eta: float, sigma_mean: float, 
                                     performance: float) -> float:
        """
        Berechnet den Multi-Objective Loss.
        
        Args:
            eta: Aktuelle Lernrate
            sigma_mean: Aktueller σ-Mittelwert
            performance: Aktuelle Performance
            
        Returns:
            float: Multi-Objective Loss
        """
        # Einzelne Ziel-Scores
        perf_score = self.calculate_performance_score()
        eta_score = self.calculate_eta_stability_score()
        sigma_score = self.calculate_sigma_health_score()
        
        # Multi-Objective Loss mit adaptiven Gewichten
        loss = (
            self.current_weights['performance'] * (1.0 - perf_score) +
            self.current_weights['eta_stability'] * (1.0 - eta_score) +
            self.current_weights['sigma_health'] * (1.0 - sigma_score)
        )
        
        return loss
    
    def optimize_parameters(self, current_eta: float, current_sigma_mean: float, 
                          current_performance: float) -> Dict[str, float]:
        """
        Optimiert η und σ basierend auf Multi-Objective-Loss.
        
        Args:
            current_eta: Aktuelle Lernrate
            current_sigma_mean: Aktueller σ-Mittelwert
            current_performance: Aktuelle Performance
            
        Returns:
            Dict mit optimierten Parametern
        """
        # Historie aktualisieren
        self.update_history(current_performance, current_eta, current_sigma_mean, 
                          0.0, 0.0)  # TD-Error und Emotion werden später aktualisiert
        
        # Adaptive Gewichte aktualisieren
        self.update_adaptive_weights()
        
        # Multi-Objective Loss berechnen
        current_loss = self.calculate_multi_objective_loss(
            current_eta, current_sigma_mean, current_performance
        )
        
        # Parameter-Optimierung basierend auf Trade-offs
        eta_adjustment = 0.0
        sigma_adjustment = 0.0
        
        # Performance-basierte Anpassungen
        if self.current_weights['performance'] > 0.5:
            # Performance ist Priorität
            if current_performance < self.config.performance_target * 0.5:
                eta_adjustment += 0.1  # Erhöhe η für besseres Lernen
                sigma_adjustment += 0.05  # Erhöhe σ für mehr Plastizität
            elif current_performance > self.config.performance_target * 0.8:
                eta_adjustment -= 0.05  # Reduziere η für Stabilität
                sigma_adjustment -= 0.02  # Reduziere σ für Konsolidierung
        
        # η-Stabilität-basierte Anpassungen
        if self.current_weights['eta_stability'] > 0.4:
            eta_variance = np.var(list(self.eta_history)[-10:]) if len(self.eta_history) >= 10 else 0.0
            if eta_variance > 1e-6:  # η ist zu volatil
                eta_adjustment -= 0.02  # Stabilisiere η
        
        # σ-Gesundheit-basierte Anpassungen
        if self.current_weights['sigma_health'] > 0.4:
            if current_sigma_mean < 0.01:  # σ ist zu niedrig
                sigma_adjustment += 0.03  # Erhöhe σ
            elif current_sigma_mean > 0.1:  # σ ist zu hoch
                sigma_adjustment -= 0.03  # Reduziere σ
        
        # Vorhersage-basierte Anpassungen
        predicted_perf, confidence = self.predict_future_performance()
        if confidence > self.config.prediction_confidence:
            if predicted_perf < current_performance * 0.9:
                # Performance wird wahrscheinlich sinken
                eta_adjustment -= 0.01  # Konservativer werden
                sigma_adjustment -= 0.01
        
        # Optimierte Parameter berechnen
        optimized_eta = current_eta * (1.0 + eta_adjustment)
        optimized_sigma_target = current_sigma_mean * (1.0 + sigma_adjustment)
        
        # Grenzen einhalten
        optimized_eta = np.clip(optimized_eta, 1e-5, 1e-2)
        optimized_sigma_target = np.clip(optimized_sigma_target, 0.001, 0.2)
        
        return {
            'optimized_eta': optimized_eta,
            'optimized_sigma_target': optimized_sigma_target,
            'eta_adjustment': eta_adjustment,
            'sigma_adjustment': sigma_adjustment,
            'multi_objective_loss': current_loss,
            'performance_score': self.calculate_performance_score(),
            'eta_stability_score': self.calculate_eta_stability_score(),
            'sigma_health_score': self.calculate_sigma_health_score(),
            'predicted_performance': predicted_perf,
            'prediction_confidence': confidence
        }
    
    def train_prediction_model(self):
        """Trainiert das Performance-Prediction-Modell."""
        if len(self.performance_history) < 20:
            return
        
        # Trainingsdaten vorbereiten
        X, y = [], []
        for i in range(len(self.performance_history) - 5):
            features = np.array([
                self.performance_history[i],
                self.eta_history[i],
                self.sigma_history[i],
                self.td_error_history[i] if i < len(self.td_error_history) else 0.0,
                self.emotion_history[i] if i < len(self.emotion_history) else 0.5
            ])
            
            # Normalisierung
            features = features / np.array([100.0, 1e-3, 0.1, 1.0, 1.0])
            
            # Target: Performance 5 Episoden später
            target = self.performance_history[i + 5] / 100.0
            
            X.append(features)
            y.append(target)
        
        if len(X) < 10:
            return
        
        # Training
        self.prediction_model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        self.prediction_optimizer.zero_grad()
        predictions = self.prediction_model(X_tensor)
        loss = nn.MSELoss()(predictions, y_tensor)
        loss.backward()
        self.prediction_optimizer.step()
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den aktuellen Optimierungsstatus zurück.
        
        Returns:
            Dict mit Optimierungs-Informationen
        """
        return {
            'current_weights': self.current_weights.copy(),
            'optimization_phase': self.optimization_phase,
            'trade_off_ratio': self.trade_off_ratio,
            'objective_scores': {
                'performance': self.calculate_performance_score(),
                'eta_stability': self.calculate_eta_stability_score(),
                'sigma_health': self.calculate_sigma_health_score()
            },
            'predicted_performance': self.predict_future_performance()[0],
            'prediction_confidence': self.predict_future_performance()[1],
            'history_size': len(self.performance_history)
        }
    
    def reset_optimization(self):
        """Setzt die Optimierung zurück."""
        self.performance_history.clear()
        self.eta_history.clear()
        self.sigma_history.clear()
        self.td_error_history.clear()
        self.emotion_history.clear()
        
        self.current_weights = {
            'performance': self.config.w_performance,
            'eta_stability': self.config.w_eta_stability,
            'sigma_health': self.config.w_sigma_health
        }
        
        self.optimization_phase = "exploration"
        self.trade_off_ratio = 0.5
        
        logger.info("MOO zurückgesetzt")
