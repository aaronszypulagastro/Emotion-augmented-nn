"""
Winner Mindset Regulator - Phase 8.0
=====================================

Emotion-Driven Meta-Learning Framework für komplexe RL-Tasks

Konzept:
--------
Emotion ist Meta-Signal für:
- Motivation (hohe Performance → Pride → kontrollierte Exploration)
- Frustration (niedrige Performance → Focus → intensiveres Lernen)
- Adaptivität (dynamische Anpassung an Task-Schwierigkeit)

Nicht für: LR-Modulation (zu instabil)
Sondern für: Exploration, Focus, Noise Scaling

Inspiriert von psychologischen "Winner Mindset" Prinzipien:
- Frustration → erhöhter Fokus (weniger Noise)
- Erfolg → kontrollierte Exploration
- Selbstregulation basierend auf Performance-Trends

Author: Phase 8.0 - Winner Mindset Framework
Date: 2025-10-16
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MindsetState(Enum):
    """Emotionale Zustände des Agenten"""
    FRUSTRATION = "frustration"  # Niedrige Performance, hohe Varianz
    CALM = "calm"                # Stabile, mittlere Performance  
    PRIDE = "pride"              # Hohe Performance, stabil
    CURIOSITY = "curiosity"      # Exploration-Phase
    FOCUS = "focus"              # Exploitation-Phase


@dataclass
class MindsetMetrics:
    """Metriken des aktuellen Mindset-Zustands"""
    state: MindsetState
    emotion_value: float         # [0, 1]
    exploration_factor: float    # Epsilon-Modulation
    noise_scale: float           # Noise für Plasticity
    focus_intensity: float       # Wie fokussiert ist der Agent
    learning_efficiency: float   # Reward Growth / Episodes


class WinnerMindsetRegulator:
    """
    Reguliert Agent-Verhalten basierend auf emotionalem Zustand
    
    Kernprinzipien:
    ---------------
    1. FRUSTRATION → FOCUS
       Niedrige Performance → Weniger Noise, mehr Exploitation
       
    2. SUCCESS → CONTROLLED EXPLORATION  
       Hohe Performance → Moderate Exploration, neue Strategien testen
       
    3. ADAPTIVE LEARNING
       System passt sich an Task-Schwierigkeit an
       
    4. STABILITY-AWARE
       Nutzt PSA-Metriken für Mindset-Entscheidungen
    """
    
    def __init__(
        self,
        epsilon_min: float = 0.01,
        epsilon_max: float = 0.3,
        noise_min: float = 0.001,
        noise_max: float = 0.05,
        frustration_threshold: float = 0.3,
        pride_threshold: float = 0.7,
        focus_decay: float = 0.95,
        history_window: int = 50
    ):
        """
        Args:
            epsilon_min: Minimale Exploration Rate
            epsilon_max: Maximale Exploration Rate
            noise_min: Minimales Noise-Level für Plasticity
            noise_max: Maximales Noise-Level
            frustration_threshold: Emotion < threshold → Frustration
            pride_threshold: Emotion > threshold → Pride
            focus_decay: Wie schnell Focus abnimmt
            history_window: Window für Performance-Tracking
        """
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.frustration_threshold = frustration_threshold
        self.pride_threshold = pride_threshold
        self.focus_decay = focus_decay
        self.history_window = history_window
        
        # State
        self.current_state = MindsetState.CALM
        self.focus_intensity = 0.5
        self.emotion_history = []
        self.performance_history = []
        
        # Dynamics Logging
        self.mindset_log = []
    
    def update(
        self,
        emotion_state: float,
        performance_metrics: Dict[str, float]
    ) -> MindsetMetrics:
        """
        Update Mindset basierend auf Emotion und Performance
        
        Args:
            emotion_state: Aktueller Emotionswert [0, 1]
            performance_metrics: Dict mit:
                - avg_return: Durchschnittlicher Return
                - stability: PSA Stability Score
                - trend: 'ascending', 'stable', 'descending'
                - td_error: Aktueller TD-Error
                
        Returns:
            MindsetMetrics mit aktuellem Zustand
        """
        # Store history
        self.emotion_history.append(emotion_state)
        self.performance_history.append(performance_metrics['avg_return'])
        
        if len(self.emotion_history) > self.history_window:
            self.emotion_history.pop(0)
            self.performance_history.pop(0)
        
        # Determine Mindset State
        self.current_state = self._determine_mindset_state(
            emotion_state,
            performance_metrics
        )
        
        # Compute Modulations
        exploration_factor = self._compute_exploration_factor(
            emotion_state,
            performance_metrics
        )
        
        noise_scale = self._compute_noise_scale(
            emotion_state,
            performance_metrics
        )
        
        # Update Focus Intensity
        self._update_focus_intensity(performance_metrics)
        
        # Compute Learning Efficiency
        learning_efficiency = self._compute_learning_efficiency()
        
        # Create metrics
        metrics = MindsetMetrics(
            state=self.current_state,
            emotion_value=emotion_state,
            exploration_factor=exploration_factor,
            noise_scale=noise_scale,
            focus_intensity=self.focus_intensity,
            learning_efficiency=learning_efficiency
        )
        
        # Log
        self.mindset_log.append({
            'emotion': emotion_state,
            'state': self.current_state.value,
            'exploration': exploration_factor,
            'noise': noise_scale,
            'focus': self.focus_intensity,
            'efficiency': learning_efficiency
        })
        
        return metrics
    
    def _determine_mindset_state(
        self,
        emotion: float,
        perf: Dict[str, float]
    ) -> MindsetState:
        """Bestimme emotionalen Zustand"""
        
        trend = perf.get('trend', 'stable')
        stability = perf.get('stability', 0.5)
        
        # FRUSTRATION: Niedrige Emotion + schlechter Trend
        if emotion < self.frustration_threshold:
            if trend == 'descending' or stability < 0.5:
                return MindsetState.FRUSTRATION
            else:
                return MindsetState.CALM
        
        # PRIDE: Hohe Emotion + guter Trend
        elif emotion > self.pride_threshold:
            if trend == 'ascending' and stability > 0.6:
                return MindsetState.PRIDE
            else:
                return MindsetState.CURIOSITY
        
        # CALM/CURIOSITY: Mittlere Emotion
        else:
            if stability > 0.6:
                return MindsetState.CALM
            else:
                return MindsetState.CURIOSITY
    
    def _compute_exploration_factor(
        self,
        emotion: float,
        perf: Dict[str, float]
    ) -> float:
        """
        Berechne Exploration Factor (Epsilon-Modulation)
        
        Strategie:
        ----------
        FRUSTRATION: Hohe Exploration (0.8 * max)
            → Suche nach neuen Lösungen
            
        PRIDE: Moderate Exploration (0.3 * max)
            → Teste neue Strategien kontrolliert
            
        CALM: Niedrige Exploration (0.1 * max)
            → Nutze Gelerntes
            
        FOCUS: Minimale Exploration (min)
            → Reine Exploitation
        """
        if self.current_state == MindsetState.FRUSTRATION:
            # Viel Exploration bei Frustration
            return 0.8
            
        elif self.current_state == MindsetState.PRIDE:
            # Moderate Exploration bei Erfolg
            return 0.3
            
        elif self.current_state == MindsetState.CURIOSITY:
            # Moderate-hohe Exploration
            return 0.6
            
        elif self.current_state == MindsetState.FOCUS:
            # Minimale Exploration
            return 0.05
            
        else:  # CALM
            # Niedrige Exploration
            return 0.2
    
    def _compute_noise_scale(
        self,
        emotion: float,
        perf: Dict[str, float]
    ) -> float:
        """
        Berechne Noise Scale für BDH-Plasticity
        
        Strategie:
        ----------
        FRUSTRATION/FOCUS: Niedriges Noise
            → Mehr Kontrolle, weniger Chaos
            
        PRIDE/CURIOSITY: Höheres Noise  
            → Erlaube Exploration in Gewichtsraum
        """
        if self.current_state in [MindsetState.FRUSTRATION, MindsetState.FOCUS]:
            # Niedriges Noise für Stabilität
            return 0.2
            
        elif self.current_state == MindsetState.PRIDE:
            # Moderates Noise
            return 0.5
            
        elif self.current_state == MindsetState.CURIOSITY:
            # Höheres Noise
            return 0.8
            
        else:  # CALM
            return 0.4
    
    def _update_focus_intensity(self, perf: Dict[str, float]):
        """
        Update Focus Intensity
        
        Focus steigt bei:
        - Frustration (Agent fokussiert sich)
        - Schlechtem Trend
        
        Focus sinkt bei:
        - Pride (Agent entspannt)
        - Gutem Trend (decay)
        """
        if self.current_state == MindsetState.FRUSTRATION:
            # Frustration → erhöhter Focus
            self.focus_intensity = min(1.0, self.focus_intensity + 0.1)
            
        elif self.current_state == MindsetState.PRIDE:
            # Pride → entspannt Focus
            self.focus_intensity = max(0.0, self.focus_intensity - 0.05)
            
        else:
            # Natürlicher Decay
            self.focus_intensity *= self.focus_decay
            self.focus_intensity = np.clip(self.focus_intensity, 0.0, 1.0)
    
    def _compute_learning_efficiency(self) -> float:
        """
        Learning Efficiency Index = Reward Growth / Episode Count
        
        Misst wie effizient der Agent lernt
        """
        if len(self.performance_history) < 10:
            return 0.0
        
        # Berechne lineare Regression Slope
        x = np.arange(len(self.performance_history))
        y = np.array(self.performance_history)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            # Normalisiere
            efficiency = np.tanh(slope / 10.0)  # [-1, 1]
            return efficiency
        
        return 0.0
    
    def modulate_exploration(self, base_epsilon: float) -> float:
        """
        Moduliere Epsilon basierend auf Mindset
        
        Args:
            base_epsilon: Basis-Epsilon vom Standard-Decay
            
        Returns:
            Moduliertes Epsilon
        """
        # Get current exploration factor
        if not self.mindset_log:
            factor = 0.5
        else:
            factor = self.mindset_log[-1]['exploration']
        
        # Modulate
        modulated = self.epsilon_min + factor * (self.epsilon_max - self.epsilon_min)
        
        # Blend mit base_epsilon (50/50)
        final_epsilon = 0.5 * base_epsilon + 0.5 * modulated
        
        return np.clip(final_epsilon, self.epsilon_min, self.epsilon_max)
    
    def modulate_noise(self, base_noise: float) -> float:
        """
        Moduliere Noise für BDH-Plasticity
        
        Args:
            base_noise: Basis-Noise-Level
            
        Returns:
            Moduliertes Noise
        """
        if not self.mindset_log:
            factor = 0.5
        else:
            factor = self.mindset_log[-1]['noise']
        
        modulated = self.noise_min + factor * (self.noise_max - self.noise_min)
        
        return np.clip(modulated, self.noise_min, self.noise_max)
    
    def get_current_mindset(self) -> MindsetMetrics:
        """Gibt aktuellen Mindset-Zustand zurück"""
        if not self.mindset_log:
            return MindsetMetrics(
                state=MindsetState.CALM,
                emotion_value=0.5,
                exploration_factor=0.5,
                noise_scale=0.5,
                focus_intensity=0.5,
                learning_efficiency=0.0
            )
        
        latest = self.mindset_log[-1]
        return MindsetMetrics(
            state=MindsetState(latest['state']),
            emotion_value=latest['emotion'],
            exploration_factor=latest['exploration'],
            noise_scale=latest['noise'],
            focus_intensity=latest['focus'],
            learning_efficiency=latest['efficiency']
        )
    
    def log_mindset_dynamics(self) -> Dict:
        """
        Exportiere Mindset-Dynamics für Visualisierung
        
        Returns:
            Dict mit allen geloggten Metriken
        """
        if not self.mindset_log:
            return {}
        
        return {
            'emotions': [log['emotion'] for log in self.mindset_log],
            'states': [log['state'] for log in self.mindset_log],
            'exploration': [log['exploration'] for log in self.mindset_log],
            'noise': [log['noise'] for log in self.mindset_log],
            'focus': [log['focus'] for log in self.mindset_log],
            'efficiency': [log['efficiency'] for log in self.mindset_log],
            'performance': self.performance_history
        }
    
    def get_state_statistics(self) -> Dict[str, int]:
        """Statistik über Mindset-States"""
        if not self.mindset_log:
            return {}
        
        states = [log['state'] for log in self.mindset_log]
        return {
            state.value: states.count(state.value)
            for state in MindsetState
        }
    
    def reset(self):
        """Reset Regulator"""
        self.current_state = MindsetState.CALM
        self.focus_intensity = 0.5
        self.emotion_history = []
        self.performance_history = []
        self.mindset_log = []


# Utility Functions für Integration

def create_winner_mindset_config(env_type: str = "cartpole") -> Dict:
    """
    Erstelle optimierte Config für verschiedene Environments
    
    Args:
        env_type: "cartpole", "lunarlander", "atari"
        
    Returns:
        Config Dict für WinnerMindsetRegulator
    """
    if env_type == "cartpole":
        return {
            'epsilon_min': 0.01,
            'epsilon_max': 0.2,
            'noise_min': 0.001,
            'noise_max': 0.03,
            'frustration_threshold': 0.35,
            'pride_threshold': 0.65,
            'focus_decay': 0.95,
            'history_window': 30
        }
    
    elif env_type == "lunarlander":
        return {
            'epsilon_min': 0.01,
            'epsilon_max': 0.4,
            'noise_min': 0.005,
            'noise_max': 0.08,
            'frustration_threshold': 0.3,
            'pride_threshold': 0.7,
            'focus_decay': 0.98,
            'history_window': 50
        }
    
    elif env_type == "atari":
        return {
            'epsilon_min': 0.01,
            'epsilon_max': 0.5,
            'noise_min': 0.01,
            'noise_max': 0.1,
            'frustration_threshold': 0.25,
            'pride_threshold': 0.75,
            'focus_decay': 0.99,
            'history_window': 100
        }
    
    else:
        # Default: Universal Config
        return {
            'epsilon_min': 0.01,
            'epsilon_max': 0.3,
            'noise_min': 0.005,
            'noise_max': 0.05,
            'frustration_threshold': 0.3,
            'pride_threshold': 0.7,
            'focus_decay': 0.97,
            'history_window': 50
        }


if __name__ == "__main__":
    # Quick Test
    print("Winner Mindset Regulator - Quick Test\n")
    
    wmr = WinnerMindsetRegulator()
    
    # Simuliere verschiedene Szenarien
    scenarios = [
        ("Frustration", 0.2, {'avg_return': 10, 'stability': 0.4, 'trend': 'descending', 'td_error': 5.0}),
        ("Calm", 0.5, {'avg_return': 150, 'stability': 0.6, 'trend': 'stable', 'td_error': 2.0}),
        ("Pride", 0.8, {'avg_return': 450, 'stability': 0.7, 'trend': 'ascending', 'td_error': 1.0}),
    ]
    
    for name, emotion, perf in scenarios:
        metrics = wmr.update(emotion, perf)
        print(f"{name}:")
        print(f"  State: {metrics.state.value}")
        print(f"  Exploration: {metrics.exploration_factor:.3f}")
        print(f"  Noise Scale: {metrics.noise_scale:.3f}")
        print(f"  Focus: {metrics.focus_intensity:.3f}")
        print()


