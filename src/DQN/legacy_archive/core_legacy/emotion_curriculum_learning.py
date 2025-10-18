"""
Emotion-basiertes Curriculum Learning (ECL) - Phase 6.2
====================================================

Adaptive Schwierigkeitsanpassung basierend auf emotionaler Verfassung,
TD-Error-Dynamik und Lernfortschritt. Das System passt die Umgebungs-
schwierigkeit dynamisch an, um optimale Lernbedingungen zu schaffen.

Key Features:
- Emotion-basierte Schwierigkeitsanpassung
- TD-Error-adaptive Komplexitätsregelung
- Lernfortschritts-basierte Curriculum-Progression
- Multi-Modal-Schwierigkeitskontrolle (Reward, Action-Space, State-Space)
- Anti-Catastrophic-Forgetting-Mechanismen
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ECLConfig:
    """Konfiguration für Emotion-basiertes Curriculum Learning."""
    
    # Emotion-basierte Schwierigkeitsanpassung
    emotion_threshold_low: float = 0.3      # Niedrige Emotion → einfachere Aufgaben
    emotion_threshold_high: float = 0.7     # Hohe Emotion → schwierigere Aufgaben
    
    # TD-Error-adaptive Regelung
    td_error_threshold_low: float = 0.5     # Niedriger TD-Error → erhöhe Schwierigkeit
    td_error_threshold_high: float = 1.5    # Hoher TD-Error → reduziere Schwierigkeit
    
    # Lernfortschritts-basierte Progression
    progress_window: int = 20               # Fenster für Fortschrittsbewertung
    progress_threshold: float = 0.1         # Mindestfortschritt für Schwierigkeitserhöhung
    
    # Schwierigkeitsstufen
    min_difficulty: float = 0.1             # Minimale Schwierigkeit
    max_difficulty: float = 1.0             # Maximale Schwierigkeit
    difficulty_step: float = 0.05           # Schrittweite für Schwierigkeitsänderungen
    
    # Anti-Catastrophic-Forgetting
    stability_window: int = 10              # Fenster für Stabilitätsbewertung
    stability_threshold: float = 0.8        # Mindeststabilität für Schwierigkeitserhöhung
    
    # Multi-Modal-Kontrolle
    reward_scaling_factor: float = 0.1      # Faktor für Reward-Skalierung
    action_noise_factor: float = 0.05       # Faktor für Action-Noise
    state_noise_factor: float = 0.02        # Faktor für State-Noise


class EmotionCurriculumLearning:
    """
    Emotion-basiertes Curriculum Learning System.
    
    Passt die Umgebungsschwierigkeit dynamisch an die emotionale Verfassung
    und den Lernfortschritt des Agenten an.
    """
    
    def __init__(self, config: ECLConfig = ECLConfig()):
        """
        Initialisiert das ECL-System.
        
        Args:
            config: ECL-Konfiguration
        """
        self.config = config
        
        # Historie für Fortschrittsbewertung
        self.reward_history = deque(maxlen=config.progress_window)
        self.td_error_history = deque(maxlen=config.progress_window)
        self.emotion_history = deque(maxlen=config.progress_window)
        self.difficulty_history = deque(maxlen=config.stability_window)
        
        # Aktuelle Schwierigkeitsstufe
        self.current_difficulty = 0.5  # Starte mit mittlerer Schwierigkeit
        
        # Curriculum-Status
        self.curriculum_phase = "exploration"  # exploration, consolidation, mastery
        self.phase_progress = 0.0
        
        # Performance-Metriken
        self.performance_ema = 0.0
        self.stability_ema = 0.0
        
        # Adaptive Parameter
        self.adaptive_reward_scale = 1.0
        self.adaptive_action_noise = 0.0
        self.adaptive_state_noise = 0.0
        
        logger.info(f"ECL initialisiert: difficulty={self.current_difficulty:.2f}, "
                   f"phase={self.curriculum_phase}")
    
    def update_history(self, reward: float, td_error: float, emotion: float):
        """Aktualisiert die Historie mit neuen Werten."""
        self.reward_history.append(reward)
        self.td_error_history.append(td_error)
        self.emotion_history.append(emotion)
        self.difficulty_history.append(self.current_difficulty)
    
    def calculate_learning_progress(self) -> float:
        """
        Berechnet den Lernfortschritt basierend auf Reward-Trend.
        
        Returns:
            float: Lernfortschritt (0.0 = kein Fortschritt, 1.0 = großer Fortschritt)
        """
        if len(self.reward_history) < 5:
            return 0.0
        
        # Trend-Analyse der Rewards
        recent_rewards = list(self.reward_history)[-10:]
        older_rewards = list(self.reward_history)[-20:-10] if len(self.reward_history) >= 20 else []
        
        if len(older_rewards) == 0:
            return 0.0
        
        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)
        
        # Fortschritt als relative Verbesserung
        if older_avg == 0:
            progress = 0.0
        else:
            progress = (recent_avg - older_avg) / abs(older_avg)
        
        return np.clip(progress, 0.0, 1.0)
    
    def calculate_emotional_stability(self) -> float:
        """
        Berechnet die emotionale Stabilität.
        
        Returns:
            float: Stabilität (0.0 = sehr instabil, 1.0 = sehr stabil)
        """
        if len(self.emotion_history) < 5:
            return 0.0
        
        # Varianz der Emotionen als Stabilitätsmaß
        emotion_variance = np.var(list(self.emotion_history))
        stability = 1.0 / (1.0 + emotion_variance * 10)  # Normalisierung
        
        return np.clip(stability, 0.0, 1.0)
    
    def calculate_td_error_stability(self) -> float:
        """
        Berechnet die TD-Error-Stabilität.
        
        Returns:
            float: TD-Error-Stabilität (0.0 = sehr instabil, 1.0 = sehr stabil)
        """
        if len(self.td_error_history) < 5:
            return 0.0
        
        # Varianz der TD-Errors als Stabilitätsmaß
        td_variance = np.var(list(self.td_error_history))
        stability = 1.0 / (1.0 + td_variance)
        
        return np.clip(stability, 0.0, 1.0)
    
    def determine_curriculum_phase(self) -> str:
        """
        Bestimmt die aktuelle Curriculum-Phase basierend auf Performance.
        
        Returns:
            str: Curriculum-Phase (exploration, consolidation, mastery)
        """
        if len(self.reward_history) < 10:
            return "exploration"
        
        # Performance-basierte Phasenbestimmung
        recent_performance = np.mean(list(self.reward_history)[-10:])
        progress = self.calculate_learning_progress()
        stability = self.calculate_emotional_stability()
        
        if recent_performance > 50 and progress > 0.1 and stability > 0.7:
            return "mastery"
        elif recent_performance > 20 and progress > 0.05 and stability > 0.5:
            return "consolidation"
        else:
            return "exploration"
    
    def update_difficulty(self, emotion: float, td_error: float, reward: float) -> float:
        """
        Aktualisiert die Schwierigkeitsstufe basierend auf aktuellen Metriken.
        
        Args:
            emotion: Aktuelle Emotion
            td_error: Aktueller TD-Error
            reward: Aktueller Reward
            
        Returns:
            float: Neue Schwierigkeitsstufe
        """
        # Historie aktualisieren
        self.update_history(reward, td_error, emotion)
        
        # Curriculum-Phase bestimmen
        self.curriculum_phase = self.determine_curriculum_phase()
        
        # Lernfortschritt und Stabilität berechnen
        progress = self.calculate_learning_progress()
        emotional_stability = self.calculate_emotional_stability()
        td_stability = self.calculate_td_error_stability()
        overall_stability = (emotional_stability + td_stability) / 2.0
        
        # Schwierigkeitsanpassung basierend auf verschiedenen Faktoren
        difficulty_change = 0.0
        
        # 1. Emotion-basierte Anpassung
        if emotion < self.config.emotion_threshold_low:
            # Niedrige Emotion → einfachere Aufgaben
            difficulty_change -= self.config.difficulty_step * 0.5
        elif emotion > self.config.emotion_threshold_high:
            # Hohe Emotion → schwierigere Aufgaben (wenn stabil)
            if overall_stability > self.config.stability_threshold:
                difficulty_change += self.config.difficulty_step * 0.5
        
        # 2. TD-Error-basierte Anpassung
        if td_error < self.config.td_error_threshold_low:
            # Niedriger TD-Error → erhöhe Schwierigkeit
            if overall_stability > self.config.stability_threshold:
                difficulty_change += self.config.difficulty_step
        elif td_error > self.config.td_error_threshold_high:
            # Hoher TD-Error → reduziere Schwierigkeit
            difficulty_change -= self.config.difficulty_step
        
        # 3. Lernfortschritts-basierte Anpassung
        if progress > self.config.progress_threshold:
            # Guter Fortschritt → erhöhe Schwierigkeit
            if overall_stability > self.config.stability_threshold:
                difficulty_change += self.config.difficulty_step * 0.3
        elif progress < -self.config.progress_threshold:
            # Schlechter Fortschritt → reduziere Schwierigkeit
            difficulty_change -= self.config.difficulty_step * 0.3
        
        # 4. Curriculum-Phase-basierte Anpassung
        if self.curriculum_phase == "exploration":
            # Exploration → moderate Schwierigkeit
            target_difficulty = 0.4
        elif self.curriculum_phase == "consolidation":
            # Konsolidierung → erhöhte Schwierigkeit
            target_difficulty = 0.7
        else:  # mastery
            # Meisterschaft → maximale Schwierigkeit
            target_difficulty = 1.0
        
        # Sanfte Anpassung zur Zielschwierigkeit
        phase_adjustment = (target_difficulty - self.current_difficulty) * 0.1
        difficulty_change += phase_adjustment
        
        # Schwierigkeit aktualisieren
        new_difficulty = self.current_difficulty + difficulty_change
        new_difficulty = np.clip(new_difficulty, 
                                self.config.min_difficulty, 
                                self.config.max_difficulty)
        
        self.current_difficulty = new_difficulty
        
        # Adaptive Parameter aktualisieren
        self._update_adaptive_parameters()
        
        return self.current_difficulty
    
    def _update_adaptive_parameters(self):
        """Aktualisiert die adaptiven Parameter basierend auf der Schwierigkeit."""
        # Reward-Skalierung
        self.adaptive_reward_scale = 1.0 + (self.current_difficulty - 0.5) * self.config.reward_scaling_factor
        
        # Action-Noise
        self.adaptive_action_noise = self.current_difficulty * self.config.action_noise_factor
        
        # State-Noise
        self.adaptive_state_noise = self.current_difficulty * self.config.state_noise_factor
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den aktuellen Curriculum-Status zurück.
        
        Returns:
            Dict mit Curriculum-Informationen
        """
        progress = self.calculate_learning_progress()
        emotional_stability = self.calculate_emotional_stability()
        td_stability = self.calculate_td_error_stability()
        
        return {
            'current_difficulty': self.current_difficulty,
            'curriculum_phase': self.curriculum_phase,
            'learning_progress': progress,
            'emotional_stability': emotional_stability,
            'td_stability': td_stability,
            'adaptive_reward_scale': self.adaptive_reward_scale,
            'adaptive_action_noise': self.adaptive_action_noise,
            'adaptive_state_noise': self.adaptive_state_noise,
            'history_size': len(self.reward_history)
        }
    
    def apply_curriculum_modifications(self, reward: float, action: np.ndarray, 
                                     state: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Wendet Curriculum-Modifikationen auf Reward, Action und State an.
        
        Args:
            reward: Originaler Reward
            action: Originale Action
            state: Originaler State
            
        Returns:
            Tuple mit modifizierten (reward, action, state)
        """
        # Reward-Skalierung
        modified_reward = reward * self.adaptive_reward_scale
        
        # Action-Noise hinzufügen
        if self.adaptive_action_noise > 0:
            action_noise = np.random.normal(0, self.adaptive_action_noise, action.shape)
            modified_action = action + action_noise
        else:
            modified_action = action
        
        # State-Noise hinzufügen
        if self.adaptive_state_noise > 0:
            state_noise = np.random.normal(0, self.adaptive_state_noise, state.shape)
            modified_state = state + state_noise
        else:
            modified_state = state
        
        return modified_reward, modified_action, modified_state
    
    def get_difficulty_recommendation(self) -> Dict[str, float]:
        """
        Gibt Empfehlungen für Umgebungsanpassungen zurück.
        
        Returns:
            Dict mit Empfehlungen für verschiedene Umgebungsparameter
        """
        return {
            'reward_scaling': self.adaptive_reward_scale,
            'action_noise_std': self.adaptive_action_noise,
            'state_noise_std': self.adaptive_state_noise,
            'episode_length_multiplier': 1.0 + (self.current_difficulty - 0.5) * 0.5,
            'exploration_bonus': max(0.0, 0.1 - self.current_difficulty * 0.05),
            'penalty_scaling': 1.0 + self.current_difficulty * 0.2
        }
    
    def reset_curriculum(self):
        """Setzt das Curriculum zurück."""
        self.current_difficulty = 0.5
        self.curriculum_phase = "exploration"
        self.phase_progress = 0.0
        self.performance_ema = 0.0
        self.stability_ema = 0.0
        
        # Historie leeren
        self.reward_history.clear()
        self.td_error_history.clear()
        self.emotion_history.clear()
        self.difficulty_history.clear()
        
        # Adaptive Parameter zurücksetzen
        self._update_adaptive_parameters()
        
        logger.info("ECL zurückgesetzt")
