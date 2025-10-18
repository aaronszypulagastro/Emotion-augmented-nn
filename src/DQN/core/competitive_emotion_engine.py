"""
Competitive Emotion Engine - Phase 8.1
=======================================

KONZEPT: Emotion durch direkten Wettbewerb
-----------------------------------------

Statt komplexer Emotion-Calibration:
→ Emotion basiert auf Win/Loss gegen Competitor!

Psychologisch fundiert:
- Gewinnen → Stolz, Selbstvertrauen (Emotion ↑)
- Verlieren → Frustration, Fokus (Emotion ↓)
- Enge Matches → Spannung, Motivation (Emotion variabel)

VORTEIL gegenüber Winner Mindset:
- KEIN Target-Return nötig
- KEIN Alpha-Tuning
- Signal ist 100% klar: Better or Worse than Competitor

Author: Phase 8.1 - Competitive Meta-Learning
Date: 2025-10-17
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CompetitionOutcome(Enum):
    """Ergebnis eines Wettbewerbs"""
    DECISIVE_WIN = "decisive_win"      # Deutlicher Sieg (>20% besser)
    WIN = "win"                        # Normaler Sieg
    DRAW = "draw"                      # Unentschieden (< 5% Differenz)
    LOSS = "loss"                      # Niederlage
    DECISIVE_LOSS = "decisive_loss"    # Deutliche Niederlage (>20% schlechter)


@dataclass
class CompetitionResult:
    """Ergebnis eines Competition-Episodes"""
    outcome: CompetitionOutcome
    score_self: float
    score_competitor: float
    score_diff: float              # Positive = gewonnen
    score_diff_relative: float     # Prozentuale Differenz
    emotion_delta: float           # Emotion-Änderung durch Competition
    new_emotion: float             # Neue Emotion nach Competition


class CompetitiveEmotionEngine:
    """
    Emotion-Engine die durch direkten Wettbewerb gesteuert wird
    
    Kernprinzip:
    ------------
    Emotion = EMA von Win/Loss-Signalen
    
    Win  → Emotion + 0.1 (Pride, Confidence)
    Loss → Emotion - 0.1 (Frustration, Determination)
    Draw → Emotion unchanged (Spannung!)
    
    Keine komplexen Target-Returns oder Calibration nötig!
    Signal ist INTRINSISCH klar.
    """
    
    def __init__(
        self,
        init_emotion: float = 0.5,
        alpha: float = 0.15,              # Reaktionsgeschwindigkeit
        bounds: Tuple[float, float] = (0.2, 0.8),
        decisive_threshold: float = 0.2,  # >20% Differenz = decisive
        draw_threshold: float = 0.05,     # <5% Differenz = draw
        momentum: float = 0.9,            # EMA für Streak-Tracking
    ):
        """
        Args:
            init_emotion: Start-Emotion (neutral bei 0.5)
            alpha: Wie schnell reagiert Emotion auf Win/Loss
            bounds: Emotion-Grenzen [min, max]
            decisive_threshold: Schwelle für "decisive" Win/Loss
            draw_threshold: Schwelle für "draw"
            momentum: Tracking von Win/Loss-Streaks
        """
        self.value = init_emotion
        self.alpha = alpha
        self.bounds = bounds
        self.decisive_threshold = decisive_threshold
        self.draw_threshold = draw_threshold
        self.momentum_factor = momentum
        
        # State
        self.win_loss_momentum = 0.0  # Positive = Win-Streak, Negative = Loss-Streak
        self.total_competitions = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        # History
        self.history = []
        self.outcome_history = []
    
    def compete(
        self,
        score_self: float,
        score_competitor: float,
        episode: int
    ) -> CompetitionResult:
        """
        Hauptfunktion: Update Emotion basierend auf Competition
        
        Args:
            score_self: Score des Haupt-Agents
            score_competitor: Score des Competitors
            episode: Aktuelle Episode (für Logging)
            
        Returns:
            CompetitionResult mit Details
        """
        self.total_competitions += 1
        
        # Compute Difference
        score_diff = score_self - score_competitor
        
        # Relative Difference (robust gegen negative Scores)
        if abs(score_competitor) > 1e-6:
            score_diff_relative = score_diff / abs(score_competitor)
        else:
            score_diff_relative = np.sign(score_diff)
        
        # Determine Outcome
        outcome = self._determine_outcome(score_diff_relative)
        
        # Update Stats
        if outcome in [CompetitionOutcome.WIN, CompetitionOutcome.DECISIVE_WIN]:
            self.wins += 1
        elif outcome in [CompetitionOutcome.LOSS, CompetitionOutcome.DECISIVE_LOSS]:
            self.losses += 1
        else:
            self.draws += 1
        
        # Compute Emotion Delta
        emotion_delta = self._compute_emotion_delta(outcome, score_diff_relative)
        
        # Update Emotion (EMA mit Bounds)
        old_emotion = self.value
        self.value = self.value + self.alpha * emotion_delta
        self.value = float(np.clip(self.value, self.bounds[0], self.bounds[1]))
        
        # Update Momentum (für Streak-Tracking)
        momentum_signal = 1.0 if outcome in [CompetitionOutcome.WIN, CompetitionOutcome.DECISIVE_WIN] else -1.0
        if outcome == CompetitionOutcome.DRAW:
            momentum_signal = 0.0
        
        self.win_loss_momentum = (
            self.momentum_factor * self.win_loss_momentum + 
            (1 - self.momentum_factor) * momentum_signal
        )
        
        # Create Result
        result = CompetitionResult(
            outcome=outcome,
            score_self=score_self,
            score_competitor=score_competitor,
            score_diff=score_diff,
            score_diff_relative=score_diff_relative,
            emotion_delta=emotion_delta,
            new_emotion=self.value
        )
        
        # Log
        self.history.append(self.value)
        self.outcome_history.append(outcome)
        
        return result
    
    def _determine_outcome(self, relative_diff: float) -> CompetitionOutcome:
        """Bestimme Competition Outcome"""
        if abs(relative_diff) < self.draw_threshold:
            return CompetitionOutcome.DRAW
        
        if relative_diff > 0:
            if relative_diff > self.decisive_threshold:
                return CompetitionOutcome.DECISIVE_WIN
            else:
                return CompetitionOutcome.WIN
        else:
            if abs(relative_diff) > self.decisive_threshold:
                return CompetitionOutcome.DECISIVE_LOSS
            else:
                return CompetitionOutcome.LOSS
    
    def _compute_emotion_delta(
        self,
        outcome: CompetitionOutcome,
        relative_diff: float
    ) -> float:
        """
        Berechne Emotion-Änderung basierend auf Outcome
        
        Strategie:
        ----------
        DECISIVE_WIN:  +0.15 (große Freude!)
        WIN:           +0.08 (Zufriedenheit)
        DRAW:          +0.00 (neutral, aber spannend)
        LOSS:          -0.08 (Frustration)
        DECISIVE_LOSS: -0.15 (starke Frustration, aber auch Determination!)
        
        Zusätzlich: Skaliere mit Margin (größerer Sieg = mehr Emotion)
        """
        base_deltas = {
            CompetitionOutcome.DECISIVE_WIN: 0.15,
            CompetitionOutcome.WIN: 0.08,
            CompetitionOutcome.DRAW: 0.00,
            CompetitionOutcome.LOSS: -0.08,
            CompetitionOutcome.DECISIVE_LOSS: -0.15,
        }
        
        delta = base_deltas[outcome]
        
        # Optional: Skaliere mit Margin (aber begrenzt)
        # Je größer der Sieg/Niederlage, desto stärker die Emotion
        margin_factor = float(np.tanh(abs(relative_diff) * 2))  # Soft scaling
        delta *= (0.7 + 0.3 * margin_factor)  # Zwischen 70% und 100%
        
        return delta
    
    def get_stats(self) -> Dict[str, float]:
        """Statistiken über Competitions"""
        total = self.total_competitions
        if total == 0:
            return {
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'draw_rate': 0.0,
                'win_loss_momentum': 0.0,
                'emotion': self.value
            }
        
        return {
            'win_rate': self.wins / total,
            'loss_rate': self.losses / total,
            'draw_rate': self.draws / total,
            'win_loss_momentum': self.win_loss_momentum,
            'emotion': self.value,
            'total_competitions': total
        }
    
    def get_competitive_mindset(self) -> str:
        """
        Bestimme aktuellen "Competitive Mindset" basierend auf Emotion + Momentum
        
        Returns:
            Mindset-String für Interpretation
        """
        emotion = self.value
        momentum = self.win_loss_momentum
        
        # Kombiniere Emotion und Momentum
        if emotion > 0.7 and momentum > 0.3:
            return "DOMINANT"        # Hohe Emotion + Win-Streak
        elif emotion > 0.6:
            return "CONFIDENT"       # Hohe Emotion
        elif emotion < 0.3 and momentum < -0.3:
            return "FRUSTRATED"      # Niedrige Emotion + Loss-Streak
        elif emotion < 0.4:
            return "DETERMINED"      # Niedrige Emotion → FOCUS!
        elif abs(momentum) < 0.2:
            return "BALANCED"        # Neutral
        else:
            return "ADAPTIVE"        # Irgendwo dazwischen


def create_competitive_config(task_type: str = "default") -> Dict:
    """
    Factory für Competition-Configs basierend auf Task
    
    Args:
        task_type: 'aggressive', 'balanced', 'conservative'
        
    Returns:
        Config-Dict für CompetitiveEmotionEngine
    """
    configs = {
        "aggressive": {
            "alpha": 0.25,              # Schnelle Reaktion
            "decisive_threshold": 0.15, # Niedrigere Schwelle
            "draw_threshold": 0.03,
            "momentum": 0.85,
        },
        "balanced": {
            "alpha": 0.15,              # Standard
            "decisive_threshold": 0.20,
            "draw_threshold": 0.05,
            "momentum": 0.90,
        },
        "conservative": {
            "alpha": 0.10,              # Langsame Reaktion
            "decisive_threshold": 0.25, # Höhere Schwelle
            "draw_threshold": 0.08,
            "momentum": 0.95,
        }
    }
    
    return configs.get(task_type, configs["balanced"])


# ==================== SELF-PLAY STRATEGY ====================

class SelfPlayCompetitor:
    """
    Competitor für Self-Play
    
    Strategien:
    -----------
    1. PAST_SELF: Spiele gegen vergangene Version (z.B. vor 100 Episodes)
    2. BEST_SELF: Spiele gegen beste gefundene Policy
    3. RANDOM_PAST: Zufällige vergangene Version
    """
    
    def __init__(self, strategy: str = "past_self", history_depth: int = 100):
        """
        Args:
            strategy: 'past_self', 'best_self', 'random_past'
            history_depth: Wie weit zurück für PAST_SELF
        """
        self.strategy = strategy
        self.history_depth = history_depth
        self.checkpoints = []  # Liste von (episode, state_dict, avg_score)
        self.best_checkpoint = None
    
    def save_checkpoint(self, episode: int, model_state_dict, avg_score: float):
        """Speichere Checkpoint für späteren Wettbewerb"""
        checkpoint = {
            'episode': episode,
            'state_dict': model_state_dict.copy(),
            'avg_score': avg_score
        }
        
        self.checkpoints.append(checkpoint)
        
        # Update Best
        if self.best_checkpoint is None or avg_score > self.best_checkpoint['avg_score']:
            self.best_checkpoint = checkpoint
    
    def get_competitor_model_state(self, current_episode: int):
        """
        Hole Competitor basierend auf Strategie
        
        Returns:
            model_state_dict oder None
        """
        if len(self.checkpoints) == 0:
            return None
        
        if self.strategy == "best_self":
            return self.best_checkpoint['state_dict'] if self.best_checkpoint else None
        
        elif self.strategy == "past_self":
            # Finde Checkpoint der ~history_depth Episodes zurückliegt
            target_episode = max(0, current_episode - self.history_depth)
            
            # Finde nächsten Checkpoint
            closest = min(self.checkpoints, key=lambda c: abs(c['episode'] - target_episode))
            return closest['state_dict']
        
        elif self.strategy == "random_past":
            # Zufälliger Checkpoint
            import random
            return random.choice(self.checkpoints)['state_dict']
        
        return None


if __name__ == "__main__":
    # Quick Test
    print("🏆 Competitive Emotion Engine - Test\n")
    
    engine = CompetitiveEmotionEngine(alpha=0.15)
    
    print("Simulating Competitions:\n")
    
    # Simulate: Win-Streak → dann Loss-Streak → dann Balance
    scenarios = [
        (100, 80, "Win 1"),
        (120, 85, "Win 2"),
        (110, 90, "Win 3"),
        (80, 100, "Loss 1"),
        (75, 95, "Loss 2"),
        (90, 92, "Draw"),
        (95, 93, "Win 4"),
    ]
    
    for i, (score_self, score_comp, label) in enumerate(scenarios):
        result = engine.compete(score_self, score_comp, episode=i)
        
        print(f"Episode {i}: {label}")
        print(f"  Outcome: {result.outcome.value}")
        print(f"  Scores: Self={score_self}, Comp={score_comp}")
        print(f"  Emotion: {result.new_emotion:.3f} (Δ{result.emotion_delta:+.3f})")
        print(f"  Mindset: {engine.get_competitive_mindset()}")
        print()
    
    stats = engine.get_stats()
    print("Final Stats:")
    for key, val in stats.items():
        print(f"  {key}: {val:.3f}")

