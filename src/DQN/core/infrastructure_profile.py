"""
Regional Infrastructure Profile - Phase 8.2
============================================

KONZEPT: Simuliert regionale Produktionsbedingungen
----------------------------------------------------

Real-World Inspiration:
- China: Räumliche Effizienz (Shenzhen-Cluster), hohe Automation
- Germany: Hohe Qualität, aber längere Lieferketten
- USA: Mittelweg, hohe Tech aber fragmentiert

Diese Bedingungen beeinflussen:
1. Reward Propagation (Feedback Loop Speed)
2. Observation Noise (Sensor/Production Variability)
3. Learning Efficiency (Automation Level)
4. Error Tolerance (Quality Standards)

Author: Phase 8.2 - Infrastructure Meta-Learning
Date: 2025-10-17
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RegionType(Enum):
    """Unterstützte Regionen"""
    CHINA = "China"
    GERMANY = "Germany"
    USA = "USA"
    BRAZIL = "Brazil"
    INDIA = "India"


@dataclass
class InfrastructureMetrics:
    """Metriken des aktuellen Infrastructure States"""
    region: str
    loop_speed: float           # [0, 1] - Feedback Geschwindigkeit
    automation: float           # [0, 1] - Automatisierungsgrad
    error_tolerance: float      # [0, 1] - Fehlertoleranz
    reward_delay_steps: int     # Schritte Delay für Reward
    observation_noise_std: float # Noise-Level in Observations
    learning_rate_modifier: float # Multiplikator für LR
    exploration_modifier: float   # Multiplikator für Epsilon


class InfrastructureProfile:
    """
    Simuliert regionale Produktionsbedingungen
    
    Kernprinzip:
    ------------
    Verschiedene Regionen haben unterschiedliche:
    - Feedback-Geschwindigkeit (Lieferketten-Effizienz)
    - Automatisierung (Roboter-Dichte, Prozess-Qualität)
    - Fehlertoleranz (Quality Standards, Regulations)
    
    Diese beeinflussen wie der Agent lernt!
    
    Example:
    --------
    >>> china = InfrastructureProfile("China")
    >>> metrics = china.get_metrics()
    >>> print(f"Loop Speed: {metrics.loop_speed}")  # 0.1 (sehr schnell!)
    
    >>> # Modulate Reward
    >>> delayed_reward = china.modulate_reward(reward=10.0, step=5)
    >>> # Reward kommt fast sofort (niedrige Latenz)
    
    >>> # Modulate Observation
    >>> noisy_obs = china.modulate_observation(observation)
    >>> # Wenig Noise (hohe Automation)
    """
    
    # Pre-defined Regional Profiles
    # Basierend auf realen Daten: Roboter-Dichte, Supply Chain Index, etc.
    REGIONAL_PROFILES = {
        "China": {
            "loop_speed": 0.1,        # Sehr schnell (Shenzhen-Cluster)
            "automation": 0.9,        # Sehr hoch (Roboter-Dichte ~200/10k)
            "error_tolerance": 0.05,  # Niedrig (schnell iterieren)
            "description": "High-speed production cluster, rapid feedback"
        },
        "Germany": {
            "loop_speed": 0.5,        # Langsam (Qualitätskontrolle)
            "automation": 0.7,        # Hoch (Industrie 4.0)
            "error_tolerance": 0.10,  # Mittel (hohe Standards)
            "description": "Quality-focused, robust processes"
        },
        "USA": {
            "loop_speed": 0.3,        # Mittel
            "automation": 0.8,        # Sehr hoch (aber fragmentiert)
            "error_tolerance": 0.08,  # Niedrig-Mittel
            "description": "High-tech but geographically distributed"
        },
        "Brazil": {
            "loop_speed": 0.6,        # Langsam (Infrastruktur-Challenges)
            "automation": 0.5,        # Mittel
            "error_tolerance": 0.15,  # Hoch (weniger strenge Regs)
            "description": "Emerging market, flexible processes"
        },
        "India": {
            "loop_speed": 0.4,        # Mittel-langsam
            "automation": 0.6,        # Mittel (stark wachsend)
            "error_tolerance": 0.12,  # Mittel
            "description": "Growing automation, mixed infrastructure"
        }
    }
    
    def __init__(
        self,
        region: str = "China",
        custom_profile: Optional[Dict] = None
    ):
        """
        Initialize Infrastructure Profile
        
        Args:
            region: Name der Region (siehe REGIONAL_PROFILES)
            custom_profile: Optional custom profile dict mit keys:
                           loop_speed, automation, error_tolerance
        """
        self.region = region
        
        if custom_profile:
            self.profile = custom_profile
        elif region in self.REGIONAL_PROFILES:
            self.profile = self.REGIONAL_PROFILES[region].copy()
        else:
            raise ValueError(f"Unknown region: {region}. Use one of {list(self.REGIONAL_PROFILES.keys())}")
        
        # Extract core parameters
        self.loop_speed = float(self.profile['loop_speed'])
        self.automation = float(self.profile['automation'])
        self.error_tolerance = float(self.profile['error_tolerance'])
        
        # Compute derived parameters
        self._compute_derived_parameters()
        
        # History
        self.reward_buffer = []  # For delayed rewards
        self.step_count = 0
    
    def _compute_derived_parameters(self):
        """Berechne abgeleitete Parameter aus Kern-Profil"""
        
        # Reward Delay: Höhere loop_speed → mehr Delay
        # loop_speed 0.1 → 0-1 steps delay
        # loop_speed 0.5 → 2-3 steps delay
        self.reward_delay_steps = int(self.loop_speed * 5)
        
        # Observation Noise: Niedrigere automation → mehr Noise
        # automation 0.9 → std 0.01 (sehr wenig Noise)
        # automation 0.5 → std 0.05 (moderater Noise)
        self.observation_noise_std = 0.1 * (1.0 - self.automation)
        
        # Learning Rate Modifier: Höhere automation → effizienter lernen
        # automation 0.9 → LR * 1.1
        # automation 0.5 → LR * 0.9
        self.lr_modifier = 0.8 + 0.4 * self.automation  # [0.8, 1.2]
        
        # Exploration Modifier: Höhere error_tolerance → mehr Exploration
        # error_tolerance 0.05 → ε * 0.9 (weniger Exploration)
        # error_tolerance 0.15 → ε * 1.1 (mehr Exploration)
        self.exploration_modifier = 0.8 + 2.0 * self.error_tolerance  # [~0.9, ~1.1]
    
    def modulate_reward(
        self,
        reward: float,
        step: int,
        flush: bool = False
    ) -> float:
        """
        Moduliere Reward basierend auf Feedback-Loop-Geschwindigkeit
        
        Konzept:
        --------
        In Regionen mit langen Lieferketten (hohe loop_speed) kommt
        Feedback verzögert an → Reward wird gepuffert und später ausgegeben
        
        Args:
            reward: Original Reward
            step: Aktueller Step
            flush: Wenn True, gebe alle gepufferten Rewards zurück
            
        Returns:
            Delayed/modulated reward
        """
        self.step_count += 1
        
        if self.reward_delay_steps == 0:
            # Kein Delay (China-style)
            return reward
        
        # Buffer Reward
        self.reward_buffer.append(reward)
        
        # Flush wenn requested oder genug gepuffert
        if flush or len(self.reward_buffer) >= self.reward_delay_steps:
            # Gebe ältesten Reward zurück
            if len(self.reward_buffer) > 0:
                delayed_reward = self.reward_buffer.pop(0)
            else:
                delayed_reward = 0.0
        else:
            # Noch im Buffer
            delayed_reward = 0.0
        
        return delayed_reward
    
    def modulate_observation(
        self,
        observation: np.ndarray
    ) -> np.ndarray:
        """
        Füge Noise zu Observation basierend auf Automatisierungsgrad
        
        Konzept:
        --------
        Niedrigere Automation → mehr Variabilität in Sensoren/Prozessen
        Höhere Automation → präzisere Messungen
        
        Args:
            observation: Original observation array
            
        Returns:
            Noisy observation
        """
        if self.observation_noise_std == 0.0:
            return observation
        
        noise = np.random.normal(
            loc=0.0,
            scale=self.observation_noise_std,
            size=observation.shape
        )
        
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def modulate_learning_rate(
        self,
        base_lr: float
    ) -> float:
        """
        Modulate Learning Rate basierend auf Automation
        
        Konzept:
        --------
        Höhere Automation → präzisere Daten → effizienter lernen → höhere LR
        Niedrigere Automation → mehr Noise → vorsichtiger lernen → niedrigere LR
        
        Args:
            base_lr: Basis Learning Rate
            
        Returns:
            Modulated learning rate
        """
        return base_lr * self.lr_modifier
    
    def modulate_exploration(
        self,
        base_epsilon: float
    ) -> float:
        """
        Modulate Exploration (Epsilon) basierend auf Error Tolerance
        
        Konzept:
        --------
        Hohe Error Tolerance → mehr Raum für Exploration
        Niedrige Error Tolerance → konservativere Policy
        
        Args:
            base_epsilon: Basis Epsilon
            
        Returns:
            Modulated epsilon
        """
        return base_epsilon * self.exploration_modifier
    
    def get_metrics(self) -> InfrastructureMetrics:
        """Hole alle aktuellen Metriken"""
        return InfrastructureMetrics(
            region=self.region,
            loop_speed=self.loop_speed,
            automation=self.automation,
            error_tolerance=self.error_tolerance,
            reward_delay_steps=self.reward_delay_steps,
            observation_noise_std=self.observation_noise_std,
            learning_rate_modifier=self.lr_modifier,
            exploration_modifier=self.exploration_modifier
        )
    
    def reset(self):
        """Reset internal state (für neue Episode)"""
        self.reward_buffer = []
        self.step_count = 0
    
    def get_infrastructure_emotion_factor(self) -> float:
        """
        Berechne Emotion-Faktor basierend auf Infrastructure
        
        Konzept:
        --------
        In effizienten Regionen (niedriges loop_speed, hohe automation)
        kann der Agent "optimistischer" sein → höherer Emotion-Faktor
        
        Returns:
            Factor in [0.8, 1.2] für Emotion-Modulation
        """
        # Combine loop_speed (invert) and automation
        efficiency_score = (1.0 - self.loop_speed) * 0.5 + self.automation * 0.5
        
        # Map to [0.8, 1.2]
        factor = 0.8 + 0.4 * efficiency_score
        
        return factor
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"InfrastructureProfile({self.region})\n"
            f"  Loop Speed: {self.loop_speed:.2f}\n"
            f"  Automation: {self.automation:.2f}\n"
            f"  Error Tolerance: {self.error_tolerance:.2f}\n"
            f"  → Reward Delay: {self.reward_delay_steps} steps\n"
            f"  → Obs Noise: {self.observation_noise_std:.3f}\n"
            f"  → LR Modifier: {self.lr_modifier:.2f}x\n"
            f"  → Epsilon Modifier: {self.exploration_modifier:.2f}x"
        )


def create_all_profiles() -> Dict[str, InfrastructureProfile]:
    """
    Factory: Erstelle alle verfügbaren Regional-Profiles
    
    Returns:
        Dict mapping region name → InfrastructureProfile
    """
    profiles = {}
    for region_name in InfrastructureProfile.REGIONAL_PROFILES.keys():
        profiles[region_name] = InfrastructureProfile(region_name)
    return profiles


def compare_profiles():
    """Vergleiche alle Profile - für Debugging/Exploration"""
    print("\n" + "="*70)
    print("REGIONAL INFRASTRUCTURE PROFILES - COMPARISON")
    print("="*70 + "\n")
    
    profiles = create_all_profiles()
    
    for region_name, profile in profiles.items():
        print(profile)
        print()


if __name__ == "__main__":
    # Quick Test
    print("Testing Infrastructure Profiles...\n")
    
    # Test China Profile
    china = InfrastructureProfile("China")
    print(china)
    print()
    
    # Test reward delay
    print("Testing Reward Delay (China):")
    for step in range(5):
        reward = 10.0 if step == 0 else 0.0
        delayed = china.modulate_reward(reward, step)
        print(f"  Step {step}: Input={reward:.1f}, Delayed={delayed:.1f}")
    print()
    
    # Test observation noise
    print("Testing Observation Noise:")
    obs = np.array([1.0, 2.0, 3.0])
    noisy = china.modulate_observation(obs)
    print(f"  Original: {obs}")
    print(f"  Noisy:    {noisy}")
    print()
    
    # Compare all
    compare_profiles()





