"""
Trading Emotion Engine für Emotion-Augmented Neural Networks
Speziell entwickelt für Trading-Emotionen und Markt-Sentiment
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from enum import Enum

class TradingEmotion(Enum):
    """Trading-spezifische Emotionen"""
    CONFIDENT = "confident"      # Gute Performance, steigende Gewinne
    CAUTIOUS = "cautious"        # Volatile Märkte, hohe Risiken
    FRUSTRATED = "frustrated"    # Verluste, schlechte Performance
    GREEDY = "greedy"           # Zu viele Gewinne, Risiko-Überschätzung
    FEARFUL = "fearful"         # Panik-Verkäufe, Risiko-Unterbewertung
    OPTIMISTIC = "optimistic"   # Positive Marktausblick
    PESSIMISTIC = "pessimistic" # Negative Marktausblick
    NEUTRAL = "neutral"         # Ausgewogene Marktlage

class TradingEmotionEngine:
    """
    Emotion Engine speziell für Trading-Anwendungen
    Passt Emotionen basierend auf Trading-Performance und Marktbedingungen an
    """
    
    def __init__(self, 
                 initial_emotion: TradingEmotion = TradingEmotion.NEUTRAL,
                 learning_rate: float = 0.01,
                 emotion_decay: float = 0.95,
                 volatility_threshold: float = 0.02):
        
        self.current_emotion = initial_emotion
        self.learning_rate = learning_rate
        self.emotion_decay = emotion_decay
        self.volatility_threshold = volatility_threshold
        
        # Emotion States (0-1 scale)
        self.emotion_states = {
            TradingEmotion.CONFIDENT: 0.5,
            TradingEmotion.CAUTIOUS: 0.5,
            TradingEmotion.FRUSTRATED: 0.0,
            TradingEmotion.GREEDY: 0.0,
            TradingEmotion.FEARFUL: 0.0,
            TradingEmotion.OPTIMISTIC: 0.5,
            TradingEmotion.PESSIMISTIC: 0.0,
            TradingEmotion.NEUTRAL: 1.0
        }
        
        # Trading Performance History
        self.performance_history = []
        self.volatility_history = []
        self.drawdown_history = []
        
        # Emotion Transition Weights
        self.transition_weights = self._initialize_transition_weights()
        
        # Market Sentiment Factors
        self.market_sentiment = {
            'trend': 0.0,      # -1 (bearish) to 1 (bullish)
            'volatility': 0.0,  # 0 (low) to 1 (high)
            'momentum': 0.0,    # -1 (negative) to 1 (positive)
            'volume': 0.0       # 0 (low) to 1 (high)
        }
        
    def _initialize_transition_weights(self) -> Dict:
        """Initialisiere Gewichte für Emotion-Übergänge"""
        return {
            # Von CONFIDENT
            (TradingEmotion.CONFIDENT, TradingEmotion.GREEDY): 0.3,
            (TradingEmotion.CONFIDENT, TradingEmotion.CAUTIOUS): 0.2,
            (TradingEmotion.CONFIDENT, TradingEmotion.FRUSTRATED): 0.1,
            
            # Von CAUTIOUS
            (TradingEmotion.CAUTIOUS, TradingEmotion.CONFIDENT): 0.3,
            (TradingEmotion.CAUTIOUS, TradingEmotion.FEARFUL): 0.2,
            (TradingEmotion.CAUTIOUS, TradingEmotion.NEUTRAL): 0.3,
            
            # Von FRUSTRATED
            (TradingEmotion.FRUSTRATED, TradingEmotion.FEARFUL): 0.4,
            (TradingEmotion.FRUSTRATED, TradingEmotion.CAUTIOUS): 0.3,
            (TradingEmotion.FRUSTRATED, TradingEmotion.NEUTRAL): 0.2,
            
            # Von GREEDY
            (TradingEmotion.GREEDY, TradingEmotion.FRUSTRATED): 0.4,
            (TradingEmotion.GREEDY, TradingEmotion.CAUTIOUS): 0.3,
            (TradingEmotion.GREEDY, TradingEmotion.CONFIDENT): 0.2,
            
            # Von FEARFUL
            (TradingEmotion.FEARFUL, TradingEmotion.CAUTIOUS): 0.4,
            (TradingEmotion.FEARFUL, TradingEmotion.NEUTRAL): 0.3,
            (TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED): 0.2,
            
            # Von NEUTRAL
            (TradingEmotion.NEUTRAL, TradingEmotion.CONFIDENT): 0.3,
            (TradingEmotion.NEUTRAL, TradingEmotion.CAUTIOUS): 0.3,
            (TradingEmotion.NEUTRAL, TradingEmotion.OPTIMISTIC): 0.2,
            (TradingEmotion.NEUTRAL, TradingEmotion.PESSIMISTIC): 0.2,
        }
    
    def update_market_sentiment(self, 
                              price_change: float,
                              volume_change: float,
                              volatility: float,
                              trend_strength: float):
        """Aktualisiere Markt-Sentiment basierend auf Marktdaten"""
        
        # Trend (basierend auf Preisänderung)
        self.market_sentiment['trend'] = np.tanh(price_change * 10)
        
        # Volatilität
        self.market_sentiment['volatility'] = np.clip(volatility / self.volatility_threshold, 0, 1)
        
        # Momentum (basierend auf Trend-Stärke)
        self.market_sentiment['momentum'] = np.tanh(trend_strength * 5)
        
        # Volume
        self.market_sentiment['volume'] = np.clip(volume_change, 0, 1)
        
        # Speichere in History
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
    
    def update_performance(self, 
                          portfolio_return: float,
                          trade_return: float,
                          drawdown: float,
                          win_rate: float):
        """Aktualisiere Emotion basierend auf Trading-Performance"""
        
        # Speichere Performance History
        self.performance_history.append({
            'portfolio_return': portfolio_return,
            'trade_return': trade_return,
            'drawdown': drawdown,
            'win_rate': win_rate
        })
        
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
        
        # Berechne Performance-Metriken
        recent_returns = [p['trade_return'] for p in self.performance_history[-10:]]
        avg_return = np.mean(recent_returns) if recent_returns else 0.0
        recent_win_rate = np.mean([p['win_rate'] for p in self.performance_history[-10:]])
        
        # Emotion Transition Logic
        self._update_emotion_states(avg_return, recent_win_rate, drawdown)
        self._transition_emotion()
        
        # Aktualisiere Emotion States
        self._decay_emotion_states()
    
    def _update_emotion_states(self, avg_return: float, win_rate: float, drawdown: float):
        """Aktualisiere Emotion States basierend auf Performance"""
        
        # CONFIDENT: Gute Performance, hohe Win Rate
        if avg_return > 0.01 and win_rate > 0.6:
            self.emotion_states[TradingEmotion.CONFIDENT] += self.learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.CONFIDENT] -= self.learning_rate * 0.1
        
        # CAUTIOUS: Hohe Volatilität oder Drawdown
        if self.market_sentiment['volatility'] > 0.7 or drawdown > 0.05:
            self.emotion_states[TradingEmotion.CAUTIOUS] += self.learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.CAUTIOUS] -= self.learning_rate * 0.1
        
        # FRUSTRATED: Schlechte Performance, niedrige Win Rate
        if avg_return < -0.01 or win_rate < 0.4:
            self.emotion_states[TradingEmotion.FRUSTRATED] += self.learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.FRUSTRATED] -= self.learning_rate * 0.1
        
        # GREEDY: Sehr gute Performance, hohe Gewinne
        if avg_return > 0.03 and win_rate > 0.7:
            self.emotion_states[TradingEmotion.GREEDY] += self.learning_rate * 0.3
        else:
            self.emotion_states[TradingEmotion.GREEDY] -= self.learning_rate * 0.1
        
        # FEARFUL: Große Verluste, hohe Drawdowns
        if drawdown > 0.1 or avg_return < -0.02:
            self.emotion_states[TradingEmotion.FEARFUL] += self.learning_rate * 0.4
        else:
            self.emotion_states[TradingEmotion.FEARFUL] -= self.learning_rate * 0.1
        
        # OPTIMISTIC: Positive Marktausblick
        if (self.market_sentiment['trend'] > 0.3 and 
            self.market_sentiment['momentum'] > 0.2):
            self.emotion_states[TradingEmotion.OPTIMISTIC] += self.learning_rate * 0.2
        else:
            self.emotion_states[TradingEmotion.OPTIMISTIC] -= self.learning_rate * 0.1
        
        # PESSIMISTIC: Negative Marktausblick
        if (self.market_sentiment['trend'] < -0.3 and 
            self.market_sentiment['momentum'] < -0.2):
            self.emotion_states[TradingEmotion.PESSIMISTIC] += self.learning_rate * 0.2
        else:
            self.emotion_states[TradingEmotion.PESSIMISTIC] -= self.learning_rate * 0.1
        
        # NEUTRAL: Ausgewogene Bedingungen
        if (abs(avg_return) < 0.005 and 
            0.4 < win_rate < 0.6 and 
            self.market_sentiment['volatility'] < 0.5):
            self.emotion_states[TradingEmotion.NEUTRAL] += self.learning_rate * 0.2
        else:
            self.emotion_states[TradingEmotion.NEUTRAL] -= self.learning_rate * 0.1
        
        # Clamp all values to [0, 1]
        for emotion in self.emotion_states:
            self.emotion_states[emotion] = np.clip(self.emotion_states[emotion], 0, 1)
    
    def _transition_emotion(self):
        """Führe Emotion-Übergang basierend auf aktuellen States durch"""
        
        # Finde Emotion mit höchstem State
        max_emotion = max(self.emotion_states.items(), key=lambda x: x[1])
        
        # Prüfe ob Transition sinnvoll ist
        if max_emotion[1] > 0.7 and max_emotion[0] != self.current_emotion:
            # Prüfe Transition Weight
            transition_key = (self.current_emotion, max_emotion[0])
            if transition_key in self.transition_weights:
                transition_prob = self.transition_weights[transition_key]
                if np.random.random() < transition_prob:
                    self.current_emotion = max_emotion[0]
    
    def _decay_emotion_states(self):
        """Lasse Emotion States langsam abklingen"""
        for emotion in self.emotion_states:
            self.emotion_states[emotion] *= self.emotion_decay
    
    def get_emotion_vector(self) -> np.ndarray:
        """Erstelle Emotion-Vektor für Neural Network"""
        emotion_vector = np.array([
            self.emotion_states[TradingEmotion.CONFIDENT],
            self.emotion_states[TradingEmotion.CAUTIOUS],
            self.emotion_states[TradingEmotion.FRUSTRATED],
            self.emotion_states[TradingEmotion.GREEDY],
            self.emotion_states[TradingEmotion.FEARFUL],
            self.emotion_states[TradingEmotion.OPTIMISTIC],
            self.emotion_states[TradingEmotion.PESSIMISTIC],
            self.emotion_states[TradingEmotion.NEUTRAL]
        ], dtype=np.float32)
        
        return emotion_vector
    
    def get_emotion_modifier(self) -> float:
        """Berechne Emotion-Modifier für Trading-Entscheidungen"""
        
        emotion_modifiers = {
            TradingEmotion.CONFIDENT: 1.2,    # Mehr Risiko
            TradingEmotion.CAUTIOUS: 0.7,     # Weniger Risiko
            TradingEmotion.FRUSTRATED: 0.5,   # Sehr wenig Risiko
            TradingEmotion.GREEDY: 1.5,       # Viel Risiko
            TradingEmotion.FEARFUL: 0.3,      # Minimales Risiko
            TradingEmotion.OPTIMISTIC: 1.1,   # Etwas mehr Risiko
            TradingEmotion.PESSIMISTIC: 0.6,  # Weniger Risiko
            TradingEmotion.NEUTRAL: 1.0       # Normales Risiko
        }
        
        return emotion_modifiers.get(self.current_emotion, 1.0)
    
    def get_risk_tolerance(self) -> float:
        """Berechne aktuelle Risikotoleranz basierend auf Emotion"""
        
        risk_tolerances = {
            TradingEmotion.CONFIDENT: 0.8,
            TradingEmotion.CAUTIOUS: 0.4,
            TradingEmotion.FRUSTRATED: 0.2,
            TradingEmotion.GREEDY: 0.9,
            TradingEmotion.FEARFUL: 0.1,
            TradingEmotion.OPTIMISTIC: 0.6,
            TradingEmotion.PESSIMISTIC: 0.3,
            TradingEmotion.NEUTRAL: 0.5
        }
        
        return risk_tolerances.get(self.current_emotion, 0.5)
    
    def get_position_sizing_modifier(self) -> float:
        """Berechne Position Sizing Modifier basierend auf Emotion"""
        
        sizing_modifiers = {
            TradingEmotion.CONFIDENT: 1.3,
            TradingEmotion.CAUTIOUS: 0.6,
            TradingEmotion.FRUSTRATED: 0.3,
            TradingEmotion.GREEDY: 1.6,
            TradingEmotion.FEARFUL: 0.2,
            TradingEmotion.OPTIMISTIC: 1.1,
            TradingEmotion.PESSIMISTIC: 0.5,
            TradingEmotion.NEUTRAL: 1.0
        }
        
        return sizing_modifiers.get(self.current_emotion, 1.0)
    
    def get_emotion_info(self) -> Dict:
        """Erstelle Info-Dictionary mit aktueller Emotion und Metriken"""
        return {
            'current_emotion': self.current_emotion.value,
            'emotion_states': {k.value: v for k, v in self.emotion_states.items()},
            'market_sentiment': self.market_sentiment.copy(),
            'emotion_modifier': self.get_emotion_modifier(),
            'risk_tolerance': self.get_risk_tolerance(),
            'position_sizing_modifier': self.get_position_sizing_modifier(),
            'performance_history_length': len(self.performance_history),
            'volatility_history_length': len(self.volatility_history)
        }
    
    def reset(self):
        """Reset Emotion Engine"""
        self.current_emotion = TradingEmotion.NEUTRAL
        self.emotion_states = {
            TradingEmotion.CONFIDENT: 0.5,
            TradingEmotion.CAUTIOUS: 0.5,
            TradingEmotion.FRUSTRATED: 0.0,
            TradingEmotion.GREEDY: 0.0,
            TradingEmotion.FEARFUL: 0.0,
            TradingEmotion.OPTIMISTIC: 0.5,
            TradingEmotion.PESSIMISTIC: 0.0,
            TradingEmotion.NEUTRAL: 1.0
        }
        self.performance_history = []
        self.volatility_history = []
        self.drawdown_history = []
        self.market_sentiment = {
            'trend': 0.0,
            'volatility': 0.0,
            'momentum': 0.0,
            'volume': 0.0
        }


if __name__ == "__main__":
    # Test das Trading Emotion Engine
    print("🚀 Teste Trading Emotion Engine...")
    
    # Erstelle Emotion Engine
    emotion_engine = TradingEmotionEngine()
    
    print(f"Initial Emotion: {emotion_engine.current_emotion.value}")
    print(f"Initial Risk Tolerance: {emotion_engine.get_risk_tolerance():.2f}")
    
    # Simuliere verschiedene Trading-Szenarien
    scenarios = [
        {"name": "Gute Performance", "return": 0.02, "win_rate": 0.7, "drawdown": 0.01},
        {"name": "Schlechte Performance", "return": -0.015, "win_rate": 0.3, "drawdown": 0.05},
        {"name": "Volatile Märkte", "return": 0.005, "win_rate": 0.5, "drawdown": 0.03},
        {"name": "Sehr gute Performance", "return": 0.04, "win_rate": 0.8, "drawdown": 0.005},
        {"name": "Große Verluste", "return": -0.03, "win_rate": 0.2, "drawdown": 0.12}
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Szenario {i+1}: {scenario['name']} ---")
        
        # Update Market Sentiment
        emotion_engine.update_market_sentiment(
            price_change=scenario['return'],
            volume_change=0.1,
            volatility=0.02,
            trend_strength=scenario['return'] * 2
        )
        
        # Update Performance
        emotion_engine.update_performance(
            portfolio_return=scenario['return'],
            trade_return=scenario['return'],
            drawdown=scenario['drawdown'],
            win_rate=scenario['win_rate']
        )
        
        # Zeige Ergebnisse
        print(f"Emotion: {emotion_engine.current_emotion.value}")
        print(f"Risk Tolerance: {emotion_engine.get_risk_tolerance():.2f}")
        print(f"Position Sizing: {emotion_engine.get_position_sizing_modifier():.2f}")
        print(f"Emotion Modifier: {emotion_engine.get_emotion_modifier():.2f}")
        
        # Zeige Emotion States
        emotion_vector = emotion_engine.get_emotion_vector()
        print(f"Emotion Vector: {emotion_vector}")
    
    print("\n✅ Trading Emotion Engine Test abgeschlossen!")
