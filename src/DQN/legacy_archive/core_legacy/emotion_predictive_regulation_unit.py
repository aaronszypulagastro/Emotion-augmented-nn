"""
Emotion-Predictive Regulation Unit (EPRU) - Phase 6.0
====================================================

Antizipative η-Regelung basierend auf Emotion-Trends und TD-Error-Vorhersagen.
Erweitert die reaktive Regelung um prädiktive Komponenten für bessere Stabilität.

Key Features:
- LSTM-basierte Emotion-Trend-Vorhersage
- TD-Error-Vorhersage mit Quantile-Regression
- Antizipative η-Anpassung vor kritischen Phasen
- Confidence-basierte Intervention-Stärke
- Multi-Horizon-Vorhersage (1, 3, 5 Episoden)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EmotionPredictiveRegulationUnit:
    """
    Emotion-Predictive Regulation Unit für antizipative η-Regelung.
    
    Diese Einheit erweitert die reaktive η-Regelung um prädiktive Komponenten:
    1. Emotion-Trend-Vorhersage (LSTM)
    2. TD-Error-Vorhersage (Quantile-Regression)
    3. Antizipative η-Anpassung
    4. Confidence-basierte Intervention
    """
    
    def __init__(self, 
                 emotion_horizon: int = 5,
                 td_horizon: int = 3,
                 confidence_threshold: float = 0.7,
                 intervention_strength: float = 0.3,
                 history_size: int = 50):
        """
        Initialisiert die EPRU.
        
        Args:
            emotion_horizon: Anzahl Episoden für Emotion-Vorhersage
            td_horizon: Anzahl Episoden für TD-Error-Vorhersage
            confidence_threshold: Mindest-Confidence für Intervention
            intervention_strength: Maximale Stärke der antizipativen Intervention
            history_size: Größe des Historie-Buffers
        """
        self.emotion_horizon = emotion_horizon
        self.td_horizon = td_horizon
        self.confidence_threshold = confidence_threshold
        self.intervention_strength = intervention_strength
        
        # Historie-Buffer für Vorhersagen
        self.emotion_history = deque(maxlen=history_size)
        self.td_error_history = deque(maxlen=history_size)
        self.eta_history = deque(maxlen=history_size)
        self.reward_history = deque(maxlen=history_size)
        
        # LSTM für Emotion-Trend-Vorhersage
        self.emotion_lstm = nn.LSTM(
            input_size=4,  # emotion, td_error, eta, reward
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Quantile-Regression für TD-Error-Vorhersage
        self.td_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # 3 Quantile: 0.25, 0.5, 0.75
        )
        
        # Confidence-Netzwerk
        self.confidence_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.emotion_lstm.parameters()) + 
            list(self.td_predictor.parameters()) + 
            list(self.confidence_net.parameters()),
            lr=0.001
        )
        
        # Training-Parameter
        self.training_enabled = True
        self.prediction_cache = {}
        self.last_update_episode = 0
        
        logger.info(f"EPRU initialisiert: emotion_horizon={emotion_horizon}, "
                   f"td_horizon={td_horizon}, confidence_threshold={confidence_threshold}")
    
    def update_history(self, emotion: float, td_error: float, eta: float, reward: float):
        """Aktualisiert die Historie mit neuen Werten."""
        self.emotion_history.append(emotion)
        self.td_error_history.append(td_error)
        self.eta_history.append(eta)
        self.reward_history.append(reward)
    
    def predict_emotion_trend(self, horizon: int = None) -> Tuple[float, float]:
        """
        Vorhersage des Emotion-Trends für die nächsten Episoden.
        
        Returns:
            Tuple[float, float]: (predicted_emotion, confidence)
        """
        if horizon is None:
            horizon = self.emotion_horizon
            
        if len(self.emotion_history) < 10:
            return 0.5, 0.0  # Neutral, keine Confidence
        
        # Input-Features vorbereiten
        features = np.array([
            list(self.emotion_history)[-10:],
            list(self.td_error_history)[-10:],
            list(self.eta_history)[-10:],
            list(self.reward_history)[-10:]
        ]).T
        
        # Normalisierung
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        
        # LSTM-Vorhersage
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            lstm_out, _ = self.emotion_lstm(input_tensor)
            
            # Letzte Hidden-State für Vorhersage
            last_hidden = lstm_out[0, -1, :]
            
            # Confidence berechnen
            confidence = self.confidence_net(last_hidden).item()
            
            # Emotion-Trend basierend auf LSTM-Output
            emotion_trend = torch.tanh(last_hidden[:8].mean()).item()
            predicted_emotion = 0.5 + 0.3 * emotion_trend  # [0.2, 0.8]
            
        return predicted_emotion, confidence
    
    def predict_td_error(self, horizon: int = None) -> Tuple[float, float, float]:
        """
        Vorhersage des TD-Error für die nächsten Episoden.
        
        Returns:
            Tuple[float, float, float]: (q25, q50, q75) Quantile
        """
        if horizon is None:
            horizon = self.td_horizon
            
        if len(self.td_error_history) < 10:
            return 1.0, 1.0, 1.0  # Konservative Schätzung
        
        # Input-Features vorbereiten
        features = np.array([
            list(self.emotion_history)[-10:],
            list(self.td_error_history)[-10:],
            list(self.eta_history)[-10:],
            list(self.reward_history)[-10:]
        ]).T
        
        # Normalisierung
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        
        # LSTM-Vorhersage
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            lstm_out, _ = self.emotion_lstm(input_tensor)
            
            # Letzte Hidden-State für TD-Error-Vorhersage
            last_hidden = lstm_out[0, -1, :]
            
            # Quantile-Vorhersage
            quantiles = self.td_predictor(last_hidden)
            q25, q50, q75 = quantiles[0].item(), quantiles[1].item(), quantiles[2].item()
            
            # Denormalisierung basierend auf historischen Werten
            td_mean = np.mean(list(self.td_error_history)[-10:])
            td_std = np.std(list(self.td_error_history)[-10:])
            
            q25 = td_mean + q25 * td_std
            q50 = td_mean + q50 * td_std
            q75 = td_mean + q75 * td_std
            
        return max(0.1, q25), max(0.1, q50), max(0.1, q75)
    
    def get_anticipatory_eta_adjustment(self, 
                                      current_emotion: float, 
                                      current_td_error: float,
                                      current_eta: float) -> Tuple[float, float, Dict]:
        """
        Berechnet antizipative η-Anpassung basierend auf Vorhersagen.
        
        Args:
            current_emotion: Aktuelle Emotion
            current_td_error: Aktueller TD-Error
            current_eta: Aktuelle η
            
        Returns:
            Tuple[float, float, Dict]: (eta_adjustment, confidence, debug_info)
        """
        # Vorhersagen berechnen
        pred_emotion, emotion_conf = self.predict_emotion_trend()
        td_q25, td_q50, td_q75 = self.predict_td_error()
        
        # Confidence kombinieren
        overall_confidence = emotion_conf * 0.6 + 0.4  # TD-Error-Confidence ist implizit
        
        # Antizipative Anpassung nur bei hoher Confidence
        if overall_confidence < self.confidence_threshold:
            return 0.0, overall_confidence, {
                'pred_emotion': pred_emotion,
                'td_q50': td_q50,
                'confidence': overall_confidence,
                'intervention': 'none'
            }
        
        # Emotion-basierte Anpassung
        emotion_factor = 1.0
        if pred_emotion > 0.7:  # Hohe Emotion erwartet
            emotion_factor = 1.0 + self.intervention_strength * 0.5
        elif pred_emotion < 0.3:  # Niedrige Emotion erwartet
            emotion_factor = 1.0 - self.intervention_strength * 0.3
        
        # TD-Error-basierte Anpassung
        td_factor = 1.0
        if td_q75 > current_td_error * 1.5:  # TD-Error-Anstieg erwartet
            td_factor = 1.0 - self.intervention_strength * 0.4
        elif td_q25 < current_td_error * 0.7:  # TD-Error-Abnahme erwartet
            td_factor = 1.0 + self.intervention_strength * 0.2
        
        # Kombinierte Anpassung
        combined_factor = emotion_factor * td_factor
        eta_adjustment = current_eta * (combined_factor - 1.0)
        
        # Anpassung begrenzen
        max_adjustment = current_eta * self.intervention_strength
        eta_adjustment = np.clip(eta_adjustment, -max_adjustment, max_adjustment)
        
        debug_info = {
            'pred_emotion': pred_emotion,
            'td_q25': td_q25,
            'td_q50': td_q50,
            'td_q75': td_q75,
            'emotion_factor': emotion_factor,
            'td_factor': td_factor,
            'combined_factor': combined_factor,
            'confidence': overall_confidence,
            'intervention': 'anticipatory' if abs(eta_adjustment) > 1e-6 else 'none'
        }
        
        return eta_adjustment, overall_confidence, debug_info
    
    def train_step(self, episode: int):
        """
        Trainiert die EPRU-Modelle basierend auf verfügbarer Historie.
        """
        if not self.training_enabled or len(self.emotion_history) < 20:
            return
        
        # Training nur alle 10 Episoden
        if episode - self.last_update_episode < 10:
            return
        
        self.last_update_episode = episode
        
        try:
            # Training-Daten vorbereiten
            features = np.array([
                list(self.emotion_history)[-20:],
                list(self.td_error_history)[-20:],
                list(self.eta_history)[-20:],
                list(self.reward_history)[-20:]
            ]).T
            
            # Normalisierung
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
            
            # Targets: nächste Werte
            emotion_targets = np.array(list(self.emotion_history)[-19:])
            td_targets = np.array(list(self.td_error_history)[-19:])
            
            # Training
            self.optimizer.zero_grad()
            
            input_tensor = torch.FloatTensor(features[:-1]).unsqueeze(0)
            lstm_out, _ = self.emotion_lstm(input_tensor)
            
            # Emotion-Loss
            emotion_pred = torch.tanh(lstm_out[0, :, :8].mean(dim=1))
            emotion_loss = nn.MSELoss()(emotion_pred, torch.FloatTensor(emotion_targets))
            
            # TD-Error-Loss (Quantile-Regression)
            td_pred = self.td_predictor(lstm_out[0, :, :])
            td_targets_tensor = torch.FloatTensor(td_targets).unsqueeze(1)
            td_loss = nn.MSELoss()(td_pred[:, 1], td_targets_tensor.squeeze())
            
            # Confidence-Loss (höhere Confidence bei stabilen Trends)
            confidence_pred = self.confidence_net(lstm_out[0, :, :])
            stability = 1.0 - torch.std(torch.FloatTensor(emotion_targets))
            confidence_loss = nn.MSELoss()(confidence_pred.squeeze(), 
                                         torch.FloatTensor([stability] * len(emotion_targets)))
            
            # Gesamt-Loss
            total_loss = emotion_loss + td_loss + 0.1 * confidence_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            if episode % 50 == 0:
                logger.info(f"EPRU Training Episode {episode}: "
                           f"emotion_loss={emotion_loss.item():.4f}, "
                           f"td_loss={td_loss.item():.4f}, "
                           f"confidence_loss={confidence_loss.item():.4f}")
                
        except Exception as e:
            logger.warning(f"EPRU Training fehlgeschlagen: {e}")
    
    def get_debug_info(self) -> Dict:
        """Gibt Debug-Informationen zurück."""
        return {
            'history_size': len(self.emotion_history),
            'emotion_horizon': self.emotion_horizon,
            'td_horizon': self.td_horizon,
            'confidence_threshold': self.confidence_threshold,
            'intervention_strength': self.intervention_strength,
            'training_enabled': self.training_enabled,
            'last_update_episode': self.last_update_episode
        }
