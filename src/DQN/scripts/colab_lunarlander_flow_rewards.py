"""
Google Colab - FLOW REWARDS + Self-Correction + Attention + Enhanced Emotion Engine + LunarLander
=================================================================================================

Phase 5.3: Flow Rewards Implementation
- Belohnt flüssige, logische Entscheidungssequenzen
- Self-Correction Mechanism Integration
- Attention Mechanisms Integration
- Enhanced Emotion Engine Integration
- LunarLander-v3 Optimization

Author: Enhanced Meta-Learning Project
Date: 2025-10-17
"""

# =============================================================================
# DEPENDENCIES (Run this cell first)
# =============================================================================

!pip install torch torchvision torchaudio
!pip install gymnasium[box2d]
!pip install pandas matplotlib seaborn tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple
import random
from tqdm import tqdm
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Test LunarLander
print("🧪 Testing LunarLander-v3...")
try:
    env = gym.make('LunarLander-v3')
    print("✅ LunarLander-v3 works!")
    env.close()
except Exception as e:
    print(f"❌ Error: {e}")

# =============================================================================
# FLOW REWARD ENGINE
# =============================================================================

class FlowRewardEngine:
    def __init__(self, window_size=10, consistency_weight=0.4, smoothness_weight=0.3, logic_weight=0.3):
        self.window_size = window_size
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.logic_weight = logic_weight
        
        # Action and state history
        self.action_history = deque(maxlen=window_size)
        self.state_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        
        # Flow metrics
        self.flow_metrics = {
            'consistency': deque(maxlen=100),
            'smoothness': deque(maxlen=100),
            'logic': deque(maxlen=100),
            'flow_reward': deque(maxlen=100)
        }
        
        # Action transition patterns for LunarLander
        self.action_patterns = {
            'smooth_landing': [0, 0, 0],  # Main engine only
            'stabilize': [1, 1, 1],       # Side engines only
            'correct_course': [2, 3, 2],  # Left-right-left
            'gentle_approach': [0, 1, 0], # Main-side-main
            'efficient_landing': [0, 2, 0] # Main-left-main
        }
        
        # State transition patterns
        self.state_patterns = {
            'stable_descent': {'velocity_y': 'decreasing', 'angle': 'stable'},
            'smooth_approach': {'position_x': 'stable', 'velocity_x': 'decreasing'},
            'controlled_landing': {'angle': 'stable', 'velocity_y': 'controlled'}
        }
        
    def calculate_flow_reward(self, state, action, next_state, reward):
        """Berechnet Flow Reward basierend auf Entscheidungssequenz"""
        # Update history
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward)
        
        if len(self.action_history) < self.window_size:
            return 0.0
        
        # Calculate flow metrics
        consistency_score = self.calculate_consistency()
        smoothness_score = self.calculate_smoothness()
        logic_score = self.calculate_logic(state, action, next_state)
        
        # Combine flow metrics
        flow_reward = (
            consistency_score * self.consistency_weight +
            smoothness_score * self.smoothness_weight +
            logic_score * self.logic_weight
        )
        
        # Store metrics
        self.flow_metrics['consistency'].append(consistency_score)
        self.flow_metrics['smoothness'].append(smoothness_score)
        self.flow_metrics['logic'].append(logic_score)
        self.flow_metrics['flow_reward'].append(flow_reward)
        
        return flow_reward
    
    def calculate_consistency(self):
        """Berechnet Konsistenz der Entscheidungen"""
        actions = list(self.action_history)
        
        # Pattern-based consistency
        pattern_consistency = 0.0
        for pattern_name, pattern in self.action_patterns.items():
            if len(actions) >= len(pattern):
                # Check if recent actions match pattern
                recent_actions = actions[-len(pattern):]
                if recent_actions == pattern:
                    pattern_consistency = 1.0
                    break
        
        # Statistical consistency
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate entropy (lower entropy = higher consistency)
        total_actions = len(actions)
        entropy = 0.0
        for count in action_counts.values():
            probability = count / total_actions
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Convert entropy to consistency (0-1 scale)
        max_entropy = np.log2(len(set(actions)))
        statistical_consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        # Combine pattern and statistical consistency
        consistency = (pattern_consistency * 0.6 + statistical_consistency * 0.4)
        
        return max(0.0, min(1.0, consistency))
    
    def calculate_smoothness(self):
        """Berechnet Glätte der Entscheidungssequenz"""
        actions = list(self.action_history)
        
        if len(actions) < 2:
            return 1.0
        
        # Calculate action transitions
        transitions = []
        for i in range(len(actions) - 1):
            transition = abs(actions[i+1] - actions[i])
            transitions.append(transition)
        
        # Smooth transitions (0 or 1) are better than abrupt (2 or 3)
        smooth_transitions = sum(1 for t in transitions if t <= 1)
        smoothness = smooth_transitions / len(transitions)
        
        # Bonus for no transitions (same action)
        no_transitions = sum(1 for t in transitions if t == 0)
        if no_transitions > len(transitions) * 0.5:
            smoothness += 0.2
        
        return max(0.0, min(1.0, smoothness))
    
    def calculate_logic(self, current_state, action, next_state):
        """Berechnet Logik der Entscheidungssequenz"""
        logic_score = 0.0
        
        # Position-based logic
        pos_x, pos_y = current_state[0], current_state[1]
        vel_x, vel_y = current_state[2], current_state[3]
        angle, angular_vel = current_state[4], current_state[5]
        
        # Logic 1: Main engine when falling too fast
        if vel_y < -0.5 and action == 0:  # Main engine
            logic_score += 0.3
        
        # Logic 2: Side engines when tilted
        if abs(angle) > 0.2 and action in [2, 3]:  # Left/Right engines
            logic_score += 0.3
        
        # Logic 3: No engine when stable
        if abs(vel_y) < 0.1 and abs(angle) < 0.1 and action == 0:
            logic_score += 0.2
        
        # Logic 4: Corrective action based on position
        if pos_x > 0.5 and action == 3:  # Right engine when too far right
            logic_score += 0.1
        elif pos_x < -0.5 and action == 2:  # Left engine when too far left
            logic_score += 0.1
        
        # Logic 5: Velocity-based corrections
        if vel_x > 0.5 and action == 2:  # Left engine to slow rightward velocity
            logic_score += 0.1
        elif vel_x < -0.5 and action == 3:  # Right engine to slow leftward velocity
            logic_score += 0.1
        
        return max(0.0, min(1.0, logic_score))
    
    def get_flow_state(self):
        """Gibt aktuellen Flow-Zustand zurück"""
        if len(self.flow_metrics['flow_reward']) < 3:
            return 'INITIALIZING'
        
        recent_flow = list(self.flow_metrics['flow_reward'])[-3:]
        avg_flow = np.mean(recent_flow)
        
        if avg_flow > 0.8:
            return 'FLOW_STATE'
        elif avg_flow > 0.6:
            return 'SMOOTH'
        elif avg_flow > 0.4:
            return 'BALANCED'
        elif avg_flow > 0.2:
            return 'ROUGH'
        else:
            return 'CHAOTIC'
    
    def get_flow_statistics(self):
        """Gibt Flow-Statistiken zurück"""
        if len(self.flow_metrics['flow_reward']) == 0:
            return {}
        
        return {
            'avg_flow_reward': np.mean(self.flow_metrics['flow_reward']),
            'avg_consistency': np.mean(self.flow_metrics['consistency']),
            'avg_smoothness': np.mean(self.flow_metrics['smoothness']),
            'avg_logic': np.mean(self.flow_metrics['logic']),
            'flow_stability': 1.0 - np.std(self.flow_metrics['flow_reward']),
            'current_flow_state': self.get_flow_state()
        }

# =============================================================================
# FLOW-AWARE AGENT
# =============================================================================

class FlowAwareAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.flow_engine = FlowRewardEngine()
        self.flow_reward_history = deque(maxlen=1000)
        self.flow_state_history = deque(maxlen=1000)
        
        # Flow statistics
        self.flow_stats = {
            'total_flow_rewards': 0.0,
            'avg_flow_reward': 0.0,
            'flow_episodes': 0,
            'flow_states': {}
        }
        
    def act(self, state, training=True):
        """Aktion mit Flow-Awareness"""
        # Normale Aktion vom Base Agent
        action = self.base_agent.act(state, training)
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """Step mit Flow Reward Berechnung"""
        # Base Agent Step
        self.base_agent.step(state, action, reward, next_state, done)
        
        # Calculate Flow Reward
        flow_reward = self.flow_engine.calculate_flow_reward(state, action, next_state, reward)
        
        # Store flow information
        self.flow_reward_history.append(flow_reward)
        self.flow_state_history.append(self.flow_engine.get_flow_state())
        
        # Update flow statistics
        self.flow_stats['total_flow_rewards'] += flow_reward
        self.flow_stats['avg_flow_reward'] = np.mean(list(self.flow_reward_history))
        
        # Count flow states
        current_flow_state = self.flow_engine.get_flow_state()
        if current_flow_state not in self.flow_stats['flow_states']:
            self.flow_stats['flow_states'][current_flow_state] = 0
        self.flow_stats['flow_states'][current_flow_state] += 1
        
        # Enhanced reward with flow component
        enhanced_reward = reward + flow_reward * 0.1  # 10% flow component
        
        return enhanced_reward
    
    def update_emotion(self, episode_reward):
        """Update emotion mit Flow-Information"""
        # Base emotion update
        emotion, mindset = self.base_agent.update_emotion(episode_reward)
        
        # Flow-based emotion adjustment
        flow_stats = self.flow_engine.get_flow_statistics()
        if flow_stats:
            flow_state = flow_stats['current_flow_state']
            
            # Adjust emotion based on flow state
            if flow_state == 'FLOW_STATE':
                emotion = min(0.9, emotion + 0.1)  # Boost confidence
            elif flow_state == 'CHAOTIC':
                emotion = max(0.1, emotion - 0.1)  # Reduce confidence
        
        return emotion, mindset
    
    def get_flow_statistics(self):
        """Gibt Flow-Statistiken zurück"""
        return self.flow_stats.copy()
    
    def get_flow_engine_stats(self):
        """Gibt Flow Engine Statistiken zurück"""
        return self.flow_engine.get_flow_statistics()

# =============================================================================
# ERROR DETECTOR (from Phase 5.2)
# =============================================================================

class ErrorDetector:
    def __init__(self):
        self.error_patterns = {
            'crash_landing': {'threshold': -200, 'pattern': 'sudden_negative'},
            'overshoot': {'threshold': 100, 'pattern': 'high_positive'},
            'oscillation': {'threshold': 0.5, 'pattern': 'high_variance'},
            'stagnation': {'threshold': 10, 'pattern': 'low_variance'},
            'inefficient': {'threshold': 0.3, 'pattern': 'low_efficiency'}
        }
        
    def analyze(self, prediction, outcome, reward, state_history=None):
        """Analysiert Vorhersage vs. Ergebnis und erkennt Fehlertypen"""
        error_type = 'correct'
        
        # Crash Landing Detection
        if reward < self.error_patterns['crash_landing']['threshold']:
            error_type = 'crash_landing'
        
        # Overshoot Detection
        elif reward > self.error_patterns['overshoot']['threshold']:
            error_type = 'overshoot'
        
        # Oscillation Detection
        elif state_history is not None and len(state_history) > 5:
            state_variance = np.var([s[0] for s in state_history[-5:]])  # Position variance
            if state_variance > self.error_patterns['oscillation']['threshold']:
                error_type = 'oscillation'
        
        # Stagnation Detection
        elif state_history is not None and len(state_history) > 5:
            state_variance = np.var([s[0] for s in state_history[-5:]])
            if state_variance < self.error_patterns['stagnation']['threshold']:
                error_type = 'stagnation'
        
        # Inefficient Action Detection
        elif abs(prediction - outcome) > self.error_patterns['inefficient']['threshold']:
            error_type = 'inefficient'
        
        return error_type

# =============================================================================
# CORRECTION LEARNER (from Phase 5.2)
# =============================================================================

class CorrectionLearner:
    def __init__(self, state_size, action_size, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Correction Networks für verschiedene Fehlertypen
        self.correction_networks = nn.ModuleDict({
            'crash_landing': nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ),
            'overshoot': nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ),
            'oscillation': nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ),
            'stagnation': nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ),
            'inefficient': nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
        }).to(device)
        
        # Optimizer für alle Correction Networks
        self.optimizer = optim.Adam(self.correction_networks.parameters(), lr=1e-4)
        
        # Error Pattern Memory
        self.error_memory = deque(maxlen=1000)
        
    def learn_corrections(self, error_patterns):
        """Lernt Korrekturen aus Fehlermustern"""
        corrections = {}
        
        for error_type, errors in error_patterns.items():
            if len(errors) < 3:  # Brauche mindestens 3 Beispiele
                continue
                
            # Extrahiere States und korrekte Aktionen
            states = []
            correct_actions = []
            
            for error in errors:
                # Simuliere korrekte Aktion basierend auf Fehlertyp
                correct_action = self.simulate_correct_action(error, error_type)
                states.append(error['state'])
                correct_actions.append(correct_action)
            
            if len(states) > 0:
                # Trainiere Correction Network
                self.train_correction_network(error_type, states, correct_actions)
                corrections[error_type] = True
        
        return corrections
    
    def simulate_correct_action(self, error, error_type):
        """Simuliert korrekte Aktion basierend auf Fehlertyp"""
        state = error['state']
        
        if error_type == 'crash_landing':
            # Bei Crash: Sanftere Landung
            return 0 if state[6] > 0 else 1  # Main engine vs. Side engine
        
        elif error_type == 'overshoot':
            # Bei Overshoot: Gegensteuern
            return 2 if state[0] > 0 else 3  # Left vs. Right
        
        elif error_type == 'oscillation':
            # Bei Oscillation: Stabilisieren
            return 0  # Main engine für Stabilität
        
        elif error_type == 'stagnation':
            # Bei Stagnation: Mehr Aktivität
            return 1  # Side engine für Bewegung
        
        elif error_type == 'inefficient':
            # Bei Ineffizienz: Optimale Aktion
            return np.argmax([abs(state[0]), abs(state[1]), abs(state[2]), abs(state[3])])
        
        return 0  # Default
    
    def train_correction_network(self, error_type, states, correct_actions):
        """Trainiert spezifisches Correction Network"""
        if error_type not in self.correction_networks:
            return
        
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(correct_actions).to(device)
        
        # Forward pass
        predicted_actions = self.correction_networks[error_type](states_tensor)
        
        # Loss
        loss = F.cross_entropy(predicted_actions, actions_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_correction(self, state, error_type):
        """Gibt Korrektur für gegebenen State und Fehlertyp"""
        if error_type not in self.correction_networks:
            return None
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            correction = self.correction_networks[error_type](state_tensor)
            return correction.argmax().item()

# =============================================================================
# SELF-CORRECTING AGENT (from Phase 5.2)
# =============================================================================

class SelfCorrectingAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.error_detector = ErrorDetector()
        self.correction_learner = CorrectionLearner(
            base_agent.state_size, 
            base_agent.action_size
        )
        self.error_history = deque(maxlen=1000)
        self.correction_history = deque(maxlen=1000)
        self.state_history = deque(maxlen=100)
        
        # Correction Statistics
        self.correction_stats = {
            'total_errors': 0,
            'corrections_applied': 0,
            'correction_success_rate': 0.0,
            'error_types': {}
        }
        
    def act(self, state, training=True):
        """Aktion mit Self-Correction"""
        # Speichere State History
        self.state_history.append(state)
        
        # Normale Aktion vom Base Agent
        action = self.base_agent.act(state, training)
        
        # Prüfe auf bekannte Fehlermuster
        if len(self.state_history) > 5:
            error_type = self.error_detector.analyze(
                action, None, None, list(self.state_history)[-5:]
            )
            
            if error_type != 'correct':
                # Versuche Korrektur
                correction = self.correction_learner.get_correction(state, error_type)
                if correction is not None:
                    action = correction
                    self.correction_history.append({
                        'state': state,
                        'original_action': self.base_agent.act(state, training),
                        'corrected_action': action,
                        'error_type': error_type,
                        'timestamp': time.time()
                    })
                    self.correction_stats['corrections_applied'] += 1
        
        return action
    
    def learn_from_errors(self, episode_reward, episode_states, episode_actions):
        """Lernt aus Episode-Fehlern"""
        if episode_reward < -150:  # Schlechte Episode
            # Analysiere Episode auf Fehler
            for i, (state, action) in enumerate(zip(episode_states, episode_actions)):
                if i < len(episode_states) - 1:
                    next_state = episode_states[i + 1]
                    # Simuliere Reward basierend auf State-Übergang
                    simulated_reward = self.simulate_reward(state, action, next_state)
                    
                    error_type = self.error_detector.analyze(
                        action, simulated_reward, simulated_reward, 
                        episode_states[max(0, i-5):i+1]
                    )
                    
                    if error_type != 'correct':
                        self.error_history.append({
                            'state': state,
                            'action': action,
                            'reward': simulated_reward,
                            'error_type': error_type,
                            'episode_reward': episode_reward,
                            'timestamp': time.time()
                        })
                        
                        self.correction_stats['total_errors'] += 1
                        if error_type not in self.correction_stats['error_types']:
                            self.correction_stats['error_types'][error_type] = 0
                        self.correction_stats['error_types'][error_type] += 1
        
        # Lerne Korrekturen aus Fehlermustern
        if len(self.error_history) > 10:
            error_patterns = self.analyze_error_patterns()
            corrections = self.correction_learner.learn_corrections(error_patterns)
            
            # Update Correction Success Rate
            if len(self.correction_history) > 0:
                recent_corrections = list(self.correction_history)[-10:]
                successful_corrections = sum(1 for c in recent_corrections 
                                           if c['error_type'] in corrections)
                self.correction_stats['correction_success_rate'] = (
                    successful_corrections / len(recent_corrections)
                )
    
    def simulate_reward(self, state, action, next_state):
        """Simuliert Reward basierend auf State-Übergang"""
        # Einfache Reward-Simulation
        position_change = abs(next_state[0] - state[0])
        velocity_change = abs(next_state[2] - state[2])
        
        # Belohne Stabilität
        reward = -position_change - velocity_change
        
        # Bestrafe extreme Werte
        if abs(next_state[0]) > 1.0 or abs(next_state[1]) > 1.0:
            reward -= 10
        
        return reward
    
    def analyze_error_patterns(self):
        """Analysiert wiederkehrende Fehlermuster"""
        patterns = {}
        for error in self.error_history:
            error_type = error['error_type']
            if error_type not in patterns:
                patterns[error_type] = []
            patterns[error_type].append(error)
        return patterns
    
    def get_correction_stats(self):
        """Gibt Korrektur-Statistiken zurück"""
        return self.correction_stats.copy()

# =============================================================================
# ATTENTION STATE ENCODER (from Phase 5.1)
# =============================================================================

class AttentionStateEncoder(nn.Module):
    def __init__(self, state_size, hidden_size=512, num_heads=8):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # State Feature Extraction
        self.state_features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Output Projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, state):
        # Extract state features
        state_features = self.state_features(state)
        
        # Add sequence dimension for attention
        state_seq = state_features.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Self-Attention
        attended_features, attention_weights = self.attention(
            state_seq, state_seq, state_seq
        )
        
        # Residual connection + Layer Norm
        attended_features = self.layer_norm1(attended_features + state_seq)
        
        # Feed Forward
        ff_output = self.feed_forward(attended_features)
        
        # Residual connection + Layer Norm
        output = self.layer_norm2(ff_output + attended_features)
        
        # Remove sequence dimension
        output = output.squeeze(1)
        
        # Final projection
        output = self.output_projection(output)
        
        return output, attention_weights

# =============================================================================
# ATTENTION DUELING NETWORK (from Phase 5.1)
# =============================================================================

class AttentionDuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512, num_heads=8):
        super().__init__()
        
        # Attention State Encoder
        self.attention_encoder = AttentionStateEncoder(
            state_size, hidden_size, num_heads
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        # Get attention-encoded features
        features, attention_weights = self.attention_encoder(x)
        
        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, attention_weights

# =============================================================================
# ENHANCED EMOTION ENGINE (from previous implementation)
# =============================================================================

class EnhancedEmotionEngine:
    def __init__(self, alpha=0.15, beta=0.85, initial_emotion=0.5, threshold=0.1, momentum=0.3, sensitivity=1.2):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.emotion = float(initial_emotion)
        self.threshold = float(threshold)
        self.momentum = float(momentum)
        self.sensitivity = float(sensitivity)
        
        self.past_scores = deque(maxlen=20)
        self.emotion_history = deque(maxlen=10)
        self.performance_trend = deque(maxlen=5)
        
    def update_parameters(self, new_params):
        self.alpha = float(new_params['alpha'].detach())
        self.beta = float(new_params['beta'].detach())
        self.emotion = float(new_params['initial_emotion'].detach())
        self.threshold = float(new_params['threshold'].detach())
        self.momentum = float(new_params['momentum'].detach())
        self.sensitivity = float(new_params['sensitivity'].detach())
        
    def update(self, current_score):
        self.past_scores.append(current_score)
        
        if len(self.past_scores) < 5:
            return self.emotion
            
        # Calculate performance metrics
        recent_avg = np.mean(list(self.past_scores)[-5:])
        older_avg = np.mean(list(self.past_scores)[-10:-5]) if len(self.past_scores) >= 10 else recent_avg
        
        # Performance trend
        trend = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)
        self.performance_trend.append(trend)
        
        # Adaptive emotion update based on trend and sensitivity
        if abs(trend) > self.threshold:
            # Strong trend detected
            if trend > 0:
                # Positive trend - increase emotion
                emotion_delta = self.alpha * self.sensitivity * min(trend, 1.0)
            else:
                # Negative trend - decrease emotion
                emotion_delta = -self.alpha * self.sensitivity * min(abs(trend), 1.0)
        else:
            # No strong trend - maintain current emotion
            emotion_delta = 0
        
        # Apply momentum
        if len(self.emotion_history) > 0:
            momentum_factor = self.momentum * (self.emotion - self.emotion_history[-1])
            emotion_delta += momentum_factor
        
        # Update emotion with bounds
        self.emotion = np.clip(self.emotion + emotion_delta, 0.1, 0.9)
        self.emotion_history.append(self.emotion)
        
        return self.emotion
    
    def get_emotion_state(self):
        """Returns detailed emotion state"""
        if len(self.emotion_history) < 3:
            return 'INITIALIZING'
        
        recent_emotions = list(self.emotion_history)[-3:]
        emotion_trend = np.mean(np.diff(recent_emotions))
        
        if self.emotion > 0.7:
            return 'CONFIDENT'
        elif self.emotion > 0.6:
            return 'BALANCED'
        elif self.emotion > 0.5:
            return 'DETERMINED'
        elif self.emotion > 0.3:
            return 'CAUTIOUS'
        else:
            return 'FRUSTRATED'

# =============================================================================
# PRIORITIZED REPLAY BUFFER (with NaN protection)
# =============================================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], [], [], []
        
        # Calculate sampling probabilities with NaN protection
        priorities = self.priorities[:len(self.buffer)]
        # Replace any NaN or zero priorities
        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1.0, neginf=1e-6)
        priorities = np.maximum(priorities, 1e-6)  # Ensure all positive
        
        probabilities = priorities ** self.alpha
        probabilities = np.nan_to_num(probabilities, nan=1e-6)
        probabilities = np.maximum(probabilities, 1e-6)
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (torch.FloatTensor(states).to(device),
                torch.LongTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(next_states).to(device),
                torch.BoolTensor(dones).to(device),
                torch.FloatTensor(weights).to(device))
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# BASE RAINBOW DQN AGENT (for Flow Rewards)
# =============================================================================

class BaseRainbowDQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.999, epsilon_min=0.01, batch_size=128, target_update=500,
                 memory_size=100000, alpha=0.6, beta=0.4, num_heads=8):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Attention Networks
        self.q_network = AttentionDuelingNetwork(
            state_size, action_size, num_heads=num_heads
        ).to(device)
        self.target_network = AttentionDuelingNetwork(
            state_size, action_size, num_heads=num_heads
        ).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(memory_size, alpha, beta)
        
        # Enhanced Emotion Engine
        self.emotion_engine = EnhancedEmotionEngine()
        
        # Training tracking
        self.step_count = 0
        self.episode_rewards = []
        self.emotion_history = []
        self.mindset_history = []
        self.attention_history = []
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values, attention_weights = self.q_network(state)
        return q_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done):
        # Store experience
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn
        if len(self.memory) > self.batch_size:
            self.learn()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn(self):
        # Sample from memory
        states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        
        if len(states) == 0:
            return
        
        # Current Q values
        current_q_values, attention_weights = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate TD error
        td_errors = current_q_values.squeeze() - target_q_values
        
        # Weighted loss
        loss = (weights * td_errors ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities with NaN protection
        td_errors_np = td_errors.detach().cpu().numpy()
        # Replace NaN and inf values with small positive number
        td_errors_np = np.nan_to_num(td_errors_np, nan=1e-6, posinf=1.0, neginf=-1.0)
        priorities = (np.abs(td_errors_np) + 1e-6) ** self.memory.alpha
        self.memory.update_priorities(range(len(priorities)), priorities)
        
        # Store attention weights for analysis
        if len(self.attention_history) < 1000:  # Keep last 1000 attention weights
            self.attention_history.append(attention_weights.detach().cpu().numpy())
    
    def update_emotion(self, episode_reward):
        """Update emotion based on episode reward"""
        emotion = self.emotion_engine.update(episode_reward)
        mindset = self.emotion_engine.get_emotion_state()
        
        self.emotion_history.append(emotion)
        self.mindset_history.append(mindset)
        
        return emotion, mindset

# =============================================================================
# FLOW REWARDS TRAINING FUNCTION
# =============================================================================

def train_flow_rewards_rainbow_lunarlander(env_name='LunarLander-v3', episodes=2000, lr=1e-4, 
                                         batch_size=128, target_update=500, epsilon_decay=0.999,
                                         gamma=0.99, memory_size=100000, num_heads=8):
    """Train Flow Rewards Rainbow DQN with Self-Correction, Attention and Enhanced Emotion Engine"""
    
    print(f"🎮 FLOW REWARDS Rainbow DQN + Self-Correction + Attention + Enhanced Emotion Engine")
    print("=" * 90)
    
    # Environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Attention heads: {num_heads}")
    
    # Base Agent
    base_agent = BaseRainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        batch_size=batch_size,
        target_update=target_update,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        memory_size=memory_size,
        num_heads=num_heads
    )
    
    # Self-Correcting Agent
    self_correcting_agent = SelfCorrectingAgent(base_agent)
    
    # Flow-Aware Agent
    agent = FlowAwareAgent(self_correcting_agent)
    
    # Training tracking
    scores = []
    avg_scores = []
    best_score = -float('inf')
    flow_stats_history = []
    correction_stats_history = []
    
    print(f"Flow Rewards Training: {episodes} episodes")
    print("Starting FLOW REWARDS training...")
    
    start_time = time.time()
    
    for episode in tqdm(range(episodes), desc="Flow Rewards Training"):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        episode_states = []
        episode_actions = []
        
        while steps < 1000:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Enhanced reward with flow component
            enhanced_reward = agent.step(state, action, reward, next_state, done)
            
            episode_states.append(state)
            episode_actions.append(action)
            
            state = next_state
            episode_reward += reward  # Use original reward for tracking
            steps += 1
            
            if done:
                break
        
        # Learn from errors
        agent.base_agent.learn_from_errors(episode_reward, episode_states, episode_actions)
        
        # Update emotion
        emotion, mindset = agent.update_emotion(episode_reward)
        
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode_reward > best_score:
            best_score = episode_reward
        
        # Store statistics
        flow_stats = agent.get_flow_statistics()
        flow_engine_stats = agent.get_flow_engine_stats()
        correction_stats = agent.base_agent.get_correction_stats()
        
        flow_stats_history.append({
            'flow_stats': flow_stats,
            'flow_engine_stats': flow_engine_stats
        })
        correction_stats_history.append(correction_stats)
        
        # Logging
        if episode % 100 == 0:
            elapsed_time = time.time() - start_time
            flow_state = flow_engine_stats.get('current_flow_state', 'UNKNOWN')
            print(f"Episode {episode+1}/{episodes}: "
                  f"Avg Score: {avg_score:.2f}, "
                  f"Emotion: {emotion:.3f}, "
                  f"Mindset: {mindset}, "
                  f"Flow State: {flow_state}, "
                  f"Flow Reward: {flow_stats['avg_flow_reward']:.3f}, "
                  f"Errors: {correction_stats['total_errors']}, "
                  f"Corrections: {correction_stats['corrections_applied']}, "
                  f"Time: {elapsed_time/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.1f} minutes")
    
    env.close()
    
    return agent, scores, avg_scores, agent.base_agent.base_agent.emotion_history, agent.base_agent.base_agent.mindset_history, flow_stats_history, correction_stats_history

# =============================================================================
# FLOW REWARDS VISUALIZATION
# =============================================================================

def plot_flow_rewards_results(scores, avg_scores, emotion_history, mindset_history, flow_stats_history, correction_stats_history):
    """Plot flow rewards training results"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Episode rewards
    axes[0, 0].plot(scores, alpha=0.3, color='blue', label='Episode Rewards')
    axes[0, 0].plot(avg_scores, color='red', linewidth=2, label='Moving Avg (100)')
    axes[0, 0].axhline(y=200, color='green', linestyle='--', label='Solved (200)')
    axes[0, 0].set_title('Flow Rewards Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Average scores
    axes[0, 1].plot(avg_scores, color='green', linewidth=2)
    axes[0, 1].axhline(y=200, color='red', linestyle='--', label='Solved (200)')
    axes[0, 1].set_title('Flow Rewards Average Scores (100 episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Emotion evolution
    axes[0, 2].plot(emotion_history, color='purple', linewidth=2)
    axes[0, 2].axhline(y=0.6, color='orange', linestyle='--', label='BALANCED')
    axes[0, 2].axhline(y=0.7, color='green', linestyle='--', label='CONFIDENT')
    axes[0, 2].set_title('Flow Rewards Emotion Evolution')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Emotion Level')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Flow rewards evolution
    if flow_stats_history:
        flow_rewards = [stats['flow_stats']['avg_flow_reward'] for stats in flow_stats_history]
        axes[1, 0].plot(flow_rewards, color='cyan', linewidth=2)
        axes[1, 0].set_title('Flow Rewards Evolution')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Flow Reward')
        axes[1, 0].grid(True)
        
        # Flow states distribution
        flow_states = {}
        for stats in flow_stats_history:
            flow_engine_stats = stats['flow_engine_stats']
            if 'current_flow_state' in flow_engine_stats:
                state = flow_engine_stats['current_flow_state']
                flow_states[state] = flow_states.get(state, 0) + 1
        
        if flow_states:
            colors = ['green', 'blue', 'orange', 'red', 'purple']
            axes[1, 1].pie(flow_states.values(), labels=flow_states.keys(), 
                           autopct='%1.1f%%', colors=colors[:len(flow_states)])
            axes[1, 1].set_title('Flow States Distribution')
        
        # Flow metrics
        if flow_stats_history:
            consistency = [stats['flow_engine_stats'].get('avg_consistency', 0) for stats in flow_stats_history]
            smoothness = [stats['flow_engine_stats'].get('avg_smoothness', 0) for stats in flow_stats_history]
            logic = [stats['flow_engine_stats'].get('avg_logic', 0) for stats in flow_stats_history]
            
            axes[1, 2].plot(consistency, label='Consistency', color='blue')
            axes[1, 2].plot(smoothness, label='Smoothness', color='green')
            axes[1, 2].plot(logic, label='Logic', color='red')
            axes[1, 2].set_title('Flow Metrics Evolution')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Metric Value')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
    
    # Correction statistics
    if correction_stats_history:
        total_errors = [stats['total_errors'] for stats in correction_stats_history]
        corrections_applied = [stats['corrections_applied'] for stats in correction_stats_history]
        success_rates = [stats['correction_success_rate'] for stats in correction_stats_history]
        
        axes[2, 0].plot(total_errors, label='Total Errors', color='red')
        axes[2, 0].plot(corrections_applied, label='Corrections Applied', color='green')
        axes[2, 0].set_title('Error Detection and Correction')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        axes[2, 1].plot(success_rates, color='blue', linewidth=2)
        axes[2, 1].set_title('Correction Success Rate')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Success Rate')
        axes[2, 1].grid(True)
        
        # Error types distribution
        if correction_stats_history:
            latest_stats = correction_stats_history[-1]
            error_types = latest_stats.get('error_types', {})
            if error_types:
                axes[2, 2].pie(error_types.values(), labels=error_types.keys(), 
                               autopct='%1.1f%%', startangle=90)
                axes[2, 2].set_title('Error Types Distribution')
            else:
                axes[2, 2].text(0.5, 0.5, 'No Error Types\nDetected Yet', 
                               ha='center', va='center', transform=axes[2, 2].transAxes)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 FLOW REWARDS + SELF-CORRECTION + ATTENTION + ENHANCED EMOTION ENGINE")
    print("=" * 90)
    
    # Flow Rewards training with optimized parameters
    results = train_flow_rewards_rainbow_lunarlander(
        episodes=2000,
        lr=1e-4,
        epsilon_decay=0.999,
        batch_size=128,
        target_update=500,
        num_heads=8
    )
    
    agent, scores, avg_scores, emotion_history, mindset_history, flow_stats_history, correction_stats_history = results
    
    # Plot results
    plot_flow_rewards_results(scores, avg_scores, emotion_history, mindset_history, flow_stats_history, correction_stats_history)
    
    # Final results
    final_avg = np.mean(scores[-100:])
    best_episode = max(scores)
    final_emotion = emotion_history[-1] if emotion_history else 0.5
    final_mindset = mindset_history[-1] if mindset_history else 'UNKNOWN'
    
    print(f"\n🏆 FLOW REWARDS Results:")
    print(f"   Final Average (100 episodes): {final_avg:.2f}")
    print(f"   Best Episode: {best_episode:.2f}")
    print(f"   Final Emotion: {final_emotion:.3f}")
    print(f"   Final Mindset: {final_mindset}")
    print(f"   Progress: {final_avg}/200 ({final_avg/200*100:.1f}% of target)")
    
    # Flow analysis
    if flow_stats_history:
        final_flow_stats = flow_stats_history[-1]
        flow_stats = final_flow_stats['flow_stats']
        flow_engine_stats = final_flow_stats['flow_engine_stats']
        
        print(f"\n🌊 Flow Rewards Analysis:")
        print(f"   Average Flow Reward: {flow_stats['avg_flow_reward']:.3f}")
        print(f"   Current Flow State: {flow_engine_stats.get('current_flow_state', 'UNKNOWN')}")
        print(f"   Flow Stability: {flow_engine_stats.get('flow_stability', 0):.3f}")
        print(f"   Average Consistency: {flow_engine_stats.get('avg_consistency', 0):.3f}")
        print(f"   Average Smoothness: {flow_engine_stats.get('avg_smoothness', 0):.3f}")
        print(f"   Average Logic: {flow_engine_stats.get('avg_logic', 0):.3f}")
    
    # Correction analysis
    if correction_stats_history:
        final_correction_stats = correction_stats_history[-1]
        print(f"\n🔧 Self-Correction Analysis:")
        print(f"   Total Errors Detected: {final_correction_stats['total_errors']}")
        print(f"   Corrections Applied: {final_correction_stats['corrections_applied']}")
        print(f"   Correction Success Rate: {final_correction_stats['correction_success_rate']:.3f}")
        print(f"   Error Types: {final_correction_stats['error_types']}")
    
    # Performance analysis
    emotion_stability = 1.0 - np.std(emotion_history) if len(emotion_history) > 1 else 0
    
    print(f"\n📊 Flow Rewards Performance Analysis:")
    print(f"   Emotion stability: {emotion_stability:.3f}")
    
    # Win rate analysis
    wins = sum(1 for score in scores if score > 0)
    win_rate = (wins / len(scores)) * 100
    print(f"   Win rate: {win_rate:.1f}%")
    
    print("\n🎉 Flow Rewards + Self-Correction + Attention + Enhanced Emotion Engine testing complete!")
    print("🚀 Ready for Phase 5.4: Emotion-Transformer!")
