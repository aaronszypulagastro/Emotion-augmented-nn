"""
Multi-Timeframe Emotion-Augmented Trading Agent
Kombiniert verschiedene Zeithorizonte für robustere Trading-Entscheidungen
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys
import os

# Import unserer Trading-Komponenten
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.multi_timeframe_environment import MultiTimeframeEnvironment
from agents.trading_emotion_engine import TradingEmotionEngine, TradingEmotion

class MultiTimeframeNetwork(nn.Module):
    """Multi-Timeframe Dueling DQN Architecture"""
    
    def __init__(self, 
                 state_size_per_timeframe: int,
                 num_timeframes: int,
                 action_size: int,
                 hidden_size: int = 128):
        super(MultiTimeframeNetwork, self).__init__()
        
        self.state_size_per_timeframe = state_size_per_timeframe
        self.num_timeframes = num_timeframes
        self.action_size = action_size
        
        # Timeframe-spezifische Feature Extractor
        self.timeframe_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_size_per_timeframe, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU()
            ) for _ in range(num_timeframes)
        ])
        
        # Timeframe Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear((hidden_size // 4) * num_timeframes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Attention Mechanism für Timeframe-Gewichtung
        self.attention = nn.Sequential(
            nn.Linear((hidden_size // 4) * num_timeframes, num_timeframes),
            nn.Softmax(dim=1)
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        # Reshape Input: (batch_size, total_state_size) -> (batch_size, num_timeframes, state_size_per_timeframe)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_timeframes, self.state_size_per_timeframe)
        
        # Extrahiere Features für jeden Timeframe
        timeframe_features = []
        for i, extractor in enumerate(self.timeframe_extractors):
            timeframe_feature = extractor(x[:, i, :])
            timeframe_features.append(timeframe_feature)
        
        # Kombiniere Timeframe Features
        combined_features = torch.cat(timeframe_features, dim=1)
        
        # Berechne Attention Weights
        attention_weights = self.attention(combined_features)
        
        # Wende Attention an
        weighted_features = []
        for i, feature in enumerate(timeframe_features):
            weighted_feature = feature * attention_weights[:, i:i+1]
            weighted_features.append(weighted_feature)
        
        # Fusioniere gewichtete Features
        fused_features = torch.cat(weighted_features, dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Dueling Architecture
        value = self.value_stream(fused_features)
        advantage = self.advantage_stream(fused_features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, attention_weights

class MultiTimeframeTradingAgent:
    """
    Multi-Timeframe Emotion-Augmented Trading Agent
    Nutzt verschiedene Zeithorizonte für robustere Entscheidungen
    """
    
    def __init__(self, 
                 state_size_per_timeframe: int,
                 num_timeframes: int,
                 action_size: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 64,
                 memory_size: int = 10000,
                 target_update: int = 100,
                 device: str = 'cpu'):
        
        self.state_size_per_timeframe = state_size_per_timeframe
        self.num_timeframes = num_timeframes
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        
        # Neural Networks
        self.q_network = MultiTimeframeNetwork(
            state_size_per_timeframe, num_timeframes, action_size
        ).to(device)
        self.target_network = MultiTimeframeNetwork(
            state_size_per_timeframe, num_timeframes, action_size
        ).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience Replay
        self.memory = deque(maxlen=memory_size)
        
        # Emotion Engine
        self.emotion_engine = TradingEmotionEngine()
        
        # Multi-Timeframe Stats
        self.training_step = 0
        self.episode_rewards = []
        self.episode_emotions = []
        self.attention_weights_history = []
        
        # Trading Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
        # Timeframe Performance Tracking
        self.timeframe_performance = {f'tf_{i}': [] for i in range(num_timeframes)}
        
    def get_emotion_augmented_state(self, state: np.ndarray) -> np.ndarray:
        """Erweitere State um Emotion-Informationen"""
        emotion_vector = self.emotion_engine.get_emotion_vector()
        
        # Für Multi-Timeframe: Erweitere jeden Timeframe um Emotionen
        state_per_timeframe = self.state_size_per_timeframe
        emotion_size = len(emotion_vector)
        
        # Reshape State: (total_state_size,) -> (num_timeframes, state_size_per_timeframe)
        state_reshaped = state.reshape(self.num_timeframes, state_per_timeframe)
        
        # Erweitere jeden Timeframe um Emotionen
        augmented_states = []
        for i in range(self.num_timeframes):
            timeframe_state = state_reshaped[i]
            # Erweitere um Emotionen (reduziere andere Features um Platz zu schaffen)
            augmented_timeframe = np.concatenate([
                timeframe_state[:-1],  # Alle Features außer dem letzten
                emotion_vector  # Emotionen hinzufügen
            ])
            augmented_states.append(augmented_timeframe)
        
        # Flatten zurück zu 1D
        augmented_state = np.concatenate(augmented_states)
        
        return augmented_state
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[float, np.ndarray]:
        """Wähle Action basierend auf Multi-Timeframe State und Emotion"""
        
        # Erweitere State um Emotionen
        augmented_state = self.get_emotion_augmented_state(state)
        augmented_state = torch.FloatTensor(augmented_state).unsqueeze(0).to(self.device)
        
        # Epsilon-Greedy mit Emotion-basierter Anpassung
        if training and np.random.random() < self.epsilon:
            # Random Action mit Emotion-basierter Bias
            emotion_modifier = self.emotion_engine.get_emotion_modifier()
            base_action = np.random.uniform(-1, 1)
            
            # Emotion-basierte Anpassung
            if self.emotion_engine.current_emotion in [TradingEmotion.CONFIDENT, TradingEmotion.GREEDY]:
                action = base_action * emotion_modifier
            elif self.emotion_engine.current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED]:
                action = base_action * (1 / emotion_modifier)
            else:
                action = base_action
                
            # Dummy Attention Weights für Random Action
            attention_weights = np.ones(self.num_timeframes) / self.num_timeframes
                
        else:
            # Q-Network Prediction
            with torch.no_grad():
                q_values, attention_weights = self.q_network(augmented_state)
                action = q_values.cpu().numpy()[0][0]  # Kontinuierliche Action
                attention_weights = attention_weights.cpu().numpy()[0]
        
        # Clamp Action basierend auf Emotion
        risk_tolerance = self.emotion_engine.get_risk_tolerance()
        max_position = risk_tolerance * 0.5  # Max 50% des Risikotoleranz
        
        action = np.clip(action, -max_position, max_position)
        
        return float(action), attention_weights
    
    def store_experience(self, state, action, reward, next_state, done, attention_weights):
        """Speichere Experience in Memory"""
        self.memory.append((state, action, reward, next_state, done, attention_weights))
    
    def update_emotion_engine(self, 
                            portfolio_return: float,
                            trade_return: float,
                            drawdown: float,
                            win_rate: float,
                            price_change: float,
                            volume_change: float,
                            volatility: float,
                            attention_weights: np.ndarray):
        """Aktualisiere Emotion Engine basierend auf Trading-Performance und Timeframe-Gewichtung"""
        
        # Update Market Sentiment
        self.emotion_engine.update_market_sentiment(
            price_change=price_change,
            volume_change=volume_change,
            volatility=volatility,
            trend_strength=price_change * 2
        )
        
        # Update Performance
        self.emotion_engine.update_performance(
            portfolio_return=portfolio_return,
            trade_return=trade_return,
            drawdown=drawdown,
            win_rate=win_rate
        )
        
        # Speichere Attention Weights für Analyse
        self.attention_weights_history.append(attention_weights.copy())
        if len(self.attention_weights_history) > 1000:
            self.attention_weights_history.pop(0)
    
    def learn(self):
        """Lerne aus Experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample Batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, attention_weights = zip(*batch)
        
        # Konvertiere zu Tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # Erweitere States um Emotionen
        augmented_states = []
        augmented_next_states = []
        
        for i in range(len(states)):
            augmented_state = self.get_emotion_augmented_state(states[i].cpu().numpy())
            augmented_next_state = self.get_emotion_augmented_state(next_states[i].cpu().numpy())
            
            augmented_states.append(augmented_state)
            augmented_next_states.append(augmented_next_state)
        
        augmented_states = torch.FloatTensor(np.array(augmented_states)).to(self.device)
        augmented_next_states = torch.FloatTensor(np.array(augmented_next_states)).to(self.device)
        
        # Current Q Values
        current_q_values, _ = self.q_network(augmented_states)
        current_q_values = current_q_values.squeeze()
        
        # Next Q Values (Double DQN)
        with torch.no_grad():
            next_q_values, _ = self.q_network(augmented_next_states)
            target_q_values = rewards + (self.gamma * next_q_values.squeeze() * ~dones)
        
        # Berechne Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Backward Pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update Target Network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_trading_stats(self, trade_return: float, attention_weights: np.ndarray):
        """Aktualisiere Trading-Statistiken und Timeframe-Performance"""
        self.total_trades += 1
        self.total_profit += trade_return
        
        if trade_return > 0:
            self.winning_trades += 1
        
        # Track Timeframe Performance
        for i, weight in enumerate(attention_weights):
            self.timeframe_performance[f'tf_{i}'].append(weight * trade_return)
    
    def get_trading_metrics(self) -> Dict:
        """Berechne Trading-Metriken inklusive Multi-Timeframe Analysis"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Timeframe Performance Analysis
        timeframe_analysis = {}
        for tf_name, performance in self.timeframe_performance.items():
            if performance:
                timeframe_analysis[tf_name] = {
                    'avg_contribution': np.mean(performance),
                    'total_contribution': np.sum(performance),
                    'num_contributions': len(performance)
                }
        
        # Attention Weights Analysis
        attention_analysis = {}
        if self.attention_weights_history:
            recent_weights = np.array(self.attention_weights_history[-100:])  # Letzte 100
            for i in range(self.num_timeframes):
                attention_analysis[f'tf_{i}'] = {
                    'avg_weight': np.mean(recent_weights[:, i]),
                    'std_weight': np.std(recent_weights[:, i]),
                    'max_weight': np.max(recent_weights[:, i]),
                    'min_weight': np.min(recent_weights[:, i])
                }
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': self.total_profit / self.total_trades if self.total_trades > 0 else 0.0,
            'current_emotion': self.emotion_engine.current_emotion.value,
            'risk_tolerance': self.emotion_engine.get_risk_tolerance(),
            'position_sizing_modifier': self.emotion_engine.get_position_sizing_modifier(),
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'timeframe_performance': timeframe_analysis,
            'attention_analysis': attention_analysis,
            'num_timeframes': self.num_timeframes
        }
    
    def get_timeframe_insights(self) -> Dict:
        """Erstelle Insights über Timeframe-Nutzung"""
        if not self.attention_weights_history:
            return {}
        
        recent_weights = np.array(self.attention_weights_history[-100:])
        
        insights = {
            'dominant_timeframe': int(np.argmax(np.mean(recent_weights, axis=0))),
            'timeframe_stability': float(np.mean(np.std(recent_weights, axis=0))),
            'timeframe_diversity': float(np.mean(np.max(recent_weights, axis=1) - np.min(recent_weights, axis=1))),
            'avg_weights': np.mean(recent_weights, axis=0).tolist()
        }
        
        return insights
    
    def save_model(self, filepath: str):
        """Speichere Model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'emotion_engine_states': self.emotion_engine.emotion_states,
            'trading_metrics': self.get_trading_metrics(),
            'timeframe_performance': self.timeframe_performance,
            'attention_weights_history': self.attention_weights_history[-100:]  # Speichere nur letzte 100
        }, filepath)
    
    def load_model(self, filepath: str):
        """Lade Model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        
        # Lade Emotion Engine States
        if 'emotion_engine_states' in checkpoint:
            self.emotion_engine.emotion_states = checkpoint['emotion_engine_states']
        
        # Lade Timeframe Performance
        if 'timeframe_performance' in checkpoint:
            self.timeframe_performance = checkpoint['timeframe_performance']
        
        # Lade Attention Weights History
        if 'attention_weights_history' in checkpoint:
            self.attention_weights_history = checkpoint['attention_weights_history']
    
    def reset(self):
        """Reset Agent"""
        self.episode_rewards = []
        self.episode_emotions = []
        self.emotion_engine.reset()
        self.epsilon = 1.0
        self.attention_weights_history = []
        self.timeframe_performance = {f'tf_{i}': [] for i in range(self.num_timeframes)}


def train_multi_timeframe_agent(env: MultiTimeframeEnvironment,
                               episodes: int = 1000,
                               save_interval: int = 100,
                               model_path: str = "multi_timeframe_agent.pth") -> Dict:
    """
    Trainiere Multi-Timeframe Emotion-Augmented Trading Agent
    """
    
    print(f"🚀 Starte Multi-Timeframe Training für {env.symbol}...")
    print(f"Episodes: {episodes}")
    print(f"Timeframes: {env.timeframes}")
    print(f"Primary Timeframe: {env.primary_timeframe}")
    print(f"Initial Capital: ${env.initial_capital:,.2f}")
    
    # Erstelle Agent
    state_size_per_timeframe = 7  # Standard State Size pro Timeframe
    num_timeframes = len(env.timeframes)
    action_size = 1  # Kontinuierliche Action
    
    agent = MultiTimeframeTradingAgent(
        state_size_per_timeframe=state_size_per_timeframe,
        num_timeframes=num_timeframes,
        action_size=action_size,
        learning_rate=1e-4,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000
    )
    
    # Training Loop
    episode_rewards = []
    episode_returns = []
    episode_emotions = []
    episode_attention_weights = []
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_return = 0
        episode_attention = []
        
        done = False
        step = 0
        
        while not done:
            # Wähle Action
            action, attention_weights = agent.select_action(state, training=True)
            action_array = np.array([action])
            
            # Environment Step
            next_state, reward, done, truncated, info = env.step(action_array)
            
            # Update Emotion Engine
            if step > 0:
                # Berechne Metriken für Emotion Engine
                portfolio_return = (info['portfolio_value'] - env.initial_capital) / env.initial_capital
                trade_return = reward
                drawdown = 0.0  # Vereinfacht
                win_rate = agent.winning_trades / max(agent.total_trades, 1)
                
                # Market Data
                if step < len(env.data[env.primary_timeframe]) - 1:
                    primary_data = env.data[env.primary_timeframe]
                    current_price = primary_data['close'].iloc[step]
                    next_price = primary_data['close'].iloc[step + 1]
                    price_change = (next_price - current_price) / current_price
                    volume_change = 0.1  # Vereinfacht
                    volatility = 0.02  # Vereinfacht
                else:
                    price_change = 0.0
                    volume_change = 0.0
                    volatility = 0.0
                
                agent.update_emotion_engine(
                    portfolio_return=portfolio_return,
                    trade_return=trade_return,
                    drawdown=drawdown,
                    win_rate=win_rate,
                    price_change=price_change,
                    volume_change=volume_change,
                    volatility=volatility,
                    attention_weights=attention_weights
                )
            
            # Speichere Experience
            agent.store_experience(state, action, reward, next_state, done, attention_weights)
            
            # Lerne
            agent.learn()
            
            # Update Stats
            agent.update_trading_stats(reward, attention_weights)
            
            # Update State
            state = next_state
            episode_reward += reward
            episode_return = (info['portfolio_value'] - env.initial_capital) / env.initial_capital
            episode_attention.append(attention_weights.copy())
            step += 1
        
        # Episode beendet
        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_emotions.append(agent.emotion_engine.current_emotion.value)
        episode_attention_weights.append(np.mean(episode_attention, axis=0))
        
        # Logging
        if episode % 10 == 0:
            metrics = agent.get_trading_metrics()
            insights = agent.get_timeframe_insights()
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Return: {episode_return*100:6.2f}% | "
                  f"Emotion: {agent.emotion_engine.current_emotion.value:12s} | "
                  f"Win Rate: {metrics['win_rate']*100:5.1f}% | "
                  f"Dominant TF: {insights.get('dominant_timeframe', 0)} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Speichere Model
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"{model_path}_episode_{episode}.pth")
    
    # Finale Metriken
    final_metrics = agent.get_trading_metrics()
    final_insights = agent.get_timeframe_insights()
    
    final_metrics.update({
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_emotions': episode_emotions,
        'episode_attention_weights': episode_attention_weights,
        'final_portfolio_value': info['portfolio_value'],
        'total_return': episode_return,
        'timeframe_insights': final_insights
    })
    
    # Speichere finales Model
    agent.save_model(f"{model_path}_final.pth")
    
    print(f"\n✅ Multi-Timeframe Training abgeschlossen!")
    print(f"Final Portfolio Value: ${info['portfolio_value']:,.2f}")
    print(f"Total Return: {episode_return*100:.2f}%")
    print(f"Final Emotion: {agent.emotion_engine.current_emotion.value}")
    print(f"Win Rate: {final_metrics['win_rate']*100:.1f}%")
    print(f"Dominant Timeframe: {final_insights.get('dominant_timeframe', 0)}")
    print(f"Timeframe Stability: {final_insights.get('timeframe_stability', 0):.3f}")
    
    return final_metrics


if __name__ == "__main__":
    # Test das Multi-Timeframe Trading Agent
    print("🚀 Teste Multi-Timeframe Trading Agent...")
    
    # Erstelle Test Environment
    from environments.multi_timeframe_environment import create_multi_timeframe_environments
    
    envs = create_multi_timeframe_environments()
    
    if envs:
        # Teste mit erstem verfügbaren Environment
        env_name = list(envs.keys())[0]
        env = envs[env_name]
        
        print(f"\n📊 Teste mit {env_name}...")
        
        # Kurzer Test
        metrics = train_multi_timeframe_agent(
            env=env,
            episodes=50,
            save_interval=25
        )
        
        print(f"\n📈 Test Results:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    print("\n✅ Multi-Timeframe Trading Agent Test abgeschlossen!")
