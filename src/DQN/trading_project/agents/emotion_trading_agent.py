"""
Emotion-Augmented Trading Agent
Kombiniert Rainbow DQN mit Trading Emotion Engine für intelligente Trading-Entscheidungen
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
from environments.trading_environment import TradingEnvironment
from agents.trading_emotion_engine import TradingEmotionEngine, TradingEmotion

class DuelingNetwork(nn.Module):
    """Dueling DQN Architecture für Trading"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DuelingNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
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
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling Architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay für Trading"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """Füge Experience hinzu"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample Batch mit Prioritäten"""
        if len(self.buffer) == 0:
            return [], [], [], [], [], []
        
        # Berechne Sampling-Wahrscheinlichkeiten
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample Indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Berechne Importance Sampling Weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Extrahiere Experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), weights, indices)
    
    def update_priorities(self, indices, priorities):
        """Aktualisiere Prioritäten"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class EmotionTradingAgent:
    """
    Emotion-Augmented Trading Agent
    Kombiniert Rainbow DQN mit Trading Emotion Engine
    """
    
    def __init__(self, 
                 state_size: int,
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
        
        self.state_size = state_size
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
        self.q_network = DuelingNetwork(state_size, action_size).to(device)
        self.target_network = DuelingNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience Replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Emotion Engine
        self.emotion_engine = TradingEmotionEngine()
        
        # Training Stats
        self.training_step = 0
        self.episode_rewards = []
        self.episode_emotions = []
        
        # Trading Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
    def get_emotion_augmented_state(self, state: np.ndarray) -> np.ndarray:
        """Erweitere State um Emotion-Informationen"""
        emotion_vector = self.emotion_engine.get_emotion_vector()
        
        # Kombiniere Market State mit Emotion State
        augmented_state = np.concatenate([state, emotion_vector])
        
        return augmented_state
    
    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Wähle Action basierend auf State und Emotion"""
        
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
                # Tendiere zu größeren Positionen
                action = base_action * emotion_modifier
            elif self.emotion_engine.current_emotion in [TradingEmotion.FEARFUL, TradingEmotion.FRUSTRATED]:
                # Tendiere zu kleineren Positionen
                action = base_action * (1 / emotion_modifier)
            else:
                action = base_action
                
        else:
            # Q-Network Prediction
            with torch.no_grad():
                q_values = self.q_network(augmented_state)
                action = q_values.cpu().numpy()[0][0]  # Kontinuierliche Action
        
        # Clamp Action basierend auf Emotion
        risk_tolerance = self.emotion_engine.get_risk_tolerance()
        max_position = risk_tolerance * 0.5  # Max 50% des Risikotoleranz
        
        action = np.clip(action, -max_position, max_position)
        
        return float(action)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Speichere Experience in Memory"""
        self.memory.add(state, action, reward, next_state, done)
    
    def update_emotion_engine(self, 
                            portfolio_return: float,
                            trade_return: float,
                            drawdown: float,
                            win_rate: float,
                            price_change: float,
                            volume_change: float,
                            volatility: float):
        """Aktualisiere Emotion Engine basierend auf Trading-Performance"""
        
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
    
    def learn(self):
        """Lerne aus Experiences"""
        if len(self.memory.buffer) < self.batch_size:
            return
        
        # Sample Batch
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
        
        if len(states) == 0:
            return
        
        # Konvertiere zu Tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Erweitere States um Emotionen
        augmented_states = []
        augmented_next_states = []
        
        for i in range(len(states)):
            # Für jeden State in Batch: Emotion Engine temporär zurücksetzen
            # und mit entsprechendem State aktualisieren
            augmented_state = self.get_emotion_augmented_state(states[i].cpu().numpy())
            augmented_next_state = self.get_emotion_augmented_state(next_states[i].cpu().numpy())
            
            augmented_states.append(augmented_state)
            augmented_next_states.append(augmented_next_state)
        
        augmented_states = torch.FloatTensor(np.array(augmented_states)).to(self.device)
        augmented_next_states = torch.FloatTensor(np.array(augmented_next_states)).to(self.device)
        
        # Current Q Values
        current_q_values = self.q_network(augmented_states).squeeze()
        
        # Next Q Values (Double DQN)
        with torch.no_grad():
            next_q_values = self.q_network(augmented_next_states).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Berechne TD-Error
        td_errors = current_q_values - target_q_values
        
        # Berechne Loss mit Importance Sampling
        loss = (weights * td_errors.pow(2)).mean()
        
        # Backward Pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Aktualisiere Prioritäten
        priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # Update Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update Target Network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_trading_stats(self, trade_return: float):
        """Aktualisiere Trading-Statistiken"""
        self.total_trades += 1
        self.total_profit += trade_return
        
        if trade_return > 0:
            self.winning_trades += 1
    
    def get_trading_metrics(self) -> Dict:
        """Berechne Trading-Metriken"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
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
            'training_steps': self.training_step
        }
    
    def save_model(self, filepath: str):
        """Speichere Model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'emotion_engine_states': self.emotion_engine.emotion_states,
            'trading_metrics': self.get_trading_metrics()
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
    
    def reset(self):
        """Reset Agent"""
        self.episode_rewards = []
        self.episode_emotions = []
        self.emotion_engine.reset()
        self.epsilon = 1.0


def train_emotion_trading_agent(env: TradingEnvironment,
                               episodes: int = 1000,
                               save_interval: int = 100,
                               model_path: str = "emotion_trading_agent.pth") -> Dict:
    """
    Trainiere Emotion-Augmented Trading Agent
    """
    
    print(f"🚀 Starte Training für {env.symbol}...")
    print(f"Episodes: {episodes}")
    print(f"Timeframe: {env.timeframe}")
    print(f"Initial Capital: ${env.initial_capital:,.2f}")
    
    # Erstelle Agent
    state_size = env.observation_space.shape[0]
    action_size = 1  # Kontinuierliche Action
    
    agent = EmotionTradingAgent(
        state_size=state_size,
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
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_return = 0
        
        done = False
        step = 0
        
        while not done:
            # Wähle Action
            action = agent.select_action(state, training=True)
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
                if step < len(env.data) - 1:
                    current_price = env.data['close'].iloc[step]
                    next_price = env.data['close'].iloc[step + 1]
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
                    volatility=volatility
                )
            
            # Speichere Experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Lerne
            agent.learn()
            
            # Update Stats
            agent.update_trading_stats(reward)
            
            # Update State
            state = next_state
            episode_reward += reward
            episode_return = (info['portfolio_value'] - env.initial_capital) / env.initial_capital
            step += 1
        
        # Episode beendet
        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_emotions.append(agent.emotion_engine.current_emotion.value)
        
        # Logging
        if episode % 10 == 0:
            metrics = agent.get_trading_metrics()
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Return: {episode_return*100:6.2f}% | "
                  f"Emotion: {agent.emotion_engine.current_emotion.value:12s} | "
                  f"Win Rate: {metrics['win_rate']*100:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Speichere Model
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"{model_path}_episode_{episode}.pth")
    
    # Finale Metriken
    final_metrics = agent.get_trading_metrics()
    final_metrics.update({
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_emotions': episode_emotions,
        'final_portfolio_value': info['portfolio_value'],
        'total_return': episode_return
    })
    
    # Speichere finales Model
    agent.save_model(f"{model_path}_final.pth")
    
    print(f"\n✅ Training abgeschlossen!")
    print(f"Final Portfolio Value: ${info['portfolio_value']:,.2f}")
    print(f"Total Return: {episode_return*100:.2f}%")
    print(f"Final Emotion: {agent.emotion_engine.current_emotion.value}")
    print(f"Win Rate: {final_metrics['win_rate']*100:.1f}%")
    
    return final_metrics


if __name__ == "__main__":
    # Test das Emotion Trading Agent
    print("🚀 Teste Emotion Trading Agent...")
    
    # Erstelle Test Environment
    from environments.trading_environment import create_trading_environments
    
    envs = create_trading_environments()
    
    if envs:
        # Teste mit erstem verfügbaren Environment
        env_name = list(envs.keys())[0]
        env = envs[env_name]
        
        print(f"\n📊 Teste mit {env_name}...")
        
        # Kurzer Test
        metrics = train_emotion_trading_agent(
            env=env,
            episodes=50,
            save_interval=25
        )
        
        print(f"\n📈 Test Results:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    print("\n✅ Emotion Trading Agent Test abgeschlossen!")
