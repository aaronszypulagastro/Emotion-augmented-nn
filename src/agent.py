# Dieses Modul definiert: 
# - QNetwork: Ein neuronales Netzwerk, das die Q-Werte für gegebene Zustände und Aktionen vorhersagt.
# - ReplayBuffer: Ein Replay-Puffer zur Speicherung und Abfrage von Erfahrungen.
# - DQNAgent: Wählt Aktionen (Epsilon-Greedy-Strategie) und lernt per Bellman-Update)



# IMPORTS 
from __future__ import annotations
from argparse import _ActionsContainer
from typing import Deque, Tuple, List
from dataclasses import dataclass

import copy 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random, collections 
import numpy as np 


# Q - NETWORK (das Funktionsapproximationsmodell)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim: int) -> None: 
        super().__init__()
        

        # Schichten definieren 
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim) # AUsgabe: ein Q-Wert pro Aktion
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x)) # 1. Schicht + ReLU
        x = F.relu(self.fc2(x)) # 2. Schicht + ReLU
        return self.fc3(x)      # rohe Q-Werte für jede Aktion
    # kein SOftmax, da wir rohe Q-Werte wollen :) 

# REPLAY BUFFER (speichert Erfahrungen)
class ReplayBuffer:
    def __init__(self, capacity: int):   # Ringpuffer mit max. Kapazität

        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float, 
        next_state: np.ndarray, 
        done: bool) -> None:  
        # Erfahrung speichern
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size): 
        # zufällige Stichprobe von Erfahrungen
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        return states, actions, rewards, next_states, done
    
    def __len__(self) -> int:  # Anzahl der gespeicherten Erfahrungen
        return len(self.buffer)

# DQN AGENT (wählt Aktionen und lernt)
class DQNConfig:
    state_dim: int
    action_dim: int 
    lr: float = 5e-4  
    gamma: float = 0.99
    replay_capacity: int = 100000
    batch_size: int = 128
    learn_starts: int = 1000
    updates_per_step: int = 2
    tau: float = 0.01 
    grad_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent:
    # Agent-Parameter
    def __init__(self, state_dim, action_dim, batch_size=None):
        self.cfg = DQNConfig()

        if batch_size is not None:
            self.cfg.batch_size = batch_size

        self.device = torch.device(self.cfg.device)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Stabilere Targets 
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.cfg.lr) 
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = \
            collections.deque(maxlen=self.cfg.replay_capacity)
        
        # ε-greedy: startet hoch (viel Exploration), wird dann immer kleiner (mehr Exploitation)
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_min = self.cfg.epsilon_end
        self.epsilon_decay = self.cfg.epsilon_decay

        self.action_dim = action_dim 
        self.step_count = 0 

     
    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.replay_buffer)

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def act(self, state):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.q_network(s)
                action = int(q.argmax(dim=1).item())

        # Decay nach JEDEM Schritt (etwas flotter)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def _soft_update(self):
        # Polyak/Soft-Target-Update
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self) -> None:
        # NUr lernen, wenn genug Erfahrungen gesammelt wurden
        if len(self.replay_buffer) < self.cfg.learn_starts:
            return

        for _ in range(self.cfg.updates_per_step):
            states, actions, rewards, next_states, dones = self.sample(self.cfg.batch_size)

            states      = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions     = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones       = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
            # Q-Werte für aktuelle Zustände und Aktionen
            q_values = self.q_network(states).gather(1, actions)

            # Ziel-Q-Werte für nächste Zustände
            with torch.no_grad():
                # Wählt die beste Aktion aus q_net 
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                # Maximale Q-Werte für nächste Zustände aus dem Zielnetzwerk
                next_q = self.target_network(next_states).gather(1, next_actions)

                # Bellman-Gleichung, aber kein Zuwachs wenn done = 1 (Ende der Episode)
                target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

            # Verlust berechnen (MSE)
            loss = self.loss_fn(q_values, target)

            # Backpropagation und Optimierung
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            
            self._soft_update()
            

        # Target alle 100 Schritte aktualisieren
        self.step_count = getattr(self, 'step_count', 0) + 1
        if self.step_count % 5000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict()) 


