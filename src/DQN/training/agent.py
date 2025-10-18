# Dieses Modul definiert:  
# - QNetwork: Ein neuronales Netzwerk, das die Q-Werte für gegebene Zustände und Aktionen vorhersagt.
# - ReplayBuffer: Ein Replay-Puffer zur Speicherung und Abfrage von Erfahrungen.
# - DQNAgent: Wählt Aktionen (Epsilon-Greedy-Strategie) und lernt per Bellman-Update)

# IMPORTS 
from __future__ import annotations
from argparse import _ActionsContainer
from typing import Deque, Tuple, List
from dataclasses import dataclass

# Emotion Engine Import mit Fallback
try:
    from ..core.emotion_engine import EmotionEngine
except ImportError:
    import os as _os, sys as _sys
    _CURR = _os.path.dirname(_os.path.abspath(__file__))
    _PKG_ROOT = _os.path.abspath(_os.path.join(_CURR, ".."))
    if _PKG_ROOT not in _sys.path:
        _sys.path.insert(0, _PKG_ROOT)
    from core.emotion_engine import EmotionEngine

import copy 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random, collections 
import numpy as np 

# BDH - artige PLastizität
class PlasticLinear (nn.Module):
    """ 
    Linear-LAyer mit additiver plastischer Synapsenkomponente (σ).
    Effektives Gewicht = W_base + σ. 
    σ wird lokal Hebbian-ähnlich upgedatet und durch Emotionen sklaiert (g(E)).
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=bias)
        # σ hat dieselbe Form wie die Basismatrix
        self.register_buffer('sigma', torch.zeros_like(self.base.weight))
        # Cache für PRe-/POst Aktivierunhgen (für HEbbian Matrix) 
        self._last_pre = None 
        self._last_post = None 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cache PRE Aktivierung 
        self._last_pre = x.detach()
        # Effektive Gewichte : W_eff = W + σ
        W_eff = self.base.weight + self.sigma
        y = F.linear(x, W_eff, self.base.bias)
        # Cache Post Aktivierung (vor Nichtlinearität) 
        self._last_post = y.detach()
        return y 

    @torch.no_grad()
    def plasticity_step(self, mod: float, eta: float = 2e-3, decay: float = 0.996, clip: float = 0.2):
        """ 
        σ_ij <- decay * σ_ij + eta * mod * <tanh(post_i) * pre_j_batch 
        - mod: g(E) (emotionaler Gain)
        - decay: Homeostase/Leck, verhindert runaway
        - clip: numerische Stabilierung 
        """
        if self._last_pre is None or self._last_post is None: 
            return 

        # Begrenzung Post_Aktivierung (biologisch plausibel, stabil)
        pre = self._last_pre                    # [B, in]
        post = torch.tanh(self._last_post / 2.0)      # [B, out]

        # Batch-Mittel des äußeren Produkts: (out x in)
        # post^T @ pre (über BAtch dimension gemittelt)
        delta = torch.einsum('bo,bi->oi', post, pre) / pre.shape[0] 
        
        # Leck + Hebbian-Update (emotion-moduliert)
        self.sigma.mul_(decay).add_(eta * float(mod) * delta)
          
        # Sicherheit
        if clip is not None: 
            self.sigma.clamp_(-clip, clip)


    @torch.no_grad()
    def reset_plasticity(self):
        self.sigma.zero_() 


# Q - NETWORK (das Funktionsapproximationsmodell)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim): 
        super().__init__()

        # Hidden LAyers als PlasticLinear (BDH_Style)
        self.fc1 = PlasticLinear(state_dim, 128)
        self.fc2 = PlasticLinear(128, 128)

        # Output bleibt klassisch 
        self.fc3 = nn.Linear(128, action_dim) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    @torch.no_grad()
    def plasticity_step(self, mod: float, eta: float, decay: float, clip: float):
        self.fc1.plasticity_step(mod=mod, eta=eta, decay=decay, clip=clip)
        self.fc2.plasticity_step(mod=mod, eta=eta, decay=decay, clip=clip)

    @torch.no_grad()
    def reset_plasticity(self):
        self.fc1.reset_plasticity()
        self.fc2.reset_plasticity()

# REPLAY BUFFER (speichert Erfahrungen)
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.5):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.pos = 0
        self.size = 0
        self.states = [None] * self.capacity
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = [None] * self.capacity
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        idx = self.pos
        self.states[idx] = np.array(state, copy=False)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = np.array(next_state, copy=False)
        self.dones[idx] = 1.0 if done else 0.0
        self.priorities[idx] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        assert self.size > 0
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        indices = np.random.choice(self.size, size=batch_size, replace=self.size < batch_size, p=probs)
        # IS Weights
        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-6)

        states = np.stack([self.states[i] for i in indices], axis=0)
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = np.stack([self.next_states[i] for i in indices], axis=0)
        dones = self.dones[indices]
        return indices, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.clip(priorities, 1e-6, 1e6)
        for idx, pr in zip(indices, priorities):
            self.priorities[int(idx)] = pr
            if pr > self.max_priority:
                self.max_priority = float(pr)

    def __len__(self) -> int:
        return self.size

# DQN AGENT (wählt Aktionen und lernt)
class DQNConfig:
    state_dim: int
    action_dim: int 
    lr: float = 5e-4  
    gamma: float = 0.99
    replay_capacity: int = 200000
    batch_size: int = 64
    learn_starts: int = 1000
    updates_per_step: int = 2
    tau: float = 0.01 
    grad_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 600
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    plastic_eta: float = 5e-4               # Lernrate σ
    plastic_decay: float = 0.997            # Homeostase (σ-LEck)
    plastic_clip: float = 0.1               # numerische Begrenzung 
    plastic_in_inference: bool = False      # optional 
    plastic_eta_infer: float = 2e-4         # kleinere Rate in Inferenz 

class DQNAgent:    # Agent-Parameter
    def __init__(self, state_dim, action_dim, config=None, emotion_engine=None):
        self.state_dim = state_dim
        self.action_dim = action_dim 

        self.cfg = DQNConfig()
        self.emotion_gain = 0.0 
        if config is not None:
        # Werte aus CONFIG überschreiben (Training Config aus train.py)
            self.cfg.gamma = config.get("gamma", self.cfg.gamma)
            self.cfg.lr = config.get("lr", self.cfg.lr)
            self.cfg.batch_size = config.get("batch_size", self.cfg.batch_size)
            self.cfg.replay_capacity = config.get("replay_capacity", self.cfg.replay_capacity)
            self.cfg.epsilon_start = config.get("epsilon_start", self.cfg.epsilon_start)
            self.cfg.epsilon_decay = config.get("epsilon_decay", self.cfg.epsilon_decay)
            if self.cfg.epsilon_decay >= 1.0:
                self.cfg.epsilon_decay = 0.995
            self.cfg.epsilon_end = config.get("epsilon_min", self.cfg.epsilon_end)

        if self.cfg.batch_size is not None:
            batch_size = self.cfg.batch_size

        self.device = torch.device(self.cfg.device)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Stabilere Targets 
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.cfg.lr) 
        # Robuster Q-Loss (Huber)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

        # Prioritized Replay Buffer
        self.prioritized_alpha = 0.5
        self.prioritized_beta_start = 0.4
        self.prioritized_beta_frames = 100000
        self.replay_buffer = PrioritizedReplayBuffer(self.cfg.replay_capacity, alpha=self.prioritized_alpha)

        # N-step settings
        self.n_step = 2
        self._nstep_queue = collections.deque(maxlen=self.n_step)
        self._frame_idx = 0
        
        # ε-greedy: startet hoch (viel Exploration), wird dann immer kleiner 
        self.epsilon = float(self.cfg.epsilon_start)
        self.epsilon_min = float(self.cfg.epsilon_end)
        self.epsilon_decay = self.cfg.epsilon_decay

        # Wenn im CONFIG eine Schritte Zahl ist, fallback. 
        if self.cfg.epsilon_decay >= 1.0:
            self.epsilon_decay = 0.995
        else: 
            self.epsilon_decay = float(self.cfg.epsilon_decay)

        # Emotion Engine optional aktivieren 
        self.emotion_enabled = True 
        print(f'[Debug] Emotion Engine aktiviert: {self.emotion_enabled}')

        if self.emotion_enabled:
            try: 
                self.emotion = EmotionEngine(init_state=0.4, alpha=0.85, target_return=60, gain=1.2, noise_std=0.05)
                self.emotion_gain = 0.4
                print("Emotion Engine aktiviert.")
                print(f'[Debug] EmotionEngine Typ: {type(self.emotion)}')
            
            except Exception as e: 
                print("Fehler beim Laden der Emotion Engine: {e}")
                self.emotion = None 
                self.emotion_gain = 0.0

        else: 
            self.emotion = None
            self.emotion_gain = 0.0
            print('Läuft im Baseline-MOdus ohne Emotion.') 
                
        self._last_eps = None

    def _emotion_gain(self) -> float:
        """
        g(E): Mappt Emotionswert (z.B. 0..1) auf einen Multiplikator ~ [0.8 .. 1.2]
        Robust gegen None/Fehler.
        """
        if not hasattr(self, "emotion_enabled") or not self.emotion_enabled or self.emotion is None:
            return 1.0
        try:
            # Dein Code nutzt self.emotion.value() ~ 0..1 (anpassbar)
            e = float(self.emotion.value())
        except Exception:
            e = 0.05
        # zentriere bei 0.05 (wie bei dir oben) und skaliere
        shift = (e - 0.05) * 2.0
        # Basiseinfluss
        gain = 1.0 + 0.6 * shift # skaliert auf [0.4 .. 1.6]
        return float(np.clip(gain, 0.7, 1.3))


    def _make_nstep_transition(self):
        # Aggregiere n-step Rückgabe aus Queue
        R = 0.0
        gamma = self.cfg.gamma
        next_state_n = None
        done_n = 0.0
        for i, (s, a, r, ns, d) in enumerate(self._nstep_queue):
            R += (gamma ** i) * float(r)
            next_state_n = ns
            done_n = 1.0 if d else done_n
            if d:
                break
        state0, action0 = self._nstep_queue[0][0], self._nstep_queue[0][1]
        return state0, action0, R, next_state_n, bool(done_n)

    def push(self, state, action, reward, next_state, done):
        # N-step Aggregation
        self._nstep_queue.append((state, action, reward, next_state, done))
        if len(self._nstep_queue) == self.n_step or done:
            s0, a0, Rn, nsn, dn = self._make_nstep_transition()
            self.replay_buffer.push(s0, a0, Rn, nsn, dn)
            # Falls nicht done, pop erstes Element weiterlaufen; bei done leeren
            if not done:
                self._nstep_queue.popleft()
            else:
                self._nstep_queue.clear()

    def __len__(self):
        return len(self.replay_buffer)

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def act(self, state):
        # Basis Epsilon 
        base_eps = float(self.epsilon)
        shift = 0.0

        # Sicherheit: Falls epsilon fehlerhaft
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            self.epsilon = self.cfg.epsilon_start

        # Emotionseinfluss
        if hasattr(self, "emotion_enabled") and self.emotion_enabled and self.emotion is not None:
            try:
                shift = (self.emotion.value() - 0.05) * 2.0
            except Exception:
                shift = 0.0

        # Emotionseinfluss begrenzen 
        shift = np.clip(shift, -1.0, 1.0)

        # Effektives Epsilon berechnen 
        eps_eff = self.epsilon * (1.0 + 0.2 * shift)
        eps_eff = max(self.cfg.epsilon_end, min(1.0, eps_eff))

        # Für Logging speichern
        self._last_eps = eps_eff

        # Epsilon-greedy mit EMOTION 
        if np.random.rand() < eps_eff: 
            action = np.random.randint(self.action_dim)
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
        # Schrittzähler für β-Annealing
        self._frame_idx += 1
        # Nur lernen, wenn genug Erfahrungen gesammelt wurden
        if len(self.replay_buffer) < self.cfg.learn_starts:
            return

        for _ in range(self.cfg.updates_per_step):
            beta = min(1.0, self.prioritized_beta_start + (1.0 - self.prioritized_beta_start) * (self._frame_idx / max(1, self.prioritized_beta_frames)))
            idxs, states, actions, rewards, next_states, dones, weights = self.replay_buffer.sample(self.cfg.batch_size, beta=beta)

            states      = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions     = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones       = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
            weights_t   = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
            # Q-Werte für aktuelle Zustände und Aktionen
            q_values = self.q_network(states).gather(1, actions)

            # Ziel-Q-Werte für nächste Zustände
            with torch.no_grad():
                # Wählt die beste Aktion aus q_net 
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                # Maximale Q-Werte für nächste Zustände aus dem Zielnetzwerk
                next_q = self.target_network(next_states).gather(1, next_actions)

                # N-Step Bellman-Target: R_n + (γ^n) * V(s_{t+n}) für nicht-terminale Sequenzen
                gamma_n = (self.cfg.gamma ** self.n_step)
                target = rewards + gamma_n * next_q * (1.0 - dones)

            # Verlust berechnen (Huber, per-sample)
            self.loss_fn.reduction = 'none'
            loss_per_sample = self.loss_fn(q_values, target)
            loss = (loss_per_sample * weights_t).mean()
            # σ-Regularisierung (falls vorhanden) zur Stabilisierung der BDH-Layer
            sigma_reg = 0.0
            if hasattr(self.q_network, 'fc1') and hasattr(self.q_network, 'fc2'):
                if hasattr(self.q_network.fc1, 'sigma') and hasattr(self.q_network.fc2, 'sigma'):
                    sigma_reg = 1e-5 * (self.q_network.fc1.sigma.pow(2).mean() + self.q_network.fc2.sigma.pow(2).mean())
            loss = loss + sigma_reg
            # einfachen mittleren Reward und TD Error berechnen
            td_errors = (target.detach() - q_values.detach()).abs()
            mean_reward = rewards.mean().item()
            td_error = td_errors.mean().item()

            if self.emotion_enabled and self.emotion is not None:
                try: 
                    self.emotion.update(mean_reward, td_error) 
                    
                except Exception as e: 
                    print(f'[WARN] EMotionEngine.update() fehlgeschlagen: {e}')

            # Backpropagation und Optimierung
            self.optimizer.zero_grad()
            loss.backward()
            # Strengeres Gradient Clipping für stabilere Updates
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
            self.optimizer.step()

            # Prioritäten aktualisieren
            with torch.no_grad():
                new_prios = td_errors.squeeze(1).clamp_min(1e-6).cpu().numpy()
                self.replay_buffer.update_priorities(idxs, new_prios)
            
            # BDH-artige Plastizität mit Emotions-Modulation 
            with torch.no_grad():
                mod = self.emotion_gain  # g(E)
                self.q_network.plasticity_step(
                    mod=mod,
                    eta=self.cfg.plastic_eta,
                    decay=self.cfg.plastic_decay,
                    clip=self.cfg.plastic_clip,
                    )

            self._soft_update()
            

        # Target alle 100 Schritte aktualisieren
        self.step_count = getattr(self, 'step_count', 0) + 1
        if self.step_count % 5000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict()) 
            
  
    def on_episode_end(self):
        # Restliche n-step Übergänge flushen
        while len(self._nstep_queue) > 0:
            s0, a0, Rn, nsn, dn = self._make_nstep_transition()
            self.replay_buffer.push(s0, a0, Rn, nsn, dn)
            self._nstep_queue.popleft()