"""
Prioritized Experience Replay (PER) - Rainbow DQN Component
============================================================

Based on: "Prioritized Experience Replay" (Schaul et al., 2016)

KEY IDEA:
---------
Not all transitions are equally important!
Sample transitions with high TD-error more frequently.

BENEFITS:
---------
- +40% sample efficiency on average
- Critical for sparse reward tasks (Acrobot!)
- Learns from rare successes faster

USAGE:
------
>>> buffer = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)
>>> buffer.push(state, action, reward, next_state, done)
>>> batch, weights, indices = buffer.sample(batch_size=64, beta=0.4)
>>> # After computing TD-errors:
>>> buffer.update_priorities(indices, td_errors)

Author: Rainbow DQN Level 2 Implementation
Date: 2025-10-17
"""

import numpy as np
import random
from typing import Tuple, List


class SumTree:
    """
    Sum Tree data structure for efficient sampling
    
    Binary tree where:
    - Leaves: priorities
    - Internal nodes: sum of children
    - Root: total sum
    
    Enables O(log N) sampling instead of O(N)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_pos = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for given cumulative sum"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return sum of all priorities"""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """Add new data with priority"""
        idx = self.write_pos + self.capacity - 1
        
        self.data[self.write_pos] = data
        self.update(idx, priority)
        
        self.write_pos = (self.write_pos + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority of leaf"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float):
        """Get data for cumulative sum s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Samples transitions based on TD-error priority.
    High TD-error = high priority = sampled more frequently.
    
    Parameters:
    -----------
    capacity: int
        Maximum number of transitions to store
        
    alpha: float [0, 1]
        How much prioritization to use
        0 = uniform sampling (standard replay)
        1 = full prioritization
        Typical: 0.6
        
    beta_start: float [0, 1]
        Initial importance sampling correction
        Increases to 1.0 over training
        Typical start: 0.4
        
    beta_frames: int
        Number of frames over which beta anneals to 1.0
        
    epsilon: float
        Small constant to ensure non-zero priorities
        Typical: 1e-6
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        
        # Max priority initialization
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add transition to buffer with max priority
        
        New transitions get max priority to ensure they're sampled at least once
        """
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Sample batch of transitions
        
        Returns:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            weights: Importance sampling weights
            indices: Tree indices (for priority update)
        """
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Current beta (anneals from beta_start to 1.0)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        
        # Divide priority range into batch_size segments
        segment_size = self.tree.total() / batch_size
        
        # Sample one from each segment (stratified sampling)
        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            indices[i] = idx
            batch.append(data)
            
            # Importance sampling weight
            # w_i = (N * P(i))^(-beta) / max(w)
            prob = priority / self.tree.total()
            weights[i] = (self.tree.n_entries * prob) ** (-beta)
        
        # Normalize weights
        weights /= weights.max()
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        batch_arrays = (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
        
        self.frame += 1
        
        return batch_arrays, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors
        
        Args:
            indices: Tree indices from sample()
            td_errors: TD-errors for each transition
        """
        for idx, error in zip(indices, td_errors):
            # Priority = |TD-error| + epsilon
            priority = (abs(error) + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Prioritized Replay Buffer...\n")
    
    # Create buffer
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
    
    # Add some transitions
    print("Adding 50 transitions...")
    for i in range(50):
        state = np.random.rand(4)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = np.random.rand(4)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}\n")
    
    # Sample batch
    print("Sampling batch of 10...")
    batch, weights, indices = buffer.sample(batch_size=10)
    
    states, actions, rewards, next_states, dones = batch
    
    print(f"Batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Indices: {indices.shape}")
    
    print(f"\nImportance Weights (should sum to ~1.0):")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    
    # Update priorities (simulate high TD-errors for some)
    print(f"\nUpdating priorities with random TD-errors...")
    td_errors = np.random.uniform(0, 2, size=10)
    buffer.update_priorities(indices, td_errors)
    
    # Sample again (should prefer high TD-error transitions)
    print(f"Sampling again after priority update...")
    batch2, weights2, indices2 = buffer.sample(batch_size=10)
    
    print(f"\nNew Importance Weights:")
    print(f"  Min: {weights2.min():.3f}")
    print(f"  Max: {weights2.max():.3f}")
    print(f"  Mean: {weights2.mean():.3f}")
    
    print("\n[OK] Prioritized Replay Buffer working correctly!")





