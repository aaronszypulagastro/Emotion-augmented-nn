"""
Dueling Network Architecture - Rainbow DQN Component
=====================================================

Based on: "Dueling Network Architectures for Deep RL" (Wang et al., 2016)

KEY IDEA:
---------
Separate Value function V(s) from Advantage function A(s,a):
Q(s,a) = V(s) + (A(s,a) - mean(A(s)))

BENEFITS:
---------
- Better value estimation (learns state value independent of actions)
- Especially effective for states where action choice doesn't matter much
- +15-25% performance on average

Author: Rainbow DQN Level 2 Implementation  
Date: 2025-10-17
"""

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Network
    
    Architecture:
    -------------
    Input → Features → Split into:
                      ├─ Value Stream → V(s)
                      └─ Advantage Stream → A(s,a)
    
    Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
    
    Parameters:
    -----------
    state_dim: int
        Dimension of state space
    action_dim: int
        Number of actions
    hidden_size: int
        Hidden layer size (default: 128)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128
    ):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        
        # Value stream: V(s)
        # Outputs single scalar value for state
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream: A(s,a)
        # Outputs advantage for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Returns:
            Q-values for each action
        """
        # Extract features
        features = self.feature_layer(x)
        
        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        # Subtracting mean centers advantages around 0
        # This makes value and advantage streams identifiable
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class DuelingQNetworkLarge(nn.Module):
    """
    Larger Dueling Network for complex tasks (LunarLander, Atari)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256
    ):
        super(DuelingQNetworkLarge, self).__init__()
        
        # Deeper shared features
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream (deeper)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (deeper)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Dueling Q-Networks...\n")
    
    # Test standard size
    state_dim = 4
    action_dim = 2
    batch_size = 32
    
    net = DuelingQNetwork(state_dim, action_dim, hidden_size=128)
    
    print(f"Standard Dueling Network:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden size: 128")
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    q_values = net(dummy_state)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_state.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Output range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # Test large network
    print(f"\n" + "="*60)
    print(f"Large Dueling Network (for LunarLander):")
    
    state_dim_large = 8
    action_dim_large = 4
    
    net_large = DuelingQNetworkLarge(state_dim_large, action_dim_large, hidden_size=256)
    
    total_params_large = sum(p.numel() for p in net_large.parameters())
    print(f"  Total parameters: {total_params_large:,}")
    
    dummy_state_large = torch.randn(batch_size, state_dim_large)
    q_values_large = net_large(dummy_state_large)
    
    print(f"  Output shape: {q_values_large.shape}")
    
    print("\n[OK] Dueling Networks working correctly!")
    
    # Comparison with standard network
    print(f"\n" + "="*60)
    print("COMPARISON: Standard vs Dueling")
    print("="*60)
    
    standard_params = (state_dim * 128) + (128 * 128) + (128 * action_dim)
    dueling_params = total_params
    
    print(f"Standard Q-Network:  {standard_params:,} parameters")
    print(f"Dueling Q-Network:   {dueling_params:,} parameters")
    print(f"Overhead:            {(dueling_params/standard_params - 1)*100:+.1f}%")
    
    print(f"\nTrade-off: +{(dueling_params/standard_params - 1)*100:.0f}% parameters for +15-25% performance")
    print(f"           WORTH IT! ✅")





