import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from collections import deque
import random


class NegotiationPolicyNetwork(nn.Module):
    """
    Neural network for the buyer agent's negotiation policy.
    
    Architecture:
    - Input: Market state (9 features)
    - Hidden layers: 2 layers with ReLU activation
    - Output: Action values (4 continuous values)
    """
    
    def __init__(self, state_dim: int = 9, action_dim: int = 4, hidden_dim: int = 128):
        super(NegotiationPolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Action output layers with specific constraints
        self.action_type_head = nn.Linear(hidden_dim, 5)  # 5 action types
        self.seller_id_head = nn.Linear(hidden_dim, 1)  # Seller selection
        self.price_head = nn.Linear(hidden_dim, 1)  # Price offer
        self.quantity_head = nn.Linear(hidden_dim, 1)  # Quantity offer
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        features = self.feature_extractor(state)
        
        # Action type (discrete, will use softmax)
        action_type_logits = self.action_type_head(features)
        action_type = torch.softmax(action_type_logits, dim=-1)
        action_type_value = torch.argmax(action_type, dim=-1, keepdim=True).float()
        
        # Seller ID (continuous, will be clamped)
        seller_id = torch.sigmoid(self.seller_id_head(features)) * 4  # 0-4 range
        
        # Price (continuous, scaled to reasonable range)
        price = torch.sigmoid(self.price_head(features)) * 20 + 5  # 5-25 range
        
        # Quantity (continuous, scaled to reasonable range)
        quantity = torch.sigmoid(self.quantity_head(features)) * 150 + 20  # 20-170 range
        
        # Concatenate all action components
        action = torch.cat([action_type_value, seller_id, price, quantity], dim=-1)
        
        return action


class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class BuyerAgent:
    """
    Reinforcement learning-based buyer agent.
    
    Uses Deep Q-Learning (DQN) to learn negotiation policy.
    The agent learns through interaction with the market environment.
    
    Key features:
    - Neural network policy
    - Experience replay
    - Epsilon-greedy exploration
    - Target network for stability
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 4,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = 'cpu'
    ):
        """
        Initialize the buyer agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_capacity: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Policy network (online)
        self.policy_net = NegotiationPolicyNetwork(state_dim, action_dim).to(device)
        
        # Target network (for stability)
        self.target_net = NegotiationPolicyNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.update_count = 0
        self.episode_count = 0
        self.total_reward = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (enables exploration)
        
        Returns:
            Action array [action_type, seller_id, price, quantity]
        """
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            requested_qty = state[0] if len(state) > 0 else 100
            action = np.array([
                random.randint(0, 4),  # action_type
                random.randint(0, 4),  # seller_id
                random.uniform(8, 15),  # price (reasonable range)
                min(requested_qty * 1.1, random.uniform(50, 120))  # quantity
            ], dtype=np.float32)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net(state_tensor)
                action = action_tensor.cpu().numpy()[0]
                # Ensure action is in valid range
                action[0] = np.clip(action[0], 0, 4)  # action_type
                action[1] = np.clip(action[1], 0, 4)  # seller_id
                action[2] = np.clip(action[2], 5, 25)  # price
                action[3] = np.clip(action[3], 20, 170)  # quantity
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_reward += reward
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states)
            # For simplicity, use the price component as Q-value proxy
            next_q_values = next_q[:, 2]  # Price component
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (MSE between current and target Q values)
        current_q_values = current_q[:, 2]  # Price component
        loss = self.criterion(current_q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Record loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """Called at the end of an episode"""
        self.episode_count += 1
        self.update_epsilon()
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'update_count': self.update_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.update_count = checkpoint['update_count']
    
    def get_statistics(self) -> dict:
        """Get training statistics"""
        return {
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'buffer_size': len(self.replay_buffer)
        }


class RuleBasedBuyerAgent:
    """
    Rule-based buyer agent for comparison.
    
    Uses fixed heuristics instead of learning:
    1. Start with low offers
    2. Gradually increase price
    3. Form coalitions when needed
    4. Prefer high-trust sellers
    """
    
    def __init__(self, initial_offer_ratio: float = 0.8, increment_ratio: float = 0.05):
        self.initial_offer_ratio = initial_offer_ratio
        self.increment_ratio = increment_ratio
        self.current_round = 0
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using fixed rules.
        
        State: [requested_qty, best_price, best_qty, round, num_available_sellers,
                avg_trust, coalition_size, current_offer_price, current_offer_qty]
        """
        requested_qty = state[0]
        best_price = state[1]
        best_qty = state[2]
        round_num = state[3]
        num_available = state[4]
        avg_trust = state[5]
        coalition_size = state[6]
        current_offer_price = state[7]
        current_offer_qty = state[8]
        
        # Determine action type
        if best_qty >= requested_qty:
            # Single seller can fulfill
            if current_offer_price > 0:
                # There's an active offer
                if current_offer_price <= best_price * 1.1:
                    # Accept if reasonable
                    action_type = 2  # Accept
                else:
                    # Counteroffer
                    action_type = 1
            else:
                # Make initial offer
                action_type = 0
        else:
            # Need coalition
            action_type = 4  # Propose coalition
        
        # Select seller (prefer high trust, low price)
        seller_id = 0  # Simplified: always select first seller
        
        # Calculate offer price
        if action_type == 0:  # Initial offer
            offer_price = best_price * self.initial_offer_ratio
        elif action_type == 1:  # Counteroffer
            offer_price = current_offer_price * (1 - self.increment_ratio)
        else:
            offer_price = best_price
        
        # Quantity
        quantity = min(requested_qty, best_qty)
        
        action = np.array([action_type, seller_id, offer_price, quantity], dtype=np.float32)
        
        return action
    
    def reset(self):
        """Reset for new episode"""
        self.current_round = 0
