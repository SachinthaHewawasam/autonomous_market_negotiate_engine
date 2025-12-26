"""
Online training utilities for RL agent
"""
import os
import torch
from datetime import datetime


def train_from_experiences(agent, experiences, negotiation_id):
    """
    Train the RL agent from a batch of experiences
    
    Args:
        agent: BuyerAgent instance
        experiences: List of experience dictionaries
        negotiation_id: ID of the negotiation for logging
    """
    print(f"[Training] Starting online training from negotiation {negotiation_id}")
    print(f"[Training] Processing {len(experiences)} experiences")
    
    # Add experiences to replay buffer
    for exp in experiences:
        agent.replay_buffer.push(
            exp['state'],
            exp['action'],
            exp['reward'],
            exp['next_state'],
            exp['done']
        )
    
    # Train if we have enough experiences
    if len(agent.replay_buffer) >= agent.batch_size:
        # Perform multiple training steps
        num_updates = min(len(experiences), 5)  # Train up to 5 times per negotiation
        total_loss = 0.0
        
        for i in range(num_updates):
            loss = agent.train_step()
            total_loss += loss
            
        avg_loss = total_loss / num_updates
        print(f"[Training] Completed {num_updates} training steps, avg loss: {avg_loss:.4f}")
        print(f"[Training] Replay buffer size: {len(agent.replay_buffer)}")
        
        return avg_loss
    else:
        print(f"[Training] Not enough experiences yet ({len(agent.replay_buffer)}/{agent.batch_size})")
        return None


def save_model_checkpoint(agent, model_path, negotiation_count):
    """
    Save model checkpoint with version number
    
    Args:
        agent: BuyerAgent instance
        model_path: Base path for model
        negotiation_count: Number of negotiations completed
    """
    # Create checkpoints directory
    checkpoint_dir = os.path.join(os.path.dirname(model_path), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save with timestamp and negotiation count
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f'buyer_agent_n{negotiation_count}_{timestamp}.pth'
    )
    
    agent.save_model(checkpoint_path)
    print(f"[Training] Saved checkpoint: {checkpoint_path}")
    
    # Also update the main model
    agent.save_model(model_path)
    print(f"[Training] Updated main model: {model_path}")
    
    return checkpoint_path
