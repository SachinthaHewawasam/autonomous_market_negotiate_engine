"""
Simplified training script with better reward shaping and debugging.
This version focuses on getting the agent to learn basic successful negotiations first.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from market_env import MarketEnv
from buyer_agent import BuyerAgent
import torch
import os


def train_buyer_agent_simple(
    num_episodes: int = 500,
    max_steps_per_episode: int = 10,
    save_interval: int = 50,
    model_save_path: str = 'models/buyer_agent.pth',
    log_file: str = 'logs/training_log.json'
):
    """
    Simplified training with better reward shaping.
    """
    print("=" * 80)
    print("SIMPLIFIED TRAINING - DEBUGGING VERSION")
    print("=" * 80)
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("\n" + "=" * 80 + "\n")
    
    # Initialize environment
    env = MarketEnv(
        num_sellers=5,
        max_quantity_per_seller=50,
        max_negotiation_rounds=max_steps_per_episode,
        seed=42
    )
    
    # Initialize agent with adjusted hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = BuyerAgent(
        state_dim=9,
        action_dim=4,
        learning_rate=0.0001,  # Lower learning rate
        gamma=0.95,  # Slightly lower discount
        epsilon_start=1.0,
        epsilon_end=0.05,  # Higher minimum exploration
        epsilon_decay=0.998,  # Slower decay
        buffer_capacity=10000,
        batch_size=32,  # Smaller batch
        target_update_freq=20,  # More frequent updates
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    training_losses = []
    
    # Detailed debugging
    action_counts = {i: 0 for i in range(5)}
    
    print("Starting training...\n")
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Debug first few episodes
        debug = episode < 3
        
        if debug:
            print(f"\n=== Episode {episode} Debug ===")
            print(f"Request: {info['requested_quantity']} units, Budget: ${info['max_budget']:.2f}")
            print(f"Sellers: {[(i, s, p) for i, (s, p) in enumerate(zip(info['seller_stocks'], info['seller_prices']))]}")
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, training=True)
            action_type = int(action[0])
            action_counts[action_type] += 1
            
            if debug:
                print(f"  Round {episode_length + 1}: Action={action_type}, Seller={int(action[1])}, "
                      f"Price=${action[2]:.2f}, Qty={int(action[3])}")
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            if debug:
                print(f"    Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}")
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                training_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # End episode
        agent.end_episode()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if done:
            success_count += 1
        
        # Periodic evaluation
        if (episode + 1) % save_interval == 0:
            avg_reward = np.mean(episode_rewards[-save_interval:])
            avg_length = np.mean(episode_lengths[-save_interval:])
            success_rate = sum(1 for i in range(max(0, episode - save_interval + 1), episode + 1) 
                             if episode_rewards[i] > 50) / min(save_interval, episode + 1)
            
            print(f"\n[Episode {episode + 1}/{num_episodes}]")
            print(f"  Avg Reward (last {save_interval}): {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            if training_losses:
                print(f"  Avg Loss: {np.mean(training_losses[-100:]):.4f}")
            print(f"  Action Distribution: {action_counts}")
            
            # Save model
            os.makedirs('models', exist_ok=True)
            agent.save_model(model_save_path)
    
    # Final save
    agent.save_model(model_save_path)
    
    # Save training log
    os.makedirs('logs', exist_ok=True)
    
    training_log = {
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'final_success_rate': success_count / num_episodes,
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'training_losses': [float(l) for l in training_losses],
        'action_distribution': action_counts,
        'agent_stats': agent.get_statistics()
    }
    
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Success Rate: {success_count / num_episodes:.2%}")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Avg Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"  Action Distribution: {action_counts}")
    print(f"\nModel saved to: {model_save_path}")
    print(f"Logs saved to: {log_file}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, training_losses)
    
    return agent, training_log


def plot_training_curves(episode_rewards, episode_lengths, training_losses):
    """Plot training metrics"""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) > 50:
        axes[0, 0].plot(
            np.convolve(episode_rewards, np.ones(50)/50, mode='valid'),
            label='Moving Average (50)',
            linewidth=2
        )
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6, label='Episode Length')
    if len(episode_lengths) > 50:
        axes[0, 1].plot(
            np.convolve(episode_lengths, np.ones(50)/50, mode='valid'),
            label='Moving Average (50)',
            linewidth=2
        )
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss
    if training_losses:
        axes[1, 0].plot(training_losses, alpha=0.4, label='Loss')
        if len(training_losses) > 100:
            axes[1, 0].plot(
                np.convolve(training_losses, np.ones(100)/100, mode='valid'),
                label='Moving Average (100)',
                linewidth=2
            )
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Reward histogram
    axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to plots/training_curves.png")
    plt.close()


if __name__ == '__main__':
    # Train with debugging
    trained_agent, log = train_buyer_agent_simple(
        num_episodes=500,
        max_steps_per_episode=10,
        save_interval=50
    )
    
    print("\n✓ Training complete! Check the debug output above.")
    print("  If success rate is still 0%, there's a fundamental issue with the environment.")
    print("  Run 'python evaluate.py' to test the trained agent.")
