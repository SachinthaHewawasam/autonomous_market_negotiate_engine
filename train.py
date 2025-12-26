import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from market_env import MarketEnv
from buyer_agent import BuyerAgent
import torch


def train_buyer_agent(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 10,
    save_interval: int = 100,
    eval_interval: int = 50,
    model_save_path: str = 'models/buyer_agent.pth',
    log_file: str = 'logs/training_log.json'
):
    """
    Train the buyer agent using reinforcement learning.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_interval: Save model every N episodes
        eval_interval: Evaluate every N episodes
        model_save_path: Path to save model
        log_file: Path to save training logs
    """
    print("=" * 80)
    print("AUTONOMOUS MARKET SIMULATION - BUYER AGENT TRAINING")
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
    
    # Initialize agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = BuyerAgent(
        state_dim=9,
        action_dim=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    training_losses = []
    
    # Training loop
    print("Starting training...\n")
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            
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
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            success_rate = success_count / (episode + 1)
            
            print(f"\n[Episode {episode + 1}/{num_episodes}]")
            print(f"  Avg Reward (last {eval_interval}): {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            if training_losses:
                print(f"  Avg Loss: {np.mean(training_losses[-100:]):.4f}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            import os
            os.makedirs('models', exist_ok=True)
            agent.save_model(model_save_path)
            print(f"\n✓ Model saved to {model_save_path}")
    
    # Final save
    agent.save_model(model_save_path)
    
    # Save training log
    import os
    os.makedirs('logs', exist_ok=True)
    
    training_log = {
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'final_success_rate': success_count / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
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
    print(f"\nModel saved to: {model_save_path}")
    print(f"Logs saved to: {log_file}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, training_losses)
    
    return agent, training_log


def plot_training_curves(episode_rewards, episode_lengths, training_losses):
    """Plot training metrics"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
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
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6, label='Episode Length')
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
    
    # Success rate over time
    window_size = 100
    success_rate = []
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - window_size)
        window_rewards = episode_rewards[start_idx:i+1]
        successes = sum(1 for r in window_rewards if r > 50)  # Threshold for success
        success_rate.append(successes / len(window_rewards))
    
    axes[1, 1].plot(success_rate, linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title(f'Success Rate (Rolling Window: {window_size})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to plots/training_curves.png")
    plt.close()


if __name__ == '__main__':
    # Train the agent
    trained_agent, log = train_buyer_agent(
        num_episodes=1000,
        max_steps_per_episode=10,
        save_interval=100,
        eval_interval=50
    )
    
    print("\n✓ Training complete! Use evaluate.py to compare with rule-based agent.")
