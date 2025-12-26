import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from market_env import MarketEnv
from buyer_agent import BuyerAgent, RuleBasedBuyerAgent
import torch


def evaluate_agent(agent, env, num_episodes: int = 100, agent_name: str = "Agent"):
    """
    Evaluate an agent's performance.
    
    Args:
        agent: Agent to evaluate
        env: Market environment
        num_episodes: Number of evaluation episodes
        agent_name: Name for logging
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_savings = 0
    negotiation_outcomes = []
    
    for episode in tqdm(range(num_episodes), desc=f"Evaluating {agent_name}"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (no exploration for evaluation)
            if isinstance(agent, BuyerAgent):
                action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if done:
            success_count += 1
            # Calculate savings (reward includes savings bonus)
            if episode_reward > 100:
                savings = (episode_reward - 100) * 10
                total_savings += savings
        
        negotiation_outcomes.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'success': done,
            'requested_qty': info['requested_quantity'],
            'max_budget': info['max_budget']
        })
    
    # Calculate metrics (convert numpy types to Python native types for JSON serialization)
    metrics = {
        'agent_name': agent_name,
        'num_episodes': int(num_episodes),
        'success_rate': float(success_count / num_episodes),
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'avg_savings': float(total_savings / success_count) if success_count > 0 else 0.0,
        'total_savings': float(total_savings),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'negotiation_outcomes': [
            {
                'episode': int(outcome['episode']),
                'reward': float(outcome['reward']),
                'length': int(outcome['length']),
                'success': bool(outcome['success']),
                'requested_qty': int(outcome['requested_qty']),
                'max_budget': float(outcome['max_budget'])
            }
            for outcome in negotiation_outcomes
        ]
    }
    
    return metrics


def compare_agents(
    num_episodes: int = 100,
    model_path: str = 'models/buyer_agent.pth',
    results_file: str = 'logs/evaluation_results.json'
):
    """
    Compare RL-based agent with rule-based agent.
    
    Args:
        num_episodes: Number of episodes for evaluation
        model_path: Path to trained RL agent model
        results_file: Path to save results
    """
    print("=" * 80)
    print("AGENT COMPARISON: RL-BASED vs RULE-BASED")
    print("=" * 80)
    print(f"\nEvaluation Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Model: {model_path}")
    print("\n" + "=" * 80 + "\n")
    
    # Initialize environment
    env = MarketEnv(
        num_sellers=5,
        max_quantity_per_seller=50,
        max_negotiation_rounds=10,
        seed=123  # Different seed for evaluation
    )
    
    # Initialize RL agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rl_agent = BuyerAgent(device=device)
    
    try:
        rl_agent.load_model(model_path)
        print(f"✓ Loaded trained RL agent from {model_path}\n")
    except FileNotFoundError:
        print(f"⚠ Warning: Could not load model from {model_path}")
        print("  Using untrained RL agent for comparison\n")
    
    # Initialize rule-based agent
    rule_agent = RuleBasedBuyerAgent()
    
    # Evaluate RL agent
    print("Evaluating RL-based agent...")
    rl_metrics = evaluate_agent(rl_agent, env, num_episodes, "RL-Based Agent")
    
    # Evaluate rule-based agent
    print("\nEvaluating rule-based agent...")
    rule_metrics = evaluate_agent(rule_agent, env, num_episodes, "Rule-Based Agent")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'RL-Based':<20} {'Rule-Based':<20} {'Improvement':<15}")
    print("-" * 85)
    
    metrics_to_compare = [
        ('Success Rate', 'success_rate', '{:.2%}'),
        ('Avg Reward', 'avg_reward', '{:.2f}'),
        ('Std Reward', 'std_reward', '{:.2f}'),
        ('Avg Episode Length', 'avg_length', '{:.2f}'),
        ('Avg Savings', 'avg_savings', '{:.2f}'),
        ('Total Savings', 'total_savings', '{:.2f}')
    ]
    
    for metric_name, metric_key, fmt in metrics_to_compare:
        rl_value = rl_metrics[metric_key]
        rule_value = rule_metrics[metric_key]
        
        if rule_value != 0:
            improvement = ((rl_value - rule_value) / rule_value) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{metric_name:<30} {fmt.format(rl_value):<20} {fmt.format(rule_value):<20} {improvement_str:<15}")
    
    # Save results
    import os
    os.makedirs('logs', exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'rl_agent': rl_metrics,
        'rule_based_agent': rule_metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Plot comparison
    plot_comparison(rl_metrics, rule_metrics)
    
    return results


def plot_comparison(rl_metrics, rule_metrics):
    """Plot comparison between RL and rule-based agents"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Success rate comparison
    agents = ['RL-Based', 'Rule-Based']
    success_rates = [rl_metrics['success_rate'], rule_metrics['success_rate']]
    
    axes[0, 0].bar(agents, success_rates, color=['#2ecc71', '#3498db'])
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(success_rates):
        axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # Average reward comparison
    avg_rewards = [rl_metrics['avg_reward'], rule_metrics['avg_reward']]
    
    axes[0, 1].bar(agents, avg_rewards, color=['#2ecc71', '#3498db'])
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Average Reward Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(avg_rewards):
        axes[0, 1].text(i, v + max(avg_rewards)*0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Reward distribution
    axes[1, 0].hist(rl_metrics['episode_rewards'], bins=20, alpha=0.7, label='RL-Based', color='#2ecc71')
    axes[1, 0].hist(rule_metrics['episode_rewards'], bins=20, alpha=0.7, label='Rule-Based', color='#3498db')
    axes[1, 0].set_xlabel('Episode Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode length comparison
    avg_lengths = [rl_metrics['avg_length'], rule_metrics['avg_length']]
    
    axes[1, 1].bar(agents, avg_lengths, color=['#2ecc71', '#3498db'])
    axes[1, 1].set_ylabel('Average Episode Length')
    axes[1, 1].set_title('Average Episode Length Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(avg_lengths):
        axes[1, 1].text(i, v + max(avg_lengths)*0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/agent_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to plots/agent_comparison.png")
    plt.close()


def detailed_analysis(results_file: str = 'logs/evaluation_results.json'):
    """Perform detailed analysis of evaluation results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    rl_outcomes = results['rl_agent']['negotiation_outcomes']
    rule_outcomes = results['rule_based_agent']['negotiation_outcomes']
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    # Analyze by request size
    print("\nPerformance by Request Size:")
    print("-" * 80)
    
    for agent_name, outcomes in [('RL-Based', rl_outcomes), ('Rule-Based', rule_outcomes)]:
        small_requests = [o for o in outcomes if o['requested_qty'] < 80]
        medium_requests = [o for o in outcomes if 80 <= o['requested_qty'] < 120]
        large_requests = [o for o in outcomes if o['requested_qty'] >= 120]
        
        print(f"\n{agent_name}:")
        for category, requests in [('Small (<80)', small_requests), 
                                   ('Medium (80-120)', medium_requests),
                                   ('Large (≥120)', large_requests)]:
            if requests:
                success_rate = sum(1 for r in requests if r['success']) / len(requests)
                avg_reward = np.mean([r['reward'] for r in requests])
                print(f"  {category:20} Success: {success_rate:.2%}  Avg Reward: {avg_reward:.2f}")


if __name__ == '__main__':
    # Compare agents
    results = compare_agents(
        num_episodes=100,
        model_path='models/buyer_agent.pth'
    )
    
    # Detailed analysis
    detailed_analysis()
    
    print("\n✓ Evaluation complete!")
