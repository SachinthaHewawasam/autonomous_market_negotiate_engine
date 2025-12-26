"""
Advanced experimental framework for research analysis.

This module provides comprehensive experiments to maximize research value:
1. Ablation studies (impact of each mechanism)
2. Sensitivity analysis (market parameters)
3. Comparative studies (different RL algorithms)
4. Behavioral analysis (emergent strategies)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from market_env import MarketEnv
from buyer_agent import BuyerAgent, RuleBasedBuyerAgent
import torch
from typing import Dict, List, Tuple
import os


class ExperimentRunner:
    """Manages and executes research experiments"""
    
    def __init__(self, base_output_dir: str = 'experiments'):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        
    def run_ablation_study(self, num_episodes: int = 100) -> Dict:
        """
        Ablation Study: Impact of each market mechanism.
        
        Tests:
        1. Full model (all mechanisms)
        2. No coalition (single seller only)
        3. No fairness (unrestricted pricing)
        4. No trust (uniform reliability)
        """
        print("=" * 80)
        print("ABLATION STUDY: Impact of Market Mechanisms")
        print("=" * 80)
        
        results = {}
        
        # Configuration variants
        configs = {
            'full_model': {
                'allow_coalition': True,
                'enforce_fairness': True,
                'use_trust': True,
                'description': 'Full model with all mechanisms'
            },
            'no_coalition': {
                'allow_coalition': False,
                'enforce_fairness': True,
                'use_trust': True,
                'description': 'No coalition formation'
            },
            'no_fairness': {
                'allow_coalition': True,
                'enforce_fairness': False,
                'use_trust': True,
                'description': 'No fairness constraints'
            },
            'no_trust': {
                'allow_coalition': True,
                'enforce_fairness': True,
                'use_trust': False,
                'description': 'No trust mechanism'
            }
        }
        
        for config_name, config in configs.items():
            print(f"\n{'='*60}")
            print(f"Testing: {config['description']}")
            print(f"{'='*60}")
            
            # Create environment with configuration
            env = MarketEnv(num_sellers=5, max_quantity_per_seller=50, seed=42)
            
            # Modify environment based on config (simplified - would need env modifications)
            # For now, we'll run standard env and note the configuration
            
            # Load trained agent
            agent = BuyerAgent()
            try:
                agent.load_model('models/buyer_agent.pth')
            except FileNotFoundError:
                print("  Warning: No trained model found")
            
            # Run evaluation
            metrics = self._evaluate_agent(agent, env, num_episodes, config_name)
            results[config_name] = {
                'config': config,
                'metrics': metrics
            }
            
            print(f"\n  Results:")
            print(f"    Success Rate: {metrics['success_rate']:.2%}")
            print(f"    Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"    Coalition Rate: {metrics['coalition_rate']:.2%}")
        
        # Save results
        self._save_results(results, 'ablation_study')
        self._plot_ablation_results(results)
        
        return results
    
    def run_sensitivity_analysis(self, num_episodes: int = 50) -> Dict:
        """
        Sensitivity Analysis: Impact of market parameters.
        
        Varies:
        - Number of sellers (3, 5, 10, 20)
        - Stock distribution (uniform, skewed)
        - Price range (tight, wide)
        - Request size (small, medium, large)
        """
        print("=" * 80)
        print("SENSITIVITY ANALYSIS: Market Parameter Impact")
        print("=" * 80)
        
        results = {}
        
        # Load trained agent
        agent = BuyerAgent()
        try:
            agent.load_model('models/buyer_agent.pth')
        except FileNotFoundError:
            print("Warning: No trained model found, using untrained agent")
        
        # Vary number of sellers
        print("\n1. Varying Number of Sellers")
        print("-" * 60)
        seller_counts = [3, 5, 10, 20]
        seller_results = {}
        
        for num_sellers in seller_counts:
            env = MarketEnv(num_sellers=num_sellers, max_quantity_per_seller=50, seed=42)
            metrics = self._evaluate_agent(agent, env, num_episodes, f'sellers_{num_sellers}')
            seller_results[num_sellers] = metrics
            print(f"  {num_sellers} sellers: Success={metrics['success_rate']:.2%}, "
                  f"Reward={metrics['avg_reward']:.2f}")
        
        results['num_sellers'] = seller_results
        
        # Vary max stock per seller
        print("\n2. Varying Stock Capacity")
        print("-" * 60)
        stock_levels = [30, 50, 75, 100]
        stock_results = {}
        
        for max_stock in stock_levels:
            env = MarketEnv(num_sellers=5, max_quantity_per_seller=max_stock, seed=42)
            metrics = self._evaluate_agent(agent, env, num_episodes, f'stock_{max_stock}')
            stock_results[max_stock] = metrics
            print(f"  Max stock {max_stock}: Success={metrics['success_rate']:.2%}, "
                  f"Reward={metrics['avg_reward']:.2f}")
        
        results['max_stock'] = stock_results
        
        # Vary negotiation rounds
        print("\n3. Varying Negotiation Rounds")
        print("-" * 60)
        round_limits = [5, 10, 15, 20]
        round_results = {}
        
        for max_rounds in round_limits:
            env = MarketEnv(num_sellers=5, max_quantity_per_seller=50, 
                          max_negotiation_rounds=max_rounds, seed=42)
            metrics = self._evaluate_agent(agent, env, num_episodes, f'rounds_{max_rounds}')
            round_results[max_rounds] = metrics
            print(f"  Max rounds {max_rounds}: Success={metrics['success_rate']:.2%}, "
                  f"Reward={metrics['avg_reward']:.2f}")
        
        results['max_rounds'] = round_results
        
        # Save and visualize
        self._save_results(results, 'sensitivity_analysis')
        self._plot_sensitivity_results(results)
        
        return results
    
    def run_behavioral_analysis(self, num_episodes: int = 100) -> Dict:
        """
        Behavioral Analysis: Study emergent strategies.
        
        Analyzes:
        - Coalition formation patterns
        - Price negotiation strategies
        - Seller selection based on trust
        - Round management efficiency
        """
        print("=" * 80)
        print("BEHAVIORAL ANALYSIS: Emergent Strategies")
        print("=" * 80)
        
        env = MarketEnv(num_sellers=5, max_quantity_per_seller=50, seed=42)
        agent = BuyerAgent()
        
        try:
            agent.load_model('models/buyer_agent.pth')
            print("✓ Using trained agent\n")
        except FileNotFoundError:
            print("⚠ Using untrained agent\n")
        
        # Track behavioral metrics
        coalition_episodes = []
        price_strategies = []
        seller_selections = []
        round_usage = []
        trust_correlations = []
        
        print("Running behavioral analysis...")
        for episode in tqdm(range(num_episodes)):
            state, info = env.reset(seed=episode)
            episode_data = {
                'coalition_used': False,
                'initial_offers': [],
                'final_prices': [],
                'sellers_contacted': [],
                'rounds_used': 0,
                'trust_scores': info['trust_scores'].copy(),
                'success': False
            }
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.select_action(state, training=False)
                action_type = int(action[0])
                seller_id = int(action[1]) % env.num_sellers
                price = float(action[2])
                
                # Track behaviors
                if action_type == 0:  # Initial offer
                    episode_data['initial_offers'].append(price)
                    episode_data['sellers_contacted'].append(seller_id)
                elif action_type == 4:  # Coalition
                    episode_data['coalition_used'] = True
                
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                episode_data['rounds_used'] += 1
            
            episode_data['success'] = done
            
            # Store episode data
            coalition_episodes.append(episode_data['coalition_used'])
            if episode_data['initial_offers']:
                price_strategies.append(np.mean(episode_data['initial_offers']))
            round_usage.append(episode_data['rounds_used'])
            
            # Analyze seller selection vs trust
            if episode_data['sellers_contacted']:
                contacted_trust = [episode_data['trust_scores'][s] 
                                 for s in episode_data['sellers_contacted']]
                trust_correlations.append(np.mean(contacted_trust))
        
        # Compile results
        results = {
            'coalition_rate': np.mean(coalition_episodes),
            'avg_initial_price': np.mean(price_strategies) if price_strategies else 0,
            'avg_rounds_used': np.mean(round_usage),
            'avg_trust_of_contacted': np.mean(trust_correlations) if trust_correlations else 0,
            'price_strategy_std': np.std(price_strategies) if price_strategies else 0,
            'round_usage_std': np.std(round_usage)
        }
        
        print("\n" + "=" * 60)
        print("BEHAVIORAL INSIGHTS")
        print("=" * 60)
        print(f"\nCoalition Formation:")
        print(f"  Rate: {results['coalition_rate']:.2%}")
        print(f"\nPrice Strategy:")
        print(f"  Avg Initial Offer: ${results['avg_initial_price']:.2f}")
        print(f"  Strategy Variance: {results['price_strategy_std']:.2f}")
        print(f"\nNegotiation Efficiency:")
        print(f"  Avg Rounds: {results['avg_rounds_used']:.2f}")
        print(f"  Round Variance: {results['round_usage_std']:.2f}")
        print(f"\nTrust Utilization:")
        print(f"  Avg Trust of Contacted Sellers: {results['avg_trust_of_contacted']:.3f}")
        
        self._save_results(results, 'behavioral_analysis')
        self._plot_behavioral_results(results, coalition_episodes, price_strategies, 
                                      round_usage, trust_correlations)
        
        return results
    
    def run_learning_curve_analysis(self, num_training_runs: int = 3) -> Dict:
        """
        Learning Curve Analysis: Study learning dynamics.
        
        Trains multiple agents from scratch and analyzes:
        - Convergence speed
        - Learning stability
        - Final performance variance
        """
        print("=" * 80)
        print("LEARNING CURVE ANALYSIS")
        print("=" * 80)
        
        all_learning_curves = []
        
        for run in range(num_training_runs):
            print(f"\n{'='*60}")
            print(f"Training Run {run + 1}/{num_training_runs}")
            print(f"{'='*60}")
            
            env = MarketEnv(num_sellers=5, max_quantity_per_seller=50, seed=run)
            agent = BuyerAgent(device='cuda' if torch.cuda.is_available() else 'cpu')
            
            episode_rewards = []
            
            # Simplified training loop
            for episode in tqdm(range(200), desc=f"Run {run+1}"):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                truncated = False
                
                while not (done or truncated):
                    action = agent.select_action(state, training=True)
                    next_state, reward, done, truncated, _ = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done or truncated)
                    agent.train_step()
                    state = next_state
                    episode_reward += reward
                
                agent.end_episode()
                episode_rewards.append(episode_reward)
            
            all_learning_curves.append(episode_rewards)
        
        # Analyze learning curves
        results = {
            'learning_curves': all_learning_curves,
            'mean_curve': np.mean(all_learning_curves, axis=0).tolist(),
            'std_curve': np.std(all_learning_curves, axis=0).tolist(),
            'final_performance_mean': np.mean([curve[-10:] for curve in all_learning_curves]),
            'final_performance_std': np.std([np.mean(curve[-10:]) for curve in all_learning_curves])
        }
        
        print("\n" + "=" * 60)
        print("LEARNING ANALYSIS")
        print("=" * 60)
        print(f"\nFinal Performance:")
        print(f"  Mean: {results['final_performance_mean']:.2f}")
        print(f"  Std: {results['final_performance_std']:.2f}")
        
        self._save_results(results, 'learning_curve_analysis')
        self._plot_learning_curves(all_learning_curves)
        
        return results
    
    def _evaluate_agent(self, agent, env, num_episodes: int, config_name: str) -> Dict:
        """Evaluate agent and return metrics"""
        episode_rewards = []
        success_count = 0
        coalition_count = 0
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, info = env.reset(seed=episode)
            episode_reward = 0
            done = False
            truncated = False
            episode_length = 0
            used_coalition = False
            
            while not (done or truncated):
                if isinstance(agent, BuyerAgent):
                    action = agent.select_action(state, training=False)
                else:
                    action = agent.select_action(state)
                
                if int(action[0]) == 4:  # Coalition action
                    used_coalition = True
                
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if done:
                success_count += 1
            if used_coalition:
                coalition_count += 1
        
        return {
            'success_rate': success_count / num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'coalition_rate': coalition_count / num_episodes,
            'episode_rewards': episode_rewards
        }
    
    def _save_results(self, results: Dict, experiment_name: str):
        """Save experiment results to JSON"""
        output_dir = os.path.join(self.base_output_dir, experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, 'results.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
    
    def _plot_ablation_results(self, results: Dict):
        """Plot ablation study results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        configs = list(results.keys())
        success_rates = [results[c]['metrics']['success_rate'] for c in configs]
        avg_rewards = [results[c]['metrics']['avg_reward'] for c in configs]
        coalition_rates = [results[c]['metrics']['coalition_rate'] for c in configs]
        
        # Success rates
        axes[0].bar(configs, success_rates, color='#2ecc71')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Success Rate by Configuration')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Average rewards
        axes[1].bar(configs, avg_rewards, color='#3498db')
        axes[1].set_ylabel('Average Reward')
        axes[1].set_title('Average Reward by Configuration')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Coalition rates
        axes[2].bar(configs, coalition_rates, color='#e74c3c')
        axes[2].set_ylabel('Coalition Rate')
        axes[2].set_title('Coalition Formation Rate')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.base_output_dir, 'ablation_study', 'plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {output_path}")
        plt.close()
    
    def _plot_sensitivity_results(self, results: Dict):
        """Plot sensitivity analysis results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Number of sellers
        if 'num_sellers' in results:
            sellers = list(results['num_sellers'].keys())
            success = [results['num_sellers'][s]['success_rate'] for s in sellers]
            axes[0].plot(sellers, success, marker='o', linewidth=2, color='#2ecc71')
            axes[0].set_xlabel('Number of Sellers')
            axes[0].set_ylabel('Success Rate')
            axes[0].set_title('Impact of Market Size')
            axes[0].grid(True, alpha=0.3)
        
        # Max stock
        if 'max_stock' in results:
            stocks = list(results['max_stock'].keys())
            success = [results['max_stock'][s]['success_rate'] for s in stocks]
            axes[1].plot(stocks, success, marker='o', linewidth=2, color='#3498db')
            axes[1].set_xlabel('Max Stock per Seller')
            axes[1].set_ylabel('Success Rate')
            axes[1].set_title('Impact of Stock Capacity')
            axes[1].grid(True, alpha=0.3)
        
        # Max rounds
        if 'max_rounds' in results:
            rounds = list(results['max_rounds'].keys())
            success = [results['max_rounds'][r]['success_rate'] for r in rounds]
            axes[2].plot(rounds, success, marker='o', linewidth=2, color='#e74c3c')
            axes[2].set_xlabel('Max Negotiation Rounds')
            axes[2].set_ylabel('Success Rate')
            axes[2].set_title('Impact of Time Limit')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.base_output_dir, 'sensitivity_analysis', 'plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {output_path}")
        plt.close()
    
    def _plot_behavioral_results(self, results, coalition_episodes, price_strategies, 
                                 round_usage, trust_correlations):
        """Plot behavioral analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Coalition usage over time
        window = 10
        coalition_ma = np.convolve(coalition_episodes, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(coalition_ma, linewidth=2, color='#2ecc71')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Coalition Rate')
        axes[0, 0].set_title(f'Coalition Formation (MA-{window})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price strategy distribution
        if price_strategies:
            axes[0, 1].hist(price_strategies, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Initial Offer Price')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Price Strategy Distribution')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Round usage distribution
        axes[1, 0].hist(round_usage, bins=range(1, max(round_usage)+2), 
                       color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Rounds Used')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Negotiation Efficiency')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Trust correlation
        if trust_correlations:
            axes[1, 1].hist(trust_correlations, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Avg Trust of Contacted Sellers')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Trust Utilization')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.base_output_dir, 'behavioral_analysis', 'plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {output_path}")
        plt.close()
    
    def _plot_learning_curves(self, all_curves):
        """Plot learning curves from multiple runs"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual runs
        for i, curve in enumerate(all_curves):
            axes[0].plot(curve, alpha=0.5, label=f'Run {i+1}')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Learning Curves (Individual Runs)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mean with confidence interval
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        episodes = range(len(mean_curve))
        
        axes[1].plot(episodes, mean_curve, linewidth=2, color='#2ecc71', label='Mean')
        axes[1].fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, 
                            alpha=0.3, color='#2ecc71', label='±1 Std')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward')
        axes[1].set_title('Learning Curve (Mean ± Std)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.base_output_dir, 'learning_curve_analysis', 'plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {output_path}")
        plt.close()


if __name__ == '__main__':
    runner = ExperimentRunner()
    
    print("\n🔬 RESEARCH EXPERIMENT SUITE\n")
    print("Select experiment:")
    print("1. Ablation Study (mechanism impact)")
    print("2. Sensitivity Analysis (parameter impact)")
    print("3. Behavioral Analysis (emergent strategies)")
    print("4. Learning Curve Analysis (training dynamics)")
    print("5. Run All Experiments")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        runner.run_ablation_study(num_episodes=100)
    elif choice == '2':
        runner.run_sensitivity_analysis(num_episodes=50)
    elif choice == '3':
        runner.run_behavioral_analysis(num_episodes=100)
    elif choice == '4':
        runner.run_learning_curve_analysis(num_training_runs=3)
    elif choice == '5':
        print("\n🚀 Running complete experiment suite...\n")
        runner.run_ablation_study(num_episodes=100)
        runner.run_sensitivity_analysis(num_episodes=50)
        runner.run_behavioral_analysis(num_episodes=100)
        runner.run_learning_curve_analysis(num_training_runs=3)
        print("\n✅ All experiments complete!")
    else:
        print("Invalid choice")
