"""
Multi-Agent Market Environment
Multiple buyer agents competing for limited resources
"""
import numpy as np
from typing import List, Dict, Tuple
from market_env import MarketEnv
from buyer_agent import BuyerAgent
import copy


class MultiAgentMarket:
    """
    Manages multiple competing buyer agents in a shared marketplace.
    Demonstrates emergent behavior, strategic competition, and Nash equilibrium.
    """
    
    def __init__(self, num_buyers: int = 3, num_sellers: int = 3, 
                 max_quantity_per_seller: int = 50):
        """
        Initialize multi-agent market
        
        Args:
            num_buyers: Number of competing buyer agents
            num_sellers: Number of sellers in the market
            max_quantity_per_seller: Maximum stock per seller
        """
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        
        # Create individual environments for each buyer
        self.buyer_envs = [
            MarketEnv(
                num_sellers=num_sellers,
                max_quantity_per_seller=max_quantity_per_seller,
                max_negotiation_rounds=10
            ) for _ in range(num_buyers)
        ]
        
        # Create buyer agents with different strategies
        self.buyer_agents = []
        self.buyer_names = []
        
        for i in range(num_buyers):
            agent = BuyerAgent(state_dim=9, action_dim=4)
            self.buyer_agents.append(agent)
            self.buyer_names.append(f"Buyer_{i+1}")
        
        # Competition state
        self.seller_inventory = {}  # Track remaining inventory
        self.buyer_requests = {}  # Track what each buyer wants
        self.competition_history = []
        
    def load_agents(self, model_paths: List[str], strategies: List[str] = None):
        """
        Load pre-trained agents with different strategies
        
        Args:
            model_paths: Paths to model files
            strategies: List of strategy types ['aggressive', 'conservative', 'balanced']
        """
        if strategies is None:
            strategies = ['aggressive', 'conservative', 'balanced']
        
        for i, (agent, path) in enumerate(zip(self.buyer_agents, model_paths)):
            try:
                agent.load_model(path)
                
                # Apply strategy modifications
                strategy = strategies[i % len(strategies)]
                
                if strategy == 'aggressive':
                    # Higher exploration, risk-taking
                    agent.epsilon = 0.3  # More random actions
                    self.buyer_names[i] = f"Buyer_{i+1}_Aggressive"
                    print(f"✓ Loaded {self.buyer_names[i]} (Aggressive: High risk, fast decisions)")
                    
                elif strategy == 'conservative':
                    # Lower exploration, safer choices
                    agent.epsilon = 0.05  # Mostly exploit learned policy
                    self.buyer_names[i] = f"Buyer_{i+1}_Conservative"
                    print(f"✓ Loaded {self.buyer_names[i]} (Conservative: Low risk, careful decisions)")
                    
                else:  # balanced
                    # Default exploration
                    agent.epsilon = 0.15  # Moderate exploration
                    self.buyer_names[i] = f"Buyer_{i+1}_Balanced"
                    print(f"✓ Loaded {self.buyer_names[i]} (Balanced: Moderate risk)")
                    
            except Exception as e:
                print(f"⚠ Could not load {self.buyer_names[i]}: {e}")
    
    def reset_competition(self, product_requests: List[Dict]):
        """
        Reset market for new competition round
        
        Args:
            product_requests: List of dicts with {buyer_id, product, quantity, budget}
        """
        # Initialize seller inventory (shared across all buyers)
        self.seller_inventory = {}
        for env in self.buyer_envs:
            for i, seller in enumerate(env.sellers):
                if i not in self.seller_inventory:
                    self.seller_inventory[i] = {
                        'stock': seller.stock,
                        'base_price': seller.base_price,
                        'reserved': 0  # Amount reserved by winning bids
                    }
        
        # Store buyer requests
        self.buyer_requests = {
            req['buyer_id']: req for req in product_requests
        }
        
        # Reset environments with requests
        self.buyer_states = []
        for i, env in enumerate(self.buyer_envs):
            if i in self.buyer_requests:
                req = self.buyer_requests[i]
                state, info = env.reset(options={
                    'quantity': req['quantity'],
                    'max_budget': req['max_budget']
                })
                self.buyer_states.append(state)
            else:
                self.buyer_states.append(None)
        
        self.competition_history = []
        
        return self.buyer_states
    
    def run_competitive_round(self) -> Dict:
        """
        Run one round of competition where all buyers act simultaneously
        
        Returns:
            Dict with round results and competition dynamics
        """
        round_actions = []
        round_results = []
        
        # All buyers select actions simultaneously
        for i, (agent, state) in enumerate(zip(self.buyer_agents, self.buyer_states)):
            if state is not None:
                # Enable exploration based on agent's epsilon (strategy)
                action = agent.select_action(state, training=True)  # Allow exploration
                round_actions.append({
                    'buyer_id': i,
                    'buyer_name': self.buyer_names[i],
                    'action': action,
                    'state': state.copy()
                })
            else:
                round_actions.append(None)
        
        # Process actions with competition logic
        for i, action_info in enumerate(round_actions):
            if action_info is None:
                continue
            
            action = action_info['action']
            env = self.buyer_envs[i]
            
            # Check if requested seller has enough inventory
            seller_id = int(action[1]) % self.num_sellers
            requested_qty = int(action[3])
            
            available = (self.seller_inventory[seller_id]['stock'] - 
                        self.seller_inventory[seller_id]['reserved'])
            
            # Competition: First come, first served (or highest bidder)
            if requested_qty > available:
                # Not enough inventory - partial fulfillment or rejection
                action[3] = max(0, available)  # Reduce quantity
                competition_penalty = -10  # Penalty for competition
            else:
                competition_penalty = 0
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Apply competition penalty
            reward += competition_penalty
            
            # Update inventory if deal was made
            action_type = int(action[0])
            if action_type == 2:  # Accept
                self.seller_inventory[seller_id]['reserved'] += int(action[3])
            
            # Store results
            round_results.append({
                'buyer_id': i,
                'buyer_name': self.buyer_names[i],
                'action': action,
                'reward': reward,
                'done': done,
                'info': info,
                'competition_penalty': competition_penalty,
                'inventory_available': available
            })
            
            # Update state
            self.buyer_states[i] = next_state if not done else None
        
        # Analyze competition dynamics
        competition_analysis = self._analyze_competition(round_results)
        
        self.competition_history.append({
            'actions': round_actions,
            'results': round_results,
            'analysis': competition_analysis
        })
        
        return {
            'round_results': round_results,
            'competition_analysis': competition_analysis,
            'remaining_inventory': copy.deepcopy(self.seller_inventory)
        }
    
    def run_full_competition(self, product_requests: List[Dict], 
                            max_rounds: int = 10) -> Dict:
        """
        Run complete competition until all buyers finish or max rounds reached
        
        Returns:
            Complete competition results with analytics
        """
        self.reset_competition(product_requests)
        
        all_rounds = []
        active_buyers = list(range(self.num_buyers))
        
        for round_num in range(max_rounds):
            # Check if all buyers are done (state is None)
            if all(state is None for state in self.buyer_states):
                break
            
            round_result = self.run_competitive_round()
            all_rounds.append(round_result)
            
            # Remove finished buyers
            active_buyers = [i for i, state in enumerate(self.buyer_states) 
                           if state is not None]
        
        # Final analysis
        final_analysis = self._compute_final_metrics(all_rounds)
        
        return {
            'rounds': all_rounds,
            'final_analysis': final_analysis,
            'competition_history': self.competition_history,
            'num_rounds': len(all_rounds)
        }
    
    def _analyze_competition(self, round_results: List[Dict]) -> Dict:
        """Analyze competitive dynamics in this round"""
        if not round_results:
            return {}
        
        # Find conflicts (multiple buyers targeting same seller)
        seller_targets = {}
        for result in round_results:
            action = result['action']
            seller_id = int(action[1]) % self.num_sellers
            
            if seller_id not in seller_targets:
                seller_targets[seller_id] = []
            seller_targets[seller_id].append(result['buyer_id'])
        
        conflicts = {sid: buyers for sid, buyers in seller_targets.items() 
                    if len(buyers) > 1}
        
        # Identify strategic behaviors
        strategies = []
        for result in round_results:
            action_type = int(result['action'][0])
            action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
            
            if action_type == 4:  # Coalition
                strategies.append(f"{result['buyer_name']}: Coalition strategy")
            elif result['competition_penalty'] < 0:
                strategies.append(f"{result['buyer_name']}: Lost in competition")
        
        return {
            'conflicts': conflicts,
            'num_conflicts': len(conflicts),
            'strategies': strategies,
            'total_rewards': {r['buyer_name']: r['reward'] for r in round_results}
        }
    
    def _compute_final_metrics(self, all_rounds: List[Dict]) -> Dict:
        """Compute final competition metrics with detailed analytics"""
        buyer_metrics = {name: {
            'total_reward': 0,
            'rounds_active': 0,
            'conflicts_faced': 0,
            'successful_deals': 0,
            'strategy_distribution': {},
            'avg_reward_per_round': 0,
            'efficiency_score': 0,
            'competitiveness_score': 0
        } for name in self.buyer_names}
        
        # Track action types for strategy analysis
        action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
        
        for round_data in all_rounds:
            for result in round_data['round_results']:
                name = result['buyer_name']
                buyer_metrics[name]['total_reward'] += result['reward']
                buyer_metrics[name]['rounds_active'] += 1
                
                # Track strategy distribution
                action_type = int(result['action'][0])
                action_name = action_names[action_type]
                if action_name not in buyer_metrics[name]['strategy_distribution']:
                    buyer_metrics[name]['strategy_distribution'][action_name] = 0
                buyer_metrics[name]['strategy_distribution'][action_name] += 1
                
                if result['competition_penalty'] < 0:
                    buyer_metrics[name]['conflicts_faced'] += 1
                
                if result['done'] and result['reward'] > 0:
                    buyer_metrics[name]['successful_deals'] += 1
        
        # Calculate derived metrics
        for name, metrics in buyer_metrics.items():
            if metrics['rounds_active'] > 0:
                metrics['avg_reward_per_round'] = metrics['total_reward'] / metrics['rounds_active']
                metrics['efficiency_score'] = (metrics['successful_deals'] / metrics['rounds_active']) * 100
                metrics['competitiveness_score'] = max(0, 100 - (metrics['conflicts_faced'] / metrics['rounds_active']) * 100)
        
        # Determine winner and rankings
        sorted_buyers = sorted(buyer_metrics.items(), 
                              key=lambda x: x[1]['total_reward'], 
                              reverse=True)
        
        winner = sorted_buyers[0]
        
        return {
            'buyer_metrics': buyer_metrics,
            'winner': winner[0],
            'winner_reward': winner[1]['total_reward'],
            'rankings': [{'name': name, 'reward': metrics['total_reward']} 
                        for name, metrics in sorted_buyers],
            'emergent_behaviors': self._identify_emergent_behaviors(all_rounds),
            'competition_stats': {
                'total_rounds': len(all_rounds),
                'total_conflicts': sum(m['conflicts_faced'] for m in buyer_metrics.values()),
                'total_deals': sum(m['successful_deals'] for m in buyer_metrics.values()),
                'avg_competitiveness': sum(m['competitiveness_score'] for m in buyer_metrics.values()) / len(buyer_metrics)
            }
        }
    
    def _identify_emergent_behaviors(self, all_rounds: List[Dict]) -> List[str]:
        """Identify interesting emergent behaviors from competition"""
        behaviors = []
        
        # Check for price wars
        price_changes = []
        for round_data in all_rounds:
            for result in round_data['round_results']:
                price = float(result['action'][2])
                price_changes.append(price)
        
        if len(price_changes) > 2:
            if price_changes[-1] < price_changes[0] * 0.9:
                behaviors.append("Price war detected: Buyers drove prices down")
        
        # Check for coalition formation
        coalition_count = sum(
            1 for round_data in all_rounds
            for result in round_data['round_results']
            if int(result['action'][0]) == 4
        )
        
        if coalition_count > 0:
            behaviors.append(f"Coalition strategy used {coalition_count} times")
        
        # Check for strategic timing
        if len(all_rounds) > 3:
            early_accepts = sum(
                1 for result in all_rounds[0]['round_results']
                if int(result['action'][0]) == 2
            )
            late_accepts = sum(
                1 for result in all_rounds[-1]['round_results']
                if int(result['action'][0]) == 2
            )
            
            if late_accepts > early_accepts:
                behaviors.append("Strategic waiting: Buyers delayed acceptance")
        
        return behaviors


def demonstrate_multi_agent():
    """Demonstration of multi-agent competition"""
    print("=" * 60)
    print("Multi-Agent Market Competition Demo")
    print("=" * 60)
    
    # Create market with 3 competing buyers
    market = MultiAgentMarket(num_buyers=3, num_sellers=3)
    
    # Define competing requests
    requests = [
        {'buyer_id': 0, 'product': 'Biscuits', 'quantity': 40, 'max_budget': 400},
        {'buyer_id': 1, 'product': 'Biscuits', 'quantity': 45, 'max_budget': 450},
        {'buyer_id': 2, 'product': 'Biscuits', 'quantity': 35, 'max_budget': 380},
    ]
    
    print(f"\n🏁 Starting competition with {len(requests)} buyers")
    for req in requests:
        print(f"  • Buyer {req['buyer_id']}: {req['quantity']} units, ${req['max_budget']} budget")
    
    # Run competition
    results = market.run_full_competition(requests)
    
    print(f"\n📊 Competition completed in {results['num_rounds']} rounds")
    print(f"\n🏆 Winner: {results['final_analysis']['winner']}")
    print(f"   Total Reward: {results['final_analysis']['winner_reward']:.2f}")
    
    print(f"\n📈 Buyer Performance:")
    for name, metrics in results['final_analysis']['buyer_metrics'].items():
        print(f"  {name}:")
        print(f"    Total Reward: {metrics['total_reward']:.2f}")
        print(f"    Conflicts Faced: {metrics['conflicts_faced']}")
        print(f"    Successful Deals: {metrics['successful_deals']}")
    
    print(f"\n🧠 Emergent Behaviors:")
    for behavior in results['final_analysis']['emergent_behaviors']:
        print(f"  • {behavior}")
    
    return results


if __name__ == "__main__":
    demonstrate_multi_agent()
