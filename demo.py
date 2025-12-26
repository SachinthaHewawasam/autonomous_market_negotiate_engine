"""
Demo script to showcase the autonomous market simulation.

This script runs a single negotiation episode with detailed output
to demonstrate how the system works.
"""

import numpy as np
from market_env import MarketEnv
from buyer_agent import BuyerAgent, RuleBasedBuyerAgent
import torch


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def print_section(title):
    """Print a section header"""
    print(f"\n{title}")
    print_separator('-')


def demo_single_negotiation(use_trained_model=False):
    """
    Run a single negotiation episode with detailed output.
    
    Args:
        use_trained_model: Whether to use a trained model (if available)
    """
    print_separator()
    print("AUTONOMOUS MARKET SIMULATION - DEMO")
    print_separator()
    
    # Initialize environment
    print("\n🏪 Initializing Market Environment...")
    env = MarketEnv(
        num_sellers=5,
        max_quantity_per_seller=50,
        max_negotiation_rounds=10,
        seed=42
    )
    
    # Initialize buyer agent
    print("🤖 Initializing Buyer Agent...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = BuyerAgent(device=device)
    
    if use_trained_model:
        try:
            agent.load_model('models/buyer_agent.pth')
            print("   ✓ Loaded trained model")
        except FileNotFoundError:
            print("   ⚠ No trained model found, using untrained agent")
    else:
        print("   Using untrained agent for demonstration")
    
    # Reset environment
    state, info = env.reset()
    
    print_section("📋 PROCUREMENT REQUEST")
    print(f"Requested Quantity: {info['requested_quantity']} units")
    print(f"Maximum Budget: ${info['max_budget']:.2f}")
    
    print_section("👥 AVAILABLE SELLERS")
    for i, (stock, price) in enumerate(zip(info['seller_stocks'], info['seller_prices'])):
        trust = info['trust_scores'][i]
        print(f"Seller {i}: Stock={stock:3d} units, Base Price=${price:5.2f}/unit, Trust={trust:.2f}")
    
    # Check if coalition needed
    max_single_stock = max(info['seller_stocks'])
    coalition_needed = max_single_stock < info['requested_quantity']
    
    if coalition_needed:
        print(f"\n⚠ Coalition Required: No single seller has {info['requested_quantity']} units")
        print(f"   (Maximum single seller stock: {max_single_stock} units)")
    else:
        print(f"\n✓ Single seller can fulfill request")
    
    # Run negotiation
    print_section("💬 NEGOTIATION PROCESS")
    
    done = False
    truncated = False
    total_reward = 0
    round_num = 0
    
    action_names = {
        0: "OFFER",
        1: "COUNTEROFFER",
        2: "ACCEPT",
        3: "REJECT",
        4: "PROPOSE COALITION"
    }
    
    while not (done or truncated):
        round_num += 1
        
        # Select action
        action = agent.select_action(state, training=False)
        action_type = int(action[0])
        seller_id = int(action[1]) % env.num_sellers
        price = float(action[2])
        quantity = int(action[3])
        
        # Display action
        print(f"\nRound {round_num}:")
        print(f"  Action: {action_names.get(action_type, 'UNKNOWN')}")
        
        if action_type in [0, 1]:  # Offer or Counteroffer
            print(f"  Target: Seller {seller_id}")
            print(f"  Price: ${price:.2f}/unit")
            print(f"  Quantity: {quantity} units")
            print(f"  Total: ${price * quantity:.2f}")
        elif action_type == 2:  # Accept
            print(f"  Accepting current offer")
        elif action_type == 3:  # Reject
            print(f"  Rejecting current offer")
        elif action_type == 4:  # Coalition
            print(f"  Proposing coalition with Seller {seller_id} as primary")
        
        # Take step
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"  Reward: {reward:+.2f}")
        
        # Update state
        state = next_state
        total_reward += reward
    
    # Display results
    print_section("📊 NEGOTIATION RESULTS")
    
    if done:
        print("✅ SUCCESS - Deal Completed!")
        print(f"\nFinal Details:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Rounds Used: {round_num}/{env.max_negotiation_rounds}")
        
        if total_reward > 100:
            savings = (total_reward - 100) * 10
            print(f"  Estimated Savings: ${savings:.2f}")
        
        if info.get('coalition_proposed', False):
            print(f"\n  Coalition Formed: Yes")
            print(f"  Coalition Members: {len(info.get('coalition_members', []))} sellers")
        else:
            print(f"\n  Coalition Formed: No (single seller)")
    else:
        print("❌ FAILED - Negotiation Incomplete")
        print(f"\nDetails:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Rounds Used: {round_num}/{env.max_negotiation_rounds}")
        print(f"  Reason: Exceeded maximum negotiation rounds")
    
    print_separator()
    
    return done, total_reward, round_num


def compare_agents_demo():
    """Compare RL-based and rule-based agents side by side"""
    print_separator()
    print("AGENT COMPARISON DEMO")
    print_separator()
    
    print("\nRunning 10 episodes for each agent...\n")
    
    # Initialize environment
    env = MarketEnv(num_sellers=5, max_quantity_per_seller=50, seed=123)
    
    # Initialize agents
    rl_agent = BuyerAgent()
    rule_agent = RuleBasedBuyerAgent()
    
    try:
        rl_agent.load_model('models/buyer_agent.pth')
        print("✓ Using trained RL agent\n")
    except FileNotFoundError:
        print("⚠ Using untrained RL agent\n")
    
    # Run episodes
    rl_successes = 0
    rule_successes = 0
    rl_rewards = []
    rule_rewards = []
    
    for episode in range(10):
        # RL agent
        state, _ = env.reset(seed=episode)
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = rl_agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rl_rewards.append(episode_reward)
        if done:
            rl_successes += 1
        
        # Rule-based agent
        state, _ = env.reset(seed=episode)
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = rule_agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rule_rewards.append(episode_reward)
        if done:
            rule_successes += 1
    
    # Display results
    print_section("COMPARISON RESULTS")
    
    print(f"\n{'Metric':<30} {'RL-Based':<20} {'Rule-Based':<20}")
    print_separator('-')
    print(f"{'Success Rate':<30} {rl_successes/10:.1%}{'':<14} {rule_successes/10:.1%}")
    print(f"{'Average Reward':<30} {np.mean(rl_rewards):>6.2f}{'':<14} {np.mean(rule_rewards):>6.2f}")
    print(f"{'Std Reward':<30} {np.std(rl_rewards):>6.2f}{'':<14} {np.std(rule_rewards):>6.2f}")
    
    print_separator()


if __name__ == '__main__':
    import sys
    
    print("\n🎮 AUTONOMOUS MARKET SIMULATION DEMO\n")
    print("Choose demo mode:")
    print("1. Single negotiation (detailed)")
    print("2. Agent comparison (10 episodes)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1' or choice == '3':
        demo_single_negotiation(use_trained_model=True)
    
    if choice == '2' or choice == '3':
        compare_agents_demo()
    
    print("\n✨ Demo complete! Run 'python train.py' to train the agent.")
    print("   Then run 'python evaluate.py' for full evaluation.\n")
