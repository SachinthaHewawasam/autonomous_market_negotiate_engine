"""
Diagnostic script to understand why negotiations are failing.
"""

from market_env import MarketEnv
from buyer_agent import RuleBasedBuyerAgent
import numpy as np

def diagnose_environment():
    """Run diagnostic tests on the environment"""
    print("=" * 80)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 80)
    
    env = MarketEnv(num_sellers=5, max_quantity_per_seller=50, seed=42)
    
    # Test 1: Can we complete a simple negotiation manually?
    print("\n1. Testing Manual Negotiation")
    print("-" * 80)
    
    state, info = env.reset()
    print(f"Request: {info['requested_quantity']} units, Budget: ${info['max_budget']:.2f}")
    print(f"\nSellers:")
    for i, (stock, price, trust) in enumerate(zip(info['seller_stocks'], info['seller_prices'], info['trust_scores'])):
        print(f"  Seller {i}: Stock={stock:3d}, Price=${price:5.2f}, Trust={trust:.2f}")
    
    # Find best single seller
    best_seller = None
    best_price = float('inf')
    for i, (stock, price) in enumerate(zip(info['seller_stocks'], info['seller_prices'])):
        if stock >= info['requested_quantity'] and price < best_price:
            best_seller = i
            best_price = price
    
    if best_seller is not None:
        print(f"\nBest single seller: Seller {best_seller} (${best_price:.2f}/unit)")
        total_cost = best_price * info['requested_quantity']
        print(f"Total cost: ${total_cost:.2f} (Budget: ${info['max_budget']:.2f})")
        
        if total_cost <= info['max_budget']:
            print("✓ Single seller can fulfill within budget!")
            
            # Try to negotiate
            print("\nAttempting negotiation:")
            for round_num in range(10):
                # Action: [action_type, seller_id, price, quantity]
                # action_type: 0=offer, 2=accept
                if round_num == 0:
                    # Make initial offer
                    action = np.array([0, best_seller, best_price, info['requested_quantity']], dtype=np.float32)
                    print(f"  Round {round_num + 1}: Making offer to Seller {best_seller}")
                else:
                    # Accept
                    action = np.array([2, best_seller, best_price, info['requested_quantity']], dtype=np.float32)
                    print(f"  Round {round_num + 1}: Accepting offer")
                
                state, reward, done, truncated, info_new = env.step(action)
                print(f"    Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}")
                
                if done:
                    print("✓ Negotiation successful!")
                    break
                if truncated:
                    print("✗ Negotiation truncated (max rounds)")
                    break
        else:
            print("✗ Over budget, need coalition")
    else:
        print("✗ No single seller can fulfill, need coalition")
    
    # Test 2: Rule-based agent
    print("\n\n2. Testing Rule-Based Agent")
    print("-" * 80)
    
    agent = RuleBasedBuyerAgent()
    successes = 0
    total_episodes = 10
    
    for episode in range(total_episodes):
        state, info = env.reset(seed=episode)
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        if done:
            successes += 1
            print(f"  Episode {episode}: SUCCESS (Reward: {episode_reward:.2f})")
        else:
            print(f"  Episode {episode}: FAILED (Reward: {episode_reward:.2f})")
    
    print(f"\nRule-based success rate: {successes}/{total_episodes} = {successes/total_episodes:.1%}")
    
    # Test 3: Check reward structure
    print("\n\n3. Reward Structure Analysis")
    print("-" * 80)
    
    state, info = env.reset(seed=100)
    print(f"Request: {info['requested_quantity']} units")
    
    # Test different actions
    test_actions = [
        ("Offer (reasonable)", [0, 0, 10.0, 50]),
        ("Offer (too low price)", [0, 0, 1.0, 50]),
        ("Offer (too high price)", [0, 0, 50.0, 50]),
        ("Offer (too much qty)", [0, 0, 10.0, 200]),
        ("Accept (no active offer)", [2, 0, 10.0, 50]),
        ("Reject", [3, 0, 10.0, 50]),
        ("Coalition", [4, 0, 10.0, 100]),
    ]
    
    for action_name, action_values in test_actions:
        env.reset(seed=100)  # Reset to same state
        action = np.array(action_values, dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        print(f"  {action_name:30s}: Reward={reward:6.2f}, Done={done}, Truncated={truncated}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    diagnose_environment()
