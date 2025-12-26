# Quick Start Guide

Get the autonomous market simulation running in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train the Buyer Agent

```bash
python train.py
```

This will:
- Train for 1000 episodes
- Save model to `models/buyer_agent.pth`
- Generate training curves in `plots/training_curves.png`
- Create logs in `logs/training_log.json`

**Expected time**: 5-10 minutes on CPU, 2-3 minutes on GPU

## Step 3: Evaluate Performance

```bash
python evaluate.py
```

This will:
- Compare RL-based vs rule-based buyer agents
- Run 100 episodes for each agent
- Generate comparison plots in `plots/agent_comparison.png`
- Save results to `logs/evaluation_results.json`

**Expected time**: 2-3 minutes

## Step 4: View Results

### Training Curves
Open `plots/training_curves.png` to see:
- Episode rewards over time
- Episode lengths over time
- Training loss progression
- Success rate evolution

### Agent Comparison
Open `plots/agent_comparison.png` to see:
- Success rate comparison
- Average reward comparison
- Reward distribution
- Episode length comparison

### Detailed Logs
Check `logs/` folder for JSON files with detailed metrics.

## Understanding the Output

### Console Output During Training
```
[Episode 50/1000]
  Avg Reward (last 50): 45.23
  Avg Length: 7.8
  Success Rate: 35.00%
  Epsilon: 0.6050
  Avg Loss: 0.0234
```

### Console Output During Evaluation
```
Metric                         RL-Based             Rule-Based           Improvement    
-------------------------------------------------------------------------------------
Success Rate                   0.68                 0.45                 +51.1%
Avg Reward                     67.34                42.18                +59.7%
Avg Savings                    234.56               145.23               +61.5%
```

## What's Happening?

1. **Training Phase**:
   - Buyer agent learns to negotiate through trial and error
   - Explores different strategies (high epsilon at start)
   - Gradually exploits learned knowledge (low epsilon at end)
   - Stores experiences in replay buffer
   - Updates neural network policy

2. **Evaluation Phase**:
   - Tests learned policy without exploration
   - Compares against simple rule-based strategy
   - Measures success rate, rewards, and efficiency

## Next Steps

### Experiment with Parameters

Edit `train.py` to change:
```python
trained_agent, log = train_buyer_agent(
    num_episodes=2000,        # More training
    max_steps_per_episode=15, # Longer negotiations
    save_interval=200,        # Save less frequently
    eval_interval=100         # Evaluate less frequently
)
```

### Modify Market Conditions

Edit environment initialization in `train.py`:
```python
env = MarketEnv(
    num_sellers=10,              # More sellers
    max_quantity_per_seller=100, # More stock
    max_negotiation_rounds=20,   # Longer rounds
    seed=42
)
```

### Analyze Specific Scenarios

Create a custom evaluation script:
```python
from market_env import MarketEnv
from buyer_agent import BuyerAgent

env = MarketEnv(num_sellers=5)
agent = BuyerAgent()
agent.load_model('models/buyer_agent.pth')

# Test specific scenario
state, info = env.reset()
print(f"Request: {info['requested_quantity']} units")
print(f"Budget: ${info['max_budget']:.2f}")

# Run negotiation
done = False
while not done:
    action = agent.select_action(state, training=False)
    state, reward, done, truncated, info = env.step(action)
    print(f"Round {info['current_round']}: Reward = {reward:.2f}")
```

## Troubleshooting

### Issue: "No module named 'gymnasium'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution**: Use CPU instead
```python
# In buyer_agent.py or train.py
device = 'cpu'  # Instead of 'cuda'
```

### Issue: Low success rate after training
**Solution**: Train longer or adjust hyperparameters
```python
# Increase episodes
num_episodes=2000

# Adjust learning rate
learning_rate=0.0005

# Slower epsilon decay
epsilon_decay=0.998
```

### Issue: Training too slow
**Solution**: Reduce episodes or use GPU
```python
# Fewer episodes for quick test
num_episodes=500

# Or enable GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Understanding the Market

### How Negotiation Works

1. **Initial Offer**: Buyer proposes price and quantity to a seller
2. **Seller Response**: Seller accepts, rejects, or counteroffers
3. **Counteroffer**: Buyer adjusts offer based on response
4. **Coalition**: If no single seller can fulfill, form coalition
5. **Accept/Reject**: Finalize deal or continue negotiating

### Market Rules (Fixed, Not Learned)

- **Fairness**: Prices must be within 70%-200% of base price
- **Trust**: Sellers have reliability scores that update based on delivery
- **Coalition**: Multiple sellers combine when needed, with fair profit distribution

### What the Agent Learns

- When to make initial offers vs counteroffers
- How to price offers for different sellers
- When to form coalitions
- How to balance speed vs savings

## Ready to Dive Deeper?

See `README.md` for:
- Detailed architecture explanation
- Research contributions
- Advanced usage examples
- Configuration options
