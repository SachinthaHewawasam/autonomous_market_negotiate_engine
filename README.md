# Autonomous Market Simulation for Bulk Procurement

A Python-based simulation of an autonomous market where a reinforcement learning buyer agent negotiates with rule-based seller agents for bulk procurement.

## 🎯 Project Overview

### The Problem
In real-world bulk procurement, businessmen must:
- Contact multiple sellers
- Negotiate prices manually
- Form coalitions when single sellers lack sufficient stock
- Avoid unfair or unreliable sellers

This process is **manual, slow, and experience-based**.

### The Solution
This project simulates an autonomous market where:
- A **Buyer Agent** negotiates automatically using reinforcement learning
- Multiple **Seller Agents** offer stock at different prices (rule-based)
- The market enforces **coalition rules, fairness rules, and trust rules**

### Key Innovation
**The market design is the research contribution** - we study how a learning agent behaves within a realistic, rule-constrained market. Reinforcement learning is simply the tool to observe adaptive behavior.

## 🏗️ Architecture

### Market Environment (`market_env.py`)
- Gym-compatible environment
- Enforces fixed market rules (not learned)
- Manages negotiation rounds and state transitions
- **Observation space**: 9 features (requested quantity, prices, trust scores, etc.)
- **Action space**: 4 continuous values (action type, seller ID, price, quantity)

### Market Rule Managers

#### Coalition Manager (`coalition_manager.py`)
- Forms coalitions when no single seller can fulfill requests
- Selects sellers based on price-to-trust ratio
- Ensures fair profit distribution
- Limits coalition size to avoid complexity

#### Fairness Checker (`fairness_checker.py`)
- Prevents exploitative pricing (max 200% markup)
- Validates coalition profit distribution
- Detects price manipulation attempts
- Calculates fairness scores

#### Trust Manager (`trust_manager.py`)
- Tracks seller reliability based on delivery outcomes
- Updates trust scores dynamically
- Influences coalition formation
- Applies time-based decay

### Agents

#### Seller Agent (`seller_agent.py`)
- **Rule-based** (no learning)
- Fixed stock and base price
- Simple negotiation logic
- Evaluates offers probabilistically

#### Buyer Agent (`buyer_agent.py`)
- **Reinforcement learning-based**
- Deep Q-Network (DQN) architecture
- Learns negotiation policy through interaction
- Features:
  - Neural network policy
  - Experience replay buffer
  - Epsilon-greedy exploration
  - Target network for stability

#### Rule-Based Buyer Agent (for comparison)
- Uses fixed heuristics
- Starts with low offers
- Gradually increases price
- Forms coalitions when needed

## 🚀 Getting Started

### Installation

```bash
# Clone or navigate to the project directory
cd finalYrproj

# Install dependencies
pip install -r requirements.txt
```

### Training the Buyer Agent

```bash
python train.py
```

**Training configuration:**
- Episodes: 1000
- Max steps per episode: 10
- Learning rate: 0.001
- Epsilon decay: 0.995
- Replay buffer: 10,000 transitions

**Output:**
- Model saved to `models/buyer_agent.pth`
- Training logs in `logs/training_log.json`
- Training curves in `plots/training_curves.png`

### Evaluating Performance

```bash
python evaluate.py
```

**Evaluation:**
- Compares RL-based agent vs rule-based agent
- 100 episodes per agent
- Metrics: success rate, average reward, savings, episode length
- Results saved to `logs/evaluation_results.json`
- Comparison plots in `plots/agent_comparison.png`

## 📊 Key Metrics

### Success Metrics
- **Success Rate**: Percentage of negotiations completed within budget
- **Average Reward**: Total reward per episode (includes savings bonus)
- **Average Savings**: Money saved compared to maximum budget
- **Episode Length**: Number of negotiation rounds

### Market Metrics
- **Coalition Formation Rate**: How often coalitions are needed
- **Trust Score Evolution**: How seller trust changes over time
- **Price Fairness**: Distribution of negotiated prices

## 🔬 Research Contributions

1. **Unified Market Design**: Integrates coalition formation, fairness constraints, and trust mechanisms in a single framework

2. **Fixed Rule Enforcement**: Market rules are deterministic and not learned, ensuring consistent behavior

3. **Adaptive Buyer Behavior**: Studies how RL agents learn to negotiate within fixed constraints

4. **No External Data**: All training data generated through simulation interaction

## 📁 Project Structure

```
finalYrproj/
├── market_env.py           # Gym environment
├── seller_agent.py         # Rule-based seller
├── buyer_agent.py          # RL-based buyer + rule-based buyer
├── coalition_manager.py    # Coalition formation logic
├── fairness_checker.py     # Fairness enforcement
├── trust_manager.py        # Trust score management
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
├── README.md             # This file
├── models/               # Saved models
├── logs/                 # Training/evaluation logs
└── plots/                # Visualization outputs
```

## 🎓 Usage Examples

### Basic Training
```python
from market_env import MarketEnv
from buyer_agent import BuyerAgent

# Initialize environment
env = MarketEnv(num_sellers=5, max_quantity_per_seller=50)

# Initialize agent
agent = BuyerAgent(learning_rate=0.001, gamma=0.99)

# Training loop
for episode in range(1000):
    state, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, truncated, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
    
    agent.end_episode()
```

### Custom Evaluation
```python
from buyer_agent import RuleBasedBuyerAgent

# Compare with rule-based agent
rule_agent = RuleBasedBuyerAgent()

# Evaluate
for episode in range(100):
    state, info = env.reset()
    done = False
    
    while not done:
        action = rule_agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
```

## 🔧 Configuration

### Environment Parameters
- `num_sellers`: Number of sellers in the market (default: 5)
- `max_quantity_per_seller`: Maximum stock per seller (default: 50)
- `max_negotiation_rounds`: Maximum rounds per episode (default: 10)

### Agent Parameters
- `learning_rate`: Learning rate for optimizer (default: 0.001)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Exploration decay rate (default: 0.995)
- `batch_size`: Training batch size (default: 64)

## 📈 Expected Results

After training, the RL-based buyer agent should:
- Achieve higher success rates than rule-based agent
- Learn to form coalitions efficiently
- Adapt to different market conditions
- Maximize savings while staying within budget

## 🤝 Contributing

This is a research project. Key areas for extension:
- Multi-agent learning (sellers also learn)
- Dynamic market conditions
- More complex coalition structures
- Real-world data integration

## 📝 Citation

If you use this project in your research, please cite:

```
Autonomous Market Simulation for Bulk Procurement
A unified market design studying reinforcement learning behavior
within fixed rules of fairness, trust, and coalition formation.
```

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

Built using:
- Gymnasium (OpenAI Gym)
- PyTorch (Deep Learning)
- NumPy (Numerical Computing)
- Matplotlib (Visualization)
#   a u t o n o m o u s _ m a r k e t _ n e g o t i a t e _ e n g i n e  
 