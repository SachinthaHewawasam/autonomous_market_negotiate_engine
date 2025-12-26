# Project Overview: Autonomous Market Simulation

## One-Sentence Summary

> **"This project designs a unified autonomous market for bulk procurement and studies how a reinforcement-learning buyer agent negotiates within fixed rules of fairness, trust, and coalition formation."**

## The Problem (Real-World Context)

When a businessman wants to buy **large quantities** (e.g., 100 packs of biscuits), he must:

- Contact many sellers manually
- Negotiate prices individually
- Combine multiple sellers if one doesn't have enough stock
- Avoid unfair or unreliable sellers
- Rely on experience and intuition

This process is **manual, slow, and experience-based**.

## The Solution (Our Approach)

We build a **simulated autonomous market** where:

- A **Buyer Agent** negotiates automatically on behalf of the businessman
- Multiple **Seller Agents** offer stock at different prices
- The market enforces **coalition rules, fairness rules, and trust rules**

The buyer agent **learns how to negotiate better over time** using **reinforcement learning**, **only through interaction**, without any real data.

## What is Actually New (Research Contribution)

⚠️ **Important Clarification**:

- We are **NOT inventing a new AI algorithm**
- We are **NOT claiming a breakthrough in reinforcement learning**
- We are **designing a new market structure**

👉 **The market design is the research contribution**

👉 **Reinforcement learning is only a tool** to observe adaptive behavior

We study **how a learning agent behaves inside a realistic, rule-constrained market**.

## Key Design Principles

### 1. Fixed Market Rules (Not Learned)

The market environment enforces these rules deterministically:

- **Coalition Formation**: When no single seller can fulfill the request, multiple sellers combine
- **Fairness Constraints**: Prices must be within 70%-200% of base price
- **Trust Mechanisms**: Seller reliability updates based on delivery outcomes
- **Profit Distribution**: Fair allocation among coalition members

These rules are **hardcoded** and **never learned**.

### 2. Single Learning Component

**Exactly ONE machine learning model is allowed:**

- The RL model is embedded inside the **BuyerAgent** as its negotiation policy
- The RL policy controls only negotiation actions (offer, counteroffer, accept, reject, propose coalition)
- Everything else is rule-based

### 3. No External Data

- No historical data is assumed
- No real-world datasets are used
- All training data is generated through **interaction with the simulated market**
- The market itself remains **fixed and deterministic**

## System Components

### Market Environment (`MarketEnv`)
- Gym-compatible environment
- Enforces all market rules
- Manages state transitions
- Calculates rewards
- **Does not learn**

### Rule Managers (Fixed Logic)

1. **CoalitionManager**: Forms coalitions when needed
2. **FairnessChecker**: Validates pricing and profit distribution
3. **TrustManager**: Tracks seller reliability

### Agents

1. **SellerAgent** (Rule-Based)
   - Fixed stock and base price
   - Simple negotiation heuristics
   - No learning

2. **BuyerAgent** (RL-Based)
   - Deep Q-Network (DQN)
   - Learns negotiation policy
   - Trained through simulation

3. **RuleBasedBuyerAgent** (Baseline)
   - Fixed negotiation heuristics
   - Used for comparison
   - No learning

## Research Questions

1. **Can an RL agent learn effective negotiation strategies within fixed market constraints?**
   - Hypothesis: Yes, through trial and error

2. **How does the learned policy compare to simple rule-based strategies?**
   - Metrics: Success rate, savings, efficiency

3. **What negotiation patterns emerge from the learning process?**
   - Analysis: Coalition usage, pricing strategies, seller selection

4. **How do market rules affect learning behavior?**
   - Study: Impact of fairness constraints, trust scores, coalition limits

## Expected Outcomes

After training, the RL-based buyer agent should:

- ✅ Achieve higher success rates than rule-based baseline
- ✅ Learn when to form coalitions vs negotiate with single sellers
- ✅ Adapt pricing strategies based on seller characteristics
- ✅ Balance negotiation speed vs cost savings
- ✅ Utilize trust scores in seller selection

## Validation Approach

### Training Phase
- 1000 episodes of simulated negotiations
- Monitor: rewards, success rate, episode length, loss

### Evaluation Phase
- 100 episodes comparing RL vs rule-based agents
- Metrics: success rate, average reward, savings, efficiency
- Statistical analysis of performance differences

### Analysis
- Reward distribution comparison
- Negotiation pattern analysis
- Coalition formation frequency
- Trust score utilization

## Technical Stack

- **Python 3.8+**
- **Gymnasium**: RL environment framework
- **PyTorch**: Deep learning for RL agent
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

## Project Structure

```
finalYrproj/
├── Core Components
│   ├── market_env.py          # Market environment (Gym-compatible)
│   ├── seller_agent.py        # Rule-based seller
│   └── buyer_agent.py         # RL-based buyer + baseline
│
├── Market Rules (Fixed)
│   ├── coalition_manager.py   # Coalition formation logic
│   ├── fairness_checker.py    # Fairness enforcement
│   └── trust_manager.py       # Trust score management
│
├── Execution Scripts
│   ├── train.py              # Train RL agent
│   ├── evaluate.py           # Compare agents
│   └── demo.py               # Interactive demo
│
├── Documentation
│   ├── README.md             # Full documentation
│   ├── QUICKSTART.md         # Quick start guide
│   └── PROJECT_OVERVIEW.md   # This file
│
└── Configuration
    ├── requirements.txt      # Dependencies
    └── .gitignore           # Git ignore rules
```

## How to Use

### Quick Start (5 minutes)
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

### Interactive Demo
```bash
python demo.py
```

### Custom Experiments
```python
from market_env import MarketEnv
from buyer_agent import BuyerAgent

# Create custom environment
env = MarketEnv(num_sellers=10, max_quantity_per_seller=100)

# Train custom agent
agent = BuyerAgent(learning_rate=0.0005, gamma=0.95)
# ... training loop ...
```

## Key Insights

### What Makes This Novel

1. **Unified Framework**: Integrates coalition formation, fairness, and trust in one system
2. **Fixed Rules**: Market rules are deterministic, not learned
3. **Adaptive Behavior**: Studies how learning emerges within constraints
4. **No Real Data**: Pure simulation-based learning
5. **Practical Application**: Models real-world bulk procurement

### What This Is NOT

- ❌ A new RL algorithm
- ❌ A production-ready system
- ❌ A real marketplace
- ❌ A data-driven approach
- ❌ A multi-agent learning system (only buyer learns)

### What This IS

- ✅ A novel market design
- ✅ A research simulation
- ✅ A study of adaptive behavior
- ✅ A proof-of-concept
- ✅ An educational tool

## Future Extensions

Potential research directions:

1. **Multi-Agent Learning**: Allow sellers to learn as well
2. **Dynamic Markets**: Time-varying prices and stock
3. **Complex Coalitions**: Hierarchical or nested coalitions
4. **Real Data Integration**: Validate with actual procurement data
5. **Strategic Behavior**: Game-theoretic analysis
6. **Scalability**: Larger markets with more agents

## Conclusion

This project demonstrates that:

- Autonomous negotiation is feasible within structured markets
- RL agents can learn effective strategies without external data
- Market design significantly influences learning outcomes
- Fixed rules can coexist with adaptive behavior

The **market design is the contribution**, and reinforcement learning is the **tool to study it**.
