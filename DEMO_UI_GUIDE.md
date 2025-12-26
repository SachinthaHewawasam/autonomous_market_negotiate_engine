# Demo UI Guide

## 🎯 Interactive Demonstration Interface

This demo UI showcases the trained RL buyer agent negotiating in real-time with multiple sellers in a bulk procurement scenario.

---

## 🚀 Quick Start

### Prerequisites

Ensure you have a trained model:
```bash
# If you haven't trained yet, run:
python train.py
```

### Launch the Demo

```bash
python demo_ui.py
```

---

## 📋 UI Overview

### Main Components

1. **Left Panel - Procurement Request**
   - Product name input
   - Quantity input (number of units)
   - Maximum budget input ($)
   - Start/Reset buttons
   - Available sellers table

2. **Right Panel - Visualization**
   - Real-time negotiation log
   - Step-by-step agent decisions
   - Final deal summary

---

## 🎮 How to Use

### Step 1: Enter Procurement Request

**Example Scenario:**
```
Product: Biscuits (Brand X)
Quantity: 120 units
Max Budget: $1200
```

**What happens:**
- The system displays 5 sellers with limited stock (20-50 units each)
- Each seller has different prices and trust scores
- No single seller can fulfill 120 units alone

### Step 2: Start Negotiation

Click **"▶ Start Negotiation"**

**The RL agent will:**
1. Analyze the market state
2. Decide which action to take (offer, accept, coalition, etc.)
3. Execute the action
4. Receive feedback from the environment
5. Repeat until deal is reached or max rounds exceeded

### Step 3: Watch the Process

**Negotiation Log shows:**
- Each round number
- Agent's chosen action
- Target seller
- Offered price and quantity
- Reward received
- Success/failure indicators

**Color coding:**
- 🟢 **Green** = Success/positive outcome
- 🟡 **Yellow** = Warning/neutral
- 🔴 **Red** = Error/penalty
- 🟣 **Purple** = Agent actions
- 🔵 **Blue** = Information

### Step 4: Review Final Deal

**If successful:**
- ✅ Deal details
- Selected sellers
- Coalition formation (if applicable)
- Final prices
- Total cost
- Savings achieved

**If failed:**
- ❌ Failure reason
- Possible causes
- Suggestions

---

## 🎭 Demo Scenarios

### Scenario 1: Simple Coalition
```
Product: Office Supplies
Quantity: 100 units
Budget: $1000
```
**Expected:** Agent forms coalition with 2-3 sellers

### Scenario 2: Tight Budget
```
Product: Electronics
Quantity: 80 units
Budget: $600
```
**Expected:** Agent negotiates aggressively for lower prices

### Scenario 3: High Quantity
```
Product: Bulk Materials
Quantity: 150 units
Budget: $2000
```
**Expected:** Agent forms large coalition (4-5 sellers)

### Scenario 4: Generous Budget
```
Product: Premium Goods
Quantity: 60 units
Budget: $1500
```
**Expected:** Quick deal with single seller or small coalition

---

## 🔍 Understanding the Agent's Behavior

### Action Types

1. **Offer (0)** - Initial price proposal to a seller
2. **Counteroffer (1)** - Revised price after seller response
3. **Accept (2)** - Accept current offer
4. **Reject (3)** - Reject current offer
5. **Coalition (4)** - Propose multi-seller coalition

### Reward Signals

| Reward Range | Meaning |
|--------------|---------|
| +100 to +150 | Coalition success with good savings |
| +50 to +100  | Successful deal |
| +1 to +10    | Positive progress |
| -1 to -10    | Minor penalty (unfair price, rejection) |
| -10 to -50   | Major penalty (over budget, failed coalition) |

### Market Rules Applied

**Fairness Constraints:**
- Prices must be within ±30% of seller's base price
- Coalition profit distribution must be balanced
- No price manipulation allowed

**Trust Mechanism:**
- Sellers have trust scores (0.0 to 1.0)
- Higher trust = more reliable delivery
- Agent prefers high-trust sellers

**Coalition Formation:**
- Triggered when no single seller can fulfill
- Selects sellers based on price-trust ratio
- Validates total cost against budget

---

## 🎨 UI Features

### Visual Indicators

**Seller Table:**
- Stock levels (units available)
- Price per unit ($)
- Trust score (0.0 - 1.0)

**Negotiation Log:**
- Timestamped actions
- Color-coded outcomes
- Detailed step breakdown

**Final Deal Panel:**
- Success/failure status
- Cost breakdown
- Savings calculation
- Coalition members (if applicable)

### Interactive Controls

**Start Button:**
- Launches negotiation
- Disabled during simulation
- Re-enabled after completion

**Reset Button:**
- Clears all logs
- Resets input fields
- Prepares for new simulation

---

## 🔧 Technical Details

### Inference Mode

**Key Points:**
- Uses trained model from `models/buyer_agent.pth`
- **No training occurs** during demo
- Agent uses epsilon=0 (pure exploitation)
- Deterministic policy network forward pass

### Threading

- Simulation runs in separate thread
- UI remains responsive
- Real-time log updates
- Smooth visualization

### Environment Integration

- Uses same `MarketEnv` as training
- Fixed market rules (deterministic)
- Reproducible results with same inputs
- Proper state/action/reward flow

---

## 📊 Interpreting Results

### Success Indicators

✅ **Successful Negotiation:**
- Deal reached within 10 rounds
- Total cost ≤ budget
- Quantity requirement met
- Fairness rules satisfied

❌ **Failed Negotiation:**
- Max rounds exceeded
- Budget constraint violated
- Insufficient total stock
- Fairness violations

### Performance Metrics

**Efficiency:**
- Fewer rounds = better
- Typical: 2-5 rounds for success

**Savings:**
- Budget - Total Cost
- Good: 10-30% savings
- Excellent: 30%+ savings

**Coalition Size:**
- Smaller = more efficient
- Typical: 2-3 sellers
- Large requests: 4-5 sellers

---

## 🎓 Educational Value

### For Presentations

**Demonstrates:**
1. RL agent decision-making process
2. Multi-agent negotiation dynamics
3. Coalition formation in action
4. Market mechanism enforcement
5. Real-time policy inference

### For Research

**Shows:**
- Learned negotiation strategies
- Emergent behavior patterns
- Trust-based seller selection
- Dynamic pricing adaptation
- Coalition optimization

### For Stakeholders

**Highlights:**
- Automation potential
- Cost savings capability
- Scalability to complex scenarios
- Transparency of decisions
- Practical applicability

---

## 🐛 Troubleshooting

### Model Not Found

**Error:** "Trained model not found"

**Solution:**
```bash
python train.py  # Train the agent first
```

### UI Not Responding

**Issue:** Simulation stuck

**Solution:**
- Wait for current simulation to complete
- Click Reset button
- Restart the application

### Invalid Input

**Error:** "Please enter valid values"

**Solution:**
- Quantity must be positive integer
- Budget must be positive number
- Use reasonable values (20-200 units, $500-$3000)

### Poor Performance

**Issue:** Agent always fails

**Solution:**
- Check if model is properly trained
- Verify budget is sufficient
- Ensure quantity is achievable (sum of all seller stocks)

---

## 💡 Tips for Best Demo

### Preparation

1. **Train a good model** (70%+ success rate)
2. **Test different scenarios** beforehand
3. **Prepare talking points** for each action type
4. **Have backup scenarios** ready

### During Demo

1. **Explain the setup** (sellers, constraints)
2. **Narrate agent actions** as they happen
3. **Highlight key decisions** (coalition formation)
4. **Discuss outcomes** (success/failure reasons)

### Follow-up

1. **Show training curves** (`plots/training_curves.png`)
2. **Discuss statistical results** (`experiments/statistical_report.txt`)
3. **Explain research contributions** (`RESEARCH_CONTRIBUTION.md`)
4. **Answer questions** about market design

---

## 🎬 Demo Script Example

**Opening:**
> "This demo shows an AI agent trained with reinforcement learning to negotiate bulk procurement deals. The agent must work with multiple sellers who each have limited stock, forming coalitions when necessary while respecting fairness and trust constraints."

**During Negotiation:**
> "Watch as the agent analyzes the market... it's choosing to propose a coalition because no single seller has enough stock. The agent selected Sellers 0, 2, and 4 based on their price-trust ratio..."

**Conclusion:**
> "The agent successfully negotiated a deal with 3 sellers, achieving 15% cost savings while respecting all market rules. This demonstrates how RL can automate complex multi-party negotiations."

---

## 📈 Next Steps

After the demo:

1. **Run experiments:** `python experiments.py`
2. **View statistical analysis:** Check `experiments/statistical_report.txt`
3. **Read research docs:** `RESEARCH_CONTRIBUTION.md`
4. **Explore code:** Review implementation details

---

## 🎯 Key Takeaways

**For Audience:**
- RL agents can learn complex negotiation strategies
- Coalition formation emerges from learning
- Market rules are enforced transparently
- Real-world applicability demonstrated

**For Research:**
- Novel market design validated
- Agent behavior is interpretable
- Framework is extensible
- Results are reproducible

---

## 📞 Support

For issues or questions:
- Check `README.md` for general information
- Review `QUICKSTART.md` for setup help
- See `RESEARCH_CONTRIBUTION.md` for research details

---

**Enjoy the demo! 🚀**
