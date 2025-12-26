# Online Training Guide for RL Agent

## 🧠 What is Online Training?

Online training allows your RL agent to **continuously learn** from real negotiations in production. Each negotiation becomes a training example that improves the agent's decision-making.

---

## 🚀 How to Enable Online Training

### **Option 1: Environment Variable (Recommended)**

Create a `.env` file in `web_app/backend/`:

```bash
ENABLE_TRAINING=true
```

Then restart the backend:
```bash
python app.py
```

### **Option 2: Direct Code Change**

Edit `web_app/backend/app.py`:

```python
ENABLE_ONLINE_TRAINING = True  # Change False to True
negotiation_service = NegotiationService(model_path, enable_training=ENABLE_ONLINE_TRAINING)
```

---

## 📊 What Happens During Online Training?

### **During Each Negotiation:**

1. **Experience Collection**
   - Every state, action, reward, next_state is stored
   - Stored in agent's replay buffer

2. **Training After Negotiation**
   - Agent performs 5 training steps using collected experiences
   - Updates neural network weights via backpropagation
   - Prints training loss to console

3. **Periodic Model Saving**
   - Every 10 negotiations (configurable)
   - Saves checkpoint: `models/checkpoints/buyer_agent_n10_20231225_120000.pth`
   - Updates main model: `models/buyer_agent.pth`

---

## 📈 Training Logs

You'll see logs like this in the backend console:

```
[Training] Starting online training from negotiation 42
[Training] Processing 8 experiences
[Training] Completed 5 training steps, avg loss: 0.0234
[Training] Replay buffer size: 156
[Training] Model saved after 10 negotiations
```

---

## ⚙️ Configuration

Edit `web_app/backend/services/negotiation_service.py`:

```python
self.save_interval = 10  # Save every N negotiations
```

Edit `web_app/backend/services/training_utils.py`:

```python
num_updates = min(len(experiences), 5)  # Training steps per negotiation
```

---

## 🎯 Training Strategy

### **Exploration vs Exploitation**

When `enable_training=True`:
- Agent uses **epsilon-greedy exploration** (10% random actions)
- Helps discover new strategies
- Prevents getting stuck in local optima

When `enable_training=False`:
- Agent uses **pure exploitation** (always best action)
- More consistent performance
- No learning or improvement

---

## 📁 Model Checkpoints

Checkpoints are saved to `models/checkpoints/`:

```
buyer_agent_n10_20231225_120000.pth   # After 10 negotiations
buyer_agent_n20_20231225_130000.pth   # After 20 negotiations
buyer_agent_n30_20231225_140000.pth   # After 30 negotiations
```

You can restore any checkpoint:

```python
agent.load_model('models/checkpoints/buyer_agent_n20_20231225_130000.pth')
```

---

## ⚠️ Important Considerations

### **Pros of Online Training:**
- ✅ Adapts to new products/sellers automatically
- ✅ Improves from real-world feedback
- ✅ Learns user preferences over time
- ✅ No need for manual retraining

### **Cons of Online Training:**
- ⚠️ Can learn bad behaviors if rewards are wrong
- ⚠️ May become unstable without monitoring
- ⚠️ Requires careful reward engineering
- ⚠️ Slower inference (exploration overhead)

---

## 🔍 Monitoring Training

### **Check Training Progress:**

1. **Watch Console Logs**
   - Training loss should decrease over time
   - Buffer size should grow initially

2. **Compare Checkpoints**
   - Test old vs new models
   - Measure success rate improvement

3. **Track Metrics**
   - Average reward per negotiation
   - Success rate over time
   - Cost savings achieved

---

## 🛡️ Safety Measures

### **Prevent Bad Learning:**

1. **Reward Clipping**
   - Already implemented in environment
   - Prevents extreme reward values

2. **Replay Buffer**
   - Stores diverse experiences
   - Prevents overfitting to recent data

3. **Periodic Checkpoints**
   - Can rollback if performance degrades
   - Keep last 5-10 checkpoints

4. **A/B Testing**
   - Run old model alongside new model
   - Compare performance before full deployment

---

## 🎓 For Your Research Paper

### **Discussion Points:**

1. **Online Learning Architecture**
   - Experience replay buffer
   - Periodic model updates
   - Checkpoint management

2. **Challenges**
   - Exploration-exploitation tradeoff
   - Stability vs adaptability
   - Reward engineering

3. **Future Work**
   - Meta-learning for fast adaptation
   - Multi-agent learning (buyer + sellers)
   - Transfer learning across product categories

---

## 🚀 Quick Start

**Enable training for testing:**

```bash
# In backend directory
echo "ENABLE_TRAINING=true" > .env
python app.py
```

**Run 10-20 negotiations, then check:**

```bash
ls models/checkpoints/
# Should see checkpoint files
```

**Disable training for production:**

```bash
echo "ENABLE_TRAINING=false" > .env
python app.py
```

---

## 📞 Troubleshooting

**Training not happening?**
- Check console for "[Training]" logs
- Verify `ENABLE_TRAINING=true` in .env
- Ensure replay buffer has enough experiences

**Model not saving?**
- Check `models/checkpoints/` directory exists
- Verify write permissions
- Check `save_interval` setting

**Performance degrading?**
- Rollback to previous checkpoint
- Reduce exploration rate
- Review reward function

---

Your RL agent can now learn continuously from real negotiations! 🎉
