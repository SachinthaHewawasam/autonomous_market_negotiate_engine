# Web Application Setup Guide

## 🎯 Full-Stack Autonomous Market Simulation

This is a production-quality web application for your final-year research project demonstration.

---

## 📋 Architecture Overview

**Backend:**
- Flask (Python) REST API
- SQLAlchemy ORM with SQLite database
- JWT authentication
- WebSocket (SocketIO) for real-time negotiation updates
- Trained RL agent integration for inference

**Frontend:**
- React 18 with modern hooks
- TailwindCSS for styling
- Axios for API calls
- Socket.IO client for real-time updates
- React Router for navigation

---

## 🚀 Quick Start

### Step 1: Install Backend Dependencies

```bash
cd web_app
pip install -r requirements_web.txt
```

### Step 2: Initialize Database

```bash
cd backend
python -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database created!')"
```

### Step 3: Seed Database with Sample Data

```bash
python -c "from app import app, db, User, Product; \
app.app_context().push(); \
seller1 = User(email='seller1@demo.com', name='ABC Supplies', role='seller'); \
seller1.set_password('demo123'); \
buyer = User(email='buyer@demo.com', name='John Businessman', role='buyer'); \
buyer.set_password('demo123'); \
db.session.add_all([seller1, buyer]); \
db.session.commit(); \
product = Product(seller_id=seller1.id, name='Biscuits', brand='Brand X', quantity=45, base_price=8.50, trust_score=0.85); \
db.session.add(product); \
db.session.commit(); \
print('Sample data created!')"
```

### Step 4: Start Backend Server

```bash
python app.py
```

Backend will run on `http://localhost:5000`

### Step 5: Install Frontend Dependencies

Open a new terminal:

```bash
cd web_app/frontend
npm install
```

### Step 6: Start Frontend Development Server

```bash
npm start
```

Frontend will run on `http://localhost:3000`

---

## 👥 Demo Accounts

**Buyer Account:**
- Email: `buyer@demo.com`
- Password: `demo123`

**Seller Account:**
- Email: `seller1@demo.com`
- Password: `demo123`

---

## 🎨 Key Features

### For Sellers:
✅ Add/edit/delete products  
✅ Set quantity and base prices  
✅ View participation in negotiations  
✅ See which products were selected in deals  
✅ Trust score management  

### For Buyers:
✅ Browse all available products  
✅ Submit bulk procurement requests  
✅ Watch real-time RL agent negotiation  
✅ See step-by-step decision explanations  
✅ Human-in-the-loop deal approval  
✅ Accept or reject proposed deals  

### Negotiation Visualization:
✅ Real-time WebSocket updates  
✅ Round-by-round action display  
✅ Reward signals visualization  
✅ Coalition formation explanation  
✅ Fairness and trust rule enforcement  
✅ Color-coded outcomes  

---

## 📁 Project Structure

```
web_app/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── models.py              # Database models
│   ├── config.py              # Configuration
│   └── services/
│       └── negotiation_service.py  # RL agent integration
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── NegotiationViewer.jsx  # Real-time negotiation UI
│   │   │   ├── BuyerDashboard.jsx     # Buyer interface
│   │   │   └── SellerDashboard.jsx    # Seller interface
│   │   ├── services/
│   │   │   └── api.js         # API client
│   │   └── App.js             # Main React app
│   └── package.json
└── requirements_web.txt       # Python dependencies
```

---

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user

### Products (Seller)
- `GET /api/products` - List all products
- `POST /api/products` - Create product
- `PUT /api/products/:id` - Update product
- `DELETE /api/products/:id` - Delete product

### Procurement Requests (Buyer)
- `GET /api/requests` - List requests
- `POST /api/requests` - Create request
- `GET /api/requests/:id` - Get request details

### Negotiation
- `POST /api/negotiate/:request_id` - Start negotiation
- `POST /api/deals/:deal_id/approve` - Approve deal
- `POST /api/deals/:deal_id/reject` - Reject deal

### WebSocket Events
- `negotiation_started` - Negotiation begins
- `market_state` - Initial market data
- `negotiation_step` - Each negotiation round
- `deal_ready` - Deal awaiting approval
- `negotiation_failed` - Negotiation failed

---

## 🎬 Demo Workflow

### 1. Seller Setup
1. Login as seller
2. Add products (name, brand, quantity, price)
3. Products become available in marketplace

### 2. Buyer Request
1. Login as buyer
2. Browse available products
3. Submit procurement request:
   - Product name and brand
   - Desired quantity
   - Maximum budget

### 3. Automated Negotiation
1. System invokes trained RL agent
2. Real-time visualization shows:
   - Agent analyzing market
   - Offers and counteroffers
   - Coalition formation (if needed)
   - Fairness/trust rule application
3. Step-by-step explanations displayed

### 4. Human Approval
1. Final deal presented to buyer
2. Shows:
   - Total cost
   - Savings achieved
   - Selected sellers
   - Quantity breakdown
3. Buyer approves or rejects

### 5. Completion
1. If approved: Deal executed
2. If rejected: Can create new request
3. Sellers see their participation

---

## 🎓 For Your Presentation

### Highlight These Points:

**1. Research Contribution:**
- Novel market design (coalition + fairness + trust)
- RL agent learns negotiation strategies
- Human-in-the-loop for real-world applicability

**2. Technical Excellence:**
- Full-stack implementation
- Real-time WebSocket communication
- Modern React architecture
- RESTful API design
- Database persistence

**3. Transparency:**
- Every decision explained
- Visual step-by-step process
- Clear separation: human vs agent vs rules

**4. Practical Value:**
- Scalable to real B2B scenarios
- Cost savings demonstrated
- Trust-based seller selection
- Coalition optimization

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 5000 is available
netstat -ano | findstr :5000

# Try different port
export FLASK_RUN_PORT=5001
python app.py
```

### Frontend can't connect to backend
```bash
# Update frontend/.env
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_SOCKET_URL=http://localhost:5000
```

### Database errors
```bash
# Reset database
rm market_simulation.db
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

### Model not found
```bash
# Ensure trained model exists
ls ../models/buyer_agent.pth

# If missing, train first
cd ..
python train.py
```

---

## 📊 Database Schema

**Users** - Buyers and sellers  
**Products** - Seller inventory  
**ProcurementRequests** - Buyer requests  
**Negotiations** - Negotiation sessions  
**NegotiationSteps** - Individual rounds  
**Deals** - Final agreements  
**DealSellers** - Sellers in each deal  

---

## 🎯 Next Steps

1. **Customize UI**: Edit React components for your branding
2. **Add Features**: Implement preference adjustments, history views
3. **Deploy**: Use Heroku/Vercel for live demo
4. **Documentation**: Add API docs with Swagger
5. **Testing**: Add unit tests for critical paths

---

## 📞 Support

For issues:
1. Check backend logs in terminal
2. Check browser console for frontend errors
3. Verify database has data: `sqlite3 market_simulation.db ".tables"`
4. Ensure trained model exists

---

**Your web app is ready for demonstration! 🚀**

This provides a professional, interactive showcase of your autonomous market simulation research.
