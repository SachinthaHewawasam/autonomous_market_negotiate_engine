"""
Flask backend for Autonomous Market Simulation
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from web_app.backend.config import Config
from web_app.backend.models import db, User, Product, ProcurementRequest, Negotiation, NegotiationStep, Deal, DealSeller
from web_app.backend.services.negotiation_service import NegotiationService
from multi_agent_market import MultiAgentMarket

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
CORS(app, resources={r"/api/*": {"origins": "*"}})
db.init_app(app)
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize negotiation service
# Enable online training (set to False for inference-only mode)
ENABLE_ONLINE_TRAINING = os.environ.get('ENABLE_TRAINING', 'False').lower() == 'true'
negotiation_service = NegotiationService(app.config['MODEL_PATH'], enable_training=ENABLE_ONLINE_TRAINING)

# Initialize multi-agent market
multi_agent_market = MultiAgentMarket(num_buyers=3, num_sellers=3)
# Load agents with different strategies for diverse competition
model_path = app.config['MODEL_PATH']
multi_agent_market.load_agents(
    [model_path, model_path, model_path],
    strategies=['aggressive', 'conservative', 'balanced']
)


# ============================================================================
# Authentication Routes
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user (buyer or seller)"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['email', 'password', 'name', 'role']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if data['role'] not in ['buyer', 'seller']:
        return jsonify({'error': 'Role must be buyer or seller'}), 400
    
    # Check if user already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    # Create new user
    user = User(
        email=data['email'],
        name=data['name'],
        role=data['role']
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    # Create access token
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'message': 'User registered successfully',
        'access_token': access_token,
        'user': user.to_dict()
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': user.to_dict()
    }), 200


@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'user': user.to_dict()}), 200


# ============================================================================
# Product Routes (Seller)
# ============================================================================

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products or filter by seller"""
    seller_id = request.args.get('seller_id', type=int)
    
    if seller_id:
        products = Product.query.filter_by(seller_id=seller_id).all()
    else:
        products = Product.query.all()
    
    return jsonify({
        'products': [p.to_dict() for p in products]
    }), 200


@app.route('/api/products', methods=['POST'])
@jwt_required()
def create_product():
    """Create a new product (seller only)"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if user.role != 'seller':
        return jsonify({'error': 'Only sellers can add products'}), 403
    
    data = request.get_json()
    required_fields = ['name', 'brand', 'quantity', 'base_price']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    product = Product(
        seller_id=user_id,
        name=data['name'],
        brand=data['brand'],
        quantity=data['quantity'],
        base_price=data['base_price'],
        trust_score=data.get('trust_score', 0.8)
    )
    
    db.session.add(product)
    db.session.commit()
    
    return jsonify({
        'message': 'Product created successfully',
        'product': product.to_dict()
    }), 201


@app.route('/api/products/<int:product_id>', methods=['PUT'])
@jwt_required()
def update_product(product_id):
    """Update a product (seller only, own products)"""
    user_id = get_jwt_identity()
    product = Product.query.get(product_id)
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    if product.seller_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    
    if 'name' in data:
        product.name = data['name']
    if 'brand' in data:
        product.brand = data['brand']
    if 'quantity' in data:
        product.quantity = data['quantity']
    if 'base_price' in data:
        product.base_price = data['base_price']
    
    product.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'message': 'Product updated successfully',
        'product': product.to_dict()
    }), 200


@app.route('/api/products/<int:product_id>', methods=['DELETE'])
@jwt_required()
def delete_product(product_id):
    """Delete a product (seller only, own products)"""
    user_id = get_jwt_identity()
    product = Product.query.get(product_id)
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    if product.seller_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(product)
    db.session.commit()
    
    return jsonify({'message': 'Product deleted successfully'}), 200


# ============================================================================
# Procurement Request Routes (Buyer)
# ============================================================================

@app.route('/api/requests', methods=['GET'])
@jwt_required()
def get_requests():
    """Get procurement requests (buyer sees own, seller sees all)"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if user.role == 'buyer':
        requests = ProcurementRequest.query.filter_by(buyer_id=user_id).order_by(ProcurementRequest.created_at.desc()).all()
    else:
        # Sellers see all requests
        requests = ProcurementRequest.query.order_by(ProcurementRequest.created_at.desc()).all()
    
    return jsonify({
        'requests': [r.to_dict() for r in requests]
    }), 200


@app.route('/api/requests', methods=['POST'])
@jwt_required()
def create_request():
    """Create a new procurement request (buyer only)"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if user.role != 'buyer':
        return jsonify({'error': 'Only buyers can create requests'}), 403
    
    data = request.get_json()
    required_fields = ['product_name', 'brand', 'quantity', 'max_budget']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    procurement_request = ProcurementRequest(
        buyer_id=user_id,
        product_name=data['product_name'],
        brand=data['brand'],
        quantity=data['quantity'],
        max_budget=data['max_budget'],
        status='pending'
    )
    
    db.session.add(procurement_request)
    db.session.commit()
    
    return jsonify({
        'message': 'Request created successfully',
        'request': procurement_request.to_dict()
    }), 201


@app.route('/api/requests/<int:request_id>', methods=['GET'])
@jwt_required()
def get_request(request_id):
    """Get a specific request with full details"""
    procurement_request = ProcurementRequest.query.get(request_id)
    
    if not procurement_request:
        return jsonify({'error': 'Request not found'}), 404
    
    return jsonify({
        'request': procurement_request.to_dict()
    }), 200


# ============================================================================
# Negotiation Routes
# ============================================================================

@app.route('/api/what-if', methods=['POST'])
@jwt_required()
def what_if_prediction():
    """Predict negotiation outcome for given parameters"""
    data = request.get_json()
    
    try:
        # Get parameters
        product_name = data.get('product_name')
        brand = data.get('brand')
        quantity = data.get('quantity')
        max_budget = data.get('max_budget')
        
        # Get matching products
        products = Product.query.filter_by(name=product_name, brand=brand).all()
        
        if not products:
            return jsonify({'error': 'No matching products found'}), 404
        
        # Run quick simulation
        prediction = negotiation_service.predict_outcome(
            products=products,
            quantity=quantity,
            max_budget=max_budget
        )
        
        return jsonify(prediction), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/negotiate/<int:request_id>', methods=['POST'])
@jwt_required()
def start_negotiation(request_id):
    """Start negotiation for a procurement request"""
    user_id = get_jwt_identity()
    procurement_request = ProcurementRequest.query.get(request_id)
    
    if not procurement_request:
        return jsonify({'error': 'Request not found'}), 404
    
    if procurement_request.buyer_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if procurement_request.status != 'pending':
        return jsonify({'error': 'Request already processed'}), 400
    
    # Update status
    procurement_request.status = 'negotiating'
    db.session.commit()
    
    # Start negotiation in background with app context
    def run_with_context():
        try:
            print(f"Starting background negotiation for request {request_id}")
            with app.app_context():
                negotiation_service.run_negotiation(request_id, socketio)
        except Exception as e:
            print(f"Error in background task: {e}")
            import traceback
            traceback.print_exc()
    
    socketio.start_background_task(run_with_context)
    
    return jsonify({
        'message': 'Negotiation started',
        'request_id': request_id
    }), 200


@app.route('/api/deals/<int:deal_id>/approve', methods=['POST'])
@jwt_required()
def approve_deal(deal_id):
    """Approve a deal (human-in-the-loop)"""
    user_id = get_jwt_identity()
    deal = Deal.query.get(deal_id)
    
    if not deal:
        return jsonify({'error': 'Deal not found'}), 404
    
    negotiation = deal.negotiation
    procurement_request = negotiation.request
    
    if procurement_request.buyer_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Approve deal
    deal.human_approved = True
    deal.approval_timestamp = datetime.utcnow()
    
    # Update request status
    procurement_request.status = 'completed'
    procurement_request.completed_at = datetime.utcnow()
    
    # Update negotiation status
    negotiation.final_status = 'approved'
    negotiation.completed_at = datetime.utcnow()
    
    db.session.commit()
    
    # Emit approval event
    socketio.emit('deal_approved', {
        'deal_id': deal_id,
        'request_id': procurement_request.id
    }, room=f'request_{procurement_request.id}')
    
    return jsonify({
        'message': 'Deal approved successfully',
        'deal': deal.to_dict()
    }), 200


@app.route('/api/deals/<int:deal_id>/reject', methods=['POST'])
@jwt_required()
def reject_deal(deal_id):
    """Reject a deal (human-in-the-loop)"""
    user_id = get_jwt_identity()
    deal = Deal.query.get(deal_id)
    
    if not deal:
        return jsonify({'error': 'Deal not found'}), 404
    
    negotiation = deal.negotiation
    procurement_request = negotiation.request
    
    if procurement_request.buyer_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Update request status
    procurement_request.status = 'rejected'
    procurement_request.completed_at = datetime.utcnow()
    
    # Update negotiation status
    negotiation.final_status = 'rejected'
    negotiation.completed_at = datetime.utcnow()
    
    db.session.commit()
    
    # Emit rejection event
    socketio.emit('deal_rejected', {
        'deal_id': deal_id,
        'request_id': procurement_request.id
    }, room=f'request_{procurement_request.id}')
    
    return jsonify({
        'message': 'Deal rejected'
    }), 200


# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to negotiation server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('join_request')
def handle_join_request(data):
    """Join a request room for real-time updates"""
    request_id = data.get('request_id')
    if request_id:
        join_room(f'request_{request_id}')
        emit('joined_request', {'request_id': request_id})


@socketio.on('leave_request')
def handle_leave_request(data):
    """Leave a request room"""
    request_id = data.get('request_id')
    if request_id:
        leave_room(f'request_{request_id}')
        emit('left_request', {'request_id': request_id})


# ============================================================================
# Database Initialization
# ============================================================================

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print('Database initialized!')


@app.cli.command()
def seed_db():
    """Seed the database with sample data"""
    # Create sample sellers
    seller1 = User(email='seller1@example.com', name='ABC Supplies', role='seller')
    seller1.set_password('password123')
    
    seller2 = User(email='seller2@example.com', name='XYZ Traders', role='seller')
    seller2.set_password('password123')
    
    seller3 = User(email='seller3@example.com', name='Global Mart', role='seller')
    seller3.set_password('password123')
    
    # Create sample buyer
    buyer = User(email='buyer@example.com', name='John Businessman', role='buyer')
    buyer.set_password('password123')
    
    db.session.add_all([seller1, seller2, seller3, buyer])
    db.session.commit()
    
    # Create sample products
    products = [
        Product(seller_id=seller1.id, name='Biscuits', brand='Brand X', quantity=45, base_price=8.50, trust_score=0.85),
        Product(seller_id=seller1.id, name='Cookies', brand='Brand Y', quantity=30, base_price=12.00, trust_score=0.85),
        Product(seller_id=seller2.id, name='Biscuits', brand='Brand X', quantity=38, base_price=9.20, trust_score=0.80),
        Product(seller_id=seller2.id, name='Crackers', brand='Brand Z', quantity=50, base_price=7.50, trust_score=0.80),
        Product(seller_id=seller3.id, name='Biscuits', brand='Brand X', quantity=42, base_price=10.00, trust_score=0.90),
        Product(seller_id=seller3.id, name='Wafers', brand='Brand A', quantity=35, base_price=11.50, trust_score=0.90),
    ]
    
    db.session.add_all(products)
    db.session.commit()
    
    print('Database seeded with sample data!')


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


# ============================================================================
# Multi-Agent Competition Routes
# ============================================================================

@app.route('/api/multi-agent/compete', methods=['POST'])
@jwt_required()
def start_multi_agent_competition():
    """
    Start a multi-agent competition with multiple buyers competing for products
    
    Request body:
    {
        "requests": [
            {"buyer_name": "Agent_1", "product_name": "Biscuits", "brand": "Brand X", "quantity": 40, "max_budget": 400},
            {"buyer_name": "Agent_2", "product_name": "Biscuits", "brand": "Brand X", "quantity": 45, "max_budget": 450},
            {"buyer_name": "Agent_3", "product_name": "Biscuits", "brand": "Brand X", "quantity": 35, "max_budget": 380}
        ]
    }
    """
    try:
        data = request.get_json()
        requests = data.get('requests', [])
        
        if len(requests) < 2:
            return jsonify({'error': 'Need at least 2 buyers to compete'}), 400
        
        # Format requests for multi-agent market
        formatted_requests = []
        for i, req in enumerate(requests):
            formatted_requests.append({
                'buyer_id': i,
                'product': req.get('product_name', 'Biscuits'),
                'quantity': req.get('quantity', 40),
                'max_budget': req.get('max_budget', 400)
            })
        
        # Run competition
        results = multi_agent_market.run_full_competition(
            formatted_requests, 
            max_rounds=10
        )
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_json_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Format response with converted types
        response_data = {
            'success': True,
            'num_rounds': int(results['num_rounds']),
            'winner': results['final_analysis']['winner'],
            'winner_reward': float(results['final_analysis']['winner_reward']),
            'buyer_metrics': convert_to_json_serializable(results['final_analysis']['buyer_metrics']),
            'emergent_behaviors': results['final_analysis']['emergent_behaviors'],
            'rounds': convert_to_json_serializable(results['rounds'])
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/multi-agent/demo', methods=['GET'])
def multi_agent_demo():
    """Run a quick demo of multi-agent competition"""
    try:
        # Demo requests
        requests = [
            {'buyer_id': 0, 'product': 'Biscuits', 'quantity': 40, 'max_budget': 400},
            {'buyer_id': 1, 'product': 'Biscuits', 'quantity': 45, 'max_budget': 450},
            {'buyer_id': 2, 'product': 'Biscuits', 'quantity': 35, 'max_budget': 380},
        ]
        
        results = multi_agent_market.run_full_competition(requests, max_rounds=5)
        
        return jsonify({
            'success': True,
            'demo': True,
            'winner': results['final_analysis']['winner'],
            'winner_reward': float(results['final_analysis']['winner_reward']),
            'emergent_behaviors': results['final_analysis']['emergent_behaviors'],
            'num_rounds': int(results['num_rounds'])
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✓ Database tables ready")
    
    print("=" * 60)
    print("🚀 Autonomous Market Simulation Backend")
    print("=" * 60)
    print("Backend API: http://localhost:5000")
    print("WebSocket: ws://localhost:5000")
    print("\nDemo Accounts:")
    print("  Buyer:  buyer@demo.com / demo123")
    print("  Seller: seller1@demo.com / demo123")
    print("=" * 60)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
