"""
Database models for the market simulation web app
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """User model for both buyers and sellers"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'buyer' or 'seller'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    products = db.relationship('Product', backref='seller', lazy=True, cascade='all, delete-orphan')
    buyer_requests = db.relationship('ProcurementRequest', backref='buyer', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'created_at': self.created_at.isoformat()
        }


class Product(db.Model):
    """Product model for seller inventory"""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    brand = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    base_price = db.Column(db.Float, nullable=False)
    trust_score = db.Column(db.Float, default=0.8)  # Default trust score
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'seller_id': self.seller_id,
            'seller_name': self.seller.name if self.seller else None,
            'name': self.name,
            'brand': self.brand,
            'quantity': self.quantity,
            'base_price': self.base_price,
            'trust_score': self.trust_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class ProcurementRequest(db.Model):
    """Procurement request from buyers"""
    __tablename__ = 'procurement_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    buyer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    product_name = db.Column(db.String(200), nullable=False)
    brand = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    max_budget = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, negotiating, completed, failed, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    negotiation = db.relationship('Negotiation', backref='request', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'buyer_id': self.buyer_id,
            'buyer_name': self.buyer.name if self.buyer else None,
            'product_name': self.product_name,
            'brand': self.brand,
            'quantity': self.quantity,
            'max_budget': self.max_budget,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'negotiation': self.negotiation.to_dict() if self.negotiation else None
        }


class Negotiation(db.Model):
    """Negotiation session details"""
    __tablename__ = 'negotiations'
    
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('procurement_requests.id'), nullable=False, unique=True)
    total_reward = db.Column(db.Float, default=0.0)
    num_rounds = db.Column(db.Integer, default=0)
    coalition_formed = db.Column(db.Boolean, default=False)
    total_cost = db.Column(db.Float, nullable=True)
    final_status = db.Column(db.String(20), nullable=True)  # success, failed, pending_approval
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    steps = db.relationship('NegotiationStep', backref='negotiation', lazy=True, cascade='all, delete-orphan')
    deal = db.relationship('Deal', backref='negotiation', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'total_reward': self.total_reward,
            'num_rounds': self.num_rounds,
            'coalition_formed': self.coalition_formed,
            'total_cost': self.total_cost,
            'final_status': self.final_status,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'steps': [step.to_dict() for step in self.steps],
            'deal': self.deal.to_dict() if self.deal else None
        }


class NegotiationStep(db.Model):
    """Individual steps in a negotiation"""
    __tablename__ = 'negotiation_steps'
    
    id = db.Column(db.Integer, primary_key=True)
    negotiation_id = db.Column(db.Integer, db.ForeignKey('negotiations.id'), nullable=False, index=True)
    round_number = db.Column(db.Integer, nullable=False)
    action_type = db.Column(db.String(50), nullable=False)  # offer, counteroffer, accept, reject, coalition
    seller_id = db.Column(db.Integer, nullable=True)
    price = db.Column(db.Float, nullable=True)
    quantity = db.Column(db.Integer, nullable=True)
    reward = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'round_number': self.round_number,
            'action_type': self.action_type,
            'seller_id': self.seller_id,
            'price': self.price,
            'quantity': self.quantity,
            'reward': self.reward,
            'explanation': self.explanation,
            'timestamp': self.timestamp.isoformat()
        }


class Deal(db.Model):
    """Final deal details"""
    __tablename__ = 'deals'
    
    id = db.Column(db.Integer, primary_key=True)
    negotiation_id = db.Column(db.Integer, db.ForeignKey('negotiations.id'), nullable=False, unique=True)
    total_cost = db.Column(db.Float, nullable=False)
    total_quantity = db.Column(db.Integer, nullable=False)
    savings = db.Column(db.Float, nullable=False)
    human_approved = db.Column(db.Boolean, default=False)
    approval_timestamp = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sellers = db.relationship('DealSeller', backref='deal', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'negotiation_id': self.negotiation_id,
            'total_cost': self.total_cost,
            'total_quantity': self.total_quantity,
            'savings': self.savings,
            'human_approved': self.human_approved,
            'approval_timestamp': self.approval_timestamp.isoformat() if self.approval_timestamp else None,
            'created_at': self.created_at.isoformat(),
            'sellers': [seller.to_dict() for seller in self.sellers]
        }


class DealSeller(db.Model):
    """Sellers participating in a deal"""
    __tablename__ = 'deal_sellers'
    
    id = db.Column(db.Integer, primary_key=True)
    deal_id = db.Column(db.Integer, db.ForeignKey('deals.id'), nullable=False, index=True)
    seller_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price_per_unit = db.Column(db.Float, nullable=False)
    subtotal = db.Column(db.Float, nullable=False)
    
    # Relationships
    seller = db.relationship('User', foreign_keys=[seller_id])
    product = db.relationship('Product', foreign_keys=[product_id])
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'seller_id': self.seller_id,
            'seller_name': self.seller.name if self.seller else None,
            'product_id': self.product_id,
            'product_name': self.product.name if self.product else None,
            'quantity': self.quantity,
            'price_per_unit': self.price_per_unit,
            'subtotal': self.subtotal
        }
