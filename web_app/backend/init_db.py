"""
Initialize database and create demo accounts
"""
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models import db, User, Product
from config import Config
from flask import Flask

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

def init_database():
    """Initialize database with demo data"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✓ Database tables created")
        
        # Check if demo users already exist
        if User.query.filter_by(email='seller1@demo.com').first():
            print("✓ Demo accounts already exist")
            return
        
        # Create demo sellers
        seller1 = User(email='seller1@demo.com', name='ABC Supplies', role='seller')
        seller1.set_password('demo123')
        
        seller2 = User(email='seller2@demo.com', name='XYZ Traders', role='seller')
        seller2.set_password('demo123')
        
        seller3 = User(email='seller3@demo.com', name='Global Mart', role='seller')
        seller3.set_password('demo123')
        
        # Create demo buyer
        buyer = User(email='buyer@demo.com', name='John Businessman', role='buyer')
        buyer.set_password('demo123')
        
        db.session.add_all([seller1, seller2, seller3, buyer])
        db.session.commit()
        print("✓ Demo accounts created:")
        print("  - seller1@demo.com / demo123")
        print("  - seller2@demo.com / demo123")
        print("  - seller3@demo.com / demo123")
        print("  - buyer@demo.com / demo123")
        
        # Create demo products
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
        print(f"✓ {len(products)} demo products created")
        
        print("\n✅ Database initialization complete!")
        print("\nYou can now:")
        print("1. Start the backend: python app.py")
        print("2. Start the frontend: npm start (in frontend folder)")
        print("3. Login with any demo account above")

if __name__ == '__main__':
    init_database()
