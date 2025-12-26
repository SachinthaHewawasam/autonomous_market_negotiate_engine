"""Quick script to check and fix database"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import db, User, Product
from config import Config
from flask import Flask

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    print("Checking database...")
    
    users = User.query.all()
    products = Product.query.all()
    
    print(f"\n✓ Found {len(users)} users:")
    for u in users:
        print(f"  - {u.email} ({u.role})")
    
    print(f"\n✓ Found {len(products)} products:")
    for p in products:
        print(f"  - {p.name} ({p.brand}) - {p.quantity} units @ ${p.base_price:.2f} - Seller: {p.seller.name}")
    
    if len(users) == 0 or len(products) == 0:
        print("\n⚠ Database is empty! Run: python init_db.py")
    else:
        print("\n✅ Database is ready for negotiation!")
