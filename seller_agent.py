import numpy as np
from typing import Optional


class SellerAgent:
    """
    Rule-based seller agent.
    
    Each seller has:
    - Fixed stock quantity
    - Base price (minimum acceptable)
    - Simple negotiation logic (no learning)
    """
    
    def __init__(self, seller_id: int, stock: int, base_price: float):
        self.seller_id = seller_id
        self.stock = stock
        self.base_price = base_price
        self.initial_stock = stock
        
        # Negotiation parameters (fixed rules)
        self.min_acceptable_margin = 0.9  # Will accept 90% of base price
        self.preferred_margin = 1.2  # Prefers 120% of base price
        self.negotiation_flexibility = 0.15  # Can negotiate within 15% range
        
    def evaluate_offer(self, offered_price: float, quantity: int) -> float:
        """
        Evaluate an offer and return acceptance probability.
        
        Uses fixed rules based on price relative to base price.
        """
        if quantity > self.stock:
            return 0.0  # Cannot fulfill
        
        min_acceptable = self.base_price * self.min_acceptable_margin
        
        if offered_price < min_acceptable:
            return 0.0  # Too low, reject
        
        if offered_price >= self.base_price * self.preferred_margin:
            return 1.0  # Excellent offer, accept immediately
        
        # Linear probability between min and preferred
        price_range = self.base_price * (self.preferred_margin - self.min_acceptable_margin)
        price_above_min = offered_price - min_acceptable
        
        acceptance_prob = min(1.0, price_above_min / price_range)
        
        # Adjust for quantity (prefer larger orders)
        quantity_bonus = min(0.2, quantity / 100)
        acceptance_prob = min(1.0, acceptance_prob + quantity_bonus)
        
        return acceptance_prob
    
    def generate_counteroffer(self, buyer_offer: float) -> float:
        """
        Generate a counteroffer based on buyer's offer.
        
        Uses fixed negotiation rules.
        """
        if buyer_offer < self.base_price * self.min_acceptable_margin:
            # Offer too low, counter with preferred price
            return self.base_price * self.preferred_margin
        
        # Meet somewhere in the middle
        target_price = (buyer_offer + self.base_price * self.preferred_margin) / 2
        
        # Add small random variation for realism
        variation = np.random.uniform(-0.05, 0.05) * target_price
        counteroffer = target_price + variation
        
        # Ensure counteroffer is above minimum
        counteroffer = max(counteroffer, self.base_price * self.min_acceptable_margin)
        
        return round(counteroffer, 2)
    
    def can_supply(self, quantity: int) -> bool:
        """Check if seller can supply the requested quantity"""
        return self.stock >= quantity
    
    def reserve_stock(self, quantity: int) -> bool:
        """Reserve stock for a potential transaction"""
        if self.can_supply(quantity):
            self.stock -= quantity
            return True
        return False
    
    def release_stock(self, quantity: int):
        """Release reserved stock back to available inventory"""
        self.stock = min(self.initial_stock, self.stock + quantity)
    
    def get_available_stock(self) -> int:
        """Get currently available stock"""
        return self.stock
    
    def get_info(self) -> dict:
        """Get seller information"""
        return {
            'seller_id': self.seller_id,
            'stock': self.stock,
            'base_price': self.base_price,
            'min_acceptable_price': self.base_price * self.min_acceptable_margin,
            'preferred_price': self.base_price * self.preferred_margin
        }
    
    def __repr__(self) -> str:
        return f"SellerAgent(id={self.seller_id}, stock={self.stock}, base_price={self.base_price:.2f})"
