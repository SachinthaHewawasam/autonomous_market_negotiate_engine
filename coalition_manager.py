import numpy as np
from typing import List, Dict, Optional
from seller_agent import SellerAgent


class CoalitionManager:
    """
    Manages coalition formation among sellers.
    
    Rules (fixed, not learned):
    1. Coalition forms when no single seller can fulfill the request
    2. Selects sellers with best price-to-trust ratio
    3. Ensures fair profit distribution among coalition members
    4. Limits coalition size to avoid complexity
    """
    
    def __init__(self, sellers: List[SellerAgent], max_coalition_size: int = 4):
        self.sellers = sellers
        self.max_coalition_size = max_coalition_size
    
    def form_coalition(
        self,
        requested_quantity: int,
        primary_seller_id: Optional[int] = None,
        trust_scores: Optional[List[float]] = None
    ) -> Optional[List[Dict]]:
        """
        Form a coalition to fulfill the requested quantity.
        
        Args:
            requested_quantity: Total quantity needed
            primary_seller_id: Preferred primary seller (optional)
            trust_scores: Trust scores for each seller (optional)
        
        Returns:
            List of dicts with seller_id, quantity, and price, or None if impossible
        """
        if trust_scores is None:
            trust_scores = [0.8] * len(self.sellers)  # Default trust
        
        # Check if any single seller can fulfill
        for seller in self.sellers:
            if seller.stock >= requested_quantity:
                return None  # No coalition needed
        
        # Build coalition using greedy algorithm
        coalition = []
        remaining_quantity = requested_quantity
        available_sellers = list(range(len(self.sellers)))
        
        # If primary seller specified, start with them
        if primary_seller_id is not None and primary_seller_id < len(self.sellers):
            seller = self.sellers[primary_seller_id]
            if seller.stock > 0:
                quantity = min(seller.stock, remaining_quantity)
                coalition.append({
                    'seller_id': primary_seller_id,
                    'quantity': quantity,
                    'price': seller.base_price,
                    'trust': trust_scores[primary_seller_id]
                })
                remaining_quantity -= quantity
                available_sellers.remove(primary_seller_id)
        
        # Add sellers based on price-trust score
        while remaining_quantity > 0 and len(coalition) < self.max_coalition_size:
            if not available_sellers:
                break
            
            # Calculate score for each available seller
            best_seller_id = None
            best_score = -float('inf')
            
            for seller_id in available_sellers:
                seller = self.sellers[seller_id]
                if seller.stock == 0:
                    continue
                
                # Score = trust / price (higher is better)
                score = trust_scores[seller_id] / seller.base_price
                
                if score > best_score:
                    best_score = score
                    best_seller_id = seller_id
            
            if best_seller_id is None:
                break  # No more sellers available
            
            seller = self.sellers[best_seller_id]
            quantity = min(seller.stock, remaining_quantity)
            
            coalition.append({
                'seller_id': best_seller_id,
                'quantity': quantity,
                'price': seller.base_price,
                'trust': trust_scores[best_seller_id]
            })
            
            remaining_quantity -= quantity
            available_sellers.remove(best_seller_id)
        
        # Check if coalition can fulfill the request
        total_quantity = sum(member['quantity'] for member in coalition)
        
        if total_quantity < requested_quantity:
            return None  # Cannot fulfill even with coalition
        
        return coalition
    
    def calculate_coalition_price(self, coalition: List[Dict]) -> float:
        """
        Calculate total price for a coalition.
        
        Applies fair pricing rules:
        - Each seller gets their base price
        - Small coordination overhead (5%)
        - Volume discount for large orders (up to 10%)
        """
        if not coalition:
            return 0.0
        
        # Base cost
        base_cost = sum(member['quantity'] * member['price'] for member in coalition)
        
        # Coordination overhead
        coordination_overhead = 1.05
        
        # Volume discount
        total_quantity = sum(member['quantity'] for member in coalition)
        volume_discount = 1.0 - min(0.1, total_quantity / 1000)
        
        total_price = base_cost * coordination_overhead * volume_discount
        
        return round(total_price, 2)
    
    def distribute_profit(self, coalition: List[Dict], total_payment: float) -> Dict[int, float]:
        """
        Distribute profit among coalition members fairly.
        
        Rules:
        - Each seller gets at least their base cost
        - Extra profit distributed proportionally to quantity supplied
        - Trust score influences profit share slightly
        """
        if not coalition:
            return {}
        
        # Calculate base costs
        base_costs = {}
        total_base_cost = 0.0
        
        for member in coalition:
            base_cost = member['quantity'] * member['price']
            base_costs[member['seller_id']] = base_cost
            total_base_cost += base_cost
        
        # Calculate extra profit
        extra_profit = max(0, total_payment - total_base_cost)
        
        # Distribute extra profit
        profit_distribution = {}
        total_quantity = sum(member['quantity'] for member in coalition)
        
        for member in coalition:
            seller_id = member['seller_id']
            
            # Base payment
            payment = base_costs[seller_id]
            
            # Share of extra profit (weighted by quantity and trust)
            quantity_share = member['quantity'] / total_quantity
            trust_weight = member['trust'] * 0.2 + 0.8  # Trust contributes 20%
            profit_share = (quantity_share * trust_weight) * extra_profit
            
            payment += profit_share
            profit_distribution[seller_id] = round(payment, 2)
        
        return profit_distribution
    
    def validate_coalition(self, coalition: List[Dict]) -> bool:
        """
        Validate that a coalition meets all rules.
        
        Checks:
        - Size within limits
        - All sellers have sufficient stock
        - No duplicate sellers
        """
        if not coalition or len(coalition) > self.max_coalition_size:
            return False
        
        seller_ids = set()
        
        for member in coalition:
            seller_id = member['seller_id']
            
            # Check for duplicates
            if seller_id in seller_ids:
                return False
            seller_ids.add(seller_id)
            
            # Check stock availability
            if seller_id >= len(self.sellers):
                return False
            
            seller = self.sellers[seller_id]
            if member['quantity'] > seller.stock:
                return False
        
        return True
    
    def get_coalition_info(self, coalition: List[Dict]) -> Dict:
        """Get detailed information about a coalition"""
        if not coalition:
            return {}
        
        total_quantity = sum(member['quantity'] for member in coalition)
        total_price = self.calculate_coalition_price(coalition)
        avg_price_per_unit = total_price / total_quantity if total_quantity > 0 else 0
        avg_trust = np.mean([member['trust'] for member in coalition])
        
        return {
            'num_members': len(coalition),
            'total_quantity': total_quantity,
            'total_price': total_price,
            'avg_price_per_unit': avg_price_per_unit,
            'avg_trust': avg_trust,
            'members': coalition
        }
