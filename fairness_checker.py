import numpy as np
from typing import List, Dict, Optional


class FairnessChecker:
    """
    Enforces fairness rules in the market.
    
    Fixed rules (not learned):
    1. Prevents exploitative pricing (too high above base price)
    2. Ensures fair profit distribution in coalitions
    3. Detects price manipulation attempts
    4. Validates reasonable price ranges
    """
    
    def __init__(
        self,
        max_markup: float = 2.0,
        min_markup: float = 0.7,
        coalition_fairness_threshold: float = 0.3
    ):
        """
        Initialize fairness checker.
        
        Args:
            max_markup: Maximum allowed markup over base price (200%)
            min_markup: Minimum allowed markup (70% - allows small discounts)
            coalition_fairness_threshold: Max allowed variance in profit distribution
        """
        self.max_markup = max_markup
        self.min_markup = min_markup
        self.coalition_fairness_threshold = coalition_fairness_threshold
    
    def is_price_fair(self, offered_price: float, base_price: float) -> bool:
        """
        Check if an offered price is fair relative to base price.
        
        Args:
            offered_price: Price being offered
            base_price: Seller's base price
        
        Returns:
            True if price is within fair range
        """
        if base_price <= 0:
            return False
        
        markup = offered_price / base_price
        
        # Check if within acceptable range
        if markup < self.min_markup or markup > self.max_markup:
            return False
        
        return True
    
    def is_coalition_fair(
        self,
        coalition: List[Dict],
        total_price: float,
        requested_quantity: Optional[int] = None
    ) -> bool:
        """
        Check if coalition pricing and profit distribution is fair.
        
        Args:
            coalition: List of coalition members with quantity and price
            total_price: Total price for the coalition
            requested_quantity: Total quantity requested (optional)
        
        Returns:
            True if coalition is fair
        """
        if not coalition:
            return False
        
        # Calculate base cost
        base_cost = sum(member['quantity'] * member['price'] for member in coalition)
        
        if base_cost <= 0:
            return False
        
        # Check overall markup
        overall_markup = total_price / base_cost
        
        if overall_markup > self.max_markup:
            return False  # Excessive markup
        
        # Check profit distribution fairness
        if len(coalition) > 1:
            # Calculate expected profit share for each member
            total_quantity = sum(member['quantity'] for member in coalition)
            profit_shares = []
            
            for member in coalition:
                quantity_share = member['quantity'] / total_quantity
                profit_shares.append(quantity_share)
            
            # Check variance in profit shares
            variance = np.var(profit_shares)
            
            if variance > self.coalition_fairness_threshold:
                return False  # Unfair profit distribution
        
        return True
    
    def detect_price_manipulation(
        self,
        price_history: List[float],
        current_price: float,
        window_size: int = 5
    ) -> bool:
        """
        Detect potential price manipulation.
        
        Args:
            price_history: Historical prices
            current_price: Current offered price
            window_size: Number of recent prices to consider
        
        Returns:
            True if manipulation detected
        """
        if len(price_history) < window_size:
            return False  # Not enough history
        
        recent_prices = price_history[-window_size:]
        avg_recent_price = np.mean(recent_prices)
        std_recent_price = np.std(recent_prices)
        
        if std_recent_price == 0:
            return False
        
        # Check if current price is abnormally different
        z_score = abs(current_price - avg_recent_price) / std_recent_price
        
        # Flag if more than 3 standard deviations away
        if z_score > 3.0:
            return True
        
        return False
    
    def validate_price_range(
        self,
        price: float,
        min_price: float = 1.0,
        max_price: float = 100.0
    ) -> bool:
        """
        Validate that price is within reasonable absolute range.
        
        Args:
            price: Price to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price
        
        Returns:
            True if price is valid
        """
        return min_price <= price <= max_price
    
    def calculate_fairness_score(
        self,
        offered_price: float,
        base_price: float,
        quantity: int,
        trust_score: float = 0.8
    ) -> float:
        """
        Calculate a fairness score for an offer.
        
        Args:
            offered_price: Offered price per unit
            base_price: Base price per unit
            quantity: Quantity being offered
            trust_score: Trust score of the seller
        
        Returns:
            Fairness score between 0 and 1 (higher is fairer)
        """
        if base_price <= 0:
            return 0.0
        
        # Price fairness component
        markup = offered_price / base_price
        
        if markup < self.min_markup or markup > self.max_markup:
            price_fairness = 0.0
        else:
            # Optimal markup is around 1.0 (base price)
            price_deviation = abs(markup - 1.0)
            price_fairness = max(0.0, 1.0 - price_deviation)
        
        # Quantity component (larger quantities are better)
        quantity_score = min(1.0, quantity / 100)
        
        # Trust component
        trust_component = trust_score
        
        # Weighted combination
        fairness_score = (
            0.5 * price_fairness +
            0.3 * quantity_score +
            0.2 * trust_component
        )
        
        return round(fairness_score, 3)
    
    def check_coalition_balance(self, coalition: List[Dict]) -> Dict:
        """
        Check balance metrics for a coalition.
        
        Returns:
            Dictionary with balance metrics
        """
        if not coalition:
            return {'balanced': False, 'reason': 'Empty coalition'}
        
        quantities = [member['quantity'] for member in coalition]
        prices = [member['price'] for member in coalition]
        
        # Check quantity balance
        quantity_variance = np.var(quantities)
        quantity_mean = np.mean(quantities)
        quantity_cv = quantity_variance / quantity_mean if quantity_mean > 0 else float('inf')
        
        # Check price balance
        price_variance = np.var(prices)
        price_mean = np.mean(prices)
        price_cv = price_variance / price_mean if price_mean > 0 else float('inf')
        
        # Coalition is balanced if coefficient of variation is low
        balanced = quantity_cv < 1.0 and price_cv < 0.5
        
        return {
            'balanced': balanced,
            'quantity_variance': round(quantity_variance, 2),
            'price_variance': round(price_variance, 2),
            'quantity_cv': round(quantity_cv, 3),
            'price_cv': round(price_cv, 3),
            'num_members': len(coalition)
        }
