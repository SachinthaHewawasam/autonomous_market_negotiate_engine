import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class TrustManager:
    """
    Manages trust scores for sellers based on delivery outcomes.
    
    Fixed rules (not learned):
    1. Trust starts at a neutral value (0.8)
    2. Successful deliveries increase trust
    3. Failed deliveries decrease trust
    4. Trust decays slowly over time without interaction
    5. Trust influences coalition formation and pricing
    """
    
    def __init__(
        self,
        num_sellers: int,
        initial_trust: float = 0.8,
        trust_increase_rate: float = 0.05,
        trust_decrease_rate: float = 0.15,
        decay_rate: float = 0.01,
        min_trust: float = 0.1,
        max_trust: float = 1.0
    ):
        """
        Initialize trust manager.
        
        Args:
            num_sellers: Number of sellers in the market
            initial_trust: Starting trust score for all sellers
            trust_increase_rate: How much trust increases on success
            trust_decrease_rate: How much trust decreases on failure
            decay_rate: Slow decay rate per time step
            min_trust: Minimum possible trust score
            max_trust: Maximum possible trust score
        """
        self.num_sellers = num_sellers
        self.initial_trust = initial_trust
        self.trust_increase_rate = trust_increase_rate
        self.trust_decrease_rate = trust_decrease_rate
        self.decay_rate = decay_rate
        self.min_trust = min_trust
        self.max_trust = max_trust
        
        # Initialize trust scores
        self.trust_scores = np.full(num_sellers, initial_trust, dtype=np.float32)
        
        # Track interaction history
        self.interaction_history = defaultdict(list)
        self.total_interactions = defaultdict(int)
        self.successful_interactions = defaultdict(int)
        
        # Time step counter for decay
        self.time_step = 0
    
    def get_trust_score(self, seller_id: int) -> float:
        """Get current trust score for a seller"""
        if seller_id < 0 or seller_id >= self.num_sellers:
            return self.initial_trust
        
        return float(self.trust_scores[seller_id])
    
    def update_trust(self, seller_id: int, delivery_successful: bool):
        """
        Update trust score based on delivery outcome.
        
        Args:
            seller_id: ID of the seller
            delivery_successful: Whether delivery was successful
        """
        if seller_id < 0 or seller_id >= self.num_sellers:
            return
        
        current_trust = self.trust_scores[seller_id]
        
        if delivery_successful:
            # Increase trust
            new_trust = current_trust + self.trust_increase_rate
            self.successful_interactions[seller_id] += 1
        else:
            # Decrease trust more significantly
            new_trust = current_trust - self.trust_decrease_rate
        
        # Clamp to valid range
        new_trust = np.clip(new_trust, self.min_trust, self.max_trust)
        
        self.trust_scores[seller_id] = new_trust
        
        # Record interaction
        self.total_interactions[seller_id] += 1
        self.interaction_history[seller_id].append({
            'time_step': self.time_step,
            'successful': delivery_successful,
            'trust_before': current_trust,
            'trust_after': new_trust
        })
    
    def apply_decay(self):
        """
        Apply time-based decay to trust scores.
        
        Trust slowly decays towards neutral (0.8) when there's no interaction.
        """
        self.time_step += 1
        
        for seller_id in range(self.num_sellers):
            current_trust = self.trust_scores[seller_id]
            
            # Decay towards neutral value
            if current_trust > self.initial_trust:
                new_trust = current_trust - self.decay_rate
                new_trust = max(new_trust, self.initial_trust)
            elif current_trust < self.initial_trust:
                new_trust = current_trust + self.decay_rate
                new_trust = min(new_trust, self.initial_trust)
            else:
                new_trust = current_trust
            
            self.trust_scores[seller_id] = new_trust
    
    def get_trust_category(self, seller_id: int) -> str:
        """
        Categorize trust level.
        
        Returns:
            'high', 'medium', 'low', or 'very_low'
        """
        trust = self.get_trust_score(seller_id)
        
        if trust >= 0.9:
            return 'high'
        elif trust >= 0.7:
            return 'medium'
        elif trust >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def get_reliability_score(self, seller_id: int) -> float:
        """
        Calculate reliability score based on interaction history.
        
        Returns:
            Reliability score between 0 and 1
        """
        if self.total_interactions[seller_id] == 0:
            return self.initial_trust  # No history, use initial trust
        
        success_rate = self.successful_interactions[seller_id] / self.total_interactions[seller_id]
        
        # Combine success rate with current trust
        reliability = 0.6 * success_rate + 0.4 * self.get_trust_score(seller_id)
        
        return round(reliability, 3)
    
    def rank_sellers_by_trust(self) -> List[int]:
        """
        Rank sellers by trust score.
        
        Returns:
            List of seller IDs sorted by trust (highest first)
        """
        seller_trust_pairs = [(i, self.trust_scores[i]) for i in range(self.num_sellers)]
        seller_trust_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [seller_id for seller_id, _ in seller_trust_pairs]
    
    def get_trust_statistics(self) -> Dict:
        """Get overall trust statistics"""
        return {
            'mean_trust': float(np.mean(self.trust_scores)),
            'std_trust': float(np.std(self.trust_scores)),
            'min_trust': float(np.min(self.trust_scores)),
            'max_trust': float(np.max(self.trust_scores)),
            'total_interactions': sum(self.total_interactions.values()),
            'total_successful': sum(self.successful_interactions.values()),
            'overall_success_rate': (
                sum(self.successful_interactions.values()) / sum(self.total_interactions.values())
                if sum(self.total_interactions.values()) > 0 else 0
            )
        }
    
    def get_seller_history(self, seller_id: int) -> Dict:
        """Get detailed history for a specific seller"""
        if seller_id < 0 or seller_id >= self.num_sellers:
            return {}
        
        return {
            'seller_id': seller_id,
            'current_trust': self.get_trust_score(seller_id),
            'trust_category': self.get_trust_category(seller_id),
            'reliability_score': self.get_reliability_score(seller_id),
            'total_interactions': self.total_interactions[seller_id],
            'successful_interactions': self.successful_interactions[seller_id],
            'success_rate': (
                self.successful_interactions[seller_id] / self.total_interactions[seller_id]
                if self.total_interactions[seller_id] > 0 else 0
            ),
            'interaction_history': self.interaction_history[seller_id][-10:]  # Last 10 interactions
        }
    
    def reset_trust(self, seller_id: Optional[int] = None):
        """
        Reset trust scores to initial values.
        
        Args:
            seller_id: Specific seller to reset, or None to reset all
        """
        if seller_id is None:
            # Reset all
            self.trust_scores = np.full(self.num_sellers, self.initial_trust, dtype=np.float32)
            self.interaction_history.clear()
            self.total_interactions.clear()
            self.successful_interactions.clear()
            self.time_step = 0
        else:
            # Reset specific seller
            if 0 <= seller_id < self.num_sellers:
                self.trust_scores[seller_id] = self.initial_trust
                self.interaction_history[seller_id] = []
                self.total_interactions[seller_id] = 0
                self.successful_interactions[seller_id] = 0
