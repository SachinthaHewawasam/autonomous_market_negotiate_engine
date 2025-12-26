import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NegotiationState:
    """Represents the current state of a negotiation"""
    requested_quantity: int
    current_offer_price: float
    current_offer_quantity: int
    seller_id: Optional[int]
    coalition_proposed: bool
    coalition_members: List[int]
    negotiation_round: int
    total_rounds: int


class MarketEnv(gym.Env):
    """
    Gym-compatible environment for bulk procurement market simulation.
    
    The environment enforces:
    - Coalition formation rules
    - Fairness constraints
    - Trust-based seller reliability
    
    The BuyerAgent learns to negotiate within these fixed rules.
    """
    
    def __init__(
        self,
        num_sellers: int = 5,
        max_quantity_per_seller: int = 50,
        max_negotiation_rounds: int = 10,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_sellers = num_sellers
        self.max_quantity_per_seller = max_quantity_per_seller
        self.max_negotiation_rounds = max_negotiation_rounds
        self.rng = np.random.default_rng(seed)
        
        # Action space: [action_type, seller_id, offer_price, quantity]
        # action_type: 0=offer, 1=counteroffer, 2=accept, 3=reject, 4=propose_coalition
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([4, num_sellers-1, 100, 200], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [requested_qty, best_price, best_qty, round, num_available_sellers, 
        #                     avg_trust, coalition_size, current_offer_price, current_offer_qty]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 100, 200, max_negotiation_rounds, num_sellers, 1, num_sellers, 100, 200], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize components (will be set by reset)
        self.sellers = []
        self.coalition_manager = None
        self.fairness_checker = None
        self.trust_manager = None
        self.current_request = None
        self.negotiation_state = None
        self.current_round = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment for a new negotiation episode
        
        Args:
            seed: Random seed for reproducibility
            options: Optional dict with 'quantity' and 'max_budget' to override defaults
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Import here to avoid circular imports
        from seller_agent import SellerAgent
        from coalition_manager import CoalitionManager
        from fairness_checker import FairnessChecker
        from trust_manager import TrustManager
        
        # Initialize sellers with random stock and base prices
        self.sellers = []
        for i in range(self.num_sellers):
            stock = self.rng.integers(20, self.max_quantity_per_seller + 1)
            base_price = self.rng.uniform(5.0, 15.0)
            self.sellers.append(SellerAgent(
                seller_id=i,
                stock=stock,
                base_price=base_price
            ))
        
        # Initialize market rule managers
        self.coalition_manager = CoalitionManager(self.sellers)
        self.fairness_checker = FairnessChecker()
        self.trust_manager = TrustManager(self.num_sellers)
        
        # Generate a new procurement request (use options if provided)
        if options and 'quantity' in options:
            quantity = options['quantity']
        else:
            quantity = self.rng.integers(60, 150)
        
        if options and 'max_budget' in options:
            max_budget = options['max_budget']
        else:
            max_budget = self.rng.uniform(800, 1500)
        
        self.current_request = {
            'quantity': quantity,
            'max_budget': max_budget
        }
        
        # Initialize negotiation state
        self.negotiation_state = NegotiationState(
            requested_quantity=self.current_request['quantity'],
            current_offer_price=0.0,
            current_offer_quantity=0,
            seller_id=None,
            coalition_proposed=False,
            coalition_members=[],
            negotiation_round=0,
            total_rounds=self.max_negotiation_rounds
        )
        
        self.current_round = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one negotiation step.
        
        Args:
            action: [action_type, seller_id, offer_price, quantity]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        action_type = int(action[0])
        seller_id = int(action[1]) % self.num_sellers
        offer_price = float(action[2])
        quantity = int(action[3])
        
        reward = 0.0
        terminated = False
        truncated = False
        
        self.current_round += 1
        self.negotiation_state.negotiation_round = self.current_round
        
        # Process action based on type
        if action_type == 0:  # Offer
            reward = self._process_offer(seller_id, offer_price, quantity)
        elif action_type == 1:  # Counteroffer
            reward = self._process_counteroffer(seller_id, offer_price, quantity)
        elif action_type == 2:  # Accept
            reward, terminated = self._process_accept(seller_id)
        elif action_type == 3:  # Reject
            reward = self._process_reject(seller_id)
        elif action_type == 4:  # Propose coalition
            reward, terminated = self._process_coalition_proposal(seller_id, quantity)
        
        # Check termination conditions
        if self.current_round >= self.max_negotiation_rounds:
            truncated = True
            reward -= 50  # Penalty for not completing negotiation
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_offer(self, seller_id: int, offer_price: float, quantity: int) -> float:
        """Process an initial offer to a seller"""
        seller = self.sellers[seller_id]
        
        # Allow partial quantities (agent can offer what seller has)
        actual_quantity = min(quantity, seller.stock)
        
        if actual_quantity == 0:
            return -5  # Seller has no stock
        
        # Check fairness
        if not self.fairness_checker.is_price_fair(offer_price, seller.base_price):
            return -3  # Penalty for unfair offer
        
        # Seller responds based on offer quality
        acceptance_prob = seller.evaluate_offer(offer_price, actual_quantity)
        
        if self.rng.random() < acceptance_prob:
            self.negotiation_state.current_offer_price = offer_price
            self.negotiation_state.current_offer_quantity = actual_quantity
            self.negotiation_state.seller_id = seller_id
            return 5  # Reward for successful offer
        else:
            # Seller counteroffers
            counter_price = seller.generate_counteroffer(offer_price)
            self.negotiation_state.current_offer_price = counter_price
            self.negotiation_state.current_offer_quantity = actual_quantity
            self.negotiation_state.seller_id = seller_id
            return 1  # Small reward for engagement
    
    def _process_counteroffer(self, seller_id: int, offer_price: float, quantity: int) -> float:
        """Process a counteroffer"""
        seller = self.sellers[seller_id]
        
        if not self.fairness_checker.is_price_fair(offer_price, seller.base_price):
            return -3
        
        acceptance_prob = seller.evaluate_offer(offer_price, quantity)
        
        if self.rng.random() < acceptance_prob:
            self.negotiation_state.current_offer_price = offer_price
            self.negotiation_state.current_offer_quantity = quantity
            self.negotiation_state.seller_id = seller_id
            return 3
        else:
            return -1  # Penalty for rejected counteroffer
    
    def _process_accept(self, seller_id: int) -> Tuple[float, bool]:
        """Process acceptance of current offer"""
        if self.negotiation_state.seller_id is None:
            return -10, False  # No active offer to accept
        
        seller = self.sellers[seller_id]
        price = self.negotiation_state.current_offer_price
        quantity = self.negotiation_state.current_offer_quantity
        
        # Check if this satisfies the request
        if quantity >= self.current_request['quantity']:
            total_cost = price * quantity
            
            # Check budget constraint
            if total_cost <= self.current_request['max_budget']:
                # Successful negotiation!
                savings = self.current_request['max_budget'] - total_cost
                reward = 100 + (savings / 10)  # Base reward + savings bonus
                
                # Update trust based on delivery (simulate)
                delivery_success = self.rng.random() < self.trust_manager.get_trust_score(seller_id)
                self.trust_manager.update_trust(seller_id, delivery_success)
                
                return reward, True
            else:
                return -20, False  # Over budget
        else:
            # Partial fulfillment - need coalition
            return -5, False
    
    def _process_reject(self, seller_id: int) -> float:
        """Process rejection of current offer"""
        self.negotiation_state.seller_id = None
        self.negotiation_state.current_offer_price = 0.0
        self.negotiation_state.current_offer_quantity = 0
        return -2  # Small penalty for rejection
    
    def _process_coalition_proposal(self, primary_seller_id: int, quantity: int) -> Tuple[float, bool]:
        """Process proposal to form a coalition of sellers"""
        # Get trust scores
        trust_scores = [self.trust_manager.get_trust_score(i) for i in range(self.num_sellers)]
        
        # Find coalition that can fulfill the request
        coalition = self.coalition_manager.form_coalition(
            requested_quantity=self.current_request['quantity'],
            primary_seller_id=primary_seller_id,
            trust_scores=trust_scores
        )
        
        if coalition is None:
            return -10, False  # No valid coalition found
        
        # Calculate coalition pricing
        total_price = self.coalition_manager.calculate_coalition_price(coalition)
        
        # Check fairness of coalition pricing
        if not self.fairness_checker.is_coalition_fair(coalition, total_price):
            return -8, False  # Unfair coalition pricing
        
        # Check budget
        if total_price <= self.current_request['max_budget']:
            self.negotiation_state.coalition_proposed = True
            self.negotiation_state.coalition_members = [s['seller_id'] for s in coalition]
            self.negotiation_state.current_offer_price = total_price / self.current_request['quantity']
            self.negotiation_state.current_offer_quantity = sum(s['quantity'] for s in coalition)
            
            # Store coalition details for deal creation
            self._coalition_details = coalition
            self._total_cost = total_price
            
            # Coalition success reward
            savings = self.current_request['max_budget'] - total_price
            reward = 100 + (savings / 10)
            
            # Update trust for all coalition members
            for seller_info in coalition:
                self.trust_manager.update_trust(seller_info['seller_id'], 'success')
            
            return reward, True  # SUCCESS!
        else:
            return -15, False  # Coalition over budget
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        # Calculate best available single-seller option
        best_price = float('inf')
        best_qty = 0
        num_available = 0
        
        for seller in self.sellers:
            if seller.stock > 0:
                num_available += 1
                if seller.base_price < best_price:
                    best_price = seller.base_price
                    best_qty = seller.stock
        
        if best_price == float('inf'):
            best_price = 0
        
        # Calculate average trust
        avg_trust = np.mean([self.trust_manager.get_trust_score(i) for i in range(self.num_sellers)])
        
        observation = np.array([
            self.current_request['quantity'],
            best_price,
            best_qty,
            self.current_round,
            num_available,
            avg_trust,
            len(self.negotiation_state.coalition_members),
            self.negotiation_state.current_offer_price,
            self.negotiation_state.current_offer_quantity
        ], dtype=np.float32)
        
        return observation
    
    def _get_info(self) -> dict:
        """Get additional information about current state"""
        info = {
            'requested_quantity': self.current_request['quantity'],
            'max_budget': self.current_request['max_budget'],
            'current_round': self.current_round,
            'coalition_proposed': self.negotiation_state.coalition_proposed,
            'num_sellers': self.num_sellers,
            'seller_stocks': [s.stock for s in self.sellers],
            'seller_prices': [s.base_price for s in self.sellers],
            'trust_scores': [self.trust_manager.get_trust_score(i) for i in range(self.num_sellers)]
        }
        
        # Add coalition details if available
        if hasattr(self, '_coalition_details'):
            info['coalition_details'] = self._coalition_details
            info['total_cost'] = self._total_cost
            info['coalition_formed'] = True
        
        return info
