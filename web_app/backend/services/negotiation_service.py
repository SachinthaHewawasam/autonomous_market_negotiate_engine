"""
Negotiation service for running RL agent inference and online training
"""
import sys
import os
import time
import numpy as np
from datetime import datetime
from .training_utils import train_from_experiences, save_model_checkpoint

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from market_env import MarketEnv
from buyer_agent import BuyerAgent
from web_app.backend.models import db, ProcurementRequest, Negotiation, NegotiationStep, Deal, DealSeller, Product
import torch


class NegotiationService:
    """Service for running negotiations with RL agent"""
    
    def __init__(self, model_path, enable_training=False):
        self.model_path = model_path
        self.env = None
        self.enable_training = enable_training
        self.training_counter = 0
        self.save_interval = 10  # Save model every 10 negotiations
        
        # Load trained agent
        try:
            self.agent = BuyerAgent(state_dim=9, action_dim=4)
            self.agent.load_model(model_path)
            print(f"✓ Loaded trained agent from {model_path}")
            if enable_training:
                print(f"✓ Online training ENABLED - Agent will learn from negotiations")
            else:
                print(f"ℹ Online training DISABLED - Agent in inference-only mode")
        except Exception as e:
            print(f"⚠ Could not load agent: {e}")
            self.agent = None
    
    def predict_outcome(self, products, quantity, max_budget):
        """Predict negotiation outcome without actually executing"""
        from market_env import MarketEnv
        from models import db
        
        # Create temporary environment
        temp_env = MarketEnv(
            num_sellers=len(products),
            max_quantity_per_seller=max(p.quantity for p in products),
            max_negotiation_rounds=10
        )
        
        # Reset with custom request
        state, info = temp_env.reset(options={
            'quantity': quantity,
            'max_budget': max_budget
        })
        
        # Override seller data
        for i, product in enumerate(products):
            temp_env.sellers[i].stock = product.quantity
            temp_env.sellers[i].base_price = product.base_price
            temp_env.trust_manager.trust_scores[i] = product.trust_score
        
        # Run RL agent inference to predict best strategy
        action = self.agent.select_action(state, training=False)
        
        # Execute predicted action in environment to get reward
        next_state, reward, done, truncated, info = temp_env.step(action)
        
        # Analyze RL agent's decision
        action_type = int(action[0])
        action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
        predicted_action = action_names[action_type]
        
        # Calculate success probability based on predicted reward
        # The RL agent's reward indicates how good this scenario is
        if reward > 50:  # High reward predicted
            success_prob = 0.90  # 90% success
        elif reward > 0:  # Positive reward
            success_prob = 0.70  # 70% success
        elif reward > -20:  # Small penalty
            success_prob = 0.40  # 40% success
        else:  # Large penalty
            success_prob = 0.15  # 15% success
        
        # Check if coalition is needed (from environment logic)
        coalition_needed = quantity > max(p.quantity for p in products)
        
        if coalition_needed:
            # Predict coalition
            from coalition_manager import CoalitionManager
            coalition_mgr = CoalitionManager(temp_env.sellers)
            trust_scores = [temp_env.trust_manager.get_trust_score(i) for i in range(len(products))]
            
            coalition = coalition_mgr.form_coalition(
                requested_quantity=quantity,
                primary_seller_id=None,
                trust_scores=trust_scores
            )
            
            if coalition:
                total_cost = coalition_mgr.calculate_coalition_price(coalition)
                # Use RL-predicted success probability, adjusted by budget feasibility
                if total_cost <= max_budget:
                    success_prob = min(0.98, success_prob + 0.1)  # Boost if within budget
                else:
                    success_prob = max(0.05, success_prob * 0.3)  # Reduce if over budget
                
                # Build detailed prediction using RL agent's decision
                return {
                    'success_probability': success_prob,
                    'predicted_action': predicted_action,  # Use RL agent's predicted action
                    'predicted_reward': float(reward),  # Show the RL agent's predicted reward
                    'estimated_cost': {
                        'min': total_cost * 0.95,
                        'max': total_cost * 1.05,
                        'most_likely': total_cost
                    },
                    'predicted_savings': max(0, max_budget - total_cost),
                    'recommended_sellers': [
                        {
                            'name': products[m['seller_id']].seller.name,
                            'quantity': m['quantity'],
                            'price': m['price'],
                            'trust': trust_scores[m['seller_id']]
                        } for m in coalition
                    ],
                    'risk_level': 'Low' if success_prob > 0.8 else 'Medium' if success_prob > 0.5 else 'High',
                    'coalition_needed': True,
                    'num_sellers_needed': len(coalition),
                    'delivery_estimate': f"{len(coalition) + 2} days"
                }
            else:
                return {
                    'success_probability': 0.1,
                    'predicted_action': 'Coalition',
                    'error': 'No valid coalition found',
                    'risk_level': 'Very High',
                    'coalition_needed': True
                }
        else:
            # Single seller possible
            best_seller = min(products, key=lambda p: p.base_price if p.quantity >= quantity else float('inf'))
            
            if best_seller.quantity >= quantity:
                total_cost = best_seller.base_price * quantity
                # Use RL-predicted success probability
                if total_cost <= max_budget:
                    success_prob = min(0.95, success_prob + 0.05)
                else:
                    success_prob = max(0.1, success_prob * 0.4)
                
                return {
                    'success_probability': success_prob,
                    'predicted_action': predicted_action,  # Use RL agent's prediction
                    'predicted_reward': float(reward),
                    'estimated_cost': {
                        'min': total_cost * 0.98,
                        'max': total_cost * 1.02,
                        'most_likely': total_cost
                    },
                    'predicted_savings': max(0, max_budget - total_cost),
                    'recommended_sellers': [{
                        'name': best_seller.seller.name,
                        'quantity': quantity,
                        'price': best_seller.base_price,
                        'trust': best_seller.trust_score
                    }],
                    'risk_level': 'Very Low',
                    'coalition_needed': False,
                    'num_sellers_needed': 1,
                    'delivery_estimate': '2 days'
                }
            else:
                return {
                    'success_probability': 0.0,
                    'predicted_action': 'None',
                    'error': 'Insufficient stock available',
                    'risk_level': 'Impossible'
                }
    
    def run_negotiation(self, request_id, socketio):
        """
        Run negotiation for a procurement request
        
        Args:
            request_id: ID of the procurement request
            socketio: SocketIO instance for real-time updates
        """
        # Note: app context is already provided by the caller
        print(f"[NegotiationService] Starting negotiation for request {request_id}")
        procurement_request = None
        try:
            # Get request
            print(f"[NegotiationService] Querying database for request {request_id}")
            procurement_request = ProcurementRequest.query.get(request_id)
            if not procurement_request:
                print(f"[NegotiationService] Request {request_id} not found in database")
                socketio.emit('negotiation_failed', {
                    'request_id': request_id,
                    'reason': 'Request not found'
                }, room=f'request_{request_id}')
                return
            
            print(f"[NegotiationService] Found request: {procurement_request.product_name} - {procurement_request.brand}")
            
            # Get matching products
            products = Product.query.filter_by(
                name=procurement_request.product_name,
                brand=procurement_request.brand
            ).all()
            
            print(f"[NegotiationService] Found {len(products)} matching products")
            
            if not products:
                print(f"[NegotiationService] No matching products found")
                self._fail_negotiation(procurement_request, "No matching products found", socketio)
                return
            
            print(f"[NegotiationService] Creating negotiation record")
            # Create negotiation record
            negotiation = Negotiation(
                request_id=request_id,
                final_status='in_progress'
            )
            db.session.add(negotiation)
            db.session.commit()
            
            # Emit negotiation started
            socketio.emit('negotiation_started', {
                'request_id': request_id,
                'negotiation_id': negotiation.id
            }, room=f'request_{request_id}')
            
            # Initialize environment with custom sellers based on products
            self.env = MarketEnv(
                num_sellers=len(products),
                max_quantity_per_seller=max(p.quantity for p in products),
                max_negotiation_rounds=10
            )
            
            # Reset environment with custom request
            state, info = self.env.reset(options={
                'quantity': procurement_request.quantity,
                'max_budget': procurement_request.max_budget
            })
            
            # Override seller data with actual products
            for i, product in enumerate(products):
                self.env.sellers[i].stock = product.quantity
                self.env.sellers[i].base_price = product.base_price
                self.env.trust_manager.trust_scores[i] = product.trust_score
            
            # Emit initial state
            socketio.emit('market_state', {
                'request_id': request_id,
                'sellers': [{
                    'id': i,
                    'product_id': products[i].id,
                    'seller_name': products[i].seller.name,
                    'stock': products[i].quantity,
                    'price': products[i].base_price,
                    'trust': products[i].trust_score
                } for i in range(len(products))],
                'requested_quantity': procurement_request.quantity,
                'max_budget': procurement_request.max_budget
            }, room=f'request_{request_id}')
            
            time.sleep(1)
            
            # Run negotiation
            done = False
            truncated = False
            round_num = 0
            total_reward = 0
            
            # Store experiences for training
            experiences = []
            
            while not (done or truncated):
                round_num += 1
                
                # Agent selects action (enable exploration if training)
                action = self.agent.select_action(state, training=self.enable_training)
                
                # Decode action
                action_type = int(action[0])
                seller_id = int(action[1]) % len(products)
                price = float(action[2])
                qty = int(action[3])
                
                action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
                
                # Generate explanation
                explanation = self._generate_explanation(
                    action_type, seller_id, price, qty, info, procurement_request.quantity
                )
                
                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                
                # Store experience for training
                if self.enable_training:
                    experiences.append({
                        'state': state.copy(),
                        'action': action.copy(),
                        'reward': reward,
                        'next_state': next_state.copy(),
                        'done': done
                    })
                
                # Save step to database
                step = NegotiationStep(
                    negotiation_id=negotiation.id,
                    round_number=round_num,
                    action_type=action_names[action_type],
                    seller_id=seller_id,
                    price=price,
                    quantity=qty,
                    reward=reward,
                    explanation=explanation
                )
                db.session.add(step)
                db.session.commit()
                
                # Emit step to client
                socketio.emit('negotiation_step', {
                    'request_id': request_id,
                    'step': step.to_dict()
                }, room=f'request_{request_id}')
                
                state = next_state
                time.sleep(1.5)  # Delay for visualization
            
            # Update negotiation
            negotiation.total_reward = total_reward
            negotiation.num_rounds = round_num
            
            if done:
                # Success - create deal
                self._create_deal(negotiation, procurement_request, products, info, socketio)
            else:
                # Failed
                self._fail_negotiation(procurement_request, "Maximum rounds exceeded", socketio)
            
            # Train agent from this negotiation if enabled
            if self.enable_training and len(experiences) > 0:
                self._train_from_experiences(experiences, negotiation.id)
            
            db.session.commit()
            
        except Exception as e:
            print(f"Error in negotiation: {e}")
            import traceback
            traceback.print_exc()
            if procurement_request:
                self._fail_negotiation(procurement_request, str(e), socketio)
            else:
                socketio.emit('negotiation_failed', {
                    'request_id': request_id,
                    'reason': f'Error: {str(e)}'
                }, room=f'request_{request_id}')
    
    def _train_from_experiences(self, experiences, negotiation_id):
        """Train agent from negotiation experiences"""
        try:
            # Train the agent
            avg_loss = train_from_experiences(self.agent, experiences, negotiation_id)
            
            if avg_loss is not None:
                # Increment counter and save periodically
                self.training_counter += 1
                
                if self.training_counter % self.save_interval == 0:
                    save_model_checkpoint(
                        self.agent, 
                        self.model_path, 
                        self.training_counter
                    )
                    print(f"[Training] Model saved after {self.training_counter} negotiations")
        
        except Exception as e:
            print(f"[Training] Error during training: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_explanation(self, action_type, seller_id, price, qty, info, requested_qty):
        """Generate human-readable explanation for action"""
        action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
        
        explanation = f"Agent chose: {action_names[action_type]}\n"
        
        if action_type == 0:  # Offer
            explanation += f"• Targeting seller with competitive price\n"
            explanation += f"• Offering ${price:.2f}/unit for {qty} units"
        
        elif action_type == 4:  # Coalition
            total_stock = sum(info.get('seller_stocks', []))
            max_single = max(info.get('seller_stocks', [0]))
            explanation += f"• No single seller can fulfill {requested_qty} units\n"
            explanation += f"• Maximum single seller stock: {max_single} units\n"
            explanation += f"• Total available: {total_stock} units\n"
            explanation += f"• Forming coalition for combined supply"
        
        elif action_type == 2:  # Accept
            explanation += f"• Current offer meets all requirements\n"
            explanation += f"• Price within budget\n"
            explanation += f"• Fairness constraints satisfied"
        
        elif action_type == 3:  # Reject
            explanation += f"• Current offer doesn't meet criteria\n"
            explanation += f"• May violate budget or fairness rules"
        
        return explanation
    
    def _create_deal(self, negotiation, procurement_request, products, info, socketio):
        """Create a deal from successful negotiation"""
        coalition_formed = info.get('coalition_formed', False)
        negotiation.coalition_formed = coalition_formed
        
        if coalition_formed:
            coalition_members = info.get('coalition_members', [])
            total_cost = info.get('total_cost', 0)
        else:
            # Single seller deal
            selected_seller = info.get('selected_seller', 0)
            coalition_members = [selected_seller]
            final_price = info.get('final_price', products[selected_seller].base_price)
            total_cost = final_price * procurement_request.quantity
        
        negotiation.total_cost = total_cost
        savings = procurement_request.max_budget - total_cost
        
        # Create deal
        deal = Deal(
            negotiation_id=negotiation.id,
            total_cost=total_cost,
            total_quantity=procurement_request.quantity,
            savings=savings,
            human_approved=False
        )
        db.session.add(deal)
        db.session.flush()
        
        # Add sellers to deal
        if coalition_formed:
            # Get the actual coalition details from the environment
            coalition_info = info.get('coalition_details', [])
            
            if coalition_info:
                # Use actual coalition distribution
                for member in coalition_info:
                    seller_id = member['seller_id']
                    product = products[seller_id]
                    quantity = member['quantity']
                    price_per_unit = member['price']
                    
                    deal_seller = DealSeller(
                        deal_id=deal.id,
                        seller_id=product.seller_id,
                        product_id=product.id,
                        quantity=quantity,
                        price_per_unit=price_per_unit,
                        subtotal=quantity * price_per_unit
                    )
                    db.session.add(deal_seller)
            else:
                # Fallback: distribute evenly among coalition members
                remaining = procurement_request.quantity
                for i, member_id in enumerate(coalition_members):
                    product = products[member_id]
                    # Last seller gets remaining quantity
                    if i == len(coalition_members) - 1:
                        quantity = remaining
                    else:
                        quantity = min(product.quantity, remaining // (len(coalition_members) - i))
                    
                    remaining -= quantity
                    price_per_unit = product.base_price
                    
                    deal_seller = DealSeller(
                        deal_id=deal.id,
                        seller_id=product.seller_id,
                        product_id=product.id,
                        quantity=quantity,
                        price_per_unit=price_per_unit,
                        subtotal=quantity * price_per_unit
                    )
                    db.session.add(deal_seller)
        else:
            # Single seller
            product = products[coalition_members[0]]
            deal_seller = DealSeller(
                deal_id=deal.id,
                seller_id=product.seller_id,
                product_id=product.id,
                quantity=procurement_request.quantity,
                price_per_unit=total_cost / procurement_request.quantity,
                subtotal=total_cost
            )
            db.session.add(deal_seller)
        
        negotiation.final_status = 'pending_approval'
        procurement_request.status = 'pending_approval'
        
        # Emit deal for approval
        socketio.emit('deal_ready', {
            'request_id': procurement_request.id,
            'deal': deal.to_dict()
        }, room=f'request_{procurement_request.id}')
    
    def _fail_negotiation(self, procurement_request, reason, socketio):
        """Mark negotiation as failed"""
        procurement_request.status = 'failed'
        procurement_request.completed_at = datetime.utcnow()
        db.session.commit()
        
        socketio.emit('negotiation_failed', {
            'request_id': procurement_request.id,
            'reason': reason
        }, room=f'request_{procurement_request.id}')
