"""
Enhanced Interactive Demo UI for Autonomous Market Simulation
with Human-in-the-Loop Controls and Decision Transparency

Features:
- Step-by-step negotiation visualization
- Decision explanation for each action
- Human review and approval of final deals
- Preference adjustment controls
- Detailed seller comparison view

Usage:
    python demo_ui_enhanced.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import numpy as np
from market_env import MarketEnv
from buyer_agent import BuyerAgent
import torch
import os


class EnhancedMarketDemo:
    """Enhanced demo with human-in-the-loop controls"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Market Simulation - Enhanced Demo")
        self.root.geometry("1600x950")
        self.root.configure(bg='#ecf0f1')
        
        # Load trained agent
        self.env = None
        self.agent = None
        self.load_trained_agent()
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.current_deal = None
        self.negotiation_history = []
        self.seller_info = None
        
        # Create UI
        self.create_ui()
        
    def load_trained_agent(self):
        """Load the trained RL buyer agent"""
        model_path = 'models/buyer_agent.pth'
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Trained model not found at {model_path}\nPlease run train.py first.")
            return
        
        self.env = MarketEnv(
            num_sellers=5,
            max_quantity_per_seller=50,
            max_negotiation_rounds=10,
            seed=42
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = BuyerAgent(state_dim=9, action_dim=4, device=device)
        
        try:
            self.agent.load_model(model_path)
            print(f"✓ Loaded trained agent from {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def create_ui(self):
        """Create the enhanced UI layout"""
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="🤖 Autonomous Market Simulation - Interactive Demo",
            font=('Arial', 22, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=10)
        
        tk.Label(
            header_frame,
            text="Human-in-the-Loop • Decision Transparency • Step-by-Step Visualization",
            font=('Arial', 10),
            bg='#34495e',
            fg='#bdc3c7'
        ).pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#ecf0f1')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Top row - Input and Sellers
        top_row = tk.Frame(main_container, bg='#ecf0f1')
        top_row.pack(fill=tk.X, pady=(0, 10))
        
        # Left: Input panel
        input_frame = tk.LabelFrame(
            top_row,
            text="📋 Procurement Request",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.create_input_section(input_frame)
        
        # Right: Seller details
        seller_frame = tk.LabelFrame(
            top_row,
            text="🏪 Available Sellers",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        seller_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.create_seller_section(seller_frame)
        
        # Middle row - Negotiation process
        negotiation_frame = tk.LabelFrame(
            main_container,
            text="📊 Negotiation Process & Decision Explanations",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        negotiation_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.create_negotiation_section(negotiation_frame)
        
        # Bottom row - Deal review
        review_frame = tk.LabelFrame(
            main_container,
            text="✅ Final Deal Review (Human Approval Required)",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        review_frame.pack(fill=tk.BOTH, pady=(0, 0))
        self.create_review_section(review_frame)
    
    def create_input_section(self, parent):
        """Create input form section"""
        form = tk.Frame(parent, bg='white')
        form.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Product
        tk.Label(form, text="Product:", font=('Arial', 10, 'bold'), bg='white').grid(row=0, column=0, sticky='w', pady=5)
        self.product_entry = tk.Entry(form, font=('Arial', 10), width=30)
        self.product_entry.insert(0, "Biscuits (Brand X)")
        self.product_entry.grid(row=0, column=1, pady=5, padx=5, sticky='ew')
        
        # Quantity
        tk.Label(form, text="Quantity:", font=('Arial', 10, 'bold'), bg='white').grid(row=1, column=0, sticky='w', pady=5)
        self.quantity_entry = tk.Entry(form, font=('Arial', 10), width=30)
        self.quantity_entry.insert(0, "120")
        self.quantity_entry.grid(row=1, column=1, pady=5, padx=5, sticky='ew')
        
        # Budget
        tk.Label(form, text="Max Budget ($):", font=('Arial', 10, 'bold'), bg='white').grid(row=2, column=0, sticky='w', pady=5)
        self.budget_entry = tk.Entry(form, font=('Arial', 10), width=30)
        self.budget_entry.insert(0, "1200")
        self.budget_entry.grid(row=2, column=1, pady=5, padx=5, sticky='ew')
        
        # Preferences
        tk.Label(form, text="Preferences:", font=('Arial', 10, 'bold'), bg='white').grid(row=3, column=0, sticky='nw', pady=5)
        pref_frame = tk.Frame(form, bg='white')
        pref_frame.grid(row=3, column=1, pady=5, padx=5, sticky='ew')
        
        self.prefer_trust = tk.BooleanVar(value=True)
        self.prefer_price = tk.BooleanVar(value=True)
        
        tk.Checkbutton(pref_frame, text="Prefer high-trust sellers", variable=self.prefer_trust, bg='white', font=('Arial', 9)).pack(anchor='w')
        tk.Checkbutton(pref_frame, text="Prefer low prices", variable=self.prefer_price, bg='white', font=('Arial', 9)).pack(anchor='w')
        
        form.columnconfigure(1, weight=1)
        
        # Buttons
        btn_frame = tk.Frame(parent, bg='white')
        btn_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ Start Negotiation",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.start_simulation,
            cursor='hand2',
            padx=15,
            pady=8
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(
            btn_frame,
            text="🔄 Reset",
            font=('Arial', 10),
            bg='#e74c3c',
            fg='white',
            command=self.reset_simulation,
            cursor='hand2',
            padx=15,
            pady=8
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
    
    def create_seller_section(self, parent):
        """Create seller information table"""
        # Table
        table_frame = tk.Frame(parent, bg='white')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Seller', 'Stock', 'Price', 'Trust', 'Status')
        self.seller_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        self.seller_tree.heading('Seller', text='Seller')
        self.seller_tree.heading('Stock', text='Stock')
        self.seller_tree.heading('Price', text='Price/unit')
        self.seller_tree.heading('Trust', text='Trust')
        self.seller_tree.heading('Status', text='Status')
        
        self.seller_tree.column('Seller', width=80, anchor='center')
        self.seller_tree.column('Stock', width=80, anchor='center')
        self.seller_tree.column('Price', width=90, anchor='center')
        self.seller_tree.column('Trust', width=80, anchor='center')
        self.seller_tree.column('Status', width=120, anchor='center')
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.seller_tree.yview)
        self.seller_tree.configure(yscroll=scrollbar.set)
        
        self.seller_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Style
        style = ttk.Style()
        style.configure("Treeview", font=('Arial', 9), rowheight=25)
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
    
    def create_negotiation_section(self, parent):
        """Create negotiation log with explanations"""
        log_frame = tk.Frame(parent, bg='white')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD,
            height=15
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Tags for colored output
        self.log_text.tag_config('header', font=('Consolas', 10, 'bold'), foreground='#2c3e50')
        self.log_text.tag_config('success', foreground='#27ae60', font=('Consolas', 9, 'bold'))
        self.log_text.tag_config('warning', foreground='#f39c12', font=('Consolas', 9, 'bold'))
        self.log_text.tag_config('error', foreground='#e74c3c', font=('Consolas', 9, 'bold'))
        self.log_text.tag_config('info', foreground='#3498db')
        self.log_text.tag_config('agent', foreground='#9b59b6', font=('Consolas', 9, 'bold'))
        self.log_text.tag_config('explanation', foreground='#16a085', font=('Consolas', 9, 'italic'))
    
    def create_review_section(self, parent):
        """Create deal review and approval section"""
        content = tk.Frame(parent, bg='white')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Deal summary (left)
        summary_frame = tk.Frame(content, bg='#ecf0f1', relief=tk.SUNKEN, borderwidth=2)
        summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(summary_frame, text="Deal Summary", font=('Arial', 10, 'bold'), bg='#ecf0f1').pack(pady=5)
        
        self.deal_text = tk.Text(
            summary_frame,
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#2c3e50',
            height=6,
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.deal_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.deal_text.config(state=tk.DISABLED)
        
        # Human controls (right)
        control_frame = tk.Frame(content, bg='white')
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        tk.Label(control_frame, text="Human Decision", font=('Arial', 10, 'bold'), bg='white').pack(pady=5)
        
        self.approve_btn = tk.Button(
            control_frame,
            text="✓ Approve Deal",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.approve_deal,
            cursor='hand2',
            state=tk.DISABLED,
            width=15,
            pady=10
        )
        self.approve_btn.pack(pady=5, padx=10)
        
        self.reject_btn = tk.Button(
            control_frame,
            text="✗ Reject Deal",
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            command=self.reject_deal,
            cursor='hand2',
            state=tk.DISABLED,
            width=15,
            pady=10
        )
        self.reject_btn.pack(pady=5, padx=10)
        
        self.modify_btn = tk.Button(
            control_frame,
            text="⚙ Adjust Preferences",
            font=('Arial', 10),
            bg='#3498db',
            fg='white',
            command=self.adjust_preferences,
            cursor='hand2',
            state=tk.DISABLED,
            width=15,
            pady=8
        )
        self.modify_btn.pack(pady=5, padx=10)
    
    def log_message(self, message, tag='info'):
        """Add message to log"""
        self.log_text.insert(tk.END, message + '\n', tag)
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_seller_table(self, info, selected_sellers=None):
        """Update seller table with status"""
        for item in self.seller_tree.get_children():
            self.seller_tree.delete(item)
        
        for i in range(len(info['seller_stocks'])):
            status = "Available"
            if selected_sellers and i in selected_sellers:
                status = "✓ Selected"
            
            self.seller_tree.insert('', tk.END, values=(
                f'Seller {i}',
                f"{info['seller_stocks'][i]} units",
                f"${info['seller_prices'][i]:.2f}",
                f"{info['trust_scores'][i]:.2f}",
                status
            ))
    
    def explain_action(self, action_type, seller_id, price, qty, info):
        """Provide explanation for agent's action"""
        action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
        
        explanation = f"\n💡 Why {action_names[action_type]}?\n"
        
        if action_type == 0:  # Offer
            seller_price = info['seller_prices'][seller_id]
            seller_trust = info['trust_scores'][seller_id]
            explanation += f"   • Seller {seller_id} has competitive price (${seller_price:.2f})\n"
            explanation += f"   • Trust score is {seller_trust:.2f} (reliable)\n"
            explanation += f"   • Offering ${price:.2f}/unit for {qty} units\n"
        
        elif action_type == 4:  # Coalition
            total_stock = sum(info['seller_stocks'])
            # Get the actual requested quantity from the environment's current request
            requested = self.env.current_request['quantity']
            max_single = max(info['seller_stocks'])
            explanation += f"   • No single seller can fulfill {requested} units\n"
            explanation += f"   • Maximum single seller stock: {max_single} units\n"
            explanation += f"   • Total available stock: {total_stock} units\n"
            explanation += f"   • Forming coalition to meet demand\n"
            explanation += f"   • Selecting sellers by price-trust optimization\n"
        
        elif action_type == 2:  # Accept
            explanation += f"   • Current offer meets requirements\n"
            explanation += f"   • Price is within budget\n"
            explanation += f"   • Fairness constraints satisfied\n"
        
        elif action_type == 3:  # Reject
            explanation += f"   • Current offer doesn't meet criteria\n"
            explanation += f"   • May violate fairness or budget constraints\n"
        
        self.log_message(explanation, 'explanation')
    
    def start_simulation(self):
        """Start negotiation simulation"""
        if self.is_running:
            messagebox.showwarning("Warning", "Simulation already running!")
            return
        
        if not self.agent or not self.env:
            messagebox.showerror("Error", "Agent not loaded!")
            return
        
        try:
            product = self.product_entry.get()
            quantity = int(self.quantity_entry.get())
            budget = float(self.budget_entry.get())
            
            if quantity <= 0 or budget <= 0:
                raise ValueError("Quantity and budget must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return
        
        self.start_btn.config(state=tk.DISABLED)
        self.is_running = True
        
        self.log_text.delete(1.0, tk.END)
        self.deal_text.config(state=tk.NORMAL)
        self.deal_text.delete(1.0, tk.END)
        self.deal_text.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_negotiation, args=(product, quantity, budget))
        thread.daemon = True
        thread.start()
    
    def run_negotiation(self, product, quantity, budget):
        """Run negotiation with explanations"""
        try:
            # Reset with custom quantity and budget
            state, info = self.env.reset(options={'quantity': quantity, 'max_budget': budget})
            
            self.seller_info = info
            self.root.after(0, self.update_seller_table, info)
            
            self.log_message("=" * 90, 'header')
            self.log_message(f"PROCUREMENT REQUEST: {product}", 'header')
            self.log_message("=" * 90, 'header')
            self.log_message(f"\n📦 Requested: {quantity} units", 'info')
            self.log_message(f"💰 Budget: ${budget:.2f}", 'info')
            self.log_message(f"🏪 Sellers: {len(info['seller_stocks'])}\n", 'info')
            
            # Analyze feasibility
            can_fulfill = any(stock >= quantity for stock in info['seller_stocks'])
            if can_fulfill:
                self.log_message("✓ At least one seller can fulfill alone\n", 'success')
            else:
                self.log_message("⚠ Coalition required - no single seller sufficient\n", 'warning')
            
            self.log_message("-" * 90, 'header')
            self.log_message("🤖 RL AGENT NEGOTIATION", 'agent')
            self.log_message("-" * 90 + "\n", 'header')
            
            time.sleep(1)
            
            done = False
            truncated = False
            step = 0
            total_reward = 0
            selected_sellers = []
            
            while not (done or truncated):
                step += 1
                
                action = self.agent.select_action(state, training=False)
                
                action_type = int(action[0])
                seller_id = int(action[1]) % self.env.num_sellers
                price = float(action[2])
                qty = int(action[3])
                
                action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
                
                self.log_message(f"Round {step}: {action_names[action_type]}", 'agent')
                self.log_message(f"  → Seller {seller_id} | Price: ${price:.2f} | Qty: {qty}", 'info')
                
                # Explain decision
                self.explain_action(action_type, seller_id, price, qty, info)
                
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                
                if reward > 50:
                    self.log_message(f"  ✓ Reward: +{reward:.2f} (Great outcome!)\n", 'success')
                elif reward > 0:
                    self.log_message(f"  ✓ Reward: +{reward:.2f}\n", 'success')
                else:
                    self.log_message(f"  ✗ Reward: {reward:.2f}\n", 'error')
                
                if action_type == 4 and reward > 50:
                    selected_sellers = info.get('coalition_members', [])
                
                state = next_state
                time.sleep(1)
            
            if done:
                self.log_message("\n🎉 NEGOTIATION SUCCESSFUL!\n", 'success')
                self.current_deal = {
                    'success': True,
                    'product': product,
                    'quantity': quantity,
                    'budget': budget,
                    'reward': total_reward,
                    'sellers': selected_sellers,
                    'info': info
                }
                self.root.after(0, self.display_deal_for_review)
            else:
                self.log_message("\n❌ Negotiation failed\n", 'error')
                self.current_deal = None
            
        except Exception as e:
            self.log_message(f"\n❌ Error: {str(e)}", 'error')
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
    
    def display_deal_for_review(self):
        """Display deal for human review"""
        if not self.current_deal:
            return
        
        deal = self.current_deal
        
        self.deal_text.config(state=tk.NORMAL)
        self.deal_text.delete(1.0, tk.END)
        
        summary = f"Product: {deal['product']}\n"
        summary += f"Quantity: {deal['quantity']} units\n"
        summary += f"Budget: ${deal['budget']:.2f}\n"
        summary += f"Total Reward: {deal['reward']:.2f}\n\n"
        
        if deal['sellers']:
            summary += f"Coalition Formed: {len(deal['sellers'])} sellers\n"
            for sid in deal['sellers']:
                summary += f"  • Seller {sid}\n"
        
        summary += f"\nAwaiting human approval..."
        
        self.deal_text.insert(1.0, summary)
        self.deal_text.config(state=tk.DISABLED)
        
        # Enable approval buttons
        self.approve_btn.config(state=tk.NORMAL)
        self.reject_btn.config(state=tk.NORMAL)
        self.modify_btn.config(state=tk.NORMAL)
        
        # Update seller table
        self.update_seller_table(self.seller_info, deal['sellers'])
    
    def approve_deal(self):
        """Human approves the deal"""
        if not self.current_deal:
            return
        
        self.log_message("\n" + "=" * 90, 'header')
        self.log_message("✅ DEAL APPROVED BY HUMAN", 'success')
        self.log_message("=" * 90, 'header')
        
        messagebox.showinfo("Deal Approved", "The procurement deal has been approved and will be executed.")
        
        self.approve_btn.config(state=tk.DISABLED)
        self.reject_btn.config(state=tk.DISABLED)
        self.modify_btn.config(state=tk.DISABLED)
    
    def reject_deal(self):
        """Human rejects the deal"""
        self.log_message("\n" + "=" * 90, 'header')
        self.log_message("❌ DEAL REJECTED BY HUMAN", 'error')
        self.log_message("=" * 90, 'header')
        
        messagebox.showinfo("Deal Rejected", "You can adjust preferences and retry negotiation.")
        
        self.approve_btn.config(state=tk.DISABLED)
        self.reject_btn.config(state=tk.DISABLED)
        self.modify_btn.config(state=tk.DISABLED)
        
        self.current_deal = None
    
    def adjust_preferences(self):
        """Allow preference adjustments"""
        msg = "Adjust your preferences in the input panel:\n\n"
        msg += "• Prefer high-trust sellers\n"
        msg += "• Prefer low prices\n\n"
        msg += "Then click 'Start Negotiation' to retry."
        
        messagebox.showinfo("Adjust Preferences", msg)
    
    def reset_simulation(self):
        """Reset simulation"""
        if self.is_running:
            messagebox.showwarning("Warning", "Cannot reset while running!")
            return
        
        self.log_text.delete(1.0, tk.END)
        self.deal_text.config(state=tk.NORMAL)
        self.deal_text.delete(1.0, tk.END)
        self.deal_text.config(state=tk.DISABLED)
        
        for item in self.seller_tree.get_children():
            self.seller_tree.delete(item)
        
        self.current_deal = None
        self.approve_btn.config(state=tk.DISABLED)
        self.reject_btn.config(state=tk.DISABLED)
        self.modify_btn.config(state=tk.DISABLED)
        
        self.log_message("System reset. Ready for new simulation.", 'info')


def main():
    root = tk.Tk()
    app = EnhancedMarketDemo(root)
    root.mainloop()


if __name__ == '__main__':
    main()
