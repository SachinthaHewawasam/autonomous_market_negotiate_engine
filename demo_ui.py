"""
Interactive Demo UI for Autonomous Market Simulation

This demo showcases the trained RL buyer agent negotiating with multiple sellers
in a bulk procurement scenario with coalition formation, fairness, and trust mechanisms.

Usage:
    python demo_ui.py
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


class MarketSimulationDemo:
    """Interactive UI for demonstrating the autonomous market simulation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Market Simulation - Demo")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Load trained agent
        self.env = None
        self.agent = None
        self.load_trained_agent()
        
        # Simulation state
        self.is_running = False
        self.current_step = 0
        self.negotiation_history = []
        
        # Create UI
        self.create_ui()
        
    def load_trained_agent(self):
        """Load the trained RL buyer agent"""
        model_path = 'models/buyer_agent.pth'
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Trained model not found at {model_path}\nPlease run train.py first.")
            return
        
        # Initialize environment
        self.env = MarketEnv(
            num_sellers=5,
            max_quantity_per_seller=50,
            max_negotiation_rounds=10,
            seed=42
        )
        
        # Initialize and load agent
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = BuyerAgent(
            state_dim=9,
            action_dim=4,
            device=device
        )
        
        try:
            self.agent.load_model(model_path)
            print(f"✓ Loaded trained agent from {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def create_ui(self):
        """Create the main UI layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🤖 Autonomous Market Simulation Demo",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Reinforcement Learning Agent • Coalition Formation • Fairness & Trust Mechanisms",
            font=('Arial', 11),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Input and Control
        left_panel = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=0)
        
        self.create_input_panel(left_panel)
        
        # Right panel - Visualization
        right_panel = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=0)
        
        self.create_visualization_panel(right_panel)
    
    def create_input_panel(self, parent):
        """Create the input and control panel"""
        # Header
        header = tk.Label(
            parent,
            text="📋 Procurement Request",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        header.pack(pady=15, padx=15, anchor='w')
        
        # Input form
        form_frame = tk.Frame(parent, bg='white')
        form_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Product name
        tk.Label(form_frame, text="Product:", font=('Arial', 11, 'bold'), bg='white').grid(row=0, column=0, sticky='w', pady=8)
        self.product_entry = tk.Entry(form_frame, font=('Arial', 11), width=25)
        self.product_entry.insert(0, "Biscuits (Brand X)")
        self.product_entry.grid(row=0, column=1, pady=8, padx=10)
        
        # Quantity
        tk.Label(form_frame, text="Quantity:", font=('Arial', 11, 'bold'), bg='white').grid(row=1, column=0, sticky='w', pady=8)
        self.quantity_entry = tk.Entry(form_frame, font=('Arial', 11), width=25)
        self.quantity_entry.insert(0, "120")
        self.quantity_entry.grid(row=1, column=1, pady=8, padx=10)
        
        # Budget
        tk.Label(form_frame, text="Max Budget ($):", font=('Arial', 11, 'bold'), bg='white').grid(row=2, column=0, sticky='w', pady=8)
        self.budget_entry = tk.Entry(form_frame, font=('Arial', 11), width=25)
        self.budget_entry.insert(0, "1200")
        self.budget_entry.grid(row=2, column=1, pady=8, padx=10)
        
        # Buttons
        button_frame = tk.Frame(parent, bg='white')
        button_frame.pack(fill=tk.X, padx=15, pady=20)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ Start Negotiation",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            command=self.start_simulation,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10
        )
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.reset_button = tk.Button(
            button_frame,
            text="🔄 Reset",
            font=('Arial', 11),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            command=self.reset_simulation,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=8
        )
        self.reset_button.pack(fill=tk.X, pady=5)
        
        # Seller Information
        seller_header = tk.Label(
            parent,
            text="🏪 Available Sellers",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        seller_header.pack(pady=(20, 10), padx=15, anchor='w')
        
        # Seller table frame
        table_frame = tk.Frame(parent, bg='white')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Create seller table
        columns = ('Seller', 'Stock', 'Price', 'Trust')
        self.seller_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=6)
        
        # Define headings
        self.seller_tree.heading('Seller', text='Seller')
        self.seller_tree.heading('Stock', text='Stock')
        self.seller_tree.heading('Price', text='Price/unit')
        self.seller_tree.heading('Trust', text='Trust Score')
        
        # Define column widths
        self.seller_tree.column('Seller', width=80, anchor='center')
        self.seller_tree.column('Stock', width=80, anchor='center')
        self.seller_tree.column('Price', width=100, anchor='center')
        self.seller_tree.column('Trust', width=100, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.seller_tree.yview)
        self.seller_tree.configure(yscroll=scrollbar.set)
        
        self.seller_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Style
        style = ttk.Style()
        style.configure("Treeview", font=('Arial', 10), rowheight=25)
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
    
    def create_visualization_panel(self, parent):
        """Create the visualization panel"""
        # Negotiation Log
        log_header = tk.Label(
            parent,
            text="📊 Negotiation Process",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        log_header.pack(pady=15, padx=15, anchor='w')
        
        # Log text area
        log_frame = tk.Frame(parent, bg='white')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for colored output
        self.log_text.tag_config('header', font=('Consolas', 11, 'bold'), foreground='#2c3e50')
        self.log_text.tag_config('success', foreground='#27ae60', font=('Consolas', 10, 'bold'))
        self.log_text.tag_config('warning', foreground='#f39c12', font=('Consolas', 10, 'bold'))
        self.log_text.tag_config('error', foreground='#e74c3c', font=('Consolas', 10, 'bold'))
        self.log_text.tag_config('info', foreground='#3498db')
        self.log_text.tag_config('agent', foreground='#9b59b6', font=('Consolas', 10, 'bold'))
        
        # Results panel
        results_header = tk.Label(
            parent,
            text="✅ Final Deal",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        results_header.pack(pady=(20, 10), padx=15, anchor='w')
        
        self.results_text = tk.Text(
            parent,
            font=('Arial', 11),
            bg='#ecf0f1',
            fg='#2c3e50',
            height=8,
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.results_text.pack(fill=tk.X, padx=15, pady=10)
        self.results_text.config(state=tk.DISABLED)
    
    def log_message(self, message, tag='info'):
        """Add a message to the log"""
        self.log_text.insert(tk.END, message + '\n', tag)
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_seller_table(self, info):
        """Update the seller information table"""
        # Clear existing items
        for item in self.seller_tree.get_children():
            self.seller_tree.delete(item)
        
        # Add seller data
        for i in range(len(info['seller_stocks'])):
            self.seller_tree.insert('', tk.END, values=(
                f'Seller {i}',
                f"{info['seller_stocks'][i]} units",
                f"${info['seller_prices'][i]:.2f}",
                f"{info['trust_scores'][i]:.2f}"
            ))
    
    def start_simulation(self):
        """Start the negotiation simulation"""
        if self.is_running:
            messagebox.showwarning("Warning", "Simulation is already running!")
            return
        
        if self.agent is None or self.env is None:
            messagebox.showerror("Error", "Agent or environment not loaded!")
            return
        
        # Get input values
        try:
            product = self.product_entry.get()
            quantity = int(self.quantity_entry.get())
            budget = float(self.budget_entry.get())
            
            if quantity <= 0 or budget <= 0:
                raise ValueError("Quantity and budget must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid values:\n{e}")
            return
        
        # Disable start button
        self.start_button.config(state=tk.DISABLED)
        self.is_running = True
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Run simulation in a separate thread
        thread = threading.Thread(target=self.run_negotiation, args=(product, quantity, budget))
        thread.daemon = True
        thread.start()
    
    def run_negotiation(self, product, quantity, budget):
        """Run the negotiation process"""
        try:
            # Reset environment with custom request
            state, info = self.env.reset(options={'quantity': quantity, 'max_budget': budget})
            
            # Update seller table
            self.root.after(0, self.update_seller_table, info)
            
            # Log initial state
            self.log_message("=" * 80, 'header')
            self.log_message(f"PROCUREMENT REQUEST: {product}", 'header')
            self.log_message("=" * 80, 'header')
            self.log_message(f"\n📦 Requested Quantity: {quantity} units", 'info')
            self.log_message(f"💰 Maximum Budget: ${budget:.2f}", 'info')
            self.log_message(f"\n🏪 Available Sellers: {len(info['seller_stocks'])}", 'info')
            
            # Check if any single seller can fulfill
            can_fulfill_alone = any(stock >= quantity for stock in info['seller_stocks'])
            if can_fulfill_alone:
                self.log_message("✓ At least one seller can fulfill the entire order", 'success')
            else:
                self.log_message("⚠ No single seller can fulfill - coalition required", 'warning')
            
            self.log_message("\n" + "-" * 80, 'header')
            self.log_message("🤖 RL AGENT STARTING NEGOTIATION", 'agent')
            self.log_message("-" * 80 + "\n", 'header')
            
            time.sleep(1)
            
            # Run negotiation
            done = False
            truncated = False
            step = 0
            total_reward = 0
            
            while not (done or truncated):
                step += 1
                
                # Agent selects action (inference only, no training)
                action = self.agent.select_action(state, training=False)
                
                # Decode action
                action_type = int(action[0])
                seller_id = int(action[1]) % self.env.num_sellers
                price = float(action[2])
                qty = int(action[3])
                
                action_names = ['Offer', 'Counteroffer', 'Accept', 'Reject', 'Coalition']
                action_name = action_names[action_type]
                
                self.log_message(f"Round {step}: Agent Action = {action_name}", 'agent')
                self.log_message(f"  → Target: Seller {seller_id}", 'info')
                self.log_message(f"  → Price: ${price:.2f}/unit", 'info')
                self.log_message(f"  → Quantity: {qty} units", 'info')
                
                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                
                # Log result
                if reward > 50:
                    self.log_message(f"  ✓ Reward: +{reward:.2f} (Positive outcome!)", 'success')
                elif reward > 0:
                    self.log_message(f"  ✓ Reward: +{reward:.2f}", 'success')
                elif reward < -10:
                    self.log_message(f"  ✗ Reward: {reward:.2f} (Penalty)", 'error')
                else:
                    self.log_message(f"  • Reward: {reward:.2f}", 'warning')
                
                if done:
                    self.log_message("\n🎉 NEGOTIATION SUCCESSFUL!", 'success')
                elif truncated:
                    self.log_message("\n⏱ Maximum rounds reached", 'warning')
                
                self.log_message("")
                
                state = next_state
                time.sleep(0.8)  # Delay for visualization
            
            # Display final results
            self.log_message("=" * 80, 'header')
            self.log_message("NEGOTIATION COMPLETED", 'header')
            self.log_message("=" * 80 + "\n", 'header')
            
            self.display_final_results(done, total_reward, info, quantity, budget)
            
        except Exception as e:
            self.log_message(f"\n❌ Error: {str(e)}", 'error')
            messagebox.showerror("Simulation Error", str(e))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
    
    def display_final_results(self, success, total_reward, info, requested_qty, budget):
        """Display the final negotiation results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if success:
            # Extract deal information
            coalition_formed = info.get('coalition_formed', False)
            
            result = f"✅ DEAL SUCCESSFUL!\n\n"
            result += f"Total Reward: {total_reward:.2f}\n"
            result += f"Requested Quantity: {requested_qty} units\n"
            result += f"Maximum Budget: ${budget:.2f}\n\n"
            
            if coalition_formed:
                result += "🤝 Coalition Formed:\n"
                coalition_members = info.get('coalition_members', [])
                for seller_id in coalition_members:
                    result += f"  • Seller {seller_id}\n"
                
                total_cost = info.get('total_cost', 0)
                savings = budget - total_cost
                result += f"\nTotal Cost: ${total_cost:.2f}\n"
                result += f"Savings: ${savings:.2f} ({savings/budget*100:.1f}%)\n"
            else:
                seller_id = info.get('selected_seller', 0)
                price = info.get('final_price', 0)
                total_cost = price * requested_qty
                savings = budget - total_cost
                
                result += f"Selected Seller: Seller {seller_id}\n"
                result += f"Price per unit: ${price:.2f}\n"
                result += f"Total Cost: ${total_cost:.2f}\n"
                result += f"Savings: ${savings:.2f} ({savings/budget*100:.1f}%)\n"
            
            self.log_message("✅ Deal finalized successfully!", 'success')
        else:
            result = f"❌ NEGOTIATION FAILED\n\n"
            result += f"Total Reward: {total_reward:.2f}\n"
            result += f"Requested Quantity: {requested_qty} units\n"
            result += f"Maximum Budget: ${budget:.2f}\n\n"
            result += "Possible reasons:\n"
            result += "  • Budget too low\n"
            result += "  • Insufficient stock across all sellers\n"
            result += "  • Fairness constraints violated\n"
            
            self.log_message("❌ Negotiation failed to reach agreement", 'error')
        
        self.results_text.insert(1.0, result)
        self.results_text.config(state=tk.DISABLED)
    
    def reset_simulation(self):
        """Reset the simulation"""
        if self.is_running:
            messagebox.showwarning("Warning", "Cannot reset while simulation is running!")
            return
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Clear results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Clear seller table
        for item in self.seller_tree.get_children():
            self.seller_tree.delete(item)
        
        # Reset input fields
        self.product_entry.delete(0, tk.END)
        self.product_entry.insert(0, "Biscuits (Brand X)")
        
        self.quantity_entry.delete(0, tk.END)
        self.quantity_entry.insert(0, "120")
        
        self.budget_entry.delete(0, tk.END)
        self.budget_entry.insert(0, "1200")
        
        self.log_message("System reset. Ready for new simulation.", 'info')


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MarketSimulationDemo(root)
    root.mainloop()


if __name__ == '__main__':
    main()
