
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


df = pd.read_csv('output.csv')


canvas_rank = None
canvas_revenue_change = None
canvas_clv_plots = None

def get_customer_data():
    customer_id = entry.get()
    customer_data = df[df['CustomerID'] == int(customer_id)]
    if not customer_data.empty:
        clv = customer_data['m6_Revenue'].values[0]
        recency = customer_data['Recency'].values[0]
        frequency = customer_data['Frequency'].values[0]
        revenue = customer_data['Revenue'].values[0]
        
       
        rank = df[df['m6_Revenue'] >= clv]['m6_Revenue'].count() + 1
        
        
        profitability = 'Profitable' if clv > revenue else 'Not Profitable'
        if clv > 300:
            customer_value = 'High Valued'
        elif clv > 100:
            customer_value = 'Mid Valued'
        else:
            customer_value = 'Low Valued'
        
        
        text_widget.config(state=tk.NORMAL)
        text_widget.delete('1.0', tk.END)
        text_widget.insert(tk.END, f'Customer ID: {customer_id}\n')
        text_widget.insert(tk.END, f'Rank (CLV): {rank}\n')
        text_widget.insert(tk.END, f'CLV: {clv}\n')
        text_widget.insert(tk.END, f'Recency: {recency}\n')
        text_widget.insert(tk.END, f'Frequency: {frequency}\n')
        text_widget.insert(tk.END, f'Revenue: {revenue}\n')
        text_widget.insert(tk.END, f'Profitability: {profitability}\n')
        text_widget.insert(tk.END, f'Customer Value: {customer_value}\n')
        text_widget.config(state=tk.DISABLED)
        
        
        display_rank_visualization(rank)
        
        
        display_revenue_change_visualization(customer_data)
    else:
        messagebox.showinfo('Customer Data', 'Customer ID not found in the database.')

def display_rank_visualization(rank):
    ranks = df['m6_Revenue'].rank(ascending=False, method='min')
    
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['CustomerID'], ranks)
    ax.set_xlabel('Customer ID')
    ax.set_ylabel('Rank')
    ax.set_title('Rank of Customers Based on CLV')
    ax.axhline(y=rank, color='r', linestyle='--', label='Selected Customer Rank')
    ax.legend()
    
    
    global canvas_rank
    canvas_rank = FigureCanvasTkAgg(fig, master=visualization_frame)
    canvas_rank.draw()
    canvas_rank.get_tk_widget().pack()

def display_revenue_change_visualization(customer_data):
    
    customer_id = customer_data['CustomerID'].values[0]
    revenue_data = customer_data[['m1_Revenue', 'm2_Revenue', 'm3_Revenue', 'm4_Revenue', 'm5_Revenue', 'm6_Revenue']]
    
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 7), revenue_data.values[0], marker='o')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue')
    ax.set_title(f'Revenue Change for Customer ID {customer_id}')
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'])
    
    
    global canvas_revenue_change
    canvas_revenue_change = FigureCanvasTkAgg(fig, master=visualization_frame)
    canvas_revenue_change.draw()
    canvas_revenue_change.get_tk_widget().pack()

def clear_screen():
    
    if canvas_rank:
        canvas_rank.get_tk_widget().pack_forget()
    if canvas_revenue_change:
        canvas_revenue_change.get_tk_widget().pack_forget()
    if canvas_clv_plots:
        canvas_clv_plots.get_tk_widget().pack_forget()
    text_widget.config(state=tk.NORMAL)
    text_widget.delete('1.0', tk.END)
    entry.delete(0, tk.END)
    text_widget.config(state=tk.DISABLED)

def display_clv_plots():
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5) 
    
    
    axs[0].hist(df['Recency'], bins=20, color='skyblue', edgecolor='black', alpha=1)
    axs[0].set_title('Recency Distribution')
    axs[0].set_xlabel('Recency')
    axs[0].set_ylabel('Frequency')
    
    
    axs[1].hist(df['Frequency'], bins=20, color='salmon', edgecolor='black', alpha=1)
    axs[1].set_title('Frequency Distribution')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Frequency')
    
    
    axs[2].hist(df['m6_Revenue'], bins=20, color='lightgreen', edgecolor='black', alpha=1)
    axs[2].set_title('Monetary Distribution')
    axs[2].set_xlabel('Monetary')
    axs[2].set_ylabel('Frequency')
    
    
    global canvas_clv_plots
    canvas_clv_plots = FigureCanvasTkAgg(fig, master=visualization_frame)
    canvas_clv_plots.draw()
    canvas_clv_plots.get_tk_widget().pack()


root = tk.Tk()
root.title('Customer Data App')
root.state('zoomed')  


dashboard_frame = ttk.Frame(root)
dashboard_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)


label = ttk.Label(dashboard_frame, text='Enter Customer ID:')
label.pack()
entry = ttk.Entry(dashboard_frame)
entry.pack()


button = ttk.Button(dashboard_frame, text='Get Customer Data', command=get_customer_data)
button.pack()


clear_button = ttk.Button(dashboard_frame, text='Clear', command=clear_screen)
clear_button.pack()


text_widget = tk.Text(dashboard_frame, height=10, width=30)
text_widget.pack()


visualization_frame = ttk.Frame(root)
visualization_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


clv_plot_button = ttk.Button(dashboard_frame, text='Display CLV Plots', command=display_clv_plots)
clv_plot_button.pack()


root.mainloop()
