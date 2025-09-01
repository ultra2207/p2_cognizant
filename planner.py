import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global reserve system
RESERVE_CAPACITY = 5000
current_reserve = RESERVE_CAPACITY

def generate_random_demand(num_weeks=4, base_regular_north=15000, base_diet_north=10000, 
                          base_regular_south=12000, base_diet_south=8000, variance=0.2, seed=None):
    """
    Generate random demand data with some variance around base values
    
    Args:
        num_weeks: Number of weeks to generate
        base_*: Base demand values for each DC/SKU combination
        variance: Percentage variance (0.2 = ±20%)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    weeks = list(range(1, num_weeks + 1))
    data = {"Week": weeks}
    
    # Generate demand with variance and growth trend
    for week in weeks:
        # Apply growth trend (5% per week)
        growth_factor = 1 + (week - 1) * 0.05
        
        # Add random variance
        variance_factor = 1 + np.random.uniform(-variance, variance)
        
        data[f"North_Regular"] = data.get("North_Regular", [])
        data[f"North_Diet"] = data.get("North_Diet", [])
        data[f"South_Regular"] = data.get("South_Regular", [])
        data[f"South_Diet"] = data.get("South_Diet", [])
        
        data["North_Regular"].append(int(base_regular_north * growth_factor * variance_factor))
        data["North_Diet"].append(int(base_diet_north * growth_factor * variance_factor))
        data["South_Regular"].append(int(base_regular_south * growth_factor * variance_factor))
        data["South_Diet"].append(int(base_diet_south * growth_factor * variance_factor))
    
    return pd.DataFrame(data)

def apply_outflow_container_rounding(demand_df, container_size=750):
    """
    Rounds up dedef run_planner(capacity, truck_size, safety, demand_df=None, container_size=750):
    if demand_df is None:
        demand_df = pd.DataFrame(demand_data)
    
    # Apply outflow container rounding to demand
    demand_df = apply_outflow_container_rounding(demand_df, container_size)
    
    prod_df = plan_production(demand_df, capacity)
    ship_df = allocate_fair_share(demand_df, prod_df)
    ship_df = apply_smart_truck_rounding(ship_df, demand_df, truck_size, safety)
    inv_df = simulate_inventory_with_buffer(ship_df, demand_df, safety)
    ful_df = fulfillment_summary_with_buffer(demand_df, inv_df)est container size multiples for customer fulfillment
    Each customer order must be in multiples of container_size (default 750 bottles)
    """
    demand_df = demand_df.copy()
    
    # Columns to round (all demand columns)
    demand_columns = [col for col in demand_df.columns if col != "Week"]
    
    for col in demand_columns:
        # Round up to nearest container multiple
        demand_df[col] = np.ceil(demand_df[col] / container_size) * container_size
    
    return demand_df

# Generate random demand data (with seed for reproducibility in testing)
demand_data = generate_random_demand(num_weeks=4, seed=42).to_dict('list')
demand_df = pd.DataFrame(demand_data)

# ---------- Planning Logic ----------
def plan_production_with_reserve(demand_df, capacity, container_size=750):
    """
    Plan production considering reserve system.
    Production is rounded to container multiples.
    When demand exceeds production, dip into reserve.
    When production exceeds demand, refill reserve.
    """
    global current_reserve
    prod = []
    reserve_transactions = []
    
    for i, row in demand_df.iterrows():
        week = row["Week"]
        reg_demand = row["North_Regular"] + row["South_Regular"]
        diet_demand = row["North_Diet"] + row["South_Diet"]
        total_demand = reg_demand + diet_demand
        
        # Calculate what we can produce this week (round to container multiples)
        if total_demand <= capacity:
            # We can meet all demand with production
            # Round each SKU production to container multiples
            reg_prod = int(np.ceil(reg_demand / container_size)) * container_size
            diet_prod = int(np.ceil(diet_demand / container_size)) * container_size
            total_prod = reg_prod + diet_prod
            
            # Check if rounded production exceeds capacity
            if total_prod > capacity:
                # Scale down proportionally and round to containers
                scale = capacity / total_prod
                reg_prod = int(reg_prod * scale // container_size) * container_size
                diet_prod = int(diet_prod * scale // container_size) * container_size
                total_prod = reg_prod + diet_prod
            
            # Calculate surplus production
            surplus = capacity - total_prod
            
            # Use surplus to refill reserve (only if we have meaningful surplus)
            if surplus >= container_size:
                reserve_refill = min(surplus, RESERVE_CAPACITY - current_reserve)
                # Round reserve refill to container multiples
                reserve_refill = int(reserve_refill // container_size) * container_size
                current_reserve += reserve_refill
                
                if reserve_refill > 0:
                    reserve_transactions.append({
                        'Week': week,
                        'Action': 'Refill',
                        'Amount': reserve_refill,
                        'Reserve_Before': current_reserve - reserve_refill,
                        'Reserve_After': current_reserve,
                        'Surplus_Used': reserve_refill,
                        'Remaining_Surplus': surplus - reserve_refill
                    })
            
        else:
            # Demand exceeds capacity - need to use reserve
            deficit = total_demand - capacity
            reserve_used = min(deficit, current_reserve)
            # Round reserve usage to container multiples
            reserve_used = int(reserve_used // container_size) * container_size
            current_reserve -= reserve_used
            
            # Scale production to capacity and round to containers
            available_production = capacity + reserve_used
            scale = available_production / total_demand
            
            reg_prod = int(reg_demand * scale // container_size) * container_size
            diet_prod = int(diet_demand * scale // container_size) * container_size
            
            # Ensure we don't exceed available production
            total_prod = reg_prod + diet_prod
            if total_prod > available_production:
                # Reduce proportionally
                scale_down = available_production / total_prod
                reg_prod = int(reg_prod * scale_down // container_size) * container_size
                diet_prod = int(diet_prod * scale_down // container_size) * container_size
            
            if reserve_used > 0:
                reserve_transactions.append({
                    'Week': week,
                    'Action': 'Use',
                    'Amount': reserve_used,
                    'Reserve_Before': current_reserve + reserve_used,
                    'Reserve_After': current_reserve,
                    'Deficit': deficit,
                    'Unmet_Demand': max(0, total_demand - capacity - reserve_used)
                })
        
        prod.append([week, reg_prod, diet_prod])
    
    prod_df = pd.DataFrame(prod, columns=["Week", "Regular", "Diet"])
    reserve_df = pd.DataFrame(reserve_transactions)
    
    return prod_df, reserve_df

def plan_production(demand_df, capacity):
    """Wrapper for backward compatibility"""
    prod_df, _ = plan_production_with_reserve(demand_df, capacity, container_size=750)
    return prod_df

def allocate_fair_share(demand_df, prod_df):
    """Allocate production to DCs, ensuring exact match with production totals"""
    shipments = []
    for i, row in demand_df.iterrows():
        week = row["Week"]
        prod_row = prod_df[prod_df["Week"] == week].iloc[0]
        for sku in ["Regular", "Diet"]:
            total_demand = row[f"North_{sku}"] + row[f"South_{sku}"]
            supply = prod_row[sku]
            
            if total_demand == 0:
                n_share, s_share = 0, 0
            elif total_demand <= supply:
                # We have enough supply, allocate exact demand
                n_share = row[f"North_{sku}"]
                s_share = row[f"South_{sku}"]
            else:
                # Supply is less than demand, allocate proportionally
                n_proportion = row[f"North_{sku}"] / total_demand
                s_proportion = row[f"South_{sku}"] / total_demand
                
                n_share = int(supply * n_proportion)
                s_share = supply - n_share  # Ensure total equals supply exactly
                
            shipments.append([week, sku, "North", n_share])
            shipments.append([week, sku, "South", s_share])
    return pd.DataFrame(shipments, columns=["Week", "SKU", "DC", "Qty"])

def apply_smart_truck_rounding_with_buffer(ship_df, demand_df, truck_size, safety, initial_inventory=None):
    """
    Apply truck rounding where:
    - Shipments are in multiples of 750 (container size)
    - Trucks are calculated to carry the required containers
    - Maintain safety stock levels
    """
    ship_df = ship_df.copy()
    
    # Initialize inventory if not provided
    if initial_inventory is None:
        inv = {("North","Regular"):safety, ("North","Diet"):safety,
               ("South","Regular"):safety, ("South","Diet"):safety}
    else:
        inv = initial_inventory.copy()
    
    # Process week by week
    for week in sorted(ship_df["Week"].unique()):
        week_shipments = ship_df[ship_df["Week"] == week].copy()
        week_demand = demand_df[demand_df["Week"] == week].iloc[0]
        
        for idx, row in week_shipments.iterrows():
            dc = row["DC"]
            sku = row["SKU"]
            allocated_qty = row["Qty"]  # This should already be in multiples of 750 from production
            
            # Calculate actual need considering current inventory and safety stock
            current_inv = inv[(dc, sku)]
            demand = week_demand[f"{dc}_{sku}"]
            
            # The allocated quantity should already be optimal from production planning
            # Just ensure it's in container multiples and calculate trucks needed
            final_qty = allocated_qty  # Already in multiples of 750
            
            # Calculate trucks needed to carry this quantity
            # Each truck can carry truck_size bottles
            trucks_needed = int(np.ceil(final_qty / truck_size)) if final_qty > 0 else 0
            
            # Update the dataframe
            ship_df.at[idx, "Trucks"] = trucks_needed
            ship_df.at[idx, "Qty"] = final_qty
            
            # Update inventory for next iteration
            inv[(dc, sku)] = max(safety, current_inv + final_qty - demand)
    
    return ship_df

def apply_smart_truck_rounding(ship_df, demand_df, truck_size, safety, initial_inventory=None):
    """Enhanced truck rounding that considers inventory levels and actual needs"""
    return apply_smart_truck_rounding_with_buffer(ship_df, demand_df, truck_size, safety, initial_inventory)

def apply_truck_rounding(ship_df, truck_size):
    """Simple truck rounding - kept for backward compatibility"""
    return apply_smart_truck_rounding(ship_df, demand_df, truck_size, 5000)

def simulate_inventory_with_buffer(ship_df, demand_df, safety):
    """Simulate inventory with safety stock protection"""
    records = []
    inv = {("North","Regular"):safety, ("North","Diet"):safety,
           ("South","Regular"):safety, ("South","Diet"):safety}
    
    for i, row in demand_df.iterrows():
        week = row["Week"]
        for dc in ["North","South"]:
            for sku in ["Regular","Diet"]:
                arrivals = ship_df[(ship_df["Week"]==week)&(ship_df["DC"]==dc)&(ship_df["SKU"]==sku)]["Qty"].sum()
                demand = row[f"{dc}_{sku}"]
                start = inv[(dc,sku)]
                
                # Calculate ending inventory
                end = start + arrivals - demand
                
                # Ensure we maintain minimum safety stock
                actual_end = max(safety, end)
                
                # Track buffer usage (when we would have gone below safety without protection)
                buffer_used = max(0, safety - end) if end < safety else 0
                
                records.append([week, dc, sku, start, arrivals, demand, actual_end, buffer_used])
                inv[(dc,sku)] = actual_end
                
    return pd.DataFrame(records, columns=["Week","DC","SKU","Start","Arrivals","Demand","End","BufferUsed"])

def simulate_inventory(ship_df, demand_df, safety):
    """Wrapper for backward compatibility"""
    df = simulate_inventory_with_buffer(ship_df, demand_df, safety)
    return df[["Week","DC","SKU","Start","Arrivals","Demand","End"]]  # Return original columns

def fulfillment_summary_with_buffer(demand_df, inv_df):
    """Calculate fulfillment considering buffer stock usage"""
    fulfilled = []
    for i, row in demand_df.iterrows():
        week = row["Week"]
        for dc in ["North","South"]:
            for sku in ["Regular","Diet"]:
                demand = row[f"{dc}_{sku}"]
                
                # Get inventory data for this week/dc/sku
                inv_row = inv_df[(inv_df["Week"]==week) & (inv_df["DC"]==dc) & (inv_df["SKU"]==sku)]
                if not inv_row.empty:
                    start = inv_row.iloc[0]["Start"]
                    arrivals = inv_row.iloc[0]["Arrivals"]
                    end = inv_row.iloc[0]["End"]
                    
                    # Calculate actual fulfillment
                    available = start + arrivals
                    actual_fulfilled = min(demand, available)
                    
                    # Check if buffer stock was used
                    buffer_used = max(0, 5000 - end) if end >= 0 else 5000
                    
                    fulfilled.append([week, dc, sku, demand, actual_fulfilled, buffer_used])
                else:
                    fulfilled.append([week, dc, sku, demand, 0, 0])
                    
    return pd.DataFrame(fulfilled, columns=["Week","DC","SKU","Demand","Fulfilled","BufferUsed"])

def fulfillment_summary(demand_df):
    """Simple fulfillment summary - kept for backward compatibility"""
    fulfilled = []
    for i, row in demand_df.iterrows():
        week = row["Week"]
        for dc in ["North","South"]:
            for sku in ["Regular","Diet"]:
                demand = row[f"{dc}_{sku}"]
                actual = demand  # simplified assumption
                fulfilled.append([week, dc, sku, demand, actual])
    return pd.DataFrame(fulfilled, columns=["Week","DC","SKU","Demand","Fulfilled"])

# ---------- Charts ----------
def plot_demand_vs_fulfilled(summary_df):
    out = []
    for sku in ["Regular","Diet"]:
        agg = summary_df[summary_df["SKU"]==sku].groupby("Week")[["Demand","Fulfilled"]].sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        agg.plot(kind="bar", ax=ax)
        ax.set_title(f"Demand vs Fulfilled - {sku}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Bottles")
        ax.tick_params(axis='x', rotation=0)  # Keep week numbers horizontal (0 degrees)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"fig_demand_vs_fulfilled_{sku}.png")
        plt.savefig(path)
        plt.close()
        out.append(path)
    return out

def plot_shipments(ship_df):
    agg = ship_df.groupby(["Week","DC"])["Qty"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots()
    agg.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Shipments by Lane")
    ax.tick_params(axis='x', rotation=0)  # Keep week numbers horizontal (0 degrees)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,"fig_shipments_by_lane.png")
    plt.savefig(path); plt.close()
    return path

def plot_inventory(inv_df, safety):
    fig, ax = plt.subplots(figsize=(8, 6))
    for dc in ["North","South"]:
        for sku in ["Regular","Diet"]:
            series = inv_df[(inv_df["DC"]==dc)&(inv_df["SKU"]==sku)].set_index("Week")["End"]
            series.plot(ax=ax, label=f"{dc}-{sku}")
    ax.axhline(safety, color="red", linestyle="--", label="Safety Stock")
    ax.set_title("Ending Inventory vs Safety Stock")
    ax.set_xlabel("Week")
    ax.set_ylabel("Bottles")
    ax.tick_params(axis='x', rotation=0)  # Keep week numbers horizontal
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,"fig_inventory_vs_safety.png")
    plt.savefig(path); plt.close()
    return path

def plot_reserve_status(reserve_df):
    """Plot reserve system transactions and status over time"""
    if reserve_df.empty:
        # Create empty plot if no transactions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(RESERVE_CAPACITY, color="blue", linestyle="-", label="Reserve Capacity")
        ax.set_title("Reserve System Status")
        ax.set_xlabel("Week")
        ax.set_ylabel("Reserve Level")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR,"fig_reserve_status.png")
        plt.savefig(path); plt.close()
        return path
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Reserve level over time
    weeks = reserve_df['Week'].values
    reserve_after = reserve_df['Reserve_After'].values
    
    ax1.plot(weeks, reserve_after, marker='o', linewidth=2, label="Reserve Level")
    ax1.axhline(RESERVE_CAPACITY, color="blue", linestyle="--", label="Max Capacity")
    ax1.axhline(0, color="red", linestyle="--", label="Empty")
    ax1.set_title("Reserve Level Over Time")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Reserve Units")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reserve transactions
    refills = reserve_df[reserve_df['Action'] == 'Refill']
    uses = reserve_df[reserve_df['Action'] == 'Use']
    
    if not refills.empty:
        ax2.bar(refills['Week'], refills['Amount'], alpha=0.7, color='green', label='Refill')
    if not uses.empty:
        ax2.bar(uses['Week'], -uses['Amount'], alpha=0.7, color='red', label='Use')
    
    ax2.set_title("Reserve Transactions")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Amount (Positive=Refill, Negative=Use)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,"fig_reserve_status.png")
    plt.savefig(path); plt.close()
    return path

def run_planner(capacity, truck_size, safety, demand_df=None, container_size=750, reset_reserve=True):
    global current_reserve
    
    if reset_reserve:
        current_reserve = RESERVE_CAPACITY
    
    if demand_df is None:
        demand_df = generate_random_demand(num_weeks=4)
    
    # Apply outflow container rounding to demand
    demand_df = apply_outflow_container_rounding(demand_df, container_size)
    
    prod_df, reserve_df = plan_production_with_reserve(demand_df, capacity, container_size)
    ship_df = allocate_fair_share(demand_df, prod_df)
    ship_df = apply_smart_truck_rounding(ship_df, demand_df, truck_size, safety)
    inv_df = simulate_inventory(ship_df, demand_df, safety)
    ful_df = fulfillment_summary(demand_df)

    # Save CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prod_df.to_csv(os.path.join(OUTPUT_DIR,"production_plan.csv"),index=False)
    ship_df.to_csv(os.path.join(OUTPUT_DIR,"shipments.csv"),index=False)
    inv_df.to_csv(os.path.join(OUTPUT_DIR,"inventory_by_dc.csv"),index=False)
    ful_df.to_csv(os.path.join(OUTPUT_DIR,"fulfillment_summary.csv"),index=False)
    reserve_df.to_csv(os.path.join(OUTPUT_DIR,"reserve_transactions.csv"),index=False)

    # Plots
    fig_paths = plot_demand_vs_fulfilled(ful_df)
    fig_paths.append(plot_shipments(ship_df))
    fig_paths.append(plot_inventory(inv_df, safety))
    fig_paths.append(plot_reserve_status(reserve_df))

    return prod_df, ship_df, inv_df, ful_df, reserve_df, fig_paths

# For tests
def default_demand_df():
    return generate_random_demand(num_weeks=4, seed=42)

def default_initial_inventory_df(safety):
    return pd.DataFrame({
        "dc": ["North", "North", "South", "South"],
        "sku": ["Regular", "Diet", "Regular", "Diet"],
        "initial": [safety, safety, safety, safety]
    })

class Config:
    pass

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--truck", type=int, required=True)
    parser.add_argument("--safety", type=int, required=True)
    parser.add_argument("--container", type=int, default=750, help="Customer container size (default: 750)")
    parser.add_argument("--demand_csv", type=str, default=None)
    parser.add_argument("--init_csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for demand generation")
    args = parser.parse_args()

    # Reset reserve system
    current_reserve = RESERVE_CAPACITY

    if args.demand_csv:
        demand_df = pd.read_csv(args.demand_csv)
    else:
        demand_df = generate_random_demand(num_weeks=4, seed=args.seed)

    # Apply outflow container rounding
    demand_df = apply_outflow_container_rounding(demand_df, args.container)

    prod_df, reserve_df = plan_production_with_reserve(demand_df, args.capacity, args.container)
    ship_df = allocate_fair_share(demand_df, prod_df)
    ship_df = apply_smart_truck_rounding(ship_df, demand_df, args.truck, args.safety)
    inv_df = simulate_inventory_with_buffer(ship_df, demand_df, args.safety)
    ful_df = fulfillment_summary_with_buffer(demand_df, inv_df)

    prod_df.to_csv(os.path.join(OUTPUT_DIR,"production_plan.csv"),index=False)
    ship_df.to_csv(os.path.join(OUTPUT_DIR,"shipments.csv"),index=False)
    inv_df.to_csv(os.path.join(OUTPUT_DIR,"inventory_by_dc.csv"),index=False)
    ful_df.to_csv(os.path.join(OUTPUT_DIR,"fulfillment_summary.csv"),index=False)
    reserve_df.to_csv(os.path.join(OUTPUT_DIR,"reserve_transactions.csv"),index=False)

    plot_demand_vs_fulfilled(ful_df)
    plot_shipments(ship_df)
    plot_inventory(inv_df,args.safety)
    plot_reserve_status(reserve_df)

    print("Plans generated successfully.")
    print(f"Final reserve level: {current_reserve}")
    if not reserve_df.empty:
        print("\nReserve transactions:")
        print(reserve_df.to_string(index=False))
