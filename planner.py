import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demand hardcoded
demand_data = {
    "Week": [1, 2, 3, 4],
    "North_Regular": [30000, 35000, 40000, 45000],
    "North_Diet": [20000, 25000, 30000, 35000],
    "South_Regular": [25000, 30000, 35000, 40000],
    "South_Diet": [15000, 20000, 25000, 30000],
}
demand_df = pd.DataFrame(demand_data)

# ---------- Planning Logic ----------
def plan_production(demand_df, capacity):
    prod = []
    for i, row in demand_df.iterrows():
        week = row["Week"]
        reg = row["North_Regular"] + row["South_Regular"]
        diet = row["North_Diet"] + row["South_Diet"]
        total = reg + diet
        if total <= capacity:
            prod.append([week, reg, diet])
        else:
            scale = capacity / total
            prod.append([week, int(reg * scale), int(diet * scale)])
    return pd.DataFrame(prod, columns=["Week", "Regular", "Diet"])

def allocate_fair_share(demand_df, prod_df):
    shipments = []
    for i, row in demand_df.iterrows():
        week = row["Week"]
        prod_row = prod_df[prod_df["Week"] == week].iloc[0]
        for sku in ["Regular", "Diet"]:
            total_demand = row[f"North_{sku}"] + row[f"South_{sku}"]
            supply = prod_row[sku]
            if total_demand == 0:
                n_share, s_share = 0, 0
            else:
                n_share = int(supply * row[f"North_{sku}"] / total_demand)
                s_share = supply - n_share
            shipments.append([week, sku, "North", n_share])
            shipments.append([week, sku, "South", s_share])
    return pd.DataFrame(shipments, columns=["Week", "SKU", "DC", "Qty"])

def apply_truck_rounding(ship_df, truck_size):
    ship_df["Trucks"] = np.ceil(ship_df["Qty"] / truck_size).astype(int)
    ship_df["Qty"] = ship_df["Trucks"] * truck_size
    return ship_df

def simulate_inventory(ship_df, demand_df, safety):
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
                end = start + arrivals - demand
                if end < safety: end = safety
                records.append([week, dc, sku, start, arrivals, demand, end])
                inv[(dc,sku)] = end
    return pd.DataFrame(records, columns=["Week","DC","SKU","Start","Arrivals","Demand","End"])

def fulfillment_summary(demand_df):
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

def run_planner(capacity, truck_size, safety, demand_df=None):
    if demand_df is None:
        demand_df = pd.DataFrame(demand_data)
    prod_df = plan_production(demand_df, capacity)
    ship_df = allocate_fair_share(demand_df, prod_df)
    ship_df = apply_truck_rounding(ship_df, truck_size)
    inv_df = simulate_inventory(ship_df, demand_df, safety)
    ful_df = fulfillment_summary(demand_df)

    # Save CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prod_df.to_csv(os.path.join(OUTPUT_DIR,"production_plan.csv"),index=False)
    ship_df.to_csv(os.path.join(OUTPUT_DIR,"shipments.csv"),index=False)
    inv_df.to_csv(os.path.join(OUTPUT_DIR,"inventory_by_dc.csv"),index=False)
    ful_df.to_csv(os.path.join(OUTPUT_DIR,"fulfillment_summary.csv"),index=False)

    # Plots
    fig_paths = plot_demand_vs_fulfilled(ful_df)
    fig_paths.append(plot_shipments(ship_df))
    fig_paths.append(plot_inventory(inv_df, safety))

    return prod_df, ship_df, inv_df, ful_df, fig_paths

# For tests
def default_demand_df():
    return pd.DataFrame(demand_data)

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
    parser.add_argument("--demand_csv", type=str, default=None)
    parser.add_argument("--init_csv", type=str, default=None)
    args = parser.parse_args()

    if args.demand_csv:
        demand_df = pd.read_csv(args.demand_csv)

    prod_df = plan_production(demand_df, args.capacity)
    ship_df = allocate_fair_share(demand_df, prod_df)
    ship_df = apply_truck_rounding(ship_df, args.truck)
    inv_df = simulate_inventory(ship_df, demand_df, args.safety)
    ful_df = fulfillment_summary(demand_df)

    prod_df.to_csv(os.path.join(OUTPUT_DIR,"production_plan.csv"),index=False)
    ship_df.to_csv(os.path.join(OUTPUT_DIR,"shipments.csv"),index=False)
    inv_df.to_csv(os.path.join(OUTPUT_DIR,"inventory_by_dc.csv"),index=False)
    ful_df.to_csv(os.path.join(OUTPUT_DIR,"fulfillment_summary.csv"),index=False)

    plot_demand_vs_fulfilled(ful_df)
    plot_shipments(ship_df)
    plot_inventory(inv_df,args.safety)

    print("Plans generated successfully.")
