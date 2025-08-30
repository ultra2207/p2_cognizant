# tests.py
import pandas as pd
from planner import (
    plan_production,
    allocate_fair_share,
    apply_truck_rounding,
    simulate_inventory,
    default_demand_df,
    default_initial_inventory_df,
    Config,
)

def test_fair_share_simple():
    demand = pd.DataFrame({
        "week": [1,1],
        "dc": ["North","South"],
        "sku": ["Regular","Regular"],
        "demand": [60_000, 40_000],
    })
    supply = pd.DataFrame({
        "week": [1],
        "sku": ["Regular"],
        "produced": [100_000],
    })
    alloc = allocate_fair_share(demand, supply)
    # Expect 60k and 40k before rounding
    n = alloc[(alloc.week==1)&(alloc.dc=="North")&(alloc.sku=="Regular")].iloc[0].alloc
    s = alloc[(alloc.week==1)&(alloc.dc=="South")&(alloc.sku=="Regular")].iloc[0].alloc
    assert n + s == 100_000
    assert abs(n - 60_000) <= 1_000  # minor rounding tolerance
    assert abs(s - 40_000) <= 1_000

def test_truck_rounding_total_preserved():
    demand = pd.DataFrame({
        "week":[1,1],
        "dc":["North","South"],
        "sku":["Diet","Diet"],
        "demand":[22_000, 18_000],
    })
    supply = pd.DataFrame({
        "week":[1],
        "sku":["Diet"],
        "produced":[40_000],
    })
    alloc = allocate_fair_share(demand, supply)
    ship = apply_truck_rounding(alloc, truck_size=10_000)
    total = ship["qty"].sum()
    assert total in (40_000, )  # preserved to nearest truck total
    # each lane must be a truck multiple
    assert all(ship["qty"] % 10_000 == 0)

def test_safety_stock_enforcement():
    # Minimal scenario: small demand, check inventory keeps safety
    demand = pd.DataFrame({
        "week":[1],
        "dc":["North"],
        "sku":["Regular"],
        "demand":[5_000],
    })
    shipments = pd.DataFrame({
        "week":[1],
        "dc":["North"],
        "sku":["Regular"],
        "qty":[5_000],
        "trucks":[0],
    })
    init = default_initial_inventory_df(5_000)
    inv, _ = simulate_inventory(demand, shipments, init, safety=5_000)
    end = inv.iloc[0].ending
    assert end >= 5_000  # safety stock maintained

def test_plan_production_capacity_cap():
    demand = default_demand_df()
    prod = plan_production(demand, capacity_per_week=150_000)
    # Each week total produced cannot exceed capacity
    check = prod.groupby("week")["produced"].sum()
    assert (check <= 150_000).all()
