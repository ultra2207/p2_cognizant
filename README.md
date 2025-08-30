# P2 – Production & Deployment Planning for The Cola Company

## Overview
This project creates a simple production and deployment planning model for 2 SKUs (Regular 500ml, Diet 500ml), 2 DCs (North, South), and 1 Plant (Plant A) across a 4-week horizon.

The code generates:
- Production plan
- Deployment (shipments) plan
- Inventory tracking at each DC
- Fulfillment summary
- Charts for visualization

## Assumptions
- Weekly plant capacity = 150,000 bottles.
- Shipments must be in multiples of 10,000 (truck rule).
- Each DC must end week with ≥ 5,000 bottles per SKU (safety stock).
- Lead time: Plant→North = 2 days, Plant→South = 4 days.
  *Weekly buckets: treated as same-week arrivals unless capacity tight.*
- Fair-share allocation used when supply < demand.
- Start inventory: 5,000 per SKU per DC.

## Setup
Run uv venv, and then run uv sync --upgrade to install dependencies.

## Usage
Run the planner with default embedded demand:

```bash
python planner.py --capacity 150000 --truck 10000 --safety 5000
```

Or launch the interactive Streamlit dashboard:

```bash
streamlit run app_streamlit.py
```
