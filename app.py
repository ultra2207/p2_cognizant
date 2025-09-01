import streamlit as st
import pandas as pd
from planner import run_planner, default_demand_df, generate_random_demand
import os

st.set_page_config(page_title="Cola Company Planner", layout="wide")

st.title("Cola Company – Production & Deployment Planner")

# Initialize session state
if 'planning_results' not in st.session_state:
    st.session_state.planning_results = None

with st.sidebar:
    st.header("Planning Parameters")
    capacity = st.number_input("Weekly Plant Capacity", value=150000, min_value=0, help="Total bottles the plant can produce per week")
    truck = st.number_input("Truck Size (bottles)", value=10000, min_value=0, help="Number of bottles per truck shipment")
    safety = st.number_input("Safety Stock / DC / SKU", value=5000, min_value=0, help="Minimum inventory to maintain at each DC per SKU")
    container = st.number_input("Customer Container Size", value=750, min_value=0, help="Customer orders must be in multiples of this size")

    st.header("Demand Generation")
    use_random = st.checkbox("Generate Random Demand", value=True, help="Generate random demand data instead of using uploaded file")
    
    if use_random:
        seed = st.number_input("Random Seed", value=42, min_value=0, help="Seed for reproducible random generation")
        num_weeks = st.number_input("Number of Weeks", value=4, min_value=1, max_value=52)
        variance = st.slider("Demand Variance", 0.0, 0.5, 0.2, help="Percentage variance in demand (±20% = 0.2)")
    else:
        seed = None
        num_weeks = 4
        variance = 0.2

    st.header("Data Input")
    demand_file = st.file_uploader("Demand CSV (optional)", type="csv", help="Upload custom demand data. Ignored if 'Generate Random Demand' is checked.")

if st.button("Run Planning", type="primary", width='stretch'):
    with st.spinner("Generating production plan..."):
        if use_random:
            demand_df = generate_random_demand(num_weeks=num_weeks, variance=variance, seed=seed)
        elif demand_file is not None:
            demand_df = pd.read_csv(demand_file)
        else:
            demand_df = default_demand_df()

        prod_df, ship_df, inv_df, ful_df, reserve_df, fig_paths = run_planner(capacity, truck, safety, demand_df, container)
        
        # Store results in session state
        st.session_state.planning_results = {
            'prod_df': prod_df,
            'ship_df': ship_df, 
            'inv_df': inv_df,
            'ful_df': ful_df,
            'reserve_df': reserve_df,
            'fig_paths': fig_paths,
            'capacity': capacity,
            'truck': truck,
            'safety': safety,
            'container': container,
            'demand_df': demand_df
        }

    st.success("Plan generated successfully!")

# Display results if available
if st.session_state.planning_results is not None:
    results = st.session_state.planning_results
    prod_df = results['prod_df']
    ship_df = results['ship_df']
    inv_df = results['inv_df']
    ful_df = results['ful_df']
    reserve_df = results['reserve_df']
    fig_paths = results['fig_paths']
    demand_df = results['demand_df']
    
    # Display key metrics including reserve status
    total_weekly_capacity = results['capacity'] * prod_df.shape[0]
    total_produced = prod_df['Regular'].sum() + prod_df['Diet'].sum()
    utilization = total_produced / total_weekly_capacity * 100
    total_shipped = ship_df['Qty'].sum()
    avg_fulfillment = (ful_df['Fulfilled'].sum() / ful_df['Demand'].sum()) * 100 if ful_df['Demand'].sum() > 0 else 0
    
    # Calculate reserve usage
    from planner import current_reserve, RESERVE_CAPACITY
    reserve_used = RESERVE_CAPACITY - current_reserve
    reserve_utilization = (reserve_used / RESERVE_CAPACITY) * 100

    st.markdown("### Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Capacity Utilization", f"{utilization:.1f}%", delta=f"of {results['capacity']:,}/week")
    kpi2.metric("Total Shipped", f"{total_shipped:,}", delta=f"over {prod_df.shape[0]} weeks")
    kpi3.metric("Avg. Fulfillment Rate", f"{avg_fulfillment:.1f}%")
    kpi4.metric("Reserve Status", f"{current_reserve:,}/{RESERVE_CAPACITY:,}", 
                delta=f"{reserve_utilization:.1f}% used" if reserve_used > 0 else "Unused")

    # Show reserve transactions summary
    if not reserve_df.empty:
        total_used = reserve_df[reserve_df['Action'] == 'Use']['Amount'].sum()
        total_refilled = reserve_df[reserve_df['Action'] == 'Refill']['Amount'].sum()
        
        st.markdown("### Reserve System Summary")
        res1, res2, res3 = st.columns(3)
        res1.metric("Total Used", f"{total_used:,}")
        res2.metric("Total Refilled", f"{total_refilled:,}")
        res3.metric("Net Change", f"{total_refilled - total_used:,}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Data Tables", "Visualizations", "Reserve System", "Downloads"])

    with tab1:
        with st.expander("Generated Demand", expanded=True):
            st.dataframe(demand_df, width='stretch')
            
        with st.expander("Production based on MOQ", expanded=True):
            st.dataframe(prod_df, width='stretch')
        
        with st.expander("Shipments", expanded=False):
            st.dataframe(ship_df, width='stretch')
        
        with st.expander("Inventory by DC", expanded=False):
            st.dataframe(inv_df, width='stretch')
        
        with st.expander("Fulfillment Summary", expanded=False):
            st.dataframe(ful_df, width='stretch')

    with tab2:
        st.subheader("Demand vs Fulfilled")
        col1, col2 = st.columns(2)
        with col1:
            if len(fig_paths) > 0 and os.path.exists(fig_paths[0]):
                st.image(fig_paths[0], caption="Regular SKU", width='stretch')
        with col2:
            if len(fig_paths) > 1 and os.path.exists(fig_paths[1]):
                st.image(fig_paths[1], caption="Diet SKU", width='stretch')

        # Put Shipments and Inventory side by side
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Shipments by Lane")
            if len(fig_paths) > 2 and os.path.exists(fig_paths[2]):
                st.image(fig_paths[2], width='stretch')
        
        with col4:
            st.subheader("Ending Inventory vs Safety Stock")
            if len(fig_paths) > 3 and os.path.exists(fig_paths[3]):
                st.image(fig_paths[3], width='stretch')

    with tab3:
        st.subheader("Reserve System Activity")
        
        if not reserve_df.empty:
            st.dataframe(reserve_df, width='stretch')
            
            # Show reserve chart
            if len(fig_paths) > 4 and os.path.exists(fig_paths[4]):
                st.image(fig_paths[4], caption="Reserve System Status", width='stretch')
        else:
            st.info("No reserve system activity recorded - demand was fully met by production capacity.")

    with tab4:
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        files = [
            ("production_plan.csv", "Production Plan"),
            ("shipments.csv", "Shipments"),
            ("inventory_by_dc.csv", "Inventory by DC"),
            ("fulfillment_summary.csv", "Fulfillment Summary"),
            ("reserve_transactions.csv", "Reserve Transactions")
        ]
        
        for idx, (file, desc) in enumerate(files):
            path = os.path.join("output", file)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    col = col1 if idx % 2 == 0 else col2
                    with col:
                        st.download_button(
                            label=f"Download {desc}",
                            data=f,
                            file_name=file,
                            mime="text/csv",
                            width='stretch',
                            key=f"download_{file}"  # Unique key to prevent conflicts
                        )

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with love using Streamlit
</div>
""", unsafe_allow_html=True)
