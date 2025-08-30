import streamlit as st
import pandas as pd
from planner import run_planner, default_demand_df
import os

st.set_page_config(page_title="Cola Company Planner", layout="wide", page_icon="🥤")

st.title("Cola Company – Production & Deployment Planner")

# Initialize session state
if 'planning_results' not in st.session_state:
    st.session_state.planning_results = None

with st.sidebar:
    st.header("⚙️ Planning Parameters")
    capacity = st.number_input("Weekly Plant Capacity", value=150000, min_value=0, help="Total bottles the plant can produce per week")
    truck = st.number_input("Truck Size (bottles)", value=10000, min_value=0, help="Number of bottles per truck shipment")
    safety = st.number_input("Safety Stock / DC / SKU", value=5000, min_value=0, help="Minimum inventory to maintain at each DC per SKU")

    st.header("📊 Data Input")
    demand_file = st.file_uploader("Demand CSV (optional)", type="csv", help="Upload custom demand data. If not provided, uses default data.")
    # init_file = st.file_uploader("Initial Inventory CSV (optional)", type="csv")

if st.button("🚀 Run Planning", type="primary", use_container_width=True):
    with st.spinner("🔄 Generating production plan..."):
        if demand_file is not None:
            demand_df = pd.read_csv(demand_file)
        else:
            demand_df = default_demand_df()

        prod_df, ship_df, inv_df, ful_df, fig_paths = run_planner(capacity, truck, safety, demand_df)
        
        # Store results in session state
        st.session_state.planning_results = {
            'prod_df': prod_df,
            'ship_df': ship_df, 
            'inv_df': inv_df,
            'ful_df': ful_df,
            'fig_paths': fig_paths,
            'capacity': capacity,
            'truck': truck,
            'safety': safety
        }

    st.success("✅ Plan generated successfully!")

# Display results if available
if st.session_state.planning_results is not None:
    results = st.session_state.planning_results
    prod_df = results['prod_df']
    ship_df = results['ship_df']
    inv_df = results['inv_df']
    ful_df = results['ful_df']
    fig_paths = results['fig_paths']
    
    # Display key metrics
    total_weekly_capacity = results['capacity'] * prod_df.shape[0]
    total_produced = prod_df['Regular'].sum() + prod_df['Diet'].sum()
    utilization = total_produced / total_weekly_capacity * 100
    total_shipped = ship_df['Qty'].sum()
    avg_fulfillment = (ful_df['Fulfilled'].sum() / ful_df['Demand'].sum()) * 100 if ful_df['Demand'].sum() > 0 else 0

    st.markdown("### 📈 Key Performance Indicators")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("🏭 Capacity Utilization", f"{utilization:.1f}%", delta=f"of {results['capacity']:,}/week")
    kpi2.metric("🚚 Total Shipped", f"{total_shipped:,}", delta=f"over {prod_df.shape[0]} weeks")
    kpi3.metric("🎯 Avg. Fulfillment Rate", f"{avg_fulfillment:.1f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Data Tables", "📊 Visualizations", "💾 Downloads"])

    with tab1:
        with st.expander("🏭 Production Plan", expanded=True):
            st.dataframe(prod_df, use_container_width=True)
        
        with st.expander("🚚 Shipments", expanded=False):
            st.dataframe(ship_df, use_container_width=True)
        
        with st.expander("📦 Inventory by DC", expanded=False):
            st.dataframe(inv_df, use_container_width=True)
        
        with st.expander("📊 Fulfillment Summary", expanded=False):
            st.dataframe(ful_df, use_container_width=True)

    with tab2:
        st.subheader("🎯 Demand vs Fulfilled")
        col1, col2 = st.columns(2)
        with col1:
            if len(fig_paths) > 0 and os.path.exists(fig_paths[0]):
                st.image(fig_paths[0], caption="Regular SKU", use_container_width=True)
        with col2:
            if len(fig_paths) > 1 and os.path.exists(fig_paths[1]):
                st.image(fig_paths[1], caption="Diet SKU", use_container_width=True)

        # Put Shipments and Inventory side by side
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🚚 Shipments by Lane")
            if len(fig_paths) > 2 and os.path.exists(fig_paths[2]):
                st.image(fig_paths[2], use_container_width=True)
        
        with col4:
            st.subheader("📦 Ending Inventory vs Safety Stock")
            if len(fig_paths) > 3 and os.path.exists(fig_paths[3]):
                st.image(fig_paths[3], use_container_width=True)

    with tab3:
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        files = [
            ("production_plan.csv", "🏭 Production Plan"),
            ("shipments.csv", "🚚 Shipments"),
            ("inventory_by_dc.csv", "📦 Inventory by DC"),
            ("fulfillment_summary.csv", "📊 Fulfillment Summary")
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
                            use_container_width=True,
                            key=f"download_{file}"  # Unique key to prevent conflicts
                        )

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
