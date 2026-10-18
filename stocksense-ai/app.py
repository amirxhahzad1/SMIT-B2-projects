"""
StockSense AI — Hackathon Edition
Intelligent Inventory & Expiry Risk Co-Pilot
Deterministic Analysis Engine + LangGraph AI Orchestration + Streamlit UI
"""

import sys
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config
from src.data_loader import (
    load_demo_data,
    validate_and_load_inventory,
    validate_and_load_sales,
    DataValidationError
)
from src.inventory_engine import (
    calculate_inventory_summary,
    identify_slow_moving_products,
    get_prioritized_reorder_list
)
from src.expiry_engine import (
    evaluate_batch_expiry,
    get_expiry_risks_dataframe
)
from src.forecasting_engine import (
    calculate_demand_forecasts,
    get_forecast_dataframe
)
from src.tools import set_active_dataset
from src.agent import ask_stocksense_agent
from src.report_generator import generate_excel_report, generate_csv_report

# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="StockSense AI — Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #1E232F;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #2E3648;
        color: #F8FAFC;
    }
    .badge-expired { background-color: #EF4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-critical { background-color: #F97316; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warning { background-color: #F59E0B; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-normal { background-color: #10B981; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-review { background-color: #6B7280; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { height: 48px; white-space: pre-wrap; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "inventory_df" not in st.session_state:
    try:
        inv_df, sales_df = load_demo_data()
        st.session_state["inventory_df"] = inv_df
        st.session_state["sales_df"] = sales_df
        st.session_state["data_source"] = "Demo Synthetic Dataset (Fictional)"
    except Exception as e:
        st.session_state["inventory_df"] = pd.DataFrame()
        st.session_state["sales_df"] = pd.DataFrame()
        st.session_state["data_source"] = "None"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am **StockSense AI**, your inventory intelligence co-pilot. "
                "I use deterministic batch calculation engines and LangGraph tools to analyze expiry risks, "
                "forecast demand, detect slow movers, and optimize reorders.\n\n"
                "**Try asking me:**\n"
                "- *Which products are at risk of expiry?*\n"
                "- *Which SKUs may run out in the next 7 days?*\n"
                "- *What should we reorder first?*\n"
                "- *Show slow-moving products and tied up capital.*"
            )
        }
    ]

# Keep agent active dataset synced with current session data
if not st.session_state["inventory_df"].empty and not st.session_state["sales_df"].empty:
    ref_d = pd.to_datetime(st.session_state["sales_df"]["date"]).dt.date.max() if "date" in st.session_state["sales_df"].columns else date.today()
    set_active_dataset(st.session_state["inventory_df"], st.session_state["sales_df"], ref_d)


# -------------------------------------------------------------
# Sidebar: Configuration & Data Ingestion
# -------------------------------------------------------------
with st.sidebar:
    st.title("📦 StockSense AI")
    st.caption("Hackathon Edition • LangGraph & Gemini")
    st.divider()

    st.subheader("📁 Data Source Selection")
    data_mode = st.radio(
        "Choose Inventory Data Source:",
        ["Use Preloaded Demo Data", "Upload Custom Spreadsheets"],
        index=0 if st.session_state.get("data_source", "").startswith("Demo") else 1
    )

    if data_mode == "Use Preloaded Demo Data":
        if st.button("🔄 Reset / Reload Demo Data", use_container_width=True):
            inv_df, sales_df = load_demo_data()
            st.session_state["inventory_df"] = inv_df
            st.session_state["sales_df"] = sales_df
            st.session_state["data_source"] = "Demo Synthetic Dataset (Fictional)"
            ref_d = pd.to_datetime(sales_df["date"]).dt.date.max()
            set_active_dataset(inv_df, sales_df, ref_d)
            st.success("Loaded synthetic demo datasets!")
            st.rerun()

    else:
        st.markdown("**Upload Inventory Batches (CSV/XLSX)**")
        uploaded_inv = st.file_uploader(
            "Inventory File",
            type=["csv", "xlsx"],
            help="Columns: sku, product_name, warehouse, batch_number, quantity, expiry_date, supplier_lead_time_days, unit_cost"
        )

        st.markdown("**Upload Historical Sales (CSV/XLSX)**")
        uploaded_sales = st.file_uploader(
            "Sales History File",
            type=["csv", "xlsx"],
            help="Columns: date, sku, units_sold"
        )

        if st.button("🚀 Validate & Ingest Files", type="primary", use_container_width=True):
            if uploaded_inv is None or uploaded_sales is None:
                st.error("Please provide both Inventory and Sales files.")
            else:
                try:
                    clean_inv, inv_warn = validate_and_load_inventory(uploaded_inv)
                    clean_sales, sales_warn = validate_and_load_sales(uploaded_sales)

                    st.session_state["inventory_df"] = clean_inv
                    st.session_state["sales_df"] = clean_sales
                    st.session_state["data_source"] = f"Custom Upload ({uploaded_inv.name})"

                    ref_d = pd.to_datetime(clean_sales["date"]).dt.date.max()
                    set_active_dataset(clean_inv, clean_sales, ref_d)

                    if inv_warn:
                        for w in inv_warn:
                            st.warning(w)
                    if sales_warn:
                        for w in sales_warn:
                            st.warning(w)

                    st.success("Successfully validated and loaded custom inventory data!")
                    st.rerun()
                except DataValidationError as dve:
                    st.error(f"Validation Error: {dve}")
                    for err in dve.errors:
                        st.markdown(f"- ❌ {err}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    st.subheader("⚙️ Business Thresholds")
    crit_days = st.number_input("Critical Expiry Threshold (Days)", min_value=1, max_value=30, value=config.CRITICAL_EXPIRY_DAYS)
    warn_days = st.number_input("Warning Expiry Threshold (Days)", min_value=8, max_value=90, value=config.WARNING_EXPIRY_DAYS)
    safety_days = st.number_input("Safety Stock Target (Days)", min_value=1, max_value=30, value=config.DEFAULT_SAFETY_STOCK_DAYS)

    st.divider()
    st.subheader("🤖 AI Engine Status")
    if config.is_gemini_configured():
        st.success(f"Gemini API Connected ({config.GEMINI_MODEL})")
    else:
        st.warning("Gemini API Key missing in `.env`. Deterministic analytics remain 100% active; AI chat will show formatted engine results.")

    st.caption("Active Dataset: " + st.session_state.get("data_source", "None"))


# -------------------------------------------------------------
# Main Application Content
# -------------------------------------------------------------
inv_df = st.session_state["inventory_df"]
sales_df = st.session_state["sales_df"]

if inv_df.empty or sales_df.empty:
    st.info("No active dataset loaded. Please select 'Use Preloaded Demo Data' or upload spreadsheets in the sidebar.")
    st.stop()

# Determine reference evaluation date (defaults to max date in sales history)
reference_date = pd.to_datetime(sales_df["date"]).dt.date.max() if "date" in sales_df.columns else date.today()

# Run Core Deterministic Analytics
summary = calculate_inventory_summary(inv_df, sales_df, reference_date=reference_date)
expiry_risks = evaluate_batch_expiry(
    inv_df, sales_df,
    reference_date=reference_date,
    critical_threshold_days=crit_days,
    warning_threshold_days=warn_days
)
forecasts = calculate_demand_forecasts(
    inv_df, sales_df,
    safety_stock_days=safety_days,
    reference_date=reference_date
)
slow_movers = identify_slow_moving_products(inv_df, sales_df, reference_date=reference_date)
reorder_schedule = get_prioritized_reorder_list(inv_df, sales_df, reference_date=reference_date, safety_stock_days=safety_days)


# -------------------------------------------------------------
# Top KPI Header Cards
# -------------------------------------------------------------
st.title("📦 StockSense AI — Inventory Intelligence")
st.markdown(f"**Baseline Evaluation Date**: `{reference_date}` • **Warehouses**: `{summary.total_warehouses}` • **Dataset**: `{st.session_state.get('data_source', 'Demo')}`")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Total Inventory Units", value=f"{summary.total_units:,}", delta=f"{summary.total_batches} Batches")
with col2:
    st.metric(label="Unique Products (SKUs)", value=f"{summary.total_skus}", delta=f"${summary.total_inventory_value:,.2f} Value")
with col3:
    st.metric(label="Expired Batches", value=f"{summary.expired_batches_count}", delta="Quarantine Required" if summary.expired_batches_count > 0 else "None", delta_color="inverse")
with col4:
    at_risk = summary.critical_batches_count + summary.warning_batches_count
    st.metric(label="Batches at Expiry Risk", value=f"{at_risk}", delta=f"{summary.critical_batches_count} Critical (≤{crit_days}d)", delta_color="inverse")
with col5:
    urgent_reorders = sum(1 for r in reorder_schedule if r["urgency"] == "URGENT")
    st.metric(label="Urgent Stockout Risks", value=f"{urgent_reorders}", delta=f"{len(reorder_schedule)} Total Reorders", delta_color="inverse")

st.markdown("---")

# -------------------------------------------------------------
# Main Navigation Tabs
# -------------------------------------------------------------
tab_overview, tab_expiry, tab_forecast, tab_reorders, tab_slow, tab_chat, tab_export = st.tabs([
    "📊 Executive Overview",
    "⚠️ Expiry Risk Radar",
    "📈 Demand Forecast & Coverage",
    "🛒 Reorder Schedule",
    "🐢 Slow-Moving & Dead Stock",
    "💬 AI Copilot Chat",
    "📥 Export Reports"
])


# -------------------------------------------------------------
# Tab 1: Executive Overview
# -------------------------------------------------------------
with tab_overview:
    st.subheader("Inventory Health & Risk Distribution")
    c1, c2 = st.columns([1, 1])

    with c1:
        # Donut Chart for Expiry Status
        cat_counts = {
            "Expired": summary.expired_batches_count,
            "Critical (≤7d)": summary.critical_batches_count,
            "Warning (8-30d)": summary.warning_batches_count,
            "Review Needed (0 Demand)": summary.review_needed_batches_count,
            "Normal / Healthy": summary.normal_batches_count
        }
        fig_donut = px.pie(
            values=list(cat_counts.values()),
            names=list(cat_counts.keys()),
            title="Batch Expiry Health Breakdown",
            hole=0.45,
            color=list(cat_counts.keys()),
            color_discrete_map={
                "Expired": "#EF4444",
                "Critical (≤7d)": "#F97316",
                "Warning (8-30d)": "#F59E0B",
                "Review Needed (0 Demand)": "#6B7280",
                "Normal / Healthy": "#10B981"
            }
        )
        fig_donut.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Warehouse Stock Breakdown
        wh_summary = inv_df.groupby("warehouse")["quantity"].sum().reset_index()
        fig_bar = px.bar(
            wh_summary,
            x="warehouse",
            y="quantity",
            title="Inventory Units by Warehouse Facility",
            color="warehouse",
            text_auto=True
        )
        fig_bar.update_layout(margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🚨 Top Urgent Action Items")
    actions = []
    if summary.expired_batches_count > 0:
        actions.append(f"🔴 **Quarantine Alert**: {summary.expired_batches_count} batch(es) have passed their expiration date. Immediately halt sales and issue discard orders.")
    if summary.critical_batches_count > 0:
        actions.append(f"🟠 **Critical Expiry**: {summary.critical_batches_count} batch(es) will expire within {crit_days} days. Apply markdown promotions or prioritize fulfillment.")
    if urgent_reorders > 0:
        actions.append(f"⚡ **Immediate Stockout Risk**: {urgent_reorders} SKU(s) have less than 7 days of supply remaining against supplier lead times. Place purchase orders today.")
    if len(slow_movers) > 0:
        actions.append(f"🟡 **Working Capital Tied Up**: ${sum(s.tied_up_capital for s in slow_movers):,.2f} locked in {len(slow_movers)} slow-moving/dormant items.")

    for act in actions:
        st.markdown(act)


# -------------------------------------------------------------
# Tab 2: Expiry Risk Radar
# -------------------------------------------------------------
with tab_expiry:
    st.subheader("⚠️ Batch Expiry & Shelf-Life Radar")
    st.caption("Evaluates each batch using FEFO demand allocation to prevent double-counting demand across lots.")

    # Filter controls
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        cat_filter = st.selectbox(
            "Filter Risk Category:",
            ["All Categories", "Expired", "Critical", "Warning", "Review Needed", "Normal"]
        )
    with fcol2:
        wh_list = ["All Warehouses"] + sorted(list(inv_df["warehouse"].unique()))
        wh_filter = st.selectbox("Filter Warehouse:", wh_list)
    with fcol3:
        sku_query = st.text_input("Search SKU or Product Name:", "")

    # Apply filters
    filtered_risks = expiry_risks
    if cat_filter != "All Categories":
        filtered_risks = [r for r in filtered_risks if r.risk_category == cat_filter]
    if wh_filter != "All Warehouses":
        filtered_risks = [r for r in filtered_risks if r.warehouse == wh_filter]
    if sku_query.strip():
        q_term = sku_query.strip().lower()
        filtered_risks = [r for r in filtered_risks if q_term in r.sku.lower() or q_term in r.product_name.lower()]

    # Render Table
    if filtered_risks:
        table_records = []
        for r in filtered_risks:
            table_records.append({
                "Risk Category": r.risk_category,
                "SKU": r.sku,
                "Product Name": r.product_name,
                "Warehouse": r.warehouse,
                "Batch #": r.batch_number,
                "Quantity": r.quantity,
                "Expiry Date": r.expiry_date,
                "Days to Expiry": r.days_until_expiry,
                "Daily Demand": r.avg_daily_demand,
                "Est. Days to Sell": f"{r.days_to_sell_batch:.1f}" if r.days_to_sell_batch else "N/A (0 Demand)",
                "Projected Expired Units": r.projected_remaining_at_expiry,
                "Reason / Recommendation": r.explanation
            })
        st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)
    else:
        st.info("No batches match the selected filters.")


# -------------------------------------------------------------
# Tab 3: Demand Forecast & Coverage
# -------------------------------------------------------------
with tab_forecast:
    st.subheader("📈 Deterministic Demand Forecast & Stock Runout")
    st.caption("7-day and 30-day moving-average baseline demand calculations with coverage analysis.")

    forecast_df = get_forecast_dataframe(forecasts)
    if not forecast_df.empty:
        # Chart: Days of Supply vs Safety Threshold
        chart_data = forecast_df.copy()
        chart_data["display_days"] = chart_data["days_of_supply"].fillna(120).clip(upper=120)

        fig_cov = px.bar(
            chart_data,
            x="sku",
            y="display_days",
            color="status",
            title="Days of Inventory Supply (Capped at 120 Days for display)",
            labels={"display_days": "Days of Supply", "sku": "Product SKU"},
            hover_data=["product_name", "current_stock", "avg_daily_demand_30d"]
        )
        fig_cov.add_hline(y=7, line_dash="dash", line_color="#EF4444", annotation_text="Immediate Stockout (7d)")
        fig_cov.add_hline(y=30, line_dash="dot", line_color="#F59E0B", annotation_text="Standard Buffer (30d)")
        fig_cov.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_cov, use_container_width=True)

        st.dataframe(
            forecast_df[[
                "sku", "product_name", "current_stock", "avg_daily_demand_7d",
                "avg_daily_demand_30d", "projected_demand_7d", "projected_demand_30d",
                "days_of_supply", "supplier_lead_time_days", "status", "notes"
            ]],
            use_container_width=True,
            hide_index=True
        )


# -------------------------------------------------------------
# Tab 4: Reorder Procurement Schedule
# -------------------------------------------------------------
with tab_reorders:
    st.subheader("🛒 Procurement Reorder Schedule")
    st.caption("Formulated using lead time demand + safety stock target - current inventory.")

    if reorder_schedule:
        total_inv_needed = sum(r["estimated_order_cost"] for r in reorder_schedule)
        st.markdown(f"**Total Estimated Replenishment Capital Needed**: `${total_inv_needed:,.2f}` across **{len(reorder_schedule)} SKUs**")

        reorder_display = pd.DataFrame(reorder_schedule)
        st.dataframe(
            reorder_display[[
                "urgency", "sku", "product_name", "current_stock", "avg_daily_demand_30d",
                "days_of_supply", "supplier_lead_time_days", "safety_stock_units",
                "recommended_reorder_qty", "unit_cost", "estimated_order_cost", "reason"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("🎉 All inventory items are currently well-stocked. No reorders needed.")


# -------------------------------------------------------------
# Tab 5: Slow-Moving & Dead Stock
# -------------------------------------------------------------
with tab_slow:
    st.subheader("🐢 Slow-Moving Products & Working Capital Optimization")
    st.caption("Detects items with >60 days coverage or zero sales. Highlights capital recovery opportunities.")

    if slow_movers:
        total_tied = sum(s.tied_up_capital for s in slow_movers)
        st.metric(label="Total Capital Locked in Slow Inventory", value=f"${total_tied:,.2f}")

        slow_records = [s.model_dump() for s in slow_movers]
        st.dataframe(pd.DataFrame(slow_records), use_container_width=True, hide_index=True)
    else:
        st.success("No stagnant or dead stock detected in the active dataset.")


# -------------------------------------------------------------
# Tab 6: AI Copilot Chat
# -------------------------------------------------------------
with tab_chat:
    st.subheader("💬 StockSense AI Chat Assistant")
    st.caption("Powered by LangGraph Agent Orchestration + Google Gemini LLM")

    # Display Conversation History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick prompt buttons
    st.markdown("**💡 Quick Prompts:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("🚨 Which batches expire soon?", use_container_width=True):
            user_input = "Which batches are at critical risk of expiry?"
    with qc2:
        if st.button("📉 What is running out in 7 days?", use_container_width=True):
            user_input = "Which products have immediate stockout risks in the next 7 days?"
    with qc3:
        if st.button("🛒 What should we reorder first?", use_container_width=True):
            user_input = "What should we reorder first and why?"
    with qc4:
        if st.button("🐢 Show slow-moving items", use_container_width=True):
            user_input = "Identify slow-moving products and tied up capital."

    # Chat Input Box
    prompt = st.chat_input("Ask about stock levels, expiry dates, forecasts, or reorders...")
    user_query = prompt or (locals().get("user_input", None))

    if user_query:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using LangGraph agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing inventory data via LangGraph tools..."):
                answer = ask_stocksense_agent(user_query)
                st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})


# -------------------------------------------------------------
# Tab 7: Export Reports
# -------------------------------------------------------------
with tab_export:
    st.subheader("📥 Export Comprehensive Inventory Intelligence Reports")
    st.markdown("Download full analytical workbooks with all calculated metrics, risk categorizations, and replenishment schedules.")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown("### 📊 Multi-Sheet Excel Workbook (.xlsx)")
        st.write("Contains dedicated tabs for Executive Summary, Expiry Risks, Demand Forecasts, Reorder Schedule, and Slow-Moving Stock.")
        try:
            excel_bytes = generate_excel_report(inv_df, sales_df, reference_date=reference_date)
            st.download_button(
                label="⬇️ Download Full Excel Report",
                data=excel_bytes,
                file_name=f"StockSense_Report_{reference_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")

    with ecol2:
        st.markdown("### 📄 Consolidated Expiry Risk CSV (.csv)")
        st.write("Fast, lightweight CSV export of all batch expiry calculations and sell-through predictions.")
        try:
            csv_str = generate_csv_report(inv_df, sales_df, reference_date=reference_date)
            st.download_button(
                label="⬇️ Download Expiry Risks CSV",
                data=csv_str,
                file_name=f"StockSense_Expiry_Risks_{reference_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"CSV export failed: {e}")
