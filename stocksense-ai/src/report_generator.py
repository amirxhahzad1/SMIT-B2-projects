"""
Report generation engine for StockSense AI.
Exports comprehensive inventory intelligence, expiry assessments, demand forecasts,
and procurement reorder schedules into multi-tab Excel workbooks or CSV files.
"""

from datetime import date
from io import BytesIO
from typing import Optional, Dict, Any, List
import pandas as pd

from src.expiry_engine import evaluate_batch_expiry, get_expiry_risks_dataframe
from src.forecasting_engine import calculate_demand_forecasts, get_forecast_dataframe
from src.inventory_engine import (
    calculate_inventory_summary,
    identify_slow_moving_products,
    get_prioritized_reorder_list
)


def generate_excel_report(
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    reference_date: Optional[date] = None
) -> bytes:
    """
    Generate an executive multi-sheet Excel (.xlsx) workbook containing:
    1. Executive Summary
    2. Expiry Risk Analysis
    3. Demand Forecasts & Coverage
    4. Procurement Reorder Schedule
    5. Slow-Moving & Dead Stock
    
    Returns:
        bytes of the Excel file ready for download.
    """
    output = BytesIO()

    # 1. Summary Sheet
    summary = calculate_inventory_summary(inventory_df, sales_df, reference_date=reference_date)
    summary_data = [
        {"Metric": "Report Reference Date", "Value": str(summary.reference_date)},
        {"Metric": "Total Inventory Units", "Value": summary.total_units},
        {"Metric": "Total Unique SKUs", "Value": summary.total_skus},
        {"Metric": "Total Batches / Lots", "Value": summary.total_batches},
        {"Metric": "Total Warehouses", "Value": summary.total_warehouses},
        {"Metric": "Total Inventory Valuation ($)", "Value": f"${summary.total_inventory_value:,.2f}"},
        {"Metric": "Expired Batches Count", "Value": summary.expired_batches_count},
        {"Metric": "Critical Expiry Batches (<= 7 days)", "Value": summary.critical_batches_count},
        {"Metric": "Warning Expiry Batches (8-30 days)", "Value": summary.warning_batches_count},
        {"Metric": "Batches Requiring Review (0 Demand)", "Value": summary.review_needed_batches_count},
        {"Metric": "SKUs with Stockout Risk", "Value": summary.stockout_risk_skus_count},
        {"Metric": "Slow-Moving / Overstocked SKUs", "Value": summary.slow_moving_skus_count},
    ]
    summary_df = pd.DataFrame(summary_data)

    # 2. Expiry Risks Sheet
    expiry_risks = evaluate_batch_expiry(inventory_df, sales_df, reference_date=reference_date)
    expiry_df = get_expiry_risks_dataframe(expiry_risks)

    # 3. Demand Forecasts Sheet
    forecasts = calculate_demand_forecasts(inventory_df, sales_df, reference_date=reference_date)
    forecast_df = get_forecast_dataframe(forecasts)

    # 4. Reorder Schedule Sheet
    reorders = get_prioritized_reorder_list(inventory_df, sales_df, reference_date=reference_date)
    reorders_df = pd.DataFrame(reorders) if reorders else pd.DataFrame(columns=["Notice"])
    if not reorders:
        reorders_df = pd.DataFrame([{"Notice": "All SKUs are currently sufficiently stocked."}])

    # 5. Slow Movers Sheet
    slow_items = identify_slow_moving_products(inventory_df, sales_df, reference_date=reference_date)
    slow_df = pd.DataFrame([s.model_dump() for s in slow_items]) if slow_items else pd.DataFrame(columns=["Notice"])
    if not slow_items:
        slow_df = pd.DataFrame([{"Notice": "No excessive or stagnant stock detected."}])

    # Write to Excel with openpyxl
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
        if not expiry_df.empty:
            expiry_df.to_excel(writer, sheet_name="Expiry Risks", index=False)
        if not forecast_df.empty:
            forecast_df.to_excel(writer, sheet_name="Demand Forecasts", index=False)
        reorders_df.to_excel(writer, sheet_name="Reorder Schedule", index=False)
        slow_df.to_excel(writer, sheet_name="Slow Moving Stock", index=False)

    output.seek(0)
    return output.getvalue()


def generate_csv_report(
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    reference_date: Optional[date] = None
) -> str:
    """
    Generate a consolidated CSV report for quick export and spreadsheet viewing.
    """
    expiry_risks = evaluate_batch_expiry(inventory_df, sales_df, reference_date=reference_date)
    expiry_df = get_expiry_risks_dataframe(expiry_risks)
    return expiry_df.to_csv(index=False)
