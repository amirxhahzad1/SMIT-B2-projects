"""
Deterministic Demand Forecasting Engine for StockSense AI.
Provides moving-average baseline demand projections (7-day and 30-day),
calculates days of supply, detects stockout and overstock risks,
and generates optimal reorder quantity recommendations.
"""

from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple, Any
import math
import pandas as pd
import numpy as np

from src.schemas import DemandForecastResult
from src.config import config


def calculate_demand_forecasts(
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    safety_stock_days: Optional[int] = None,
    reference_date: Optional[date] = None,
    incoming_inventory_map: Optional[Dict[str, int]] = None
) -> List[DemandForecastResult]:
    """
    Calculate deterministic moving-average demand forecasts and reorder recommendations for all SKUs.
    
    Formulae:
    - 7-Day Average Daily Demand = Total Units Sold (Last 7 Days) / 7
    - 30-Day Average Daily Demand = Total Units Sold (Last 30 Days) / 30
    - Projected 7-Day Demand = 7-Day Avg Daily Demand * 7
    - Projected 30-Day Demand = 30-Day Avg Daily Demand * 30
    - Days of Supply = Available Usable Stock / 30-Day Avg Daily Demand
    - Reorder Target = (Demand during Lead Time) + (Safety Stock Demand)
    - Recommended Reorder Qty = max(0, ceil(Reorder Target - Available Stock - Incoming Stock))
    
    Returns:
        List of DemandForecastResult objects.
    """
    if inventory_df.empty:
        return []

    safety_days = safety_stock_days if safety_stock_days is not None else config.DEFAULT_SAFETY_STOCK_DAYS
    incoming_map = incoming_inventory_map or {}

    # Total usable current stock per SKU (excluding expired batches if evaluated)
    # Group inventory by SKU to get total quantity and supplier lead time
    inv_agg = inventory_df.groupby("sku").agg({
        "product_name": "first",
        "quantity": "sum",
        "supplier_lead_time_days": "max",
        "unit_cost": "mean"
    }).reset_index()

    # Determine reference date
    if reference_date is None:
        if sales_df is not None and not sales_df.empty:
            reference_date = pd.to_datetime(sales_df["date"]).dt.date.max()
        else:
            reference_date = date.today()

    # Prepare sales lookup
    sales = sales_df.copy() if (sales_df is not None and not sales_df.empty) else pd.DataFrame(columns=["date", "sku", "units_sold"])
    if not sales.empty:
        sales["date"] = pd.to_datetime(sales["date"]).dt.date

    results: List[DemandForecastResult] = []

    for _, row in inv_agg.iterrows():
        sku = str(row["sku"])
        prod_name = str(row["product_name"])
        current_stock = int(row["quantity"])
        lead_time = int(row["supplier_lead_time_days"])
        incoming_stock = incoming_map.get(sku, 0)

        # Filter sales for this SKU
        sku_sales = sales[sales["sku"] == sku] if not sales.empty else pd.DataFrame()

        # Check sufficiency of historical data
        if sku_sales.empty:
            results.append(DemandForecastResult(
                sku=sku,
                product_name=prod_name,
                current_stock=current_stock,
                avg_daily_demand_7d=0.0,
                avg_daily_demand_30d=0.0,
                projected_demand_7d=0.0,
                projected_demand_30d=0.0,
                days_of_supply=None,
                stockout_risk_7d=False,
                stockout_risk_30d=False,
                overstock_risk=(current_stock > 0),
                supplier_lead_time_days=lead_time,
                safety_stock_units=0,
                recommended_reorder_qty=0,
                status="No Sales History",
                notes="Zero historical sales found. Review demand or marketing before placing purchase orders."
            ))
            continue

        min_sales_date = sku_sales["date"].min()
        max_sales_date = sku_sales["date"].max()
        history_span_days = (max_sales_date - min_sales_date).days + 1

        is_insufficient = history_span_days < 7

        # 7-day lookback window: [reference_date - 6 days, reference_date]
        start_7d = reference_date - timedelta(days=6)
        sales_7d_subset = sku_sales[(sku_sales["date"] >= start_7d) & (sku_sales["date"] <= reference_date)]
        total_7d_units = sales_7d_subset["units_sold"].sum()
        avg_daily_7d = float(total_7d_units) / 7.0

        # 30-day lookback window: [reference_date - 29 days, reference_date]
        start_30d = reference_date - timedelta(days=29)
        sales_30d_subset = sku_sales[(sku_sales["date"] >= start_30d) & (sku_sales["date"] <= reference_date)]
        total_30d_units = sales_30d_subset["units_sold"].sum()
        avg_daily_30d = float(total_30d_units) / 30.0

        # Projections
        proj_7d = round(avg_daily_7d * 7.0, 1)
        proj_30d = round(avg_daily_30d * 30.0, 1)

        # Baseline demand for replenishment logic: primary is 30d, fallback to 7d
        effective_daily_demand = avg_daily_30d if avg_daily_30d > 0 else avg_daily_7d

        # Days of supply (runout time)
        days_of_supply = round(current_stock / effective_daily_demand, 1) if effective_daily_demand > 0 else None

        # Risk assessments
        stockout_7d = (current_stock < proj_7d)
        stockout_30d = (current_stock < proj_30d)
        overstock = (days_of_supply is not None and days_of_supply > config.SLOW_MOVING_DAYS_THRESHOLD) or (days_of_supply is None and current_stock > 0)

        # Reorder calculation:
        # Lead time demand = effective_daily_demand * lead_time
        # Safety stock demand = effective_daily_demand * safety_days
        # Target stock = lead_time_demand + safety_stock
        # Shortfall = target_stock - (current_stock + incoming_stock)
        lead_time_demand = effective_daily_demand * float(lead_time)
        safety_stock_units = int(math.ceil(effective_daily_demand * float(safety_days)))
        reorder_target = lead_time_demand + safety_stock_units
        net_shortfall = reorder_target - (current_stock + incoming_stock)
        recommended_reorder = max(0, int(math.ceil(net_shortfall)))

        # Status and notes
        if is_insufficient:
            status = "Insufficient Data"
            notes = f"Only {history_span_days} day(s) of sales data available. Baseline may be volatile."
        elif stockout_7d:
            status = "Immediate Stockout Risk (7d)"
            notes = f"Current inventory ({current_stock} units) covers only {days_of_supply:.1f} days. Reorder {recommended_reorder} units immediately."
        elif stockout_30d:
            status = "Upcoming Stockout Risk (30d)"
            notes = f"Inventory covers {days_of_supply:.1f} days against supplier lead time of {lead_time} days. Place reorder soon."
        elif overstock:
            status = "Overstocked / Slow Moving"
            notes = f"Inventory covers {days_of_supply:.1f} days of demand. Exceeds standard buffer threshold."
        else:
            status = "Healthy Balance"
            notes = f"Stock level provides {days_of_supply:.1f} days of coverage. Balanced replenishment status."

        results.append(DemandForecastResult(
            sku=sku,
            product_name=prod_name,
            current_stock=current_stock,
            avg_daily_demand_7d=round(avg_daily_7d, 2),
            avg_daily_demand_30d=round(avg_daily_30d, 2),
            projected_demand_7d=proj_7d,
            projected_demand_30d=proj_30d,
            days_of_supply=days_of_supply,
            stockout_risk_7d=stockout_7d,
            stockout_risk_30d=stockout_30d,
            overstock_risk=overstock,
            supplier_lead_time_days=lead_time,
            safety_stock_units=safety_stock_units,
            recommended_reorder_qty=recommended_reorder,
            status=status,
            notes=notes
        ))

    # Priority sorting: immediate stockouts first, then 30d stockouts, then overstocks, then healthy
    def sort_key(item: DemandForecastResult) -> int:
        if item.stockout_risk_7d:
            return 0
        if item.stockout_risk_30d:
            return 1
        if item.overstock_risk:
            return 2
        return 3

    results.sort(key=sort_key)
    return results


def get_forecast_dataframe(forecasts: List[DemandForecastResult]) -> pd.DataFrame:
    """Convert forecast results list into pandas DataFrame for UI and charting."""
    if not forecasts:
        return pd.DataFrame()
    records = [f.model_dump() for f in forecasts]
    return pd.DataFrame(records)
