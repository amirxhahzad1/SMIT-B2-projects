"""
Structured tools for StockSense AI's LangGraph Agent.
Provides 5 validated inventory intelligence tools:
1. get_inventory_summary
2. get_expiry_risks
3. get_slow_moving_products
4. get_demand_forecast
5. get_reorder_recommendations
"""

from typing import Dict, Any, List, Optional
from datetime import date
import pandas as pd
from langchain_core.tools import tool

from src.data_loader import load_demo_data
from src.inventory_engine import (
    calculate_inventory_summary,
    identify_slow_moving_products,
    get_prioritized_reorder_list
)
from src.expiry_engine import evaluate_batch_expiry
from src.forecasting_engine import calculate_demand_forecasts


# Global/Session dataset registry for tool execution
class ActiveDatasetManager:
    _inventory_df: Optional[pd.DataFrame] = None
    _sales_df: Optional[pd.DataFrame] = None
    _reference_date: Optional[date] = None

    @classmethod
    def set_datasets(cls, inv_df: pd.DataFrame, sales_df: pd.DataFrame, ref_date: Optional[date] = None):
        cls._inventory_df = inv_df.copy()
        cls._sales_df = sales_df.copy()
        cls._reference_date = ref_date

    @classmethod
    def get_datasets(cls) -> tuple[pd.DataFrame, pd.DataFrame, Optional[date]]:
        if cls._inventory_df is None or cls._sales_df is None:
            # Fallback to demo datasets automatically if not initialized
            inv_df, sales_df = load_demo_data()
            cls._inventory_df = inv_df
            cls._sales_df = sales_df
        return cls._inventory_df, cls._sales_df, cls._reference_date


def set_active_dataset(inv_df: pd.DataFrame, sales_df: pd.DataFrame, ref_date: Optional[date] = None):
    """Set the active dataset context for agent tool queries."""
    ActiveDatasetManager.set_datasets(inv_df, sales_df, ref_date)


def get_active_dataset() -> tuple[pd.DataFrame, pd.DataFrame, Optional[date]]:
    """Retrieve currently loaded inventory, sales dataframes, and reference date."""
    return ActiveDatasetManager.get_datasets()


# -------------------------------------------------------------
# Tool 1: get_inventory_summary
# -------------------------------------------------------------
@tool
def get_inventory_summary() -> Dict[str, Any]:
    """
    Get a high-level executive summary of current inventory health across all warehouses.
    Returns total stock count, unique SKUs, total valuation, expired batch count,
    critical expiry count, stockout risks, and slow-moving items count.
    """
    inv_df, sales_df, ref_date = get_active_dataset()
    summary = calculate_inventory_summary(inv_df, sales_df, reference_date=ref_date)
    return summary.model_dump()


# -------------------------------------------------------------
# Tool 2: get_expiry_risks
# -------------------------------------------------------------
@tool
def get_expiry_risks(
    risk_category: Optional[str] = None,
    warehouse: Optional[str] = None,
    sku: Optional[str] = None,
    limit: int = 15
) -> Dict[str, Any]:
    """
    Find batches at risk of expiring, including Expired, Critical (<= 7 days),
    Warning (8-30 days), or Review Needed batches.
    
    Args:
        risk_category: Filter by category ('Expired', 'Critical', 'Warning', 'Review Needed', 'Normal').
        warehouse: Optional warehouse filter (e.g. 'WH-North').
        sku: Optional specific SKU filter.
        limit: Max number of batches to return (default 15).
    """
    inv_df, sales_df, ref_date = get_active_dataset()
    risks = evaluate_batch_expiry(inv_df, sales_df, reference_date=ref_date)

    filtered = risks
    if risk_category:
        rc_clean = risk_category.strip().title()
        filtered = [r for r in filtered if r.risk_category.title() == rc_clean]
    if warehouse:
        wh_clean = warehouse.strip().lower()
        filtered = [r for r in filtered if r.warehouse.lower() == wh_clean]
    if sku:
        sku_clean = sku.strip().upper()
        filtered = [r for r in filtered if r.sku.upper() == sku_clean]

    total_matched = len(filtered)
    trimmed = filtered[:limit]

    return {
        "total_at_risk_batches_found": total_matched,
        "returned_count": len(trimmed),
        "batches": [b.model_dump() for b in trimmed]
    }


# -------------------------------------------------------------
# Tool 3: get_slow_moving_products
# -------------------------------------------------------------
@tool
def get_slow_moving_products(limit: int = 10) -> Dict[str, Any]:
    """
    Identify slow-moving products, dormant SKUs, and dead stock with excessive coverage or zero sales.
    Returns list of items with tied up capital and actionable recommendations.
    
    Args:
        limit: Maximum number of slow-moving products to return (default 10).
    """
    inv_df, sales_df, ref_date = get_active_dataset()
    slow_items = identify_slow_moving_products(inv_df, sales_df, reference_date=ref_date)

    trimmed = slow_items[:limit]
    total_tied_capital = sum(item.tied_up_capital for item in slow_items)

    return {
        "total_slow_moving_skus": len(slow_items),
        "total_tied_up_capital": round(total_tied_capital, 2),
        "items": [item.model_dump() for item in trimmed]
    }


# -------------------------------------------------------------
# Tool 4: get_demand_forecast
# -------------------------------------------------------------
@tool
def get_demand_forecast(
    sku: Optional[str] = None,
    filter_risk: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get 7-day and 30-day moving-average demand forecasts, stock coverage (days of supply),
    and stockout risks for products.
    
    Args:
        sku: Optional specific product SKU to inspect.
        filter_risk: Filter for 'stockout_7d', 'stockout_30d', or 'overstock'.
        limit: Maximum items to return (default 10).
    """
    inv_df, sales_df, ref_date = get_active_dataset()
    forecasts = calculate_demand_forecasts(inv_df, sales_df, reference_date=ref_date)

    filtered = forecasts
    if sku:
        sku_clean = sku.strip().upper()
        filtered = [f for f in filtered if f.sku.upper() == sku_clean]

    if filter_risk:
        fr = filter_risk.lower().strip()
        if "7" in fr or "immediate" in fr:
            filtered = [f for f in filtered if f.stockout_risk_7d]
        elif "30" in fr:
            filtered = [f for f in filtered if f.stockout_risk_30d]
        elif "overstock" in fr or "slow" in fr:
            filtered = [f for f in filtered if f.overstock_risk]

    total_matched = len(filtered)
    trimmed = filtered[:limit]

    return {
        "total_matched_forecasts": total_matched,
        "returned_count": len(trimmed),
        "forecasts": [f.model_dump() for f in trimmed]
    }


# -------------------------------------------------------------
# Tool 5: get_reorder_recommendations
# -------------------------------------------------------------
@tool
def get_reorder_recommendations(limit: int = 10) -> Dict[str, Any]:
    """
    Get prioritized purchase replenishment recommendations based on lead times,
    safety stock, and projected demand shortfalls.
    
    Args:
        limit: Maximum number of reorder items to return (default 10).
    """
    inv_df, sales_df, ref_date = get_active_dataset()
    reorders = get_prioritized_reorder_list(inv_df, sales_df, reference_date=ref_date)

    total_cost = sum(r["estimated_order_cost"] for r in reorders)
    trimmed = reorders[:limit]

    return {
        "total_skus_needing_reorder": len(reorders),
        "total_estimated_reorder_investment": round(total_cost, 2),
        "recommendations": trimmed
    }


ALL_TOOLS = [
    get_inventory_summary,
    get_expiry_risks,
    get_slow_moving_products,
    get_demand_forecast,
    get_reorder_recommendations
]
