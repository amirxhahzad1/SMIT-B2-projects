"""
Inventory Intelligence Engine for StockSense AI.
Provides inventory summarization, dead/slow-moving stock identification,
and prioritizes purchase reorder recommendations.
"""

from datetime import date
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from src.schemas import InventorySummary, SlowMovingProduct, DemandForecastResult
from src.expiry_engine import evaluate_batch_expiry
from src.forecasting_engine import calculate_demand_forecasts
from src.config import config


def calculate_inventory_summary(
    inventory_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[date] = None
) -> InventorySummary:
    """
    Generate an executive summary of the inventory portfolio.
    Combines batch-level expiry evaluation and SKU-level demand forecasting.
    """
    if inventory_df.empty:
        return InventorySummary(
            total_units=0,
            total_skus=0,
            total_batches=0,
            total_warehouses=0,
            total_inventory_value=0.0,
            expired_batches_count=0,
            critical_batches_count=0,
            warning_batches_count=0,
            normal_batches_count=0,
            review_needed_batches_count=0,
            stockout_risk_skus_count=0,
            slow_moving_skus_count=0,
            reference_date=(reference_date or date.today()).isoformat()
        )

    # Reference date
    if reference_date is None:
        if sales_df is not None and not sales_df.empty:
            reference_date = pd.to_datetime(sales_df["date"]).dt.date.max()
        else:
            reference_date = date.today()

    total_units = int(inventory_df["quantity"].sum())
    total_skus = int(inventory_df["sku"].nunique())
    total_batches = len(inventory_df)
    total_warehouses = int(inventory_df["warehouse"].nunique())

    # Total valuation
    val_series = inventory_df["quantity"] * inventory_df["unit_cost"]
    total_inventory_value = round(float(val_series.sum()), 2)

    # Batch expiry breakdown
    expiry_risks = evaluate_batch_expiry(inventory_df, sales_df, reference_date=reference_date)
    expired_cnt = sum(1 for r in expiry_risks if r.risk_category == "Expired")
    crit_cnt = sum(1 for r in expiry_risks if r.risk_category == "Critical")
    warn_cnt = sum(1 for r in expiry_risks if r.risk_category == "Warning")
    norm_cnt = sum(1 for r in expiry_risks if r.risk_category == "Normal")
    review_cnt = sum(1 for r in expiry_risks if r.risk_category == "Review Needed")

    # Demand forecasts breakdown
    sales_clean = sales_df if (sales_df is not None and not sales_df.empty) else pd.DataFrame()
    forecasts = calculate_demand_forecasts(inventory_df, sales_clean, reference_date=reference_date)
    stockout_cnt = sum(1 for f in forecasts if (f.stockout_risk_7d or f.stockout_risk_30d))
    slow_cnt = sum(1 for f in forecasts if f.overstock_risk)

    return InventorySummary(
        total_units=total_units,
        total_skus=total_skus,
        total_batches=total_batches,
        total_warehouses=total_warehouses,
        total_inventory_value=total_inventory_value,
        expired_batches_count=expired_cnt,
        critical_batches_count=crit_cnt,
        warning_batches_count=warn_cnt,
        normal_batches_count=norm_cnt,
        review_needed_batches_count=review_cnt,
        stockout_risk_skus_count=stockout_cnt,
        slow_moving_skus_count=slow_cnt,
        reference_date=reference_date.isoformat()
    )


def identify_slow_moving_products(
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    reference_date: Optional[date] = None,
    threshold_days: Optional[int] = None
) -> List[SlowMovingProduct]:
    """
    Detect slow-moving, dormant, and dead-stock inventory.
    Identifies items where stock exceeds healthy days of supply or zero sales occurred.
    Calculates tied-up working capital and recommends corrective commercial actions.
    """
    if inventory_df.empty:
        return []

    cutoff_days = threshold_days if threshold_days is not None else config.SLOW_MOVING_DAYS_THRESHOLD
    forecasts = calculate_demand_forecasts(inventory_df, sales_df, reference_date=reference_date)

    # Cost map per SKU
    sku_costs = inventory_df.groupby("sku")["unit_cost"].mean().to_dict()

    slow_movers: List[SlowMovingProduct] = []

    for f in forecasts:
        cost = float(sku_costs.get(f.sku, 0.0))
        tied_capital = round(f.current_stock * cost, 2)

        is_slow = False
        status_label = ""
        action = ""

        if f.avg_daily_demand_30d == 0.0 and f.current_stock > 0:
            is_slow = True
            status_label = "Dead Stock (0 Sales in 30d)"
            action = "Zero customer demand. Initiate flash sale, liquidation, or vendor return before degradation."
        elif f.days_of_supply is not None and f.days_of_supply > cutoff_days:
            is_slow = True
            status_label = f"Overstocked ({f.days_of_supply:.0f} Days Coverage)"
            action = f"Inventory covers {f.days_of_supply:.0f} days. Halt replenishment orders and bundle with high-velocity items."

        if is_slow:
            slow_movers.append(SlowMovingProduct(
                sku=f.sku,
                product_name=f.product_name,
                total_stock=f.current_stock,
                avg_daily_demand_30d=f.avg_daily_demand_30d,
                days_of_supply=f.days_of_supply,
                unit_cost=cost,
                tied_up_capital=tied_capital,
                status=status_label,
                recommendation=action
            ))

    # Sort descending by tied up capital
    slow_movers.sort(key=lambda x: x.tied_up_capital, reverse=True)
    return slow_movers


def get_prioritized_reorder_list(
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    reference_date: Optional[date] = None,
    safety_stock_days: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate an actionable, prioritized procurement reorder schedule.
    Filters for items needing replenishment and orders them by lead-time runout urgency.
    """
    forecasts = calculate_demand_forecasts(
        inventory_df,
        sales_df,
        safety_stock_days=safety_stock_days,
        reference_date=reference_date
    )

    sku_costs = inventory_df.groupby("sku")["unit_cost"].mean().to_dict()

    reorders: List[Dict[str, Any]] = []
    for f in forecasts:
        if f.recommended_reorder_qty > 0:
            unit_cost = float(sku_costs.get(f.sku, 0.0))
            est_cost = round(f.recommended_reorder_qty * unit_cost, 2)
            urgency = "URGENT" if f.stockout_risk_7d else ("MEDIUM" if f.stockout_risk_30d else "ROUTINE")

            reorders.append({
                "sku": f.sku,
                "product_name": f.product_name,
                "urgency": urgency,
                "current_stock": f.current_stock,
                "avg_daily_demand_30d": f.avg_daily_demand_30d,
                "days_of_supply": f.days_of_supply,
                "supplier_lead_time_days": f.supplier_lead_time_days,
                "safety_stock_units": f.safety_stock_units,
                "recommended_reorder_qty": f.recommended_reorder_qty,
                "unit_cost": unit_cost,
                "estimated_order_cost": est_cost,
                "reason": f.notes
            })

    # Sort: URGENT first, then by days_of_supply ascending
    urgency_prio = {"URGENT": 0, "MEDIUM": 1, "ROUTINE": 2}
    reorders.sort(key=lambda x: (urgency_prio.get(x["urgency"], 9), x["days_of_supply"] if x["days_of_supply"] is not None else 9999))
    return reorders
