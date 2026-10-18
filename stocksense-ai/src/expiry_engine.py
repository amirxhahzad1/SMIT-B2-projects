"""
Deterministic Expiry Risk Engine for StockSense AI.
Calculates days until expiry, demand-backed sell-through rates, and FEFO (First-Expired, First-Out)
depletion projections to flag expired, critical, warning, and review-needed batches.
"""

from datetime import date, datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.schemas import BatchExpiryRisk, RiskCategoryType
from src.config import config


def calculate_sku_daily_demand(sales_df: pd.DataFrame, window_days: int = 30, reference_date: Optional[date] = None) -> Dict[str, float]:
    """
    Calculate average daily sales per SKU over a specified lookback window.
    Fills missing dates with zero sales so average demand is not artificially inflated.
    
    Returns:
        Dict mapping sku -> avg_daily_demand
    """
    if sales_df.empty:
        return {}

    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    ref_date = reference_date or df["date"].max()
    min_date = ref_date - pd.Timedelta(days=window_days - 1)

    # Filter sales within the lookback window
    window_sales = df[(df["date"] >= min_date) & (df["date"] <= ref_date)]

    sku_demand: Dict[str, float] = {}
    
    # Calculate across all unique SKUs in the sales dataset
    all_skus = df["sku"].unique()
    for sku in all_skus:
        sku_records = window_sales[window_sales["sku"] == sku]
        total_units = sku_records["units_sold"].sum()
        # Divide by window_days (not just days with sales) to avoid zero-day bias
        avg_demand = float(total_units) / float(window_days)
        sku_demand[sku] = round(avg_demand, 3)

    return sku_demand


def evaluate_batch_expiry(
    inventory_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[date] = None,
    critical_threshold_days: Optional[int] = None,
    warning_threshold_days: Optional[int] = None
) -> List[BatchExpiryRisk]:
    """
    Evaluate expiry risks for each inventory batch using First-Expired, First-Out (FEFO) demand allocation.
    
    Rules:
    - Expired: days_until_expiry < 0
    - Critical: 0 <= days_until_expiry <= critical_threshold_days (default 7)
    - Warning: critical_threshold_days < days_until_expiry <= warning_threshold_days (default 30)
    - Normal: days_until_expiry > warning_threshold_days AND projected to sell before expiry
    - Elevated Risk: If days_until_expiry > warning_threshold_days but demand is too low to sell
      before expiry, category is elevated to 'Warning' or 'Critical' based on remaining unsold units.
    - Review Needed: If demand is zero or no sales history exists.
    
    Assumption:
    - Expiry date is the final valid selling date.
    
    Returns:
        List of BatchExpiryRisk objects sorted by urgency.
    """
    if inventory_df.empty:
        return []

    crit_thresh = critical_threshold_days if critical_threshold_days is not None else config.CRITICAL_EXPIRY_DAYS
    warn_thresh = warning_threshold_days if warning_threshold_days is not None else config.WARNING_EXPIRY_DAYS

    # Determine reference date (defaults to today or max sales date if available)
    if reference_date is None:
        if sales_df is not None and not sales_df.empty:
            reference_date = pd.to_datetime(sales_df["date"]).dt.date.max()
        else:
            reference_date = date.today()

    # Calculate 30-day demand baseline
    sku_demand_map = calculate_sku_daily_demand(sales_df, window_days=30, reference_date=reference_date) if sales_df is not None else {}

    results: List[BatchExpiryRisk] = []

    # Sort inventory by SKU and expiry date ascending (FEFO queue)
    inv = inventory_df.copy()
    inv["expiry_date_dt"] = pd.to_datetime(inv["expiry_date"]).dt.date
    inv = inv.sort_values(by=["sku", "expiry_date_dt", "quantity"], ascending=[True, True, True])

    # Group by SKU to simulate sequential depletion without double counting
    for sku, group in inv.groupby("sku"):
        avg_demand = sku_demand_map.get(str(sku), 0.0)
        cumulative_consumed_stock = 0.0

        for _, row in group.iterrows():
            batch_num = str(row["batch_number"])
            prod_name = str(row["product_name"])
            warehouse = str(row["warehouse"])
            qty = int(row["quantity"])
            batch_exp = row["expiry_date_dt"]

            days_until_exp = (batch_exp - reference_date).days

            # Case 1: Already expired
            if days_until_exp < 0:
                results.append(BatchExpiryRisk(
                    sku=sku,
                    product_name=prod_name,
                    warehouse=warehouse,
                    batch_number=batch_num,
                    quantity=qty,
                    expiry_date=batch_exp.isoformat(),
                    days_until_expiry=days_until_exp,
                    avg_daily_demand=avg_demand,
                    allocated_demand=0.0,
                    days_to_sell_batch=None,
                    projected_remaining_at_expiry=float(qty),
                    risk_category="Expired",
                    explanation=f"Batch expired {abs(days_until_exp)} day(s) ago on {batch_exp}. Quarantine and remove from active sales immediately."
                ))
                continue

            # Case 2: Zero or unknown demand
            if avg_demand <= 0.0:
                risk_cat: RiskCategoryType = "Critical" if days_until_exp <= crit_thresh else ("Warning" if days_until_exp <= warn_thresh else "Review Needed")
                results.append(BatchExpiryRisk(
                    sku=sku,
                    product_name=prod_name,
                    warehouse=warehouse,
                    batch_number=batch_num,
                    quantity=qty,
                    expiry_date=batch_exp.isoformat(),
                    days_until_expiry=days_until_exp,
                    avg_daily_demand=0.0,
                    allocated_demand=0.0,
                    days_to_sell_batch=None,
                    projected_remaining_at_expiry=float(qty),
                    risk_category=risk_cat,
                    explanation=f"Zero historical demand detected for this SKU. {qty} units at risk of expiring in {days_until_exp} days. Flagged for commercial review / promo."
                ))
                continue

            # Case 3: Positive demand with FEFO sequential depletion
            # Time required to sell earlier batches in queue:
            time_to_reach_batch = cumulative_consumed_stock / avg_demand
            # Time to sell this batch:
            time_to_sell_this = qty / avg_demand
            total_time_to_deplete = time_to_reach_batch + time_to_sell_this

            # Total units of this SKU that can be sold between now and batch expiry:
            total_sellable_capacity = max(0.0, days_until_exp * avg_demand)
            # Units allocated to earlier batches:
            earlier_absorbed = min(cumulative_consumed_stock, total_sellable_capacity)
            remaining_capacity_for_batch = max(0.0, total_sellable_capacity - earlier_absorbed)

            projected_sold = min(float(qty), remaining_capacity_for_batch)
            projected_remaining = round(max(0.0, float(qty) - projected_sold), 1)

            # Update cumulative stock tracker for subsequent batches
            cumulative_consumed_stock += qty

            # Determine category and explanation
            if days_until_exp <= crit_thresh:
                risk_cat = "Critical"
                exp_msg = f"Critical expiry: Expires in {days_until_exp} day(s). Projected remaining unsold: {projected_remaining:.0f} units."
            elif days_until_exp <= warn_thresh:
                risk_cat = "Warning"
                exp_msg = f"Warning: Expires in {days_until_exp} day(s). Projected remaining unsold: {projected_remaining:.0f} units."
            elif projected_remaining > 0:
                # Expiring after warning threshold, but sell-through is too slow to absorb it before expiry!
                risk_cat = "Warning"
                exp_msg = f"Sell-through deficit: Batch expires in {days_until_exp} days, but estimated depletion takes {total_time_to_deplete:.1f} days. Estimated {projected_remaining:.0f} units will expire unsold."
            else:
                risk_cat = "Normal"
                exp_msg = f"Healthy batch: Expires in {days_until_exp} days. Demand of {avg_demand:.1f} units/day is sufficient to fully sell through before expiry."

            results.append(BatchExpiryRisk(
                sku=sku,
                product_name=prod_name,
                warehouse=warehouse,
                batch_number=batch_num,
                quantity=qty,
                expiry_date=batch_exp.isoformat(),
                days_until_expiry=days_until_exp,
                avg_daily_demand=round(avg_demand, 2),
                allocated_demand=round(projected_sold, 1),
                days_to_sell_batch=round(total_time_to_deplete, 1),
                projected_remaining_at_expiry=projected_remaining,
                risk_category=risk_cat,
                explanation=exp_msg
            ))

    # Priority sorting: Expired first, then Critical, then Warning, Review Needed, then Normal
    priority_map = {"Expired": 0, "Critical": 1, "Warning": 2, "Review Needed": 3, "Normal": 4}
    results.sort(key=lambda x: (priority_map.get(x.risk_category, 99), x.days_until_expiry))

    return results


def get_expiry_risks_dataframe(risks: List[BatchExpiryRisk]) -> pd.DataFrame:
    """Convert a list of BatchExpiryRisk models into a display-ready pandas DataFrame."""
    if not risks:
        return pd.DataFrame()
    records = [r.model_dump() for r in risks]
    return pd.DataFrame(records)
