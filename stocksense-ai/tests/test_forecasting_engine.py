"""
Tests for Demand Forecasting Engine in StockSense AI.
Verifies moving average projections, stockout risks, safety stock, and reorder logic.
"""

import pytest
import pandas as pd
from datetime import date, timedelta

from src.forecasting_engine import calculate_demand_forecasts


@pytest.fixture
def base_ref_date():
    return date(2026, 10, 18)


@pytest.fixture
def forecasting_inventory():
    return pd.DataFrame([
        {
            "sku": "SKU-FAST",
            "product_name": "Fast Moving Beverage",
            "warehouse": "WH-1",
            "batch_number": "B-01",
            "quantity": 30, # Low stock vs 20/day sales -> Stockout!
            "expiry_date": "2027-01-01",
            "supplier_lead_time_days": 7,
            "unit_cost": 2.50
        },
        {
            "sku": "SKU-NORMAL",
            "product_name": "Balanced Grocery Item",
            "warehouse": "WH-1",
            "batch_number": "B-02",
            "quantity": 150, # 5/day sales = 30 days supply
            "expiry_date": "2027-01-01",
            "supplier_lead_time_days": 7,
            "unit_cost": 5.00
        },
        {
            "sku": "SKU-OVERSTOCK",
            "product_name": "Excessive Inventory Item",
            "warehouse": "WH-1",
            "batch_number": "B-03",
            "quantity": 1000, # 1/day sales = 1000 days supply
            "expiry_date": "2027-01-01",
            "supplier_lead_time_days": 10,
            "unit_cost": 3.00
        },
        {
            "sku": "SKU-NODATA",
            "product_name": "Unsold Item",
            "warehouse": "WH-1",
            "batch_number": "B-04",
            "quantity": 80,
            "expiry_date": "2027-01-01",
            "supplier_lead_time_days": 5,
            "unit_cost": 4.00
        }
    ])


@pytest.fixture
def forecasting_sales(base_ref_date):
    records = []
    # 30 days of sales history
    for i in range(30):
        d = base_ref_date - timedelta(days=i)
        d_str = d.isoformat()
        records.append({"date": d_str, "sku": "SKU-FAST", "units_sold": 20})
        records.append({"date": d_str, "sku": "SKU-NORMAL", "units_sold": 5})
        records.append({"date": d_str, "sku": "SKU-OVERSTOCK", "units_sold": 1})
        # SKU-NODATA has no sales
    return pd.DataFrame(records)


def test_forecast_projections_and_stockout(forecasting_inventory, forecasting_sales, base_ref_date):
    forecasts = calculate_demand_forecasts(
        forecasting_inventory,
        forecasting_sales,
        safety_stock_days=7,
        reference_date=base_ref_date
    )

    f_map = {f.sku: f for f in forecasts}

    # Fast mover: 20 units/day
    fast = f_map["SKU-FAST"]
    assert fast.avg_daily_demand_30d == 20.0
    assert fast.projected_demand_7d == 140.0
    assert fast.projected_demand_30d == 600.0
    assert fast.current_stock == 30
    assert fast.stockout_risk_7d is True
    assert fast.stockout_risk_30d is True
    assert fast.days_of_supply == 1.5 # 30 / 20 = 1.5 days

    # Reorder calculation for SKU-FAST:
    # Lead time = 7 days -> 7 * 20 = 140
    # Safety stock = 7 days -> 7 * 20 = 140
    # Target = 280
    # Current stock = 30 -> Shortfall = 250 units
    assert fast.recommended_reorder_qty == 250


def test_reorder_quantity_never_negative(forecasting_inventory, forecasting_sales, base_ref_date):
    forecasts = calculate_demand_forecasts(
        forecasting_inventory,
        forecasting_sales,
        safety_stock_days=7,
        reference_date=base_ref_date
    )

    f_map = {f.sku: f for f in forecasts}
    overstock = f_map["SKU-OVERSTOCK"]

    # Target is (10 days lead * 1) + (7 days safety * 1) = 17 units.
    # Current stock is 1000 units. Shortfall is -983.
    # Must NEVER be negative; should be 0.
    assert overstock.recommended_reorder_qty == 0
    assert overstock.overstock_risk is True
    assert overstock.days_of_supply == 1000.0


def test_zero_or_insufficient_sales_data(forecasting_inventory, forecasting_sales, base_ref_date):
    forecasts = calculate_demand_forecasts(
        forecasting_inventory,
        forecasting_sales,
        safety_stock_days=7,
        reference_date=base_ref_date
    )

    f_map = {f.sku: f for f in forecasts}
    nodata = f_map["SKU-NODATA"]

    assert nodata.status == "No Sales History"
    assert nodata.avg_daily_demand_30d == 0.0
    assert nodata.recommended_reorder_qty == 0
