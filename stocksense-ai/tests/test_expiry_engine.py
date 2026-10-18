"""
Tests for Expiry Risk Engine in StockSense AI.
Verifies FEFO allocation, expired batch classification, critical/warning thresholds,
and zero-demand handling using deterministic reference dates.
"""

import pytest
import pandas as pd
from datetime import date

from src.expiry_engine import evaluate_batch_expiry, calculate_sku_daily_demand


@pytest.fixture
def sample_inventory():
    return pd.DataFrame([
        {
            "sku": "SKU-BEV",
            "product_name": "Test Drink",
            "warehouse": "WH-1",
            "batch_number": "B-EXPIRED",
            "quantity": 30,
            "expiry_date": "2026-10-10", # -8 days (Expired)
            "supplier_lead_time_days": 5,
            "unit_cost": 2.0
        },
        {
            "sku": "SKU-BEV",
            "product_name": "Test Drink",
            "warehouse": "WH-1",
            "batch_number": "B-CRITICAL",
            "quantity": 50,
            "expiry_date": "2026-10-22", # +4 days (Critical)
            "supplier_lead_time_days": 5,
            "unit_cost": 2.0
        },
        {
            "sku": "SKU-BEV",
            "product_name": "Test Drink",
            "warehouse": "WH-1",
            "batch_number": "B-WARNING",
            "quantity": 100,
            "expiry_date": "2026-11-05", # +18 days (Warning)
            "supplier_lead_time_days": 5,
            "unit_cost": 2.0
        },
        {
            "sku": "SKU-BEV",
            "product_name": "Test Drink",
            "warehouse": "WH-1",
            "batch_number": "B-NORMAL",
            "quantity": 200,
            "expiry_date": "2027-02-15", # +120 days (Normal)
            "supplier_lead_time_days": 5,
            "unit_cost": 2.0
        },
        {
            "sku": "SKU-ZERO",
            "product_name": "Zero Demand Item",
            "warehouse": "WH-2",
            "batch_number": "B-ZERO-DEMAND",
            "quantity": 75,
            "expiry_date": "2026-12-15", # +58 days, but 0 demand
            "supplier_lead_time_days": 10,
            "unit_cost": 5.0
        }
    ])


@pytest.fixture
def sample_sales():
    # 30 days of 10 units/day for SKU-BEV = 300 units total (10 units/day)
    # SKU-ZERO has no sales
    records = []
    base_date = date(2026, 10, 18)
    for i in range(30):
        d = date.fromordinal(base_date.toordinal() - i)
        records.append({"date": d.isoformat(), "sku": "SKU-BEV", "units_sold": 10})
    return pd.DataFrame(records)


def test_calculate_sku_daily_demand(sample_sales):
    ref_date = date(2026, 10, 18)
    demand = calculate_sku_daily_demand(sample_sales, window_days=30, reference_date=ref_date)
    assert demand["SKU-BEV"] == 10.0


def test_evaluate_batch_expiry_categories(sample_inventory, sample_sales):
    ref_date = date(2026, 10, 18)
    results = evaluate_batch_expiry(
        sample_inventory,
        sample_sales,
        reference_date=ref_date,
        critical_threshold_days=7,
        warning_threshold_days=30
    )

    batch_map = {r.batch_number: r for r in results}

    # B-EXPIRED must be classified as Expired
    assert batch_map["B-EXPIRED"].risk_category == "Expired"
    assert batch_map["B-EXPIRED"].days_until_expiry < 0

    # B-CRITICAL (+4 days) must be classified as Critical
    assert batch_map["B-CRITICAL"].risk_category == "Critical"
    assert batch_map["B-CRITICAL"].days_until_expiry == 4

    # B-WARNING (+18 days) must be classified as Warning
    assert batch_map["B-WARNING"].risk_category == "Warning"
    assert batch_map["B-WARNING"].days_until_expiry == 18

    # B-NORMAL (+120 days with 10 units/day demand) must be Normal
    assert batch_map["B-NORMAL"].risk_category == "Normal"

    # B-ZERO-DEMAND has zero sales, so it must be flagged for review
    assert batch_map["B-ZERO-DEMAND"].risk_category == "Review Needed"
    assert batch_map["B-ZERO-DEMAND"].avg_daily_demand == 0.0


def test_fefo_avoids_double_counting():
    """Verify earlier batches consume demand before later batches."""
    ref_date = date(2026, 10, 18)
    # 2 batches of 100 units. Demand is 10 units/day.
    # Batch 1 expires in 5 days (capacity 50 units sold -> 50 remaining)
    # Batch 2 expires in 12 days
    inv = pd.DataFrame([
        {
            "sku": "SKU-X",
            "product_name": "Test X",
            "warehouse": "WH-1",
            "batch_number": "BATCH-1",
            "quantity": 100,
            "expiry_date": "2026-10-23", # 5 days
            "supplier_lead_time_days": 7,
            "unit_cost": 1.0
        },
        {
            "sku": "SKU-X",
            "product_name": "Test X",
            "warehouse": "WH-1",
            "batch_number": "BATCH-2",
            "quantity": 100,
            "expiry_date": "2026-10-30", # 12 days
            "supplier_lead_time_days": 7,
            "unit_cost": 1.0
        }
    ])
    # Sales: 10 units/day over 30 days
    sales = pd.DataFrame([
        {"date": date.fromordinal(ref_date.toordinal() - i).isoformat(), "sku": "SKU-X", "units_sold": 10}
        for i in range(30)
    ])

    results = evaluate_batch_expiry(inv, sales, reference_date=ref_date)
    b1 = next(r for r in results if r.batch_number == "BATCH-1")
    b2 = next(r for r in results if r.batch_number == "BATCH-2")

    # B1 can only sell 5 * 10 = 50 units before expiry; 50 units remaining
    assert b1.projected_remaining_at_expiry == 50.0
    # B2 cannot sell before B1 is depleted. In 12 days, total capacity is 120 units.
    # First 100 units go to B1. That leaves only 20 units capacity for B2.
    # B2 has 100 units, so 80 units remain unsold.
    assert b2.projected_remaining_at_expiry == 80.0
