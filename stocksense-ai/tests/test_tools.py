"""
Tests for Agent Tools and Orchestration Utilities in StockSense AI.
Verifies all 5 tools return structured, validated dictionaries using fixture data.
Runs completely offline without requiring Google Gemini API keys.
"""

import pytest
import pandas as pd
from datetime import date

from src.tools import (
    set_active_dataset,
    get_inventory_summary,
    get_expiry_risks,
    get_slow_moving_products,
    get_demand_forecast,
    get_reorder_recommendations
)


@pytest.fixture(autouse=True)
def setup_test_context():
    ref_date = date(2026, 10, 18)
    inv = pd.DataFrame([
        {
            "sku": "SKU-A",
            "product_name": "Product Alpha",
            "warehouse": "WH-North",
            "batch_number": "B-A1",
            "quantity": 40,
            "expiry_date": "2026-10-20", # Expiring in 2 days -> Critical
            "supplier_lead_time_days": 7,
            "unit_cost": 10.0
        },
        {
            "sku": "SKU-A",
            "product_name": "Product Alpha",
            "warehouse": "WH-Central",
            "batch_number": "B-A2",
            "quantity": 100,
            "expiry_date": "2026-12-31", # Normal
            "supplier_lead_time_days": 7,
            "unit_cost": 10.0
        },
        {
            "sku": "SKU-B",
            "product_name": "Product Beta (Dead)",
            "warehouse": "WH-South",
            "batch_number": "B-B1",
            "quantity": 250,
            "expiry_date": "2027-06-30",
            "supplier_lead_time_days": 14,
            "unit_cost": 20.0
        }
    ])

    # Sales: 10 units/day for SKU-A; 0 for SKU-B
    sales = pd.DataFrame([
        {"date": date.fromordinal(ref_date.toordinal() - i).isoformat(), "sku": "SKU-A", "units_sold": 10}
        for i in range(30)
    ])

    set_active_dataset(inv, sales, ref_date)


def test_tool_inventory_summary():
    result = get_inventory_summary.invoke({})
    assert result["total_units"] == 390
    assert result["total_skus"] == 2
    assert result["total_batches"] == 3
    assert result["critical_batches_count"] >= 1


def test_tool_expiry_risks():
    result = get_expiry_risks.invoke({"risk_category": "Critical"})
    assert result["total_at_risk_batches_found"] >= 1
    assert result["batches"][0]["batch_number"] == "B-A1"
    assert result["batches"][0]["risk_category"] == "Critical"


def test_tool_slow_moving_products():
    result = get_slow_moving_products.invoke({"limit": 5})
    assert result["total_slow_moving_skus"] >= 1
    # SKU-B has 0 sales and 250 units at $20 = $5,000 tied up capital
    item = next(i for i in result["items"] if i["sku"] == "SKU-B")
    assert item["tied_up_capital"] == 5000.0


def test_tool_demand_forecast():
    result = get_demand_forecast.invoke({"sku": "SKU-A"})
    assert result["total_matched_forecasts"] == 1
    forecast = result["forecasts"][0]
    assert forecast["sku"] == "SKU-A"
    assert forecast["avg_daily_demand_30d"] == 10.0


def test_tool_reorder_recommendations():
    result = get_reorder_recommendations.invoke({"limit": 5})
    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)
