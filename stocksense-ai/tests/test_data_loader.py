"""
Tests for Data Loading and Spreadsheet Validation in StockSense AI.
Verifies column normalization, schema checks, negative values, and date parsing.
"""

import io
import pytest
import pandas as pd
from datetime import date

from src.data_loader import (
    validate_and_load_inventory,
    validate_and_load_sales,
    load_demo_data,
    DataValidationError,
    clean_dataframe_columns
)


def test_clean_dataframe_columns():
    """Verify column normalization converts spaces and hyphens to lowercase underscores."""
    df = pd.DataFrame(columns=[" SKU ", "Product-Name", "Batch Number", "UNIT_COST"])
    cleaned = clean_dataframe_columns(df)
    assert list(cleaned.columns) == ["sku", "product_name", "batch_number", "unit_cost"]


def test_valid_inventory_csv_loading():
    """Verify loading a valid inventory CSV with multiple batches."""
    csv_data = """sku,product_name,warehouse,batch_number,quantity,expiry_date,supplier_lead_time_days,unit_cost
SKU-001,Test Product,WH-1,BATCH-A,100,2026-12-31,7,10.50
SKU-001,Test Product,WH-1,BATCH-B,50,2026-11-30,7,10.50
"""
    df, warnings = validate_and_load_inventory(io.StringIO(csv_data))
    assert len(df) == 2
    assert df["quantity"].sum() == 150
    assert df["sku"].iloc[0] == "SKU-001"
    assert df["unit_cost"].iloc[0] == 10.50


def test_missing_required_columns_fails():
    """Verify missing required columns raises DataValidationError."""
    csv_data = """sku,product_name,quantity
SKU-001,Test Product,100
"""
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_load_inventory(io.StringIO(csv_data))
    assert "Missing required inventory columns" in str(exc_info.value)


def test_negative_quantity_fails():
    """Verify negative stock quantities trigger a validation error."""
    csv_data = """sku,product_name,warehouse,batch_number,quantity,expiry_date,supplier_lead_time_days,unit_cost
SKU-001,Test Product,WH-1,BATCH-A,-25,2026-12-31,7,10.50
"""
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_load_inventory(io.StringIO(csv_data))
    assert "negative stock quantity" in str(exc_info.value).lower()


def test_invalid_expiry_date_fails():
    """Verify unparseable expiry dates trigger a validation error."""
    csv_data = """sku,product_name,warehouse,batch_number,quantity,expiry_date,supplier_lead_time_days,unit_cost
SKU-001,Test Product,WH-1,BATCH-A,100,NOT_A_DATE,7,10.50
"""
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_load_inventory(io.StringIO(csv_data))
    assert "invalid or missing dates" in str(exc_info.value).lower()


def test_valid_sales_csv_loading():
    """Verify loading valid sales history data."""
    csv_data = """date,sku,units_sold
2026-10-01,SKU-001,15
2026-10-02,SKU-001,20
"""
    df, warnings = validate_and_load_sales(io.StringIO(csv_data))
    assert len(df) == 2
    assert df["units_sold"].sum() == 35


def test_negative_sales_units_fails():
    """Verify negative sales numbers trigger a validation error."""
    csv_data = """date,sku,units_sold
2026-10-01,SKU-001,-5
"""
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_load_sales(io.StringIO(csv_data))
    assert "negative" in str(exc_info.value).lower()


def test_load_demo_datasets():
    """Verify bundled demo datasets load and validate cleanly."""
    inv_df, sales_df = load_demo_data()
    assert not inv_df.empty
    assert not sales_df.empty
    assert "sku" in inv_df.columns
    assert "units_sold" in sales_df.columns
