"""
Data loading and validation engine for StockSense AI.
Supports CSV and Excel (.xlsx) file uploads, column normalization,
data type coercions, validation against Pydantic schemas, and demo dataset loading.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import io
import pandas as pd

from src.schemas import InventoryItemSchema, SalesRecordSchema
from src.config import DATA_DIR


REQUIRED_INVENTORY_COLUMNS = [
    "sku",
    "product_name",
    "warehouse",
    "batch_number",
    "quantity",
    "expiry_date",
    "supplier_lead_time_days",
    "unit_cost"
]

REQUIRED_SALES_COLUMNS = [
    "date",
    "sku",
    "units_sold"
]


def normalize_column_name(col_name: str) -> str:
    """Normalize raw column name: strip whitespace, lowercase, replace spaces/dashes with underscores."""
    col = str(col_name).strip().lower()
    col = col.replace(" ", "_").replace("-", "_")
    return col


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across an uploaded or loaded DataFrame."""
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


class DataValidationError(Exception):
    """Custom exception raised when uploaded inventory or sales data fails validation."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]

    def __str__(self) -> str:
        if self.errors:
            return f"{super().__str__()} Details: {'; '.join(self.errors)}"
        return super().__str__()


def read_file_to_df(file_source: Any, file_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read uploaded file (BytesIO, file-like object, or file path) into a pandas DataFrame.
    Supports CSV and Excel (.xlsx).
    """
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        return pd.read_csv(path)

    # In-memory Streamlit UploadedFile or BytesIO
    fname = getattr(file_source, "name", file_name or "file.csv").lower()
    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        return pd.read_excel(file_source)
    return pd.read_csv(file_source)


def validate_and_load_inventory(file_source: Any) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and parse inventory data from CSV/Excel.
    
    Returns:
        Tuple of (clean_df, list_of_warning_messages)
    
    Raises:
        DataValidationError: If critical columns are missing or values are invalid.
    """
    df = read_file_to_df(file_source)
    if df.empty:
        raise DataValidationError("Uploaded inventory file is empty.")

    df = clean_dataframe_columns(df)

    # Check for missing required columns
    missing_cols = [c for c in REQUIRED_INVENTORY_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing required inventory columns: {', '.join(missing_cols)}. "
            f"Required columns are: {', '.join(REQUIRED_INVENTORY_COLUMNS)}"
        )

    errors: List[str] = []
    warnings: List[str] = []

    # Clean and coerce types
    df["sku"] = df["sku"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["warehouse"] = df["warehouse"].astype(str).str.strip()
    df["batch_number"] = df["batch_number"].astype(str).str.strip()

    # Empty string checks
    if (df["sku"] == "").any():
        errors.append("Found rows with empty SKU.")
    if (df["batch_number"] == "").any():
        errors.append("Found rows with empty batch number.")

    # Numeric conversions
    try:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    except Exception as e:
        errors.append(f"Could not convert 'quantity' to numeric: {str(e)}")

    try:
        df["supplier_lead_time_days"] = pd.to_numeric(df["supplier_lead_time_days"], errors="coerce").fillna(7)
    except Exception as e:
        df["supplier_lead_time_days"] = 7

    try:
        df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0)
    except Exception as e:
        df["unit_cost"] = 0.0

    # Check for NaN in essential quantities
    if df["quantity"].isna().any():
        errors.append("Found invalid or non-numeric values in 'quantity'.")

    # Negative stock check
    if (df["quantity"] < 0).any():
        neg_count = (df["quantity"] < 0).sum()
        errors.append(f"Found {neg_count} row(s) with negative stock quantity. Quantities must be >= 0.")

    # Negative lead time check
    if (df["supplier_lead_time_days"] < 0).any():
        errors.append("Supplier lead time cannot be negative.")

    # Negative cost check
    if (df["unit_cost"] < 0).any():
        errors.append("Unit cost cannot be negative.")

    # Expiry date parsing
    try:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    except Exception as e:
        errors.append(f"Failed to parse 'expiry_date': {str(e)}")

    if df["expiry_date"].isna().any():
        errors.append("Found invalid or missing dates in 'expiry_date' column. Dates must be formatted as YYYY-MM-DD.")

    if errors:
        raise DataValidationError("Inventory data validation failed.", errors=errors)

    # Cast integer types
    df["quantity"] = df["quantity"].astype(int)
    df["supplier_lead_time_days"] = df["supplier_lead_time_days"].astype(int)
    df["unit_cost"] = df["unit_cost"].astype(float)

    # Check for duplicate batch records within the same warehouse
    dup_mask = df.duplicated(subset=["sku", "warehouse", "batch_number"], keep=False)
    if dup_mask.any():
        warnings.append("Warning: Duplicate records found for the same SKU, Warehouse, and Batch Number. They will be aggregated.")
        df = df.groupby(["sku", "product_name", "warehouse", "batch_number", "expiry_date"], as_index=False).agg({
            "quantity": "sum",
            "supplier_lead_time_days": "max",
            "unit_cost": "mean"
        })

    return df, warnings


def validate_and_load_sales(file_source: Any) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and parse historical sales data from CSV/Excel.
    
    Returns:
        Tuple of (clean_df, list_of_warning_messages)
    
    Raises:
        DataValidationError: If critical columns are missing or values are invalid.
    """
    df = read_file_to_df(file_source)
    if df.empty:
        raise DataValidationError("Uploaded sales history file is empty.")

    df = clean_dataframe_columns(df)

    missing_cols = [c for c in REQUIRED_SALES_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing required sales columns: {', '.join(missing_cols)}. "
            f"Required columns are: {', '.join(REQUIRED_SALES_COLUMNS)}"
        )

    errors: List[str] = []
    warnings: List[str] = []

    # Clean text columns
    df["sku"] = df["sku"].astype(str).str.strip()
    if (df["sku"] == "").any():
        errors.append("Found rows with empty SKU in sales data.")

    # Parse date
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    except Exception as e:
        errors.append(f"Failed to parse 'date' column: {str(e)}")

    if df["date"].isna().any():
        errors.append("Found invalid or missing dates in 'date' column.")

    # Units sold numeric
    try:
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    except Exception as e:
        errors.append(f"Could not convert 'units_sold' to numeric: {str(e)}")

    if df["units_sold"].isna().any():
        errors.append("Found invalid or non-numeric values in 'units_sold'.")

    # Negative sales check
    if (df["units_sold"] < 0).any():
        errors.append("Sales units cannot be negative. Negative quantities indicate returns or errors.")

    if errors:
        raise DataValidationError("Sales history data validation failed.", errors=errors)

    df["units_sold"] = df["units_sold"].astype(int)

    # Sort chronologically
    df = df.sort_values(by=["sku", "date"]).reset_index(drop=True)

    return df, warnings


def load_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the included synthetic demo datasets.
    
    Returns:
        Tuple of (inventory_df, sales_df)
    """
    inv_path = DATA_DIR / "demo_inventory.csv"
    sales_path = DATA_DIR / "demo_sales_history.csv"

    if not inv_path.exists() or not sales_path.exists():
        raise FileNotFoundError(f"Demo datasets missing from {DATA_DIR}")

    inv_df, _ = validate_and_load_inventory(inv_path)
    sales_df, _ = validate_and_load_sales(sales_path)
    return inv_df, sales_df
