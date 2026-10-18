"""
Data schemas and Pydantic models for StockSense AI.
Provides strict validation for spreadsheet uploads, intermediate calculations,
and structured outputs for LangGraph agent tools.
"""

import datetime as dt
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class InventoryItemSchema(BaseModel):
    """Schema representing a single inventory batch record."""
    model_config = ConfigDict(coerce_numbers_to_str=False)

    sku: str = Field(..., min_length=1, description="Unique product SKU")
    product_name: str = Field(..., min_length=1, description="Display name of the product")
    warehouse: str = Field(..., min_length=1, description="Warehouse facility identifier")
    batch_number: str = Field(..., min_length=1, description="Batch / Lot number")
    quantity: int = Field(..., ge=0, description="Available batch inventory count")
    expiry_date: dt.date = Field(..., description="Expiration date of this batch")
    supplier_lead_time_days: int = Field(default=7, ge=0, description="Lead time in days for replenishment")
    unit_cost: float = Field(default=0.0, ge=0.0, description="Unit cost in currency")

    @field_validator("quantity")
    @classmethod
    def validate_quantity_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Quantity cannot be negative")
        return v

    @field_validator("unit_cost")
    @classmethod
    def validate_cost_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Unit cost cannot be negative")
        return v


class SalesRecordSchema(BaseModel):
    """Schema representing a single daily sales record for an SKU."""
    date: dt.date = Field(..., description="Date of sales transaction")
    sku: str = Field(..., min_length=1, description="Product SKU")
    units_sold: int = Field(..., ge=0, description="Units sold on this date")

    @field_validator("units_sold")
    @classmethod
    def validate_units_sold(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Units sold cannot be negative")
        return v


RiskCategoryType = Literal["Expired", "Critical", "Warning", "Normal", "Review Needed"]


class BatchExpiryRisk(BaseModel):
    """Calculated expiry risk indicators for a specific batch."""
    sku: str
    product_name: str
    warehouse: str
    batch_number: str
    quantity: int
    expiry_date: str
    days_until_expiry: int
    avg_daily_demand: float
    allocated_demand: float
    days_to_sell_batch: Optional[float]
    projected_remaining_at_expiry: float
    risk_category: RiskCategoryType
    explanation: str


class DemandForecastResult(BaseModel):
    """Calculated deterministic moving-average forecast and replenishment advice."""
    sku: str
    product_name: str
    current_stock: int
    avg_daily_demand_7d: float
    avg_daily_demand_30d: float
    projected_demand_7d: float
    projected_demand_30d: float
    days_of_supply: Optional[float]
    stockout_risk_7d: bool
    stockout_risk_30d: bool
    overstock_risk: bool
    supplier_lead_time_days: int
    safety_stock_units: int
    recommended_reorder_qty: int
    status: str
    notes: str


class SlowMovingProduct(BaseModel):
    """Model for slow-moving inventory items."""
    sku: str
    product_name: str
    total_stock: int
    avg_daily_demand_30d: float
    days_of_supply: Optional[float]
    unit_cost: float
    tied_up_capital: float
    status: str
    recommendation: str


class InventorySummary(BaseModel):
    """High-level snapshot of inventory health."""
    total_units: int
    total_skus: int
    total_batches: int
    total_warehouses: int
    total_inventory_value: float
    expired_batches_count: int
    critical_batches_count: int
    warning_batches_count: int
    normal_batches_count: int
    review_needed_batches_count: int
    stockout_risk_skus_count: int
    slow_moving_skus_count: int
    reference_date: str
