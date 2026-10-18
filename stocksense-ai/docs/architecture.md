# StockSense AI — System Architecture & Workflow

## 1. High-Level Overview

StockSense AI is an intelligent inventory co-pilot built for retailers, distributors, and supply chain analysts. It addresses four critical operational bottlenecks:
1. **Spooling Expiry & Food/Pharma Waste**: Batches expiring before they can be sold through.
2. **Stockouts & Lost Sales**: Running out of critical SKUs due to unmonitored supplier lead times.
3. **Trapped Working Capital**: Money tied up in dead or dormant inventory.
4. **Information Asymmetry**: Inability to quickly query spreadsheet data in natural language.

---

## 2. Core Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            STREAMLIT USER INTERFACE                               |
|   - Executive Dashboard    - Expiry Radar        - Demand Forecast Charts         |
|   - Reorder Schedule       - Slow Mover Tracker  - AI Copilot Chat & File Upload  |
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
+───────────────────────────────────────+ +────────────────────────────────────────+
|      DETERMINISTIC ENGINE TIER        | |         LANGGRAPH AI AGENT TIER        |
|                                       | |                                        |
| 1. data_loader.py                     | |  [START]                               |
|    - Normalization & Pydantic checks  | |     │                                  |
| 2. expiry_engine.py                   | |     ▼                                  |
|    - FEFO Sequential Depletion        | |  [validate_question]                   |
| 3. forecasting_engine.py              | |     │                                  |
|    - 7d & 30d Moving Averages         | |     ▼                                  |
|    - Days of Supply & Reorders        | |  [identify_task] (Tool Selection)      |
| 4. inventory_engine.py                | |     │                                  |
|    - Slow mover & Capital Analysis    | |     ▼                                  |
| 5. report_generator.py                | |  [execute_tool] (Approved Tools)       |
|    - Multi-sheet Excel & CSV Exports  | |     │                                  |
|                                       | |     ▼                                  |
+───────────────────────────────────────+ |  [verify_tool_results]                 |
                    │                     |     │                                  |
                    └────────────────────►|     ▼                                  |
                     Shared Dataset State |  [generate_answer] (Gemini Model)      |
                                          |     │                                  |
                                          |     ▼                                  |
                                          |   [END]                                |
                                          +────────────────────────────────────────+
```

---

## 3. Module Breakdown

### `src/config.py`
- Manages application settings, paths, and business logic thresholds.
- Reads `GOOGLE_API_KEY` and `GEMINI_MODEL` from `.env`.
- Defines default business thresholds:
  - `CRITICAL_EXPIRY_DAYS = 7`: Batches expiring in $\le 7$ days.
  - `WARNING_EXPIRY_DAYS = 30`: Batches expiring in $8-30$ days.
  - `SLOW_MOVING_THRESHOLD_DAYS = 60`: Inventory exceeding 60 days of demand.
  - `DEFAULT_SAFETY_STOCK_DAYS = 7`: Buffer days used in replenishment target.

### `src/schemas.py`
- Strict Pydantic models ensuring data integrity at every layer.
- `InventoryItemSchema`: Validates `sku`, `product_name`, `warehouse`, `batch_number`, `quantity` ($\ge 0$), `expiry_date`, `supplier_lead_time_days` ($\ge 0$), `unit_cost` ($\ge 0$).
- `SalesRecordSchema`: Validates daily transactions `date`, `sku`, `units_sold` ($\ge 0$).
- `BatchExpiryRisk`, `DemandForecastResult`, `SlowMovingProduct`, `InventorySummary`.

### `src/data_loader.py`
- Handles CSV and Excel (`.xlsx`) file ingestion.
- Normalizes column names (e.g. `"Product-Name "` $\rightarrow$ `"product_name"`).
- Coerces data types and surfaces human-readable error messages for missing columns, negative stock, or corrupt dates.
- Detects duplicate batch records per warehouse and safely aggregates them.
- Provides `load_demo_data()` to ensure out-of-the-box hackathon readiness.

### `src/expiry_engine.py`
- **FEFO (First-Expired, First-Out) Demand Allocation**:
  - Batches of the same SKU are ordered chronologically by `expiry_date`.
  - Demand is allocated sequentially to the earliest-expiring batch first.
  - Avoids double-counting SKU demand across multiple lots.
  - If a batch expires later than 30 days, but total stock in the queue cannot be sold before its expiry date, the engine elevates the risk category to **Warning** with a clear explanation of the sell-through deficit.
  - Batches with 0 historical demand are flagged as **Review Needed**.

### `src/forecasting_engine.py`
- Transparent, deterministic moving-average baseline:
  $$\text{Daily Demand (7d)} = \frac{\sum_{t=1}^{7} \text{Sales}_t}{7}$$
  $$\text{Daily Demand (30d)} = \frac{\sum_{t=1}^{30} \text{Sales}_t}{30}$$
- **Days of Supply (Runout Time)**:
  $$\text{Days of Supply} = \frac{\text{Usable Stock}}{\text{30d Daily Demand}}$$
- **Replenishment Reorder Calculation**:
  $$\text{Lead Time Demand} = \text{Daily Demand} \times \text{Supplier Lead Time Days}$$
  $$\text{Safety Stock Target} = \text{Daily Demand} \times \text{Safety Stock Days}$$
  $$\text{Target Stock} = \text{Lead Time Demand} + \text{Safety Stock Target}$$
  $$\text{Recommended Reorder} = \max(0, \lceil \text{Target Stock} - \text{Usable Stock} - \text{Incoming Stock} \rceil)$$
  *Reorder quantities are strictly guaranteed to never be negative.*

### `src/inventory_engine.py`
- Consolidates executive KPI metrics (units, valuation, risk counts).
- Calculates tied-up working capital in slow-moving and dead stock ($\text{Quantity} \times \text{Unit Cost}$).
- Prioritizes procurement reorders into **URGENT** (7-day stockout), **MEDIUM** (30-day stockout), and **ROUTINE**.

### `src/tools.py`
- Implements 5 structured LangChain `@tool` functions:
  1. `get_inventory_summary()`
  2. `get_expiry_risks(risk_category, warehouse, sku, limit)`
  3. `get_slow_moving_products(limit)`
  4. `get_demand_forecast(sku, filter_risk, limit)`
  5. `get_reorder_recommendations(limit)`
- Tools operate on the active dataset in memory and return concise dictionaries formatted specifically to minimize LLM token usage.

### `src/agent.py`
- Orchestrates the state graph using **LangGraph**:
  1. `validate_question`: Guards against empty or gibberish input.
  2. `identify_task`: Classifies intent and determines the appropriate tool and arguments.
  3. `execute_tool`: Runs the approved Python tool deterministically.
  4. `verify_tool_results`: Confirms valid output before LLM consumption.
  5. `generate_answer`: Invocates Gemini to synthesize verified numbers into a clear direct answer, supporting evidence, and next actions.
- Graceful offline fallback: If the Gemini API key is missing, the agent outputs formatted deterministic tool data without crashing the dashboard.

### `src/report_generator.py`
- Generates professional multi-sheet Excel reports (`openpyxl`) and CSV files for immediate export and stakeholder presentation.

---

## 4. Key Design Decisions

1. **Deterministic Calculations vs LLM Hallucinations**:
   Mathematical formulas (FEFO allocation, days of supply, reorder quantities) are executed entirely in Python. The LLM is used strictly for intent classification and natural language synthesis of verified data.
2. **Session Memory & Offline Resilience**:
   The dashboard runs 100% offline for calculations and charting. If Gemini is unavailable, users still receive all numerical intelligence without interruption.
3. **Data Minimization & Security**:
   Entire spreadsheets are never passed to the LLM. Only the concise, filtered output of the executed tool is sent to Gemini.
