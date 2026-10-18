# StockSense AI — 3-Minute Hackathon Demo Script

## Elevator Pitch (30 Seconds)

> "Distributors and retailers lose billions each year to two preventable problems: throwing away expired perishable stock, and losing revenue to unexpected stockouts. 
> 
> Spreadsheet formulas are static and cannot predict whether a lot will actually sell through before it expires. 
> 
> **StockSense AI** combines deterministic inventory engines with LangGraph-orchestrated Gemini intelligence to automatically flag shelf-life risks, forecast replenishment needs, and let managers speak to their inventory data in plain English."

---

## 3-Minute Demo Walkthrough

### ⏱️ Minute 0:00 – 0:45: The Problem & Executive Dashboard
1. **Open the App**:
   - Show the Streamlit dashboard on `http://localhost:8501`.
   - Highlight the **Top KPI Cards**:
     - *Total Inventory Units*: 2,755 units across 16 batches.
     - *Total Valuation*: Over $18,000 in working capital.
     - *Expired Batches*: 1 batch flagged for immediate quarantine (`LOT-N201` expired 8 days ago).
     - *Critical Expiry*: Batches expiring in $\le 7$ days.
2. **Visual Charts**:
   - Point out the **Batch Expiry Health Breakdown** donut chart.
   - Show the **Warehouse Distribution** bar chart showing inventory allocation between North, Central, and South warehouses.

---

### ⏱️ Minute 0:45 – 1:30: Expiry Radar & FEFO Intelligence
1. **Navigate to "⚠️ Expiry Risk Radar" Tab**:
   - Explain **FEFO (First-Expired, First-Out)**:
     *"Unlike simple spreadsheet formulas that look only at the date, StockSense AI calculates whether customer demand will actually absorb the batch before it expires."*
2. **Highlight Key Scenarios**:
   - `SKU-BEV-001` (Aura Spark): Batch `LOT-A101` expires in 4 days (Critical).
   - `SKU-SNK-003` (VitaCrisp Apple Chips): Batch `LOT-V301` expires in 21 days, but daily demand is only 0.25 units/day. The engine calculates that 214 units will expire unsold!
   - `SKU-NUT-005`: Zero sales recorded in 30 days — flagged as **Review Needed**.

---

### ⏱️ Minute 1:30 – 2:15: Demand Forecasting & Urgent Reorders
1. **Navigate to "📈 Demand Forecast & Coverage" Tab**:
   - Show the **Days of Supply Chart**:
     - Red dotted line shows the 7-day immediate stockout risk threshold.
     - `SKU-TEA-004` (Ceremonial Matcha) has only **1.5 days of supply** left (15 units in stock vs 10 units/day demand)!
2. **Navigate to "🛒 Reorder Schedule" Tab**:
   - Point to the **URGENT** replenishment recommendation:
     - Supplier lead time is 14 days.
     - Recommended reorder quantity: **250 units** ($3,625 estimated order cost).
     - Emphasize: *The algorithm accounts for lead-time demand plus safety buffer, preventing costly stockouts.*
3. **Navigate to "🐢 Slow-Moving & Dead Stock" Tab**:
   - Show how the business has **$4,000+** locked in dormant inventory (`SKU-NUT-005` and `SKU-SNK-003`).

---

### ⏱️ Minute 2:15 – 2:50: Live LangGraph AI Copilot Chat
1. **Navigate to "💬 AI Copilot Chat" Tab**:
2. **Ask Question 1**:
   > *"Which products are at risk of expiry?"*
   - Show the agent executing `get_expiry_risks` and Gemini synthesizing:
     - Directly naming expired lot `LOT-N201` and critical lot `LOT-A101`.
     - Suggesting immediate quarantine and flash promotions.
3. **Ask Question 2**:
   > *"What should we reorder first and why?"*
   - Show the agent executing `get_reorder_recommendations`.
   - The assistant answers:
     - `SKU-TEA-004` is the #1 priority with only 1.5 days of coverage remaining against a 14-day lead time.

---

### ⏱️ Minute 2:50 – 3:00: Export & Closing
1. **Navigate to "📥 Export Reports" Tab**:
   - Click **Download Full Excel Report (.xlsx)** to show multi-tab audit-ready spreadsheets generated with `openpyxl`.
2. **Wrap Up**:
   > *"StockSense AI delivers actionable intelligence in seconds, saving distributors thousands in wasted inventory and unfulfilled orders. Built with Python 3.12, Streamlit, LangGraph, and Google Gemini."*
