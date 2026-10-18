# 📦 StockSense AI — Hackathon Edition
### Intelligent Inventory & Expiry Risk Co-Pilot
*Built with Python 3.12, Streamlit, Pandas, LangGraph, and Google Gemini 3.8 Flash*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

---

## 1. What StockSense AI Does

Retailers, distributors, and food/pharma warehouses face two recurring, expensive dilemmas:
1. **Perishable Expiry & Spoilage**: Millions of dollars in inventory expire before it can be sold because traditional ERP software only flags the calendar date without evaluating real customer demand velocity.
2. **Stockouts & Emergency Orders**: High-demand products abruptly run out because lead times and safety stock buffers were calculated inconsistently.

**StockSense AI** is a runnable, beginner-friendly inventory intelligence system that ingests inventory batch spreadsheets and historical sales data to:
- Detect expired batches and critical expiry windows ($<7$ days and $8-30$ days).
- Project whether each batch will sell through before expiring using **FEFO (First-Expired, First-Out)** demand allocation without double-counting demand.
- Forecast 7-day and 30-day moving-average demand baselines and compute exact **Days of Supply**.
- Calculate deterministic, non-negative procurement replenishment reorders accounting for supplier lead times and safety buffers.
- Uncover tied-up working capital in slow-moving and dead stock items.
- Provide a conversational **AI Inventory Co-pilot** powered by **LangGraph** and **Google Gemini** that answers inventory queries with verified numerical evidence.
- Export full multi-sheet audit workbooks (`.xlsx`) and `.csv` files.

---

## 2. Key Features

- **🛡️ Bulletproof Data Ingestion**: Supports `.csv` and `.xlsx` uploads with strict Pydantic validation, column normalization, and explicit error feedback for negative stock, invalid dates, or missing headers.
- **⚡ Deterministic Expiry Engine**: Calculates days to expiry, depletion runout, and projected remaining unsold units across multiple batches of the same SKU.
- **📈 Transparent Demand Forecasting**: Deterministic moving-average baseline with zero-divide safety, stockout risk flags, and supplier lead-time coverage.
- **🤖 LangGraph Agent Pipeline**: Structured intent classification $\rightarrow$ verified tool execution $\rightarrow$ grounded Gemini answer synthesis with next-action recommendations.
- **🔒 Enterprise Security & Privacy**: Never executes arbitrary code, never passes raw spreadsheet dumps to the LLM (passes only compact verified tool results), and reads keys strictly from `.env`.
- **📊 Polished Interactive Visuals**: Plotly charts for batch risk distributions, days-of-supply thresholds, and warehouse inventory allocation.
- **🧪 100% Offline Test Suite**: Pytest tests run without needing any API key or active internet connection.

---

## 3. Project Structure

```
stocksense-ai/
│
├── app.py                      # Streamlit interactive dashboard & AI chat application
├── requirements.txt            # Pinned, compatible Python dependencies
├── .env.example                # Template for Google Gemini API configuration
├── .gitignore                  # Git ignore rules for virtualenvs, keys, and caches
├── README.md                   # Comprehensive technical documentation & setup guide
│
├── data/
│   ├── demo_inventory.csv      # Synthetic multi-batch inventory dataset (fictional)
│   └── demo_sales_history.csv  # 60-day historical daily transactions dataset
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Environment variables, file paths, and business thresholds
│   ├── schemas.py              # Pydantic validation models for batches, sales, and forecasts
│   ├── data_loader.py          # CSV/Excel parsing, column standardizer, and validation logic
│   ├── expiry_engine.py        # FEFO batch expiry and sell-through calculation engine
│   ├── forecasting_engine.py   # Moving-average forecasting, days of supply, and reorder math
│   ├── inventory_engine.py     # Executive KPI summarizer, slow mover and capital analyzer
│   ├── tools.py                # 5 LangChain/LangGraph structured inventory tools
│   ├── agent.py                # LangGraph state machine, intent router, and Gemini synthesis
│   └── report_generator.py     # Multi-sheet Excel (.xlsx) and CSV export generator
│
├── tests/
│   ├── test_data_loader.py         # Tests for file upload, validation, and error guards
│   ├── test_expiry_engine.py       # Tests for FEFO depletion, expired batches, and 0-demand
│   ├── test_forecasting_engine.py  # Tests for moving averages, stockouts, and non-negative reorders
│   └── test_tools.py               # Tests for all 5 LangGraph agent tools against fixture data
│
└── docs/
    ├── architecture.md         # In-depth architectural blueprint and formula guide
    └── demo_script.md          # 3-minute hackathon pitch and presentation script
```

---

## 4. Prerequisites

- **Operating System**: Windows 10/11 (or macOS / Linux)
- **Python**: Version **3.12** installed (verify with `py --version` or `python --version`)
- **Google Gemini API Key**: Free API key from [Google AI Studio](https://aistudio.google.com/apikey)

---

## 5. Windows PowerShell Setup (Step-by-Step)

Open **Windows PowerShell** and run the following commands:

### Step 1: Navigate to the Project Directory
```powershell
cd d:\01_langgraph_assistant\stocksense-ai
```

### Step 2: Create a Virtual Environment
```powershell
py -3.12 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

> 💡 **Troubleshooting PowerShell Script Execution**:
> If Windows blocks activating the virtual environment with `cannot be loaded because running scripts is disabled on this system`, allow scripts **for the current PowerShell window only** (safe, temporary, does not alter global system security):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\.venv\Scripts\Activate.ps1
> ```

### Step 4: Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Your Gemini API Key
Create your local `.env` file from `.env.example`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in any text editor (e.g. Notepad or VS Code) and insert your Google Gemini API key:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

---

## 6. How to Run the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```
Streamlit will automatically open your browser at `http://localhost:8501`.

---

## 7. How to Run Automated Tests

The test suite runs 100% offline and requires no internet access or API credentials:
```powershell
python -m pytest -q
```
To run tests with detailed verbosity:
```powershell
python -m pytest -v
```

---

## 8. Example Questions to Ask the AI Copilot

Once the dashboard is running, navigate to the **"💬 AI Copilot Chat"** tab and try asking:

1. *"Which products are at critical risk of expiry?"*
   - Explains which lots will expire in $\le 7$ days and highlights sell-through deficits.
2. *"Which SKUs may run out of stock in the next seven days?"*
   - Identifies high-velocity items where current inventory is below 7 days of supply.
3. *"What should we reorder first and why?"*
   - Surfaces prioritized replenishment items ordered by lead-time runout urgency.
4. *"Show slow-moving products and tied up capital."*
   - Pinpoints dormant items with $>60$ days coverage and calculates total locked cash.
5. *"Summarize our top five inventory problems right now."*
   - Provides an executive summary of expired lots, critical stockouts, and dead stock.

---

## 9. Calculation Methodology & Business Formulas

| Metric | Business Formula | Notes |
|---|---|---|
| **Days to Expiry** | $\text{Expiry Date} - \text{Reference Date}$ | $<0$ is Expired; $1-7$ is Critical; $8-30$ is Warning. |
| **FEFO Allocation** | $\text{Depletion Queue Sorted by Expiry Date ASC}$ | Early batches absorb daily demand first; avoids double-counting. |
| **30-Day Avg Demand** | $\frac{\sum \text{Units Sold in Last 30 Days}}{30}$ | Fills missing dates with 0 to prevent artificial inflation. |
| **Days of Supply** | $\frac{\text{Current Usable Stock}}{\text{30-Day Avg Daily Demand}}$ | Runout duration in days. |
| **Safety Stock Units** | $\lceil \text{Daily Demand} \times \text{Safety Buffer Days} \rceil$ | Configurable buffer (default 7 days). |
| **Recommended Reorder** | $\max(0, \lceil \text{Target Stock} - \text{Available Stock} - \text{Incoming Stock} \rceil)$ | Strictly non-negative. |
| **Tied Up Capital** | $\text{Batch Quantity} \times \text{Unit Cost}$ | Quantifies cash locked in sluggish inventory. |

---

## 10. Security & Privacy Guarantees

- **No Code Execution**: The agent never executes arbitrary Python code from the LLM. All calculations run through fixed, tested Python functions.
- **Data Minimization**: Entire spreadsheets are never sent over the network to Google Gemini. Only the concise, relevant output of the executed tool is sent.
- **Read-Only Operation**: StockSense AI produces recommendations and analysis reports. It does not place orders, send external emails, or write to live ERP databases.
- **Zero Hardcoded Secrets**: Secrets are read exclusively from environment variables via `python-dotenv`.

---

## 11. Troubleshooting Common Issues

1. **`ModuleNotFoundError: No module named 'streamlit'`**:
   - Ensure your virtual environment is activated: `.\.venv\Scripts\Activate.ps1`.
   - Run `pip install -r requirements.txt`.
2. **`NotFound: 404 This model models/gemini-2.5-flash is no longer available`**:
   - Google AI Studio updated active model endpoints. In `.env`, ensure `GEMINI_MODEL=gemini-3.8-flash`.
3. **`DataValidationError: Missing required inventory columns`**:
   - Ensure your custom inventory file contains the 8 mandatory headers:
     `sku`, `product_name`, `warehouse`, `batch_number`, `quantity`, `expiry_date`, `supplier_lead_time_days`, `unit_cost`.
4. **Offline Mode / Missing API Key**:
   - If `GOOGLE_API_KEY` is not provided, the dashboard, Plotly charts, risk tables, and Excel exporters continue to function 100%. The chat interface will display formatted engine results with a reminder to add your key.

---

## 12. License & Synthetic Data Disclaimer

All sample company, product, and batch names in `data/demo_inventory.csv` and `data/demo_sales_history.csv` are entirely fictional and generated synthetically for demonstration and educational purposes.
