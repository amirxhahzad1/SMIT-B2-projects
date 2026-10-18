"""
Configuration management for StockSense AI.
Reads environment variables securely and provides default thresholds for inventory logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find directory paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file if present
load_dotenv(PROJECT_ROOT / ".env")
# Also fallback to root .env if running from parent directory
load_dotenv(PROJECT_ROOT.parent / ".env")

class AppConfig:
    """Application configuration and business thresholds."""

    # Google Gemini settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

    # Expiry Risk Thresholds (in days)
    # Expired: <= 0 days
    # Critical: 1 to 7 days
    # Warning: 8 to 30 days
    # Normal: > 30 days
    CRITICAL_EXPIRY_DAYS: int = int(os.getenv("CRITICAL_EXPIRY_DAYS", "7"))
    WARNING_EXPIRY_DAYS: int = int(os.getenv("WARNING_EXPIRY_DAYS", "30"))

    # Inventory Analysis Thresholds
    # Slow moving if stock covers more than this number of days of demand or demand is near zero
    SLOW_MOVING_DAYS_THRESHOLD: int = int(os.getenv("SLOW_MOVING_THRESHOLD_DAYS", "60"))
    DEFAULT_SAFETY_STOCK_DAYS: int = int(os.getenv("DEFAULT_SAFETY_STOCK_DAYS", "7"))

    # File upload constraints
    MAX_UPLOAD_SIZE_MB: int = 15

    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Check if a Gemini API key is provided."""
        return bool(cls.GOOGLE_API_KEY and len(cls.GOOGLE_API_KEY) > 5)

config = AppConfig()
