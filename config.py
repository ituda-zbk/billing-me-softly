# config.py

import os
from pathlib import Path

# --- Data direktorij (OS-agnostic: uvijek u Users home folderu) ---
# Override: BILLING_DATA_DIR env varijabla (postavljena prije importa configa).
DATA_DIR = Path(os.environ.get("BILLING_DATA_DIR") or (Path.home() / "BillingMeSoftly"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Učitaj .env iz data direktorija (ako postoji) — ne prepisuje već postavljene
# prave environment varijable (override=False), samo im daje fallback.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=DATA_DIR / ".env", override=False)
except ImportError:
    pass

# Koliko novih tokena Gemini model smije generirati
MAX_NEW_TOKENS = 8000

# Tolerancija razlike između total i zbroja stavki
TOTAL_TOLERANCE = 0.05

# --- Upload / rad s datotekama ---
UPLOAD_DIR = str(DATA_DIR / "uploads")
DEFAULT_DB_PATH = str(DATA_DIR / "receipts.db")
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", "100"))

# --- Obrada slike ---
DEFAULT_RESIZE_MAX = (800, 1800)  # širina, visina
ENABLE_IMAGE_NORMALIZATION = True

# --- UI defaulti ---
DEFAULT_LANG = "hrv"

# --- Gemini API ---
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MAX_PARALLEL = int(os.environ.get("GEMINI_MAX_PARALLEL", "1"))
GEMINI_REQUESTS_PER_MINUTE = int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "2"))
GEMINI_MAX_429_RETRIES = int(os.environ.get("GEMINI_MAX_429_RETRIES", "4"))
# Koliko slika se šalje u jednom Gemini zahtjevu (RPD kvota je po zahtjevu, ne po slici)
GEMINI_IMAGES_PER_REQUEST = int(os.environ.get("GEMINI_IMAGES_PER_REQUEST", "5"))

# Lokacija lokalno syncanog OneDrive foldera za uvoz računa
ONEDRIVE_IMPORT_DIR = os.environ.get("ONEDRIVE_IMPORT_DIR", "")
