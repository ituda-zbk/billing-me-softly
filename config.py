# config.py

import os

# Koliko novih tokena Gemini model smije generirati
MAX_NEW_TOKENS = 8000

# Tolerancija razlike između total i zbroja stavki
TOTAL_TOLERANCE = 0.05

# --- Upload / rad s datotekama ---
UPLOAD_DIR = "uploads"
MAX_UPLOAD_FILES = 20

# --- Obrada slike ---
DEFAULT_RESIZE_MAX = (800, 1800)  # širina, visina
ENABLE_IMAGE_NORMALIZATION = True

# --- UI defaulti ---
DEFAULT_LANG = "hrv"

# --- Gemini API ---
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MAX_PARALLEL = int(os.environ.get("GEMINI_MAX_PARALLEL", "2"))
GEMINI_REQUESTS_PER_MINUTE = int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "5"))
GEMINI_MAX_429_RETRIES = int(os.environ.get("GEMINI_MAX_429_RETRIES", "6"))

# Lokacija lokalno syncanog OneDrive foldera za uvoz računa
ONEDRIVE_IMPORT_DIR = os.environ.get("ONEDRIVE_IMPORT_DIR", "")
