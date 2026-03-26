# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Receipt OCR and expense tracking application. Uses Google Gemini Vision API to parse receipt images, extract items with categories, and store them in SQLite. Provides a Flask web UI for reviewing, editing, and analyzing expenses. All UI text and OCR prompts are in Croatian.

## Running the Application

```bash
# Web server (primary usage)
python receipt_ocr.py --serve --host 0.0.0.0 --port 5000

# Single image OCR
python receipt_ocr.py /path/to/receipt.jpg --lang hrv --db-path receipts.db
```

**Required environment variables:**
- `GEMINI_API_KEY` — Google Gemini API key
- `GEMINI_MODEL` — model name (default: `gemini-2.5-flash`)

**Dependencies:** `flask`, `werkzeug`, `pillow`, `requests`

```bash
pip install flask werkzeug pillow requests
```

## Architecture

The entire application lives in a single file: `receipt_ocr.py` (~3100 lines). Configuration constants are in `config.py`.

**Data flow:** Image upload → EXIF normalization & resize → Base64 encode → Gemini Vision API → JSON parse & repair → ReceiptData dataclass → SQLite + JSON file in `uploads/`

**Key components within `receipt_ocr.py`:**
- `create_app()` — Flask app factory with all route definitions
- `build_receipt_data()` — orchestrates the full OCR pipeline for one image
- `call_gemini_vision_parser()` — sends image to Gemini API, handles retries
- `parse_llm_json()` — multi-attempt JSON parsing with auto-repair (code fence stripping, trailing comma fixes)
- `process_images_batch()` — parallel processing via ThreadPoolExecutor
- `_reserve_gemini_request_slot()` — thread-safe rate limiting with exponential backoff on 429s
- `apply_form_updates()` — merges web form edits into receipt payload
- `normalize_image_orientation()` / `resize_image()` / `ensure_gemini_compatible_image()` — image preprocessing

**Database:** Single SQLite table `receipts` with auto-initialization via `init_db()`. Schema includes `data_json` TEXT column holding the full parsed receipt as JSON blob.

**Web routes:** `/` (dashboard with filters), `/upload`, `/receipt/<id>` (view/edit), `/import_onedrive` (bulk import), `/rotate_image_action`, `/category_items`.

## Expense Categories (predefined in Gemini prompt)

Hrana, Cigarete/alkohol/kave, Kućne potrepštine, Lijekovi, Odjeća i obuća, Škola i dječje aktivnosti, Sport, Automobili, Osiguranja, Internet/mobitel/TV, Struja, Voda, Plin, Smeće, Komunalni doprinos, Vodni doprinos, Putovanja/izleti/ručkovi, Ostalo.

## Key Configuration (`config.py`)

- `TOTAL_TOLERANCE` (0.05) — max acceptable EUR difference between items sum and receipt total
- `MAX_UPLOAD_FILES` (20) — batch upload limit
- `DEFAULT_RESIZE_MAX` (800, 1800) — image resize for storage
- `GEMINI_REQUESTS_PER_MINUTE` (5) — API rate limit
- `GEMINI_MAX_PARALLEL` (2) — concurrent API worker threads

## Notes

- No test suite exists. Test changes manually through the web UI or CLI.
- No linter/formatter configured.
- Templates are rendered inline via Jinja2 (no separate template files).
- `old_scripts/` contains legacy backups — do not modify.
- `uploads/` contains receipt images and `_parsed.json` files — gitignored.
