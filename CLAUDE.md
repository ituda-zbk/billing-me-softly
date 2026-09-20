# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Receipt OCR and expense tracking application. Uses Google Gemini Vision API to parse receipt images, extract items with categories, and store them in SQLite. Provides a Flask web UI for reviewing, editing, and analyzing expenses. All UI text and OCR prompts are in Croatian.

## Running the Application

OS-agnostic (Linux, macOS, Windows — same commands everywhere). First-time setup:

```bash
python bootstrap.py   # creates ~/BillingMeSoftly/ (data dir), installs requirements.txt, prompts for GEMINI_API_KEY
```

```bash
# Web server (primary usage)
python receipt_ocr.py --serve --host 0.0.0.0 --port 5000

# Single image OCR
python receipt_ocr.py /path/to/receipt.jpg --lang hrv
```

**Required environment variables:**
- `GEMINI_API_KEY` — Google Gemini API key
- `GEMINI_MODEL` — model name (default: `gemini-2.5-flash`)

These can be set as real environment variables, or once in `~/BillingMeSoftly/.env` (created by `bootstrap.py`) — loaded via `python-dotenv` in `config.py`, and never overrides a real env var that's already set.

**Dependencies:** `flask`, `werkzeug`, `pillow`, `pillow-heif`, `requests`, `python-dotenv` — see `requirements.txt` (`pip install -r requirements.txt`).

## Architecture

The entire application lives in a single file: `receipt_ocr.py` (~3400 lines). Configuration constants are in `config.py`.

**Data directory (OS-agnostic):** `config.py` resolves `DATA_DIR` to `Path.home() / "BillingMeSoftly"` (override with `BILLING_DATA_DIR`) at import time and creates it if missing. `UPLOAD_DIR` and `DEFAULT_DB_PATH` (the `--db-path` default) both live under `DATA_DIR`, so behavior no longer depends on the process's working directory. `_migrate_legacy_repo_data()` in `receipt_ocr.py` runs once on `main()` startup (only when `--db-path` is left at its default) and moves a pre-existing repo-root `receipts.db`/`uploads/` into `DATA_DIR` if found — a one-time upgrade path for installs that predate this change (it also rewrites the absolute `image_path` prefix in the DB, since `receipts.image_path` stores absolute paths). `_repair_broken_image_paths()` runs on every startup (default `--db-path` only): for any receipt whose stored `image_path` doesn't exist on disk, it looks for the same *basename* in `UPLOAD_DIR` and repoints the row — this is what makes copying `receipts.db` + `uploads/` between machines/OSes (e.g. Linux → Windows, where the stored `/home/...` paths are meaningless) work. `manual://` / `excel://` placeholder rows never match a file and are skipped.

**Data flow:** Image upload → EXIF normalization & resize → Base64 encode → Gemini Vision API → JSON parse & repair → ReceiptData dataclass → normalized rows in SQLite (image itself stays a file in `uploads/`, but parsed data lives only in the DB). Batch uploads (`/upload`, `/import_onedrive`) run OCR in a background thread — the request returns immediately with a redirect to a live status page (`/batch/<id>`) that polls per-image progress and auto-continues to the review form once done. Images are grouped into chunks of `GEMINI_IMAGES_PER_REQUEST` and sent to Gemini as one multi-image request per chunk (Gemini's free-tier daily quota is per-*request*, not per-image, so batching multiple images per call multiplies daily throughput) — see `call_gemini_vision_batch_parser()`.

**Key components within `receipt_ocr.py`:**
- `create_app()` — Flask app factory with all route definitions
- `build_receipt_data()` — single-image pipeline: calls `call_gemini_vision_parser()` then `_build_receipt_data_from_payload()` to build a `ReceiptData`
- `_build_receipt_data_from_payload()` — converts a raw Gemini JSON payload into `ReceiptData` (item cleaning, `items_sum`, date normalization); shared by both the single-image and batch paths
- `call_gemini_vision_parser()` — sends ONE image to Gemini API (single-image path: CLI, manual retry)
- `call_gemini_vision_batch_parser()` — sends up to `GEMINI_IMAGES_PER_REQUEST` images in a SINGLE Gemini request; requires the model to tag each result with a matching `index` and returns `{index: raw_payload | None}` — an index that's missing, duplicated, out-of-range, or malformed maps to `None` and is **never** treated as processed (caller must surface it as a per-image error, never save it)
- `_post_gemini_generate_content()` — shared POST-with-429-retry/daily-quota-detection loop used by both `call_gemini_vision_parser()` and `call_gemini_vision_batch_parser()`
- `parse_llm_json()` / `parse_llm_batch_json()` — multi-attempt JSON parsing with auto-repair (code fence stripping, trailing comma fixes); batch variant salvages partial `"receipts"` arrays on truncation instead of `"items"`
- `process_images_batch()` — parallel processing via ThreadPoolExecutor; jobs are grouped into chunks of `GEMINI_IMAGES_PER_REQUEST` and each chunk is processed as ONE Gemini request via `process_image_chunk()`; vraća `(results, errors)` tuple; circuit breaker zaustavlja preostale chunkove kad detektira dnevnu kvotu; opcionalni `on_status(idx, status, error)` callback i dalje javlja "processing"/"done"/"error" **po pojedinačnoj slici** (ne po chunku)
- `process_image_chunk()` — obrađuje jedan chunk: prvo svaku sliku provjeri kroz `_validate_image_readable()` (postoji, nije 0 bajtova, PIL je može otvoriti) — neispravna slika odmah postaje greška SAMO za taj index i izbacuje se iz zahtjeva (cijeli chunk je JEDAN HTTP poziv pa bi inače jedna prazna slika srušila Gemini 400 za svih N slika); zatim poziva `call_gemini_vision_batch_parser()` i za svaki index s valjanim rezultatom gradi `ReceiptData`/`_package_processed_entry()`; index bez rezultata postaje eksplicitna greška za tu sliku
- `_package_processed_entry()` — builds the `{image_path, preview_path, payload, ...}` result dict; shared by single-image and batch paths
- `_reserve_gemini_request_slot()` — thread-safe sliding-window rate limiting (60 s prozor, `deque`); trošak je po *zahtjevu* bez obzira nosi li 1 ili N slika
- `GeminiDailyQuotaExceeded` — iznimka za iscrpljenu dnevnu kvotu; detektira se via `_classify_429()`
- `_truncate_broken_items_array(s, key='"items"')` — depth-aware popravak skraćenih Gemini odgovora (radi i za ugniježđene nizove, npr. `"receipts"` čiji elementi sami sadrže `"items"`); parametrizirano preko `key` da ga koriste i `parse_llm_json()` i `parse_llm_batch_json()`
- `apply_form_updates()` — merges web form edits into receipt payload
- `normalize_image_orientation()` / `resize_image()` / `ensure_gemini_compatible_image()` — image preprocessing
- `render_review_page()` / `batch_failure_message()` — dijeljena logika za review flow nakon uspješnog/neuspješnog batcha (koriste je i `/upload` i `/import_onedrive`, posredno preko `/batch/<id>/review`)
- `_upload_batches` (+ `_create_batch()`/`_set_batch_status()`/`_finish_batch()`/`_get_batch()`/`_pop_batch()`/`_run_batch_in_background()`) — in-memory store (thread-safe, TTL 2h) koji prati status live-progress batch obrade po slici; koristi ga `/batch/<id>` status stranica

**Database:** Two SQLite tables via `init_db()`: `receipts` (scalar fields — `image_path`, `language`, `total`, `items_sum`, `date`, `time`, `warranty`, timestamps) and `receipt_items` (one row per line item — `receipt_id` FK with `ON DELETE CASCADE`, `position` for ordering, `description`, `category`, `quantity`, `unit_price`, `total_price`). No JSON blob is stored anywhere — `fetch_receipt_payload()` reassembles the same `{image, language, items, ...}` dict shape the templates expect by joining the two tables; `save_receipt_to_db()`/`update_receipt_record()` decompose that dict back into rows (`_replace_receipt_items()` does a delete+reinsert of a receipt's items). `init_db()` also runs a one-time migration (`_migrate_legacy_data_json()`) that backfills `receipt_items` from any pre-existing `data_json` column and drops the old `data_json`/`json_path` columns (requires SQLite ≥3.35 for `DROP COLUMN`).

**Web routes:** `/` (dashboard s filterima uključujući `item_search` — `EXISTS` pretraga po `receipt_items.description`/`category`), `/upload`, `/import_onedrive` (bulk import) — obje pokreću batch u pozadini i redirectaju na `/batch/<id>`; `/batch/<id>` (live status stranica, polla `/batch/<id>/status.json`), `/batch/<id>/review` (konzumira gotov batch → postojeći review/edit flow), `/receipt/<id>` (view/edit), `/rotate_image_action`, `/category_items`.

**Frontend:** sav HTML/CSS je inline u Jinja2 template stringovima (`INDEX_TEMPLATE`, `DETAIL_TEMPLATE`, `CATEGORY_ITEMS_TEMPLATE`, `BATCH_STATUS_TEMPLATE`) građenim konkatenacijom `"""...""" + FONT_LINKS + """<style>""" + BASE_STYLE + """...""`. `FONT_LINKS`/`BASE_STYLE` su dijeljene konstante (definirane prije `INDEX_TEMPLATE`) koje nose zajednički "Ledger & Ink" dizajn sustav — CSS varijable (`--paper`, `--ink`, `--stamp`, `--ledger-stripe`, ...), monospace `IBM Plex Mono` za brojeve/naslove, `.button-primary`/`.button-secondary`/`.button-onbar` "pečat" gumbe, ledger zebra tablice, `.warranty-pill` stil statusnih oznaka. Nove stranice/template-e treba nadovezati na iste konstante umjesto uvoditi paralelan stil. Ikona aplikacije (TornReceipt, 128px PNG s prozirnim kutovima) ugrađena je kao base64 (`_APP_ICON_PNG_B64`) i služi se na `/favicon.png` i `/favicon.ico` (rute `app_icon`/`app_icon_ico` u `create_app()`); `ICON_LINKS` (favicon `<link>`) ide u `<head>` svakog templatea nakon `FONT_LINKS`, a `.topbar-brand`/`.topbar-logo` stavlja logo lijevo od naslova u svakom `.topbar`-u — novi template mora oboje preuzeti.

## Expense Categories (predefined in Gemini prompt)

Hrana, Cigarete, alkohol, kave,..., Kućne potrepštine, Kućni ljubimci, Lijekovi, troškovi liječenja, Odjeća i obuća, Škola i dječje aktivnosti, Sport, Automobili, Osiguranja, Internet/mobitel/TV, Struja, Voda, Plin, Smeće, Komunalni doprinos, Vodni doprinos, Putovanja, izleti, ručkovi, Ostalo. (Exact strings as used in the Gemini prompt and the dashboard — note the commas; the DB stores these literal names, so don't "normalize" them.)

## Key Configuration (`config.py`)

- `TOTAL_TOLERANCE` (0.05) — max acceptable EUR difference between items sum and receipt total
- `MAX_UPLOAD_FILES` (20) — batch upload limit
- `DEFAULT_RESIZE_MAX` (800, 1800) — image resize for storage
- `GEMINI_REQUESTS_PER_MINUTE` (2) — API rate limit (sliding window, 60 s)
- `GEMINI_MAX_PARALLEL` (1) — concurrent API worker threads (defaultno sekvencijalna obrada)
- `GEMINI_IMAGES_PER_REQUEST` (5) — koliko slika ide u jedan Gemini `generateContent` poziv (chunk size u `process_images_batch()`); dnevna RPD kvota je po zahtjevu pa ovo direktno množi dnevni kapacitet obrade
- `_GEMINI_MAX_OUTPUT_TOKENS` (4096) — `min(MAX_NEW_TOKENS, 4096)`, single-image cap, definirano u `receipt_ocr.py`, ne u `config.py`
- `_gemini_batch_max_output_tokens(n)` — batch cap, `min(MAX_NEW_TOKENS, max(4096, 700 * n))`, raste s brojem slika u chunku (definirano u `receipt_ocr.py`)

## Notes

- No test suite exists. Test changes manually through the web UI or CLI.
- No linter/formatter configured.
- Templates are rendered inline via Jinja2 (no separate template files). JS regexes embedded in these non-raw Python strings must double their backslashes (`\\s`, `\\d`, `\\.`) — a single backslash triggers a `SyntaxWarning` on newer Python (3.12+) and will become an error in future versions.
- Development happens on Linux/WSL, but the app is also run on the Windows host (`C:\Users\<user>\...`); when testing OS-specific behavior, run the Windows side via `powershell.exe` from WSL.
- `old_scripts/` contains legacy backups — do not modify.
- `README.md` is the public GitHub landing page; `docs/screenshots/` are generated from a fictional demo database (never real receipts) — regenerate the same way if the UI changes.
- `uploads/` (under `DATA_DIR`, i.e. `~/BillingMeSoftly/uploads/`) contains only receipt images. No `_parsed.json` files are written; parsed data lives exclusively in `receipts.db`.
