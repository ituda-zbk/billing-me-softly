#!/usr/bin/env python3
"""Receipt parser powered by Gemini Vision."""

from __future__ import annotations
import argparse
import base64
from collections import deque
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sqlite3
import requests
import shutil
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional
from uuid import uuid4
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, Response, request, redirect, url_for, render_template_string, send_file, abort, jsonify
from werkzeug.utils import secure_filename
from config import (
    MAX_NEW_TOKENS,
    TOTAL_TOLERANCE,
    MAX_UPLOAD_FILES,
    UPLOAD_DIR,
    DATA_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_LANG,
    DEFAULT_RESIZE_MAX,
    ENABLE_IMAGE_NORMALIZATION,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    GEMINI_ENDPOINT,
    GEMINI_MAX_PARALLEL,
    GEMINI_REQUESTS_PER_MINUTE,
    GEMINI_MAX_429_RETRIES,
    GEMINI_IMAGES_PER_REQUEST,
    ONEDRIVE_IMPORT_DIR,
)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass



def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[INFO {timestamp}] {message}"
    print(line, flush=True)


# Pravila za datum dijele jednoslikovni i batch prompt. Ne smije sadržavati vitičaste
# zagrade jer se batch predložak provlači kroz str.format().
# Model se ovdje koristi samo kao OCR datuma ("date_candidates"); odabir pravog datuma
# među njima radi _resolve_receipt_date() jer model bez "thinking" budžeta zna prepisati
# oba datuma ispravno, a onda ih krivo protumačiti (godina <-> dan).
_GEMINI_DATE_RULES = """
PRAVILA ZA DATUM:
- Račun često ima VIŠE datuma u RAZLIČITIM formatima (npr. DD.MM.YY u fiskalnom dijelu računa i YY/MM/DD na slipu kartičnog terminala).
- U "date_candidates" doslovno prepiši SVE datume s računa, znak po znak, točno kako su ispisani (isti separatori i isti broj znamenki). Samo datume, bez vremena; svaki različiti zapis samo jednom. Ako datum nije čitljiv ili ga nema na slici, vrati "date_candidates": [] i "date": null; NIKAD ne izmišljaj datum.
- U "date" upiši najbolju procjenu kao "YYYY-MM-DD". Točke znače DD.MM.YY(YY), dan je prvi; "YY/MM/DD" na slipovima terminala ima godinu prvu. Ako nisi siguran, vrati null.
""".strip()


GEMINI_VISION_PROMPT = """
Pročitaj račun sa slike i vrati isključivo valjani JSON bez markdowna.
JSON shema:
{
  "items": [
    {
      "description": string,
      "category": "Hrana | Cigarete, alkohol, kave,... | Kućne potrepštine | Kućni ljubimci | Lijekovi, troškovi liječenja | Odjeća i obuća | Škola i dječje aktivnosti | Sport | Automobili | Osiguranja | Internet/mobitel/TV | Struja | Voda | Plin | Smeće | Komunalni doprinos | Vodni doprinos | Putovanja, izleti, ručkovi | Ostalo" | null,
      "quantity": number | null,
      "unit_price": number | null,
      "total_price": number | null
    }
  ],
  "total": number | null,
  "date_candidates": [string],
  "date": "YYYY-MM-DD" | null,
  "time": "HH:MM[:SS]" | null
}
Pravila:
- decimalne zareze pretvori u točku
- nepoznate vrijednosti postavi na null

""".lstrip() + _GEMINI_DATE_RULES


GEMINI_VISION_BATCH_PROMPT_TEMPLATE = """
Dobit ćeš {n} slika računa, tim redoslijedom kojim su poslane (prva slika = index 0, druga = index 1, itd).
Za SVAKU sliku vrati zaseban objekt u polju "receipts", s poljem "index" koje odgovara redoslijedu slike.
Ako neku sliku ne možeš pročitati, svejedno vrati objekt za taj index s praznim "items": [] i "unreadable": true.

Vrati isključivo valjani JSON bez markdowna, ove strukture:
{{
  "receipts": [
    {{
      "index": number,
      "items": [
        {{
          "description": string,
          "category": "Hrana | Cigarete, alkohol, kave,... | Kućne potrepštine | Kućni ljubimci | Lijekovi, troškovi liječenja | Odjeća i obuća | Škola i dječje aktivnosti | Sport | Automobili | Osiguranja | Internet/mobitel/TV | Struja | Voda | Plin | Smeće | Komunalni doprinos | Vodni doprinos | Putovanja, izleti, ručkovi | Ostalo" | null,
          "quantity": number | null,
          "unit_price": number | null,
          "total_price": number | null
        }}
      ],
      "total": number | null,
      "date_candidates": [string],
      "date": "YYYY-MM-DD" | null,
      "time": "HH:MM[:SS]" | null
    }}
  ]
}}
Pravila:
- decimalne zareze pretvori u točku
- nepoznate vrijednosti postavi na null
- datum određuj za SVAKU sliku zasebno, samo iz te slike

{date_rules}
""".strip()


@dataclass
class ReceiptItem:
    description: str
    total_price: Optional[float]
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    category: Optional[str] = None


@dataclass
class ReceiptData:
    image: str
    language: str
    items: List[ReceiptItem]
    items_sum: float
    total: Optional[float]
    date: Optional[str]
    time: Optional[str]
    warranty: bool = False
    image_hash: Optional[str] = None


_THREAD_LOCAL = threading.local()
_GEMINI_MAX_OUTPUT_TOKENS = min(MAX_NEW_TOKENS, 4096)
_GEMINI_IMAGE_MAX_SIZE = (1400, 2600)
_GEMINI_RATE_LOCK = threading.Lock()
_GEMINI_RECENT_REQUESTS: deque[float] = deque()
_GEMINI_WINDOW_SECONDS = 60.0
_GEMINI_MAX_PER_WINDOW = max(1, GEMINI_REQUESTS_PER_MINUTE)
_GEMINI_MIN_INTERVAL_SECONDS = _GEMINI_WINDOW_SECONDS / _GEMINI_MAX_PER_WINDOW


def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE NOT NULL,
                language TEXT,
                total REAL,
                items_sum REAL,
                date DATE,
                time TIME,
                warranty INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Ensure warranty column exists for older schemas
        try:
            conn.execute("ALTER TABLE receipts ADD COLUMN warranty INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        # image_hash: sha256 izvornih bajtova slike (prije EXIF/resize/format obrade),
        # koristi se za otkrivanje da je ista datoteka već uvezena (vidi find_receipt_by_hash).
        try:
            conn.execute("ALTER TABLE receipts ADD COLUMN image_hash TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipts_image_hash ON receipts(image_hash)"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS receipt_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                quantity REAL,
                unit_price REAL,
                total_price REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipt_items_receipt_id ON receipt_items(receipt_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipt_items_category ON receipt_items(category)"
        )
        conn.commit()

        _migrate_legacy_data_json(conn)
    finally:
        conn.close()


def _migrate_legacy_data_json(conn: sqlite3.Connection) -> None:
    """Jednokratna migracija sa starije sheme (data_json/json_path blob) na
    normalizirane retke u receipt_items. Sigurno se poziva i kad stupci
    data_json/json_path ne postoje (no-op)."""
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(receipts)")}
    if "data_json" not in existing_columns:
        return

    cur = conn.execute("SELECT id, data_json FROM receipts")
    rows = cur.fetchall()
    for row in rows:
        already_has_items = conn.execute(
            "SELECT 1 FROM receipt_items WHERE receipt_id = ? LIMIT 1", (row["id"],)
        ).fetchone()
        if already_has_items:
            continue
        try:
            data = json.loads(row["data_json"]) if row["data_json"] else {}
        except json.JSONDecodeError:
            data = {}
        for position, item in enumerate(data.get("items", [])):
            conn.execute(
                """
                INSERT INTO receipt_items
                    (receipt_id, position, description, category, quantity, unit_price, total_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    position,
                    item.get("description") or "",
                    item.get("category"),
                    _to_optional_float(item.get("quantity")),
                    _to_optional_float(item.get("unit_price")),
                    _to_optional_float(item.get("total_price")),
                ),
            )
    conn.commit()

    sqlite_version = tuple(int(part) for part in sqlite3.sqlite_version.split("."))
    if sqlite_version >= (3, 35, 0):
        for column in ("data_json", "json_path"):
            if column in existing_columns:
                try:
                    conn.execute(f"ALTER TABLE receipts DROP COLUMN {column}")
                except sqlite3.OperationalError:
                    pass
        conn.commit()


def _replace_receipt_items(conn: sqlite3.Connection, receipt_id: int, items: List[dict]) -> None:
    """Zamijeni sve stavke računa novim skupom (unutar postojeće transakcije)."""
    conn.execute("DELETE FROM receipt_items WHERE receipt_id = ?", (receipt_id,))
    conn.executemany(
        """
        INSERT INTO receipt_items
            (receipt_id, position, description, category, quantity, unit_price, total_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                receipt_id,
                position,
                item.get("description") or "",
                item.get("category"),
                _to_optional_float(item.get("quantity")),
                _to_optional_float(item.get("unit_price")),
                _to_optional_float(item.get("total_price")),
            )
            for position, item in enumerate(items)
        ],
    )


def fetch_receipt_items(receipt_id: int, db_path: str) -> List[dict]:
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            """
            SELECT description, category, quantity, unit_price, total_price
            FROM receipt_items
            WHERE receipt_id = ?
            ORDER BY position ASC
            """,
            (receipt_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_receipt_payload(row: sqlite3.Row, db_path: str) -> dict:
    """Rekonstruira payload dict (image/language/items/...) iz normaliziranih
    stupaca — isti oblik koji su template-i prije dobivali iz data_json bloba."""
    return {
        "image": row["image_path"],
        "language": row["language"],
        "items": fetch_receipt_items(row["id"], db_path),
        "items_sum": row["items_sum"],
        "total": row["total"],
        "date": row["date"],
        "time": row["time"],
        "warranty": bool(row["warranty"]),
    }


def save_receipt_to_db(receipt: ReceiptData, db_path: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO receipts (image_path, language, total, items_sum, date, time, warranty, image_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                language=excluded.language,
                total=excluded.total,
                items_sum=excluded.items_sum,
                date=excluded.date,
                time=excluded.time,
                warranty=excluded.warranty,
                image_hash=excluded.image_hash,
                updated_at=excluded.updated_at
            RETURNING id
            """,
            (
                receipt.image,
                receipt.language,
                receipt.total,
                receipt.items_sum,
                receipt.date,
                receipt.time,
                1 if receipt.warranty else 0,
                receipt.image_hash,
                now,
                now,
            ),
        )
        receipt_id = cur.fetchone()["id"]
        _replace_receipt_items(conn, receipt_id, [asdict(item) for item in receipt.items])
        conn.commit()
    finally:
        conn.close()


def fetch_all_receipts(
    db_path: str,
    sort_by: str = "updated_at",
    direction: str = "desc",
    limit: Optional[int] = None,
    filters: Optional[dict] = None,
) -> List[sqlite3.Row]:
    allowed_columns = {
        "id": "id",
        "image_path": "image_path",
        "total": "total",
        "items_sum": "items_sum",
        "date": "date",
        "time": "time",
        "updated_at": "updated_at",
        "warranty": "warranty",
    }
    sort_column = allowed_columns.get(sort_by, "updated_at")
    sort_direction = "ASC" if direction.lower() == "asc" else "DESC"

    def normalize_date(value: Optional[str]) -> Optional[str]:
        return _normalize_date_for_db(value)

    filter_clauses = []
    params: List[object] = []
    if filters:
        date_from = normalize_date(filters.get("date_from"))
        date_to = normalize_date(filters.get("date_to"))
        warranty = filters.get("warranty")
        total_min = filters.get("total_min")
        total_max = filters.get("total_max")
        if date_from:
            filter_clauses.append("date >= ?")
            params.append(date_from)
        if date_to:
            filter_clauses.append("date <= ?")
            params.append(date_to)
        if warranty in {"0", "1"}:
            filter_clauses.append("warranty = ?")
            params.append(int(warranty))
        if total_min is not None:
            filter_clauses.append("total >= ?")
            params.append(total_min)
        if total_max is not None:
            filter_clauses.append("total <= ?")
            params.append(total_max)
        item_search = filters.get("item_search")
        if item_search:
            filter_clauses.append(
                """EXISTS (
                    SELECT 1 FROM receipt_items ri
                    WHERE ri.receipt_id = receipts.id
                    AND (ri.description LIKE ? OR ri.category LIKE ?)
                )"""
            )
            params.append(f"%{item_search}%")
            params.append(f"%{item_search}%")

    where_clause = f"WHERE {' AND '.join(filter_clauses)}" if filter_clauses else ""

    conn = get_db_connection(db_path)
    try:
        query_params = list(params)
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ?"
            query_params.append(limit)
        cur = conn.execute(
            f"""
            SELECT id, image_path, total, items_sum, date, time, warranty, updated_at
            FROM receipts
            {where_clause}
            ORDER BY {sort_column} {sort_direction}{limit_clause}
            """,
            tuple(query_params),
        )
        return cur.fetchall()
    finally:
        conn.close()


def category_month_summary(db_path: str, year: int) -> tuple[dict, List[float]]:
    categories = [
        "Hrana","Cigarete, alkohol, kave,...","Kućne potrepštine","Kućni ljubimci","Lijekovi, troškovi liječenja",
        "Odjeća i obuća","Škola i dječje aktivnosti","Sport","Automobili","Osiguranja",
        "Internet/mobitel/TV","Struja","Voda","Plin","Smeće","Komunalni doprinos",
        "Vodni doprinos","Putovanja, izleti, ručkovi","Ostalo"
    ]
    summary = {category: [0.0] * 12 for category in categories}

    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            """
            SELECT r.date AS entry_date, ri.category AS category, ri.total_price AS total_price
            FROM receipts r
            JOIN receipt_items ri ON ri.receipt_id = r.id
            """
        )
        rows = cur.fetchall()
        for row in rows:
            entry_date = row["entry_date"]
            if not entry_date:
                continue
            parsed = None
            for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
                try:
                    parsed = datetime.strptime(entry_date, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None or parsed.year != year:
                continue
            month_index = parsed.month - 1
            category = row["category"] or "Ostalo"
            total_price = row["total_price"] or 0.0
            if category not in summary:
                summary[category] = [0.0] * 12
            summary[category][month_index] += float(total_price or 0.0)
    finally:
        conn.close()
    monthly_totals = [0.0] * 12
    for values in summary.values():
        for idx, value in enumerate(values):
            monthly_totals[idx] += value
    return summary, monthly_totals


def fetch_category_items_for_month(
    db_path: str,
    year: int,
    month: int,
    category: str,
) -> List[dict]:
    """
    Vrati sve stavke iz svih računa za zadani year+month i zadanu kategoriju.
    Svaka stavka nosi i ID računa, datum i vrijeme radi lakšeg klikanja natrag.
    """
    items: List[dict] = []
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            """
            SELECT r.id AS receipt_id, r.date AS entry_date, r.time AS entry_time,
                   ri.description AS description, ri.category AS category,
                   ri.quantity AS quantity, ri.unit_price AS unit_price, ri.total_price AS total_price
            FROM receipts r
            JOIN receipt_items ri ON ri.receipt_id = r.id
            """
        )
        for row in cur:
            entry_date = row["entry_date"]
            if not entry_date:
                continue

            parsed = None
            for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
                try:
                    parsed = datetime.strptime(entry_date, fmt)
                    break
                except ValueError:
                    continue

            if parsed is None:
                continue

            if parsed.year != year or parsed.month != month:
                continue

            item_category = row["category"] or "Ostalo"
            if item_category != category:
                continue

            items.append(
                {
                    "receipt_id": row["receipt_id"],
                    "description": row["description"] or "",
                    "category": item_category,
                    "quantity": row["quantity"],
                    "unit_price": row["unit_price"],
                    "total_price": row["total_price"],
                    "raw_date": entry_date,
                    "time": row["entry_time"],
                }
            )
    finally:
        conn.close()

    # Sortiraj po datumu/vremenu čisto radi UX-a
    items.sort(key=lambda x: (x["raw_date"] or "", x["time"] or ""))
    return items



def fetch_years(db_path: str) -> List[int]:
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT date FROM receipts WHERE date IS NOT NULL")
        years = set()
        for (date_value,) in cur:
            if not date_value:
                continue
            parsed_year = None
            for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
                try:
                    parsed_year = datetime.strptime(date_value, fmt).year
                    break
                except ValueError:
                    continue
            if parsed_year:
                years.add(parsed_year)
        return sorted(years)
    finally:
        conn.close()


def fetch_receipt_record(receipt_id: int, db_path: str) -> sqlite3.Row:
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM receipts WHERE id = ?", (receipt_id,))
        row = cur.fetchone()
        return row
    finally:
        conn.close()


def _sha256_of_file(path: str) -> str:
    """Sha256 sadržaja datoteke. Mora se zvati NA IZVORNIM bajtovima, prije
    normalize_image_orientation/ensure_gemini_compatible_image/resize_image —
    te funkcije mijenjaju sadržaj datoteke, pa bi hash izračunat nakon njih
    promašio duplikat kod ponovnog uploada iste izvorne slike."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_receipt_by_hash(image_hash: Optional[str], db_path: str) -> Optional[sqlite3.Row]:
    """Vrati postojeći račun s istim image_hash (bajt-identična slika već uvezena), ili None."""
    if not image_hash:
        return None
    conn = get_db_connection(db_path)
    try:
        return conn.execute(
            "SELECT id, date, total FROM receipts WHERE image_hash = ? LIMIT 1",
            (image_hash,),
        ).fetchone()
    finally:
        conn.close()


def update_receipt_record(receipt_id: int, data: dict, db_path: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE receipts
            SET total = ?, items_sum = ?, date = ?, time = ?, warranty = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                data.get("total"),
                data.get("items_sum"),
                data.get("date"),
                data.get("time"),
                1 if data.get("warranty") else 0,
                now,
                receipt_id,
            ),
        )
        _replace_receipt_items(conn, receipt_id, data.get("items", []))
        conn.commit()
    finally:
        conn.close()


def delete_receipt_record(receipt_id: int, db_path: str) -> None:
    """Briše račun iz baze; receipt_items se briše automatski (ON DELETE CASCADE, FK uključen)."""
    conn = get_db_connection(db_path)
    try:
        conn.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
        conn.commit()
    finally:
        conn.close()


def ensure_gemini_compatible_image(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return image_path

    target_path = os.path.splitext(image_path)[0] + ".jpg"
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(target_path, format="JPEG", optimize=True, quality=88)
    try:
        os.remove(image_path)
    except OSError:
        pass
    return target_path

def resize_image(image_path: str, max_size: tuple[int, int] = None) -> None:
    """
    Smanjuje sliku proporcionalno tako da širina/visina ne prelaze max_size.
    Ako je slika već manja od tih dimenzija – ne radi ništa.
    max_size dolazi iz configa ako nije ručno zadano.
    """

    if max_size is None:
        max_size = DEFAULT_RESIZE_MAX

    max_w, max_h = max_size

    try:
        with Image.open(image_path) as img:
            w, h = img.size

            # Ako je već dovoljno mala
            if w <= max_w and h <= max_h:
                return

            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = img.resize((new_w, new_h), Image.LANCZOS)

            # Ako je PNG → spremi kao PNG bez quality parametra
            if image_path.lower().endswith(".png"):
                resized.save(image_path, optimize=True)
            else:
                # Za JPG/WebP/TIF → koristimo kvalitetu
                resized.save(image_path, optimize=True, quality=85)

    except Exception as exc:
        # Ne prekida rad aplikacije
        log_progress(f"Neuspjelo smanjenje slike '{image_path}': {exc}")



def rotate_image_file(image_path: str, direction: str = "right") -> None:
    """
    Rotira sliku za 90° ulijevo ili udesno.
    direction: "left" ili "right" (default: "right")
    """
    try:
        with Image.open(image_path) as img:
            if direction == "left":
                angle = 90
            else:
                # "right" ili bilo što drugo -> udesno
                angle = -90
            img = img.rotate(angle, expand=True)
            img.save(image_path)
    except Exception as exc:
        log_progress(f"Neuspjela rotacija slike '{image_path}' ({direction}): {exc}")



def normalize_image_orientation(image_path: str) -> None:
    """
    Normalizira orijentaciju slike prema EXIF oznaci (ako postoji),
    i sprema rezultat nazad. Nakon ovoga pikseli su 'kako treba',
    a EXIF orijentacija više ne utječe na prikaz.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img.save(image_path)
    except Exception as exc:
        log_progress(f"Neuspjelo normaliziranje orijentacije '{image_path}': {exc}")


def _clean_string(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = f"{value}"
    text = str(value).strip()
    return text or None


def _to_optional_float(value: Optional[object]) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _format_decimal(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}".replace(".", ",")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*", "", stripped, count=1).strip()
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].strip()
    return stripped


def _extract_json_object(s: str) -> str | None:
    """
    Pokušaj izvući jedan JSON objekt {...} iz stringa, uzimajući u obzir ugniježđene
    zagrade i string literal-e. Ne oslanja se na regex ".*" koji zna biti previše pohlepan.
    """
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s[start:], start=start):
        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def _truncate_broken_items_array(s: str, key: str = '"items"') -> str:
    """
    Ako postoji polje "items": [ ... ] (ili "receipts": [ ... ] za batch odgovore,
    gdje svaki element sam sadrži svoj ugniježđeni "items": [...]) i zadnji
    element liste je napola (ili je model nakon toga nastavio brbljati),
    odreži sve iza zadnjeg POTPUNO zatvorenog top-level elementa te liste,
    zatvori listu, pa zatvori sve zagrade koje su bile otvorene prije nje.

    Depth-aware (prati ugnježđivanje i ignorira zagrade unutar stringova) —
    naivan rfind("]") bi za "receipts" slučaj znao pogoditi zatvaranje
    UNUTARNJEG "items" niza umjesto vanjskog "receipts" niza.

    Ideja: bolje izgubiti 1 polu-razvijeni element nego cijeli JSON.
    """
    key_idx = s.find(key)
    if key_idx == -1:
        return s

    bracket_start = s.find("[", key_idx)
    if bracket_start == -1:
        return s

    depth = 0
    in_string = False
    escape = False
    last_complete_end = None
    for i in range(bracket_start + 1, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            if ch == "}" and depth == 1:
                last_complete_end = i
            depth -= 1
            if depth <= 0 and ch == "]":
                # Lista se čisto zatvorila — ništa nije bilo slomljeno.
                return s

    if last_complete_end is None:
        return s

    prefix = s[:bracket_start]
    open_count = 0
    p_in_string = False
    p_escape = False
    for ch in prefix:
        if p_escape:
            p_escape = False
            continue
        if ch == "\\":
            p_escape = True
            continue
        if ch == '"':
            p_in_string = not p_in_string
            continue
        if p_in_string:
            continue
        if ch in "{[":
            open_count += 1
        elif ch in "}]":
            open_count -= 1

    # Sve zagrade otvorene prije ove liste su, za naš JSON oblik
    # ({"items"/"receipts": [...]} , po potrebi ugniježđeno), uvijek objekti.
    new_s = prefix + s[bracket_start : last_complete_end + 1] + "]" + ("}" * max(open_count, 0))

    return re.sub(r",(\s*[}\]])", r"\1", new_s)


def _repair_json_str(s: str) -> str:
    """
    Pokušaj “popraviti” tipične sitne greške:
    - u tekstu ima svega i svačega prije/iza JSON-a → uzmi samo objekt
    - trailing comma:  {...,} ili [...,]
    """
    s = s.strip()

    # uzmi samo dio od prvog '{' do zadnjeg '}' ako ništa pametnije ne znamo
    extracted = _extract_json_object(s)
    if extracted is not None:
        s = extracted
    else:
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            s = s[first : last + 1]

    # makni višak zareza prije '}' ili ']'
    s = re.sub(r",(\s*[}\]])", r"\1", s)

    return s


def parse_llm_json(text: str) -> dict:
    """
    Pokušaj više puta parsirati JSON:

    1) direktno (ako model već vraća čist JSON)
    2) skini code fence + opet probaj
    3) “repair” tipične greške (višak teksta, trailing comma, ...) i probaj treći put
    4) ako i to padne, pokušaj odrezati zadnju napola napisanu stavku u items[]

    Ako sve padne, digne ValueError s originalnim sadržajem (skraćenim u poruci).
    """
    raw = text
    stripped = _strip_code_fences(raw)

    # 1) direktno na stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2) pokušaj s pametnijim izdvajanjem i popravkom
    repaired = _repair_json_str(stripped)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 3) dodatni pokušaj: odreži potencijalno razvaljeni zadnji element u "items"
    salvaged = _truncate_broken_items_array(repaired)
    try:
        return json.loads(salvaged)
    except json.JSONDecodeError as exc:
        # zadnja šansa – digni “ljepšu” grešku, ali bez kilometarskog teksta
        preview = raw
        if len(preview) > 1200:
            preview = preview[:1200] + "... [skraceno]"
        raise ValueError(
            f"LLM response is not valid JSON ni nakon pokušaja popravka: {exc}\n\nSirovi odgovor modela:\n{preview}"
        )


def parse_llm_batch_json(text: str) -> dict:
    """
    Isto kao parse_llm_json, ali salvage korak (3) reže po "receipts" polju
    umjesto po "items" polju — koristi se za multi-image batch odgovore.
    """
    raw = text
    stripped = _strip_code_fences(raw)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    repaired = _repair_json_str(stripped)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    salvaged = _truncate_broken_items_array(repaired, key='"receipts"')
    try:
        return json.loads(salvaged)
    except json.JSONDecodeError as exc:
        preview = raw
        if len(preview) > 1200:
            preview = preview[:1200] + "... [skraceno]"
        raise ValueError(
            f"LLM batch response is not valid JSON ni nakon pokušaja popravka: {exc}\n\nSirovi odgovor modela:\n{preview}"
        )



def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png"}:
        return "image/png"
    if ext in {".webp"}:
        return "image/webp"
    if ext in {".jpg", ".jpeg", ".jfif"}:
        return "image/jpeg"
    # fallback – većina računa će biti jpg/png, ali neka
    return "image/jpeg"


class GeminiDailyQuotaExceeded(RuntimeError):
    """Iscrpljena dnevna Gemini kvota — retry nema smisla, čekaj do reset-a."""

    def __init__(self, reason: str, retry_after_seconds: Optional[float] = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


def _gemini_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "gemini_session", None)
    if session is not None:
        return session

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _THREAD_LOCAL.gemini_session = session
    return session


def _reserve_gemini_request_slot() -> None:
    """Klizni prozor: najviše _GEMINI_MAX_PER_WINDOW zahtjeva u 60 s, uz razmak između njih."""
    with _GEMINI_RATE_LOCK:
        while True:
            now = time.monotonic()
            cutoff = now - _GEMINI_WINDOW_SECONDS
            while _GEMINI_RECENT_REQUESTS and _GEMINI_RECENT_REQUESTS[0] <= cutoff:
                _GEMINI_RECENT_REQUESTS.popleft()

            if len(_GEMINI_RECENT_REQUESTS) >= _GEMINI_MAX_PER_WINDOW:
                # Prozor je pun – čekaj da najstariji zahtjev ispadne iz njega
                time.sleep(_GEMINI_RECENT_REQUESTS[0] + _GEMINI_WINDOW_SECONDS - now + 0.5)
                continue

            if _GEMINI_RECENT_REQUESTS:
                since_last = now - _GEMINI_RECENT_REQUESTS[-1]
                if since_last < _GEMINI_MIN_INTERVAL_SECONDS:
                    time.sleep(_GEMINI_MIN_INTERVAL_SECONDS - since_last)
                    continue

            _GEMINI_RECENT_REQUESTS.append(now)
            return


def _gemini_error_object(payload: object) -> dict:
    """Sigurno izvuci `error` objekt iz Gemini odgovora (prazan dict ako ga nema)."""
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error")
    return error if isinstance(error, dict) else {}


def _gemini_error_details(payload: object) -> List[dict]:
    """Vrati `error.details` unose koji su dictovi (prazna lista ako ih nema)."""
    details = _gemini_error_object(payload).get("details")
    if not isinstance(details, list):
        return []
    return [detail for detail in details if isinstance(detail, dict)]


def _parse_retry_delay_seconds(response: requests.Response) -> float:
    default_delay = _GEMINI_MIN_INTERVAL_SECONDS
    try:
        payload = response.json()
    except ValueError:
        payload = None

    for detail in _gemini_error_details(payload):
        retry_value = detail.get("retryDelay")
        if isinstance(retry_value, str) and retry_value.endswith("s"):
            try:
                return max(float(retry_value[:-1]), 0.5)
            except ValueError:
                continue

    text = response.text or ""
    match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", text)
    if match:
        try:
            return max(float(match.group(1)), 0.5)
        except ValueError:
            pass
    return default_delay


def _summarize_429_reason(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:200]

    message = _gemini_error_object(payload).get("message", "")
    quota_id = ""
    quota_metric = ""
    for detail in _gemini_error_details(payload):
        violations = detail.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if isinstance(violation, dict):
                quota_id = violation.get("quotaId") or quota_id
                quota_metric = violation.get("quotaMetric") or quota_metric

    parts = [part for part in (quota_id, quota_metric, message) if part]
    summary = " | ".join(parts) if parts else (response.text or "")
    return summary[:300]


def _classify_429(response: requests.Response) -> tuple[bool, str, float]:
    """Vrati (is_daily_quota, reason_summary, retry_after_seconds)."""
    reason = _summarize_429_reason(response)
    retry_after = _parse_retry_delay_seconds(response)
    is_daily = (
        "PerDay" in reason
        or "PerProjectPerModel-FreeTier" in reason
        or "free_tier_requests" in reason
    )
    return is_daily, reason, retry_after


def _gemini_api_key() -> str:
    api_key = os.environ.get(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Očekujem varijablu okoline {GEMINI_API_KEY_ENV} s Gemini API ključem."
        )
    return api_key


def _post_gemini_generate_content(payload: dict) -> dict:
    """POST na generateContent uz 429 retry i detekciju dnevne kvote.

    Zajednički za jednu-sliku i batch (multi-image) pozive — RPM/RPD budžet
    se troši po zahtjevu, bez obzira nosi li taj zahtjev 1 ili N slika.
    """
    api_key = _gemini_api_key()
    url = f"{GEMINI_ENDPOINT}/{GEMINI_MODEL}:generateContent?key={api_key}"
    for attempt in range(GEMINI_MAX_429_RETRIES + 1):
        _reserve_gemini_request_slot()
        resp = _gemini_session().post(url, json=payload, timeout=(10, 90))
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code != 429 or attempt >= GEMINI_MAX_429_RETRIES:
            raise RuntimeError(f"Gemini Vision API error {resp.status_code}: {resp.text}")
        is_daily, reason, sleep_seconds = _classify_429(resp)
        if is_daily:
            log_progress(f"Gemini DNEVNA kvota iscrpljena, prekidam batch. Razlog: {reason}")
            raise GeminiDailyQuotaExceeded(reason, retry_after_seconds=sleep_seconds)
        log_progress(
            f"Gemini quota/rate limit (429), čekam {sleep_seconds:.1f}s "
            f"prije ponovnog pokušaja ({attempt + 1}/{GEMINI_MAX_429_RETRIES}). "
            f"Razlog: {reason}"
        )
        time.sleep(sleep_seconds)
    raise AssertionError("unreachable")


def _image_to_inline_part(image_path: str) -> dict:
    mime_type = _guess_mime_type(image_path)
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    return {"inline_data": {"mime_type": mime_type, "data": img_base64}}


def call_gemini_vision_parser(image_path: str) -> dict:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": GEMINI_VISION_PROMPT},
                    _image_to_inline_part(image_path),
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": _GEMINI_MAX_OUTPUT_TOKENS,
            "thinkingConfig": {
                "thinkingBudget": 0
            },
        },
    }

    data = _post_gemini_generate_content(payload)
    try:
        model_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError(f"Neočekivan odgovor Gemini Vision modela: {data}") from exc

    return parse_llm_json(model_text)


def _gemini_batch_max_output_tokens(n: int) -> int:
    """Output-token budžet raste s brojem slika u zahtjevu (~300 tok./račun izmjereno)."""
    return min(MAX_NEW_TOKENS, max(4096, 700 * n))


def call_gemini_vision_batch_parser(image_paths: List[str]) -> Dict[int, Optional[dict]]:
    """Pošalji N slika u JEDNOM Gemini zahtjevu i vrati {index: raw_llm_payload}.

    KRITIČNO: ako Gemini ne vrati jasan, jedinstven "index" za neku sliku
    (nedostaje, dupliciran je, izvan raspona, ili je odgovor skraćen prije
    nego što je taj receipt stigao), ta slika NIJE obrađena — mapira se na
    None, i pozivatelj je NIKAD ne smije spremiti kao uspješan rezultat.
    """
    n = len(image_paths)
    batch_prompt = GEMINI_VISION_BATCH_PROMPT_TEMPLATE.format(n=n, date_rules=_GEMINI_DATE_RULES)
    parts = [{"text": batch_prompt}]
    parts.extend(_image_to_inline_part(p) for p in image_paths)

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": _gemini_batch_max_output_tokens(n),
            "thinkingConfig": {
                "thinkingBudget": 0
            },
        },
    }

    data = _post_gemini_generate_content(payload)
    try:
        model_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError(f"Neočekivan odgovor Gemini Vision modela (batch): {data}") from exc

    parsed = parse_llm_batch_json(model_text)
    receipts = parsed.get("receipts", []) if isinstance(parsed, dict) else []

    by_index: Dict[int, dict] = {}
    for r in receipts:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < n and idx not in by_index:
            by_index[idx] = r
        else:
            log_progress(f"Gemini batch: ignoriram receipt s nevaljanim/dupliciranim indexom {idx!r}")

    missing = [i for i in range(n) if i not in by_index]
    if missing:
        log_progress(f"Gemini batch: nije vraćen rezultat za {len(missing)}/{n} slika (indexi: {missing})")

    return {i: by_index.get(i) for i in range(n)}



def serialise_receipt(data: ReceiptData) -> dict:
    payload = asdict(data)
    payload["items"] = [asdict(item) for item in data.items]
    return payload


def prepare_image_for_gemini(image_path: str) -> str:
    if ENABLE_IMAGE_NORMALIZATION:
        normalize_image_orientation(image_path)
    image_path = ensure_gemini_compatible_image(image_path)
    resize_image(image_path, max_size=_GEMINI_IMAGE_MAX_SIZE)
    return image_path


def _package_processed_entry(used_path: str, receipt: ReceiptData, source_path: Optional[str]) -> dict:
    payload = serialise_receipt(receipt)
    payload.setdefault("warranty", False)
    return {
        "image_path": used_path,
        "preview_path": used_path,
        "payload": payload,
        "progress": [],
        "warranty": False,
        "source_path": source_path,
    }


def process_single_image(image_path: str, language: str, source_path: Optional[str] = None) -> dict:
    # Hash IZVORNIH bajtova, prije nego što prepare_image_for_gemini nešto promijeni na disku.
    image_hash = _sha256_of_file(image_path)
    used_path = prepare_image_for_gemini(image_path)
    receipt = build_receipt_data(used_path, language, image_hash=image_hash)
    return _package_processed_entry(used_path, receipt, source_path)


def _validate_image_readable(path: str) -> None:
    """Baca iznimku ako datoteka ne postoji, prazna je (0 bajtova) ili je PIL
    ne prepoznaje kao sliku. Koristi se da jedna neispravna slika (npr. 0-byte
    datoteka nastala neuspjelim uploadom) ne obori CIJELI Gemini batch zahtjev
    — cijeli chunk inače ide kao JEDAN HTTP poziv, pa neispravna slika unutar
    njega uzrokuje 400 za sve slike iz tog chunka, ne samo za neispravnu.
    """
    if not os.path.exists(path):
        raise ValueError("datoteka ne postoji na disku")
    if os.path.getsize(path) == 0:
        raise ValueError("datoteka je prazna (0 bajtova) — upload vjerojatno nije uspio")
    with Image.open(path) as img:
        img.verify()


def process_image_chunk(
    chunk: List[tuple[int, str, Optional[str], Optional[str]]], language: str
) -> Dict[int, dict]:
    """Obradi do GEMINI_IMAGES_PER_REQUEST slika kroz JEDAN Gemini zahtjev.

    `chunk`: [(global_idx, image_path, source_path, image_hash), ...]
    Vraća {global_idx: {"ok": entry} | {"error": poruka}}.

    Slika za koju Gemini nije vratio jasan index (vidi
    call_gemini_vision_batch_parser) NIKAD ne prolazi kroz
    _build_receipt_data_from_payload/_package_processed_entry — uvijek
    završava kao "error", nikad kao lažno uspješan rezultat.

    Prije slanja Gemini-ju svaka se slika provjerava (_validate_image_readable)
    — neispravna/prazna slika odmah postaje "error" SAMO za taj indeks i
    izbacuje se iz zahtjeva, umjesto da sruši cijeli chunk (i time i ostale,
    ispravne slike iz istog zahtjeva).
    """
    out: Dict[int, dict] = {}
    prepared: List[tuple[int, str, Optional[str], Optional[str]]] = []
    for gidx, path, src, image_hash in chunk:
        used_path = prepare_image_for_gemini(path)
        try:
            _validate_image_readable(used_path)
        except Exception as exc:
            out[gidx] = {"error": f"Slika je neispravna i preskočena: {exc}"}
            continue
        prepared.append((gidx, used_path, src, image_hash))

    if not prepared:
        return out

    batch_payloads = call_gemini_vision_batch_parser([p for _, p, _, _ in prepared])

    for local_i, (gidx, used_path, src, image_hash) in enumerate(prepared):
        llm_payload = batch_payloads.get(local_i)
        if llm_payload is None:
            out[gidx] = {"error": "Gemini nije vratio rezultat za ovu sliku u ovom batchu"}
            continue
        receipt = _build_receipt_data_from_payload(llm_payload, used_path, language, image_hash=image_hash)
        out[gidx] = {"ok": _package_processed_entry(used_path, receipt, src)}
    return out


def process_images_batch(
    jobs: List[tuple[str, Optional[str], Optional[str]]],
    language: str,
    on_status: Optional[Callable[[int, str, Optional[str]], None]] = None,
) -> tuple[List[dict], List[str]]:
    """Vraća (uspješni_rezultati, lista_grešaka).

    `jobs`: [(image_path, source_path, image_hash), ...] — image_hash je sha256
    izvornih bajtova (izračunat prije ove funkcije, prije bilo kakve obrade slike).

    `on_status(idx, status, error)` je opcionalni callback (status je
    "processing" | "done" | "error") kojim pozivatelj može pratiti napredak
    po pojedinačnoj slici dok batch teče (npr. za prikaz u UI-u).

    Slike se grupiraju u chunkove od GEMINI_IMAGES_PER_REQUEST i šalju
    Gemini-ju kao jedan zahtjev po chunku (RPD kvota je po zahtjevu, ne po
    slici) — svaki chunk se izvršava kao jedan posao u thread poolu, ali
    on_status/rezultati/greške i dalje su po pojedinačnoj slici.
    """
    if not jobs:
        return [], []

    indexed_jobs = list(enumerate(jobs))
    chunk_size = max(1, GEMINI_IMAGES_PER_REQUEST)
    chunks = [
        [
            (gidx, path, source_path, image_hash)
            for gidx, (path, source_path, image_hash) in indexed_jobs[i : i + chunk_size]
        ]
        for i in range(0, len(indexed_jobs), chunk_size)
    ]

    workers = min(max(1, GEMINI_MAX_PARALLEL), len(chunks))
    results: List[Optional[dict]] = [None] * len(jobs)
    errors: List[str] = []
    daily_quota_hit = threading.Event()
    quota_reason = ""

    def process_chunk_job(chunk: List[tuple[int, str, Optional[str], Optional[str]]]) -> Dict[int, dict]:
        """Circuit breaker: kad je dnevna kvota potrošena, preostali chunkovi odmah odustaju."""
        nonlocal quota_reason
        if daily_quota_hit.is_set():
            raise GeminiDailyQuotaExceeded(quota_reason or "preskočeno zbog dnevne kvote")
        if on_status:
            for gidx, _path, _src, _hash in chunk:
                on_status(gidx, "processing", None)
        try:
            return process_image_chunk(chunk, language)
        except GeminiDailyQuotaExceeded as exc:
            quota_reason = exc.reason
            daily_quota_hit.set()
            raise

    log_progress(
        f"Paralelna obrada računa: {len(jobs)} datoteka u {len(chunks)} chunk(ova) "
        f"(do {chunk_size} slika/zahtjev), {workers} radnika."
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_chunk_job, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk_results = future.result()
                for gidx, outcome in chunk_results.items():
                    if "ok" in outcome:
                        results[gidx] = outcome["ok"]
                        if on_status:
                            on_status(gidx, "done", None)
                    else:
                        failed_file = os.path.basename(jobs[gidx][0])
                        errors.append(f"{failed_file}: {outcome['error']}")
                        if on_status:
                            on_status(gidx, "error", outcome["error"])
            except GeminiDailyQuotaExceeded as exc:
                error_text = f"dnevna kvota ({exc.reason})"
                for gidx, _path, _src, _hash in chunk:
                    failed_file = os.path.basename(jobs[gidx][0])
                    errors.append(f"{failed_file}: {error_text}")
                    if on_status:
                        on_status(gidx, "error", error_text)
            except Exception as exc:
                log_progress(f"Greška pri obradi chunka: {exc}")
                for gidx, _path, _src, _hash in chunk:
                    failed_file = os.path.basename(jobs[gidx][0])
                    errors.append(f"{failed_file}: {exc}")
                    if on_status:
                        on_status(gidx, "error", str(exc))

    return [entry for entry in results if entry is not None], errors


# --- In-memory store za status batch obrade (za /batch/<id> live-status stranicu) ---
_upload_batches: Dict[str, dict] = {}
_upload_batches_lock = threading.Lock()
_BATCH_TTL_SECONDS = 7200  # 2h — čisti napuštene batcheve koje korisnik nikad nije pregledao


def _cleanup_expired_batches() -> None:
    now = time.time()
    with _upload_batches_lock:
        expired = [
            batch_id
            for batch_id, batch in _upload_batches.items()
            if now - batch["created_at"] > _BATCH_TTL_SECONDS
        ]
        for batch_id in expired:
            del _upload_batches[batch_id]


def _create_batch(
    jobs: List[tuple[str, Optional[str], Optional[str]]],
    lang: str,
    model: str,
    stored_verb: str,
    duplicates: Optional[List[dict]] = None,
) -> str:
    """`duplicates`: datoteke izbačene PRIJE OCR-a jer im je hash već u bazi —
    prikazuju se odvojeno na /batch/<id> (vidi BATCH_STATUS_TEMPLATE), nisu dio
    `jobs`/`files` liste pa ne ometaju idx-adresiranje u on_status callbacku."""
    _cleanup_expired_batches()
    batch_id = uuid4().hex
    files = [
        {"idx": idx, "name": os.path.basename(path), "status": "queued", "error": None}
        for idx, (path, _source, _hash) in enumerate(jobs)
    ]
    with _upload_batches_lock:
        _upload_batches[batch_id] = {
            "id": batch_id,
            "created_at": time.time(),
            "lang": lang,
            "model": model,
            "stored_verb": stored_verb,
            "files": files,
            "duplicates": duplicates or [],
            "status": "running",
            "processed_entries": None,
            "batch_errors": None,
        }
    return batch_id


def _set_batch_status(batch_id: str, idx: int, status: str, error: Optional[str] = None) -> None:
    with _upload_batches_lock:
        batch = _upload_batches.get(batch_id)
        if not batch:
            return
        batch["files"][idx]["status"] = status
        batch["files"][idx]["error"] = error


def _finish_batch(
    batch_id: str, processed_entries: List[dict], batch_errors: List[str]
) -> None:
    with _upload_batches_lock:
        batch = _upload_batches.get(batch_id)
        if not batch:
            return
        batch["status"] = "done"
        batch["processed_entries"] = processed_entries
        batch["batch_errors"] = batch_errors


def _get_batch(batch_id: str) -> Optional[dict]:
    with _upload_batches_lock:
        return _upload_batches.get(batch_id)


def _pop_batch(batch_id: str) -> Optional[dict]:
    with _upload_batches_lock:
        return _upload_batches.pop(batch_id, None)


def _run_batch_in_background(
    batch_id: str, jobs: List[tuple[str, Optional[str], Optional[str]]], lang: str
) -> None:
    def on_status(idx: int, status: str, error: Optional[str] = None) -> None:
        _set_batch_status(batch_id, idx, status, error)

    try:
        processed_entries, batch_errors = process_images_batch(jobs, lang, on_status=on_status)
    except Exception as exc:
        log_progress(f"Batch {batch_id} neočekivano pukao: {exc}")
        processed_entries, batch_errors = [], [str(exc)]
    _finish_batch(batch_id, processed_entries, batch_errors)


def _start_batch_or_redirect(
    upload_jobs: List[tuple[str, Optional[str], Optional[str]]],
    duplicate_notices: List[dict],
    upload_lang: str,
    upload_model: str,
    stored_verb: str,
):
    """Zajednička logika za /upload i /import_onedrive nakon što su datoteke
    spremljene/kopirane na disk i podijeljene po image_hash provjeri u:
    `upload_jobs` (idu na OCR) i `duplicate_notices` (bajt-identične već
    postojećem računu, preskočene PRIJE OCR-a).

    - Točno jedna poslana datoteka i ta je duplikat -> direktno na tu postojeću
      /receipt/<id> stranicu (nema smisla prikazivati prazan batch UI).
    - Ima nešto za OCR -> normalan batch u pozadini; duplikati (ako ih ima) se
      prikazuju uz njega na /batch/<id>.
    - Sve je duplikat (upload_jobs prazan, a duplicate_notices nije) -> batch
      se odmah označi gotovim (bez pozadinske dretve), redirect i dalje ide na
      /batch/<id> da korisnik vidi popis s linkovima na postojeće račune.
    """
    if not upload_jobs and len(duplicate_notices) == 1:
        return redirect(url_for("receipt_detail", receipt_id=duplicate_notices[0]["receipt_id"]))

    batch_id = _create_batch(
        upload_jobs, upload_lang, upload_model, stored_verb, duplicates=duplicate_notices
    )
    if upload_jobs:
        threading.Thread(
            target=_run_batch_in_background,
            args=(batch_id, upload_jobs, upload_lang),
            daemon=True,
        ).start()
    else:
        _finish_batch(batch_id, [], [])
    return redirect(url_for("upload_batch_status", batch_id=batch_id))


def _delete_onedrive_source(source_path: str) -> None:
    """Obriši originalnu datoteku (source_path) iz OneDrive foldera, ako postoji.

    Koristi se i kad je uvezeni račun uspješno spremljen, i kad je odbačen —
    u oba slučaja ne želimo da se ista datoteka ponovno uveze (i reprocesira)
    kod sljedećeg /import_onedrive. Za obične uploade source_path je prazan.
    """
    if not source_path:
        return
    abs_source = os.path.abspath(source_path)
    try:
        if os.path.exists(abs_source):
            os.remove(abs_source)
            log_progress(f"Obrisan originalni OneDrive fajl: {abs_source}")
    except OSError as exc:
        log_progress(f"Ne mogu obrisati izvorni OneDrive fajl '{abs_source}': {exc}")


def batch_failure_message(batch_errors: List[str], image_count: int, stored_verb: str) -> str:
    """Poruka za korisnika kad nijedna slika iz batcha nije obrađena."""
    if any("dnevna kvota" in error for error in batch_errors):
        return (
            f"Iscrpljena dnevna Gemini kvota. {image_count} slika je {stored_verb} "
            f"u uploads/ ali nije obrađeno. Pokušaj ponovno nakon resetiranja kvote "
            f"(~09:00 hrvatskog vremena)."
        )
    if batch_errors:
        return "Nijedna slika nije uspješno obrađena. Greške: " + "; ".join(batch_errors)
    return "Nijedna slika nije uspješno obrađena."


def _render_pending_entry_page(
    entry: dict,
    pending_payloads: List[dict],
    upload_lang: str,
    upload_model: str,
    extra_progress: Optional[List[str]] = None,
) -> str:
    """Prikaži jedan već-obrađen (ali još nespremljen) račun za review.

    Dijeljeno između: prvog prikaza review stranice nakon batcha, i nastavka
    na sljedeći pending račun nakon što je korisnik spremio ili odbacio prethodni.
    """
    preview_image_path = entry.get("preview_path", entry["image_path"])

    preview_image_mtime = None
    if preview_image_path and os.path.exists(preview_image_path):
        preview_image_mtime = int(os.path.getmtime(preview_image_path))

    rotate_target = (
        preview_image_path
        if preview_image_path and not preview_image_path.startswith("manual://")
        else None
    )

    progress = list(entry.get("progress", []))
    if extra_progress:
        progress.extend(extra_progress)

    return render_template_string(
        DETAIL_TEMPLATE,
        receipt={"id": None, "image_path": entry["image_path"]},
        data=entry["payload"],
        items=entry["payload"]["items"],
        saved=False,
        is_new=True,
        image_path=entry["image_path"],
        preview_image_path=preview_image_path,
        preview_image_mtime=preview_image_mtime,
        base_payload=entry["payload"],
        progress=progress,
        default_lang=upload_lang,
        default_model=upload_model,
        pending_payloads=pending_payloads,
        pending_count=len(pending_payloads),
        rotate_target=rotate_target,
        current_url=request.url,
        format_date=_format_date_for_display,
        form_error=None,
        source_path=entry.get("source_path"),
    )


def _render_next_pending_entry(
    pending_payloads: List[dict], upload_lang: str, upload_model: str
) -> str:
    """Skida sljedeći račun s pending liste (in-place) i prikazuje ga za review."""
    next_entry = pending_payloads.pop(0)
    resize_image(next_entry["image_path"])
    return _render_pending_entry_page(next_entry, pending_payloads, upload_lang, upload_model)


def render_review_page(
    processed_entries: List[dict],
    batch_errors: List[str],
    upload_lang: str,
    upload_model: str,
) -> str:
    """Prikaži prvi obrađeni račun za pregled; ostali čekaju u pending listi."""
    first_entry = processed_entries[0]
    pending_entries = processed_entries[1:]

    extra_progress = []
    if batch_errors:
        extra_progress.append(f"⚠ Neuspjelo ({len(batch_errors)}): " + "; ".join(batch_errors))

    return _render_pending_entry_page(
        first_entry, pending_entries, upload_lang, upload_model, extra_progress
    )


def create_app(db_path: str, default_lang: str, default_model: str) -> Flask:
    app = Flask(__name__)

    def _icon_response() -> Response:
        return Response(
            _APP_ICON_PNG,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.route("/favicon.png")
    def app_icon() -> Response:
        return _icon_response()

    # Preglednici sami traže /favicon.ico; isti PNG (moderni preglednici ga prihvaćaju).
    @app.route("/favicon.ico")
    def app_icon_ico() -> Response:
        return _icon_response()

    @app.route("/", methods=["GET"])
    def index() -> str:
        # Defaultno: sortiraj po datumu (noviji prvo) i prikaži zadnjih 100 računa
        sort_by = request.args.get("sort", "date")
        direction = request.args.get("dir", "desc")
        limit_param = request.args.get("limit", "100")
        allowed_limits = ["100", "200", "500", "all"]

        if limit_param not in allowed_limits:
            limit_param = "all"
        limit_value = int(limit_param) if limit_param != "all" else None
        total_min_raw = request.args.get("total_min")
        total_max_raw = request.args.get("total_max")
        filters = {
            "date_from": request.args.get("date_from") or None,
            "date_to": request.args.get("date_to") or None,
            "total_min": _to_optional_float(total_min_raw) if total_min_raw else None,
            "total_max": _to_optional_float(total_max_raw) if total_max_raw else None,
            "warranty": request.args.get("warranty_filter") if request.args.get("warranty_filter") in {"0", "1"} else None,
            "item_search": request.args.get("item_search", "").strip() or None,
        }
        receipts = fetch_all_receipts(db_path, sort_by, direction, limit_value, filters)
        available_years = fetch_years(db_path)
        if available_years:
            try:
                selected_year = int(request.args.get("year", available_years[-1]))
            except ValueError:
                selected_year = available_years[-1]
            if selected_year not in available_years:
                selected_year = available_years[-1]
        else:
            selected_year = datetime.now().year
        category_summary, monthly_totals = category_month_summary(db_path, selected_year)
                # --- Statistika za OneDrive folder (broj slikovnih datoteka) ---
        onedrive_total_files = None
        if ONEDRIVE_IMPORT_DIR:
            abs_source = os.path.abspath(ONEDRIVE_IMPORT_DIR)
            if os.path.isdir(abs_source):
                exts = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".tif", ".tiff"}
                try:
                    count = 0
                    for entry in os.listdir(abs_source):
                        full_path = os.path.join(abs_source, entry)
                        if not os.path.isfile(full_path):
                            continue
                        _, ext = os.path.splitext(entry)
                        if ext.lower() in exts:
                            count += 1
                    onedrive_total_files = count
                except OSError as exc:
                    log_progress(f"Ne mogu pročitati OneDrive folder '{abs_source}': {exc}")
                    onedrive_total_files = None

        return render_template_string(
            INDEX_TEMPLATE,
            receipts=receipts,
            error_message=request.args.get("error"),
            progress=[],
            sort_by=sort_by,
            direction=direction,
            filters=filters,
            limit_value=limit_param,
            allowed_limits=allowed_limits,
            category_summary=category_summary,
            current_year=selected_year,
            available_years=available_years,
            monthly_totals=monthly_totals,
            month_names=["Siječanj","Veljača","Ožujak","Travanj","Svibanj","Lipanj","Srpanj","Kolovoz","Rujan","Listopad","Studeni","Prosinac"],
            format_date=_format_date_for_display,
            onedrive_default_path=ONEDRIVE_IMPORT_DIR,
            onedrive_total_files=onedrive_total_files,
            onedrive_import_limit=MAX_UPLOAD_FILES,
        )

    @app.route("/category_items", methods=["GET"])
    def category_items() -> str:
        category = request.args.get("category") or ""
        year = request.args.get("year", type=int)
        month = request.args.get("month", type=int)

        if not category or not year or not month:
            # Ako nešto fali, samo nazad na početnu
            return redirect(url_for("index"))

        items = fetch_category_items_for_month(db_path, year, month, category)

        month_names = [
            "Siječanj","Veljača","Ožujak","Travanj","Svibanj","Lipanj",
            "Srpanj","Kolovoz","Rujan","Listopad","Studeni","Prosinac",
        ]
        if 1 <= month <= 12:
            month_name = month_names[month - 1]
        else:
            month_name = f"Mjesec {month}"

        total_amount = sum((item["total_price"] or 0.0) for item in items)
        back_url = url_for("index", year=year) + "#categories-section"

        return render_template_string(
            CATEGORY_ITEMS_TEMPLATE,
            category=category,
            year=year,
            month_name=month_name,
            items=items,
            total_amount=total_amount,
            back_url=back_url,
            format_date=_format_date_for_display,
        )


    @app.route("/upload", methods=["POST"])
    def upload_receipt() -> str:
        upload_files = [f for f in request.files.getlist("image") if f and f.filename]
        manual_entry = request.form.get("manual") == "on"

        # Ograničenje: max MAX_UPLOAD_FILES datoteka odjednom
        if upload_files and len(upload_files) > MAX_UPLOAD_FILES:
            return redirect(
                url_for(
                    "index",
                    error=f"Maksimalno je dopušteno učitati {MAX_UPLOAD_FILES} datoteka odjednom (pokušali ste {len(upload_files)}).",
                )
            )

        if not upload_files and not manual_entry:
            return redirect(url_for("index"))

        upload_lang = default_lang
        upload_model = default_model

        upload_dir = UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)

        # --- Ručni unos bez slike ---
        if manual_entry and not upload_files:
            manual_identifier = f"manual://{int(datetime.now().timestamp() * 1000)}_{uuid4().hex[:6]}"
            empty_payload = {
                "image": manual_identifier,
                "language": upload_lang,
                "items": [],
                "items_sum": 0.0,
                "total": None,
                "date": datetime.now().strftime("%d.%m.%Y"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "warranty": False,
            }
            return render_template_string(
                DETAIL_TEMPLATE,
                receipt={"id": None, "image_path": manual_identifier},
                data=empty_payload,
                items=[],
                saved=False,
                is_new=True,
                image_path=manual_identifier,
                preview_image_path="",
                base_payload=empty_payload,
                progress=[],
                default_lang=upload_lang,
                default_model=upload_model,
                pending_payloads=[],
                pending_count=0,
                rotate_target=None,
                current_url=request.url,
                format_date=_format_date_for_display,
                form_error=None,
            )

        # --- Obrada uploadanih slika ---
        upload_jobs: List[tuple[str, Optional[str], Optional[str]]] = []
        duplicate_notices: List[dict] = []
        for upload_file in upload_files:
            original_name = secure_filename(upload_file.filename)
            if not original_name:
                original_name = "receipt.png"

            name, ext = os.path.splitext(original_name)
            if not ext:
                ext = ".png"

            # Jedinstveni sufiks: timestamp + par znakova iz UUID-a
            unique_suffix = f"{int(datetime.now().timestamp() * 1000)}_{uuid4().hex[:6]}"
            filename = f"{name}_{unique_suffix}{ext}"

            temp_path = os.path.join(upload_dir, filename)
            upload_file.save(temp_path)

            # Hash IZVORNIH bajtova (prije OCR obrade) — ako je bajt-identična slika
            # već negdje spremljena kao račun, preskoči OCR i ne troši Gemini kvotu.
            file_hash = _sha256_of_file(temp_path)
            existing = find_receipt_by_hash(file_hash, db_path)
            if existing:
                os.remove(temp_path)
                duplicate_notices.append(
                    {
                        "name": original_name,
                        "receipt_id": existing["id"],
                        "date": existing["date"],
                        "total": existing["total"],
                    }
                )
                continue

            upload_jobs.append((temp_path, None, file_hash))

        return _start_batch_or_redirect(
            upload_jobs, duplicate_notices, upload_lang, upload_model, "spremljeno"
        )

    @app.route("/import_onedrive", methods=["POST"])
    def import_onedrive() -> str:
        """
        Uvezi slike računa iz lokalno syncanog OneDrive foldera
        i pripremi ih za review, kao da su upload-ane kroz formu.
        """
        source_dir = (request.form.get("onedrive_path") or ONEDRIVE_IMPORT_DIR or "").strip()
        upload_lang = default_lang
        upload_model = default_model

        if not source_dir:
            return redirect(
                url_for(
                    "index",
                    error="OneDrive putanja nije postavljena. Unesi putanju ili postavi ONEDRIVE_IMPORT_DIR.",
                )
            )

        abs_source = os.path.abspath(source_dir)
        if not os.path.isdir(abs_source):
            return redirect(
                url_for(
                    "index",
                    error=f"OneDrive putanja ne postoji ili nije direktorij: {abs_source}",
                )
            )

        # Skupi sve image fajlove iz tog foldera
        exts = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".tif", ".tiff"}
        all_files = []
        for entry in sorted(os.listdir(abs_source)):
            full_path = os.path.join(abs_source, entry)
            if not os.path.isfile(full_path):
                continue
            _, ext = os.path.splitext(entry)
            if ext.lower() in exts:
                all_files.append(full_path)

        if not all_files:
            return redirect(
                url_for(
                    "index",
                    error=f"Nema slikovnih datoteka u folderu: {abs_source}",
                )
            )

        # Poštuj ograničenje MAX_UPLOAD_FILES
        image_paths = all_files[:MAX_UPLOAD_FILES]

        upload_dir = UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)

        upload_jobs: List[tuple[str, Optional[str], Optional[str]]] = []
        duplicate_notices: List[dict] = []

        for src_path in image_paths:
            original_name = secure_filename(os.path.basename(src_path)) or "receipt.png"

            # Hash izvornika PRIJE kopiranja/OCR-a — ako je već uvezen, preskoči
            # kopiranje u uploads/ i odmah očisti izvornik (inače bi se svaki
            # sljedeći import ponovno "spotaknuo" o istu datoteku).
            file_hash = _sha256_of_file(src_path)
            existing = find_receipt_by_hash(file_hash, db_path)
            if existing:
                duplicate_notices.append(
                    {
                        "name": original_name,
                        "receipt_id": existing["id"],
                        "date": existing["date"],
                        "total": existing["total"],
                    }
                )
                _delete_onedrive_source(src_path)
                continue

            name, ext = os.path.splitext(original_name)
            if not ext:
                ext = ".png"

            unique_suffix = f"{int(datetime.now().timestamp() * 1000)}_{uuid4().hex[:6]}"
            filename = f"{name}_{unique_suffix}{ext}"

            temp_path = os.path.join(upload_dir, filename)
            # Kopiraj iz OneDrive foldera u uploads/
            shutil.copy2(src_path, temp_path)
            upload_jobs.append((temp_path, src_path, file_hash))

        return _start_batch_or_redirect(
            upload_jobs, duplicate_notices, upload_lang, upload_model, "kopirano"
        )

    @app.route("/batch/<batch_id>", methods=["GET"])
    def upload_batch_status(batch_id: str) -> str:
        batch = _get_batch(batch_id)
        if not batch:
            return redirect(
                url_for("index", error="Obrada nije pronađena (možda je istekla).")
            )
        return render_template_string(
            BATCH_STATUS_TEMPLATE,
            batch_id=batch_id,
            files=batch["files"],
            duplicates=batch.get("duplicates") or [],
            format_date=_format_date_for_display,
        )

    @app.route("/batch/<batch_id>/status.json", methods=["GET"])
    def upload_batch_status_json(batch_id: str):
        batch = _get_batch(batch_id)
        if not batch:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"done": batch["status"] == "done", "files": batch["files"]})

    @app.route("/batch/<batch_id>/review", methods=["GET"])
    def upload_batch_review(batch_id: str) -> str:
        batch = _get_batch(batch_id)
        if not batch:
            return redirect(
                url_for("index", error="Obrada nije pronađena (možda je istekla).")
            )
        if batch["status"] != "done":
            return redirect(url_for("upload_batch_status", batch_id=batch_id))

        batch = _pop_batch(batch_id) or batch
        processed_entries = batch["processed_entries"] or []
        batch_errors = batch["batch_errors"] or []

        if not processed_entries:
            duplicates = batch.get("duplicates") or []
            if duplicates and not batch_errors:
                # Sve slike u ovom batchu su prepoznate kao već uvezene (isti image_hash) —
                # nema "greške", samo nema ničeg novog za pregled.
                error_msg = (
                    f"Sve {len(duplicates)} slika(e) su prepoznate kao već uvezene (identičan "
                    f"sadržaj datoteke) — ništa novo za pregled."
                )
            else:
                error_msg = batch_failure_message(
                    batch_errors, len(batch["files"]), batch["stored_verb"]
                )
            return redirect(url_for("index", error=error_msg))

        return render_review_page(
            processed_entries, batch_errors, batch["lang"], batch["model"]
        )

    @app.route("/receipt/save_new", methods=["POST"])
    def save_new_receipt() -> str:
        image_path = request.form.get("image_path") or ""
        base_payload_raw = request.form.get("base_payload")
        pending_payloads_raw = request.form.get("pending_payloads", "[]")
        if not base_payload_raw:
            abort(400)
        try:
            base_payload = json.loads(base_payload_raw)
        except json.JSONDecodeError:
            abort(400)
        try:
            pending_payloads = json.loads(pending_payloads_raw) if pending_payloads_raw else []
        except json.JSONDecodeError:
            pending_payloads = []

        updated_payload = apply_form_updates(base_payload, request.form)
        updated_payload["image"] = os.path.abspath(image_path)

        # --- VALIDACIJA: datum & total (obavezni) + datum ne u budućnosti ---
        error_msg = _validate_receipt_payload(updated_payload, require_date=True, require_total=True)
        if error_msg:
            # Rekonstruiraj preview za ponovno prikazivanje forme
            preview_image_path = request.form.get("preview_image_path") or image_path
            preview_image_mtime = None
            if preview_image_path and os.path.exists(preview_image_path):
                preview_image_mtime = int(os.path.getmtime(preview_image_path))

            items = updated_payload.get("items", [])

            return render_template_string(
                DETAIL_TEMPLATE,
                receipt={"id": None, "image_path": image_path},
                data=updated_payload,
                items=items,
                saved=False,
                is_new=True,
                image_path=image_path,
                preview_image_path=preview_image_path,
                preview_image_mtime=preview_image_mtime,
                base_payload=updated_payload,
                progress=[],
                default_lang=default_lang,
                default_model=default_model,
                pending_payloads=pending_payloads,
                pending_count=len(pending_payloads),
                rotate_target=(preview_image_path if preview_image_path and not preview_image_path.startswith("manual://") else None),
                current_url=request.url,
                format_date=_format_date_for_display,
                form_error=error_msg,
                source_path=request.form.get("source_path") or "",
            )

        # --- Ako je sve u redu, spremi u DB ---
        receipt_obj = receipt_from_payload(updated_payload)
        save_receipt_to_db(receipt_obj, db_path)

        # Ako je ovaj račun uvezen iz OneDrive-a i sad je uspješno spremljen,
        # obriši originalnu datoteku (source_path) iz OneDrive foldera.
        _delete_onedrive_source(request.form.get("source_path") or "")


        if pending_payloads:
            return _render_next_pending_entry(pending_payloads, default_lang, default_model)

        if image_path and not image_path.startswith("manual://"):
            resize_image(image_path)
        return redirect(url_for("index"))

    @app.route("/receipt/discard_new", methods=["POST"])
    def discard_new_receipt() -> str:
        """Odbaci trenutni (još nespremljeni) račun iz review niza.

        Račun se NE sprema u bazu, a njegova slika u uploads/ se briše s diska.
        Ako je uvezen iz OneDrive-a, briše se i izvornik u OneDrive folderu
        (isto kao kod uspješnog spremanja) — inače bi se kod sljedećeg importa
        isti (npr. duplicirani ili neispravni) račun samo ponovno uvezao.
        Koristi se kad je slika neispravna ili je isti račun već unesen (duplikat).
        """
        image_path = request.form.get("image_path") or ""
        pending_payloads_raw = request.form.get("pending_payloads", "[]")
        try:
            pending_payloads = json.loads(pending_payloads_raw) if pending_payloads_raw else []
        except json.JSONDecodeError:
            pending_payloads = []

        if image_path and not image_path.startswith("manual://"):
            abs_path = os.path.abspath(image_path)
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    log_progress(f"Odbačen račun, obrisana slika: {abs_path}")
            except OSError as exc:
                log_progress(f"Ne mogu obrisati odbačenu sliku '{abs_path}': {exc}")

        _delete_onedrive_source(request.form.get("source_path") or "")

        if pending_payloads:
            return _render_next_pending_entry(pending_payloads, default_lang, default_model)

        return redirect(url_for("index"))

    @app.route("/receipt/<int:receipt_id>/attach_image", methods=["POST"])
    def attach_image(receipt_id: int):
        """Pridružuje uploadanu sliku postojećem računu (npr. ručnom unosu)."""
        row = fetch_receipt_record(receipt_id, db_path)
        if row is None:
            abort(404)

        upload_file = request.files.get("image")
        if not upload_file or not upload_file.filename:
            return redirect(url_for("receipt_detail", receipt_id=receipt_id))

        original_name = secure_filename(upload_file.filename) or "receipt.png"
        name, ext = os.path.splitext(original_name)
        if not ext:
            ext = ".png"

        unique_suffix = f"{int(datetime.now().timestamp() * 1000)}_{uuid4().hex[:6]}"
        filename = f"receipt_{receipt_id}_{unique_suffix}{ext}"

        upload_dir = UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        saved_path = os.path.join(upload_dir, filename)
        upload_file.save(saved_path)

        # Hash izvornih bajtova, prije nego što ih normalize/resize promijeni na disku.
        file_hash = _sha256_of_file(saved_path)

        normalize_image_orientation(saved_path)
        saved_path = ensure_gemini_compatible_image(saved_path)
        resize_image(saved_path)

        abs_path = os.path.abspath(saved_path)

        conn = get_db_connection(db_path)
        try:
            conn.execute(
                "UPDATE receipts SET image_path = ?, image_hash = ?, updated_at = ? WHERE id = ?",
                (abs_path, file_hash, datetime.now().isoformat(timespec="seconds"), receipt_id),
            )
            conn.commit()
        finally:
            conn.close()

        return redirect(url_for("receipt_detail", receipt_id=receipt_id, saved=1))

    @app.route("/receipt/<int:receipt_id>/image")
    def receipt_image(receipt_id: int):
        row = fetch_receipt_record(receipt_id, db_path)
        if row is None:
            abort(404)
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            abort(404)
        return send_file(image_path)

    @app.route("/preview_image")
    def preview_image():
        path = request.args.get("path")
        if not path:
            abort(404)
        abs_path = os.path.abspath(path)
        upload_dir = os.path.abspath(UPLOAD_DIR)
        if not abs_path.startswith(upload_dir):
            abort(403)
        if not os.path.exists(abs_path):
            abort(404)
        return send_file(abs_path)

    @app.route("/rotate_image_action", methods=["GET", "POST"])
    def rotate_image_action():
        # POST branch: new/unsaved receipt (coming from upload preview)
        if request.method == "POST":
            path = request.form.get("path")
            if not path or path.startswith("manual://"):
                return redirect(url_for("index"))

            abs_path = os.path.abspath(path)
            uploads_dir = os.path.abspath(UPLOAD_DIR)
            if not os.path.exists(abs_path) or not abs_path.startswith(uploads_dir):
                return redirect(url_for("index"))

            direction = request.form.get("direction", "right")
            # Rotate the image file on disk
            rotate_image_file(abs_path, direction=direction)

            # Rebuild context for DETAIL_TEMPLATE
            image_path = request.form.get("image_path") or path
            preview_image_path = request.form.get("preview_image_path") or image_path

            base_payload_raw = request.form.get("base_payload") or "{}"
            pending_payloads_raw = request.form.get("pending_payloads") or "[]"
            source_path = request.form.get("source_path") or ""

            try:
                base_payload = json.loads(base_payload_raw)
            except json.JSONDecodeError:
                base_payload = {}

            try:
                pending_payloads = json.loads(pending_payloads_raw)
            except json.JSONDecodeError:
                pending_payloads = []

            items = base_payload.get("items", [])

            resolved_lang = request.form.get("default_lang") or default_lang
            resolved_model = request.form.get("default_model") or default_model

            preview_image_mtime = None
            if preview_image_path and os.path.exists(preview_image_path):
                preview_image_mtime = int(os.path.getmtime(preview_image_path))

            rotate_target = (
                preview_image_path
                if preview_image_path and not preview_image_path.startswith("manual://")
                else None
            )

            return render_template_string(
                DETAIL_TEMPLATE,
                receipt={"id": None, "image_path": image_path},
                data=base_payload,
                items=items,
                saved=False,
                is_new=True,
                image_path=image_path,
                preview_image_path=preview_image_path,
                preview_image_mtime=preview_image_mtime,
                base_payload=base_payload,
                progress=[],
                default_lang=resolved_lang,
                default_model=resolved_model,
                pending_payloads=pending_payloads,
                pending_count=len(pending_payloads),
                rotate_target=rotate_target,
                current_url=request.url,
                format_date=_format_date_for_display,
                form_error=None,
                source_path=source_path,
            )

        # GET branch: existing saved receipts
        path = request.args.get("path")
        next_url = request.args.get("next")
        direction = request.args.get("direction", "right")
        fallback = url_for("index")

        if not path or path.startswith("manual://"):
            return redirect(next_url or fallback)

        abs_path = os.path.abspath(path)
        uploads_dir = os.path.abspath(UPLOAD_DIR)
        if not os.path.exists(abs_path) or not abs_path.startswith(uploads_dir):
            return redirect(next_url or fallback)

        rotate_image_file(abs_path, direction=direction)

        if next_url and not next_url.endswith("/upload"):
            return redirect(next_url)

        return redirect(fallback)



    @app.route("/receipt/<int:receipt_id>", methods=["GET", "POST"])
    def receipt_detail(receipt_id: int):
        row = fetch_receipt_record(receipt_id, db_path)
        if row is None:
            abort(404)
        data = fetch_receipt_payload(row, db_path)
        items = data.get("items", [])
        rotate_target = row["image_path"] if row["image_path"] and not row["image_path"].startswith("manual://") else None

        if request.method == "POST":
            updated_payload = apply_form_updates(data, request.form)

            # --- VALIDACIJA: dopusti prazno, ali ne dopuštaj budući datum ---
            error_msg = _validate_receipt_payload(updated_payload, require_date=False, require_total=False)
            if error_msg:
                items = updated_payload.get("items", [])
                preview_image_path = row["image_path"]
                preview_image_mtime = None
                if preview_image_path and os.path.exists(preview_image_path):
                    preview_image_mtime = int(os.path.getmtime(preview_image_path))

                return render_template_string(
                    DETAIL_TEMPLATE,
                    receipt=row,
                    data=updated_payload,
                    items=items,
                    saved=False,
                    is_new=False,
                    progress=[],
                    image_path=row["image_path"],
                    preview_image_path=preview_image_path,
                    preview_image_mtime=preview_image_mtime,
                    base_payload=updated_payload,
                    default_lang=default_lang,
                    default_model=default_model,
                    pending_payloads=[],
                    pending_count=0,
                    rotate_target=rotate_target,
                    current_url=request.url,
                    format_date=_format_date_for_display,
                    form_error=error_msg,
                )

            update_receipt_record(receipt_id, updated_payload, db_path)
            return redirect(url_for("receipt_detail", receipt_id=receipt_id, saved=1))


        preview_image_path = row["image_path"]
        preview_image_mtime = None
        if preview_image_path and os.path.exists(preview_image_path):
            preview_image_mtime = int(os.path.getmtime(preview_image_path))

        saved_flag = request.args.get("saved") == "1"
        return render_template_string(
            DETAIL_TEMPLATE,
            receipt=row,
            data=data,
            items=items,
            saved=saved_flag,
            is_new=False,
            progress=[],
            image_path=row["image_path"],
            preview_image_path=preview_image_path,
            preview_image_mtime=preview_image_mtime,
            base_payload=data,
            default_lang=default_lang,
            default_model=default_model,
            pending_payloads=[],
            pending_count=0,
            rotate_target=rotate_target,
            current_url=request.url,
            format_date=_format_date_for_display,
            form_error=None,
        )

    @app.route("/receipt/<int:receipt_id>/delete", methods=["POST"])
    def delete_receipt(receipt_id: int):
        """Trajno briše spremljeni račun (i njegove stavke) te njegovu sliku s diska."""
        row = fetch_receipt_record(receipt_id, db_path)
        if row is None:
            abort(404)

        image_path = row["image_path"]
        delete_receipt_record(receipt_id, db_path)

        if image_path and not image_path.startswith(("manual://", "excel://")):
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    log_progress(f"Obrisan račun #{receipt_id}, obrisana slika: {image_path}")
            except OSError as exc:
                log_progress(f"Ne mogu obrisati sliku obrisanog računa '{image_path}': {exc}")

        return redirect(url_for("index"))

    return app


# Ikona aplikacije (TornReceipt): 128px PNG s prozirnim kutovima, ugrađena kao base64 da
# aplikacija ostane samostalna (radi i kad se kopira samo receipt_ocr.py + config.py).
# Služi se na /favicon.png i /favicon.ico; koristi se kao favicon i u zaglavlju stranica.
_APP_ICON_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAARS0lEQVR42u1dbYwd1Xl+3nNmzr137t6912vvLv7aZW0gfH8LB/Nl"
    "giogNIBAiURc2a2BCDeQWhVNlAIWouqH1EpEQqr5AaqaNmmVVm2lQJsfVTctkAZKKDTBAgMh4IC/8H7cz5kz55y3P+auWS+2WWzD"
    "3nv3vNJoV7M7c6/mec77Pu9zzswQjiOYmQBIIjIz+2q1vcN5Ci9mx5eBaD0JcSZAq9kZACD4OK5LTSIAwLvZudfA/BMS9ELM6Uul"
    "0in7ZuERALBExJ/0A+g4wJdEZAGA9+3rs0ujLzptvkrOXR6EwRBUDnAOSFOkqfbYnzgHEIYKCENACEAnMKnZz0L8t5DB9w9OvfOv"
    "w8Pn1udic9IJwMwCAIjITU29syRS/XcJIbfKXGEMYCBpQWvNAFz7vEREHv2TQQFmBjCzCaUUIVcAQLCt5tsObkdTV5+oVEYnZ+N0"
    "0ggwm1mmObEZMnxEqmgEugGdJDOMEx7wz5QQDgBULiehirC68S6seSiIBr77SbIBzePDAiIyral9p0mV+05YiG6CjqGTxLRBFx6S"
    "BSWDA+BULhdA5ZG2mk9bnWwrVIbfnMHuWMeLY598PCAiE0/uvVUVij8NC9FNujZtkzhmIgo8+AsfRCSIKEjimHVt2oaF4k2qEP00"
    "nth7MxGZ8fHx4LgywAx7dPXAPWGhuMPZFEZrQ0SBv+wdnRFMoFRAMoRpNbaq/sHHj5UJxDHBrx+4PyyVd6RJyxmtnQe/KzJCYLR2"
    "Jmm5sFTeoesH7ici024VPz4DzKr5W/PlJX+ZNmqWrRUkhBd43ZQJnGOS0oXFkkynJ7aqyilHzAR0JLUfT+//zVyx9MM0bll21qv7"
    "riUBZyTI52XSqH8pXx56am53QHP6fI7j6bEA9KIglE2iQcILvS4ngZMqJAZNGfCl+Xz57bZH4z6iAYiIRaqfDPKFJSZJ2IPfA5pA"
    "kLBauyBfWCJS/eRcu1jMSv0umd6/RZUGNujatCEhpL98vUICIXVt2qjSwIZkev8WInLMLAGA2hM7mJyc7C+FvFOE4XKjE/Y9fs+1"
    "hy5QObJp8n41obOXLl1am8kAkog4kvauoK+yIk1i58HvyfZQpEnswr6Blf053NkuBVIQwTJzXhDuRdpkr/h7mgSEtMnM7veYOU8E"
    "KwDipPbBDWFf+VTdavnU3+NZQLdarPoqo0ntgxsAyqYOBYmNIGKQcP4y9bwidCBiImwEADE19e4As7uKdZMA9sq/9+WgZN0kMK6a"
    "mnp3ICgGxXUyDIbTRPv6v0h0QJpoDnNqmGNeJxy7K0jlgPYCAx+LIhypPASJywQYF4AZ8Iv3FlUiADuAcIkA6AxOU0+ARUYANgYA"
    "Tiddn2Cwz/6LVBBAePAXc0PA8KbPIg9PAE8AH54APjwBfHgC+PAE8OEJ4MMTwMfiia681YtdZ7qXJIQnwGdxkWW+kD0qoaO+GOB0"
    "0rHk7AkCEAmYVhPNN3ai09auMDOiVaciyEfgLppf6RoCsHOQhQJab+7Ey9++C0Ll0F7H0AnMhNMJLvrTJ1A++yKYVqNrykH3aQAi"
    "CJXrOAIc9tOXgE893364ddJ38m2gD08AH74ELHQ9/rQ6BO60suMJcGQ1ztae/CWuDJCUnSU+PQEOD6c1opWnIiyVT3ovTiSQ1qto"
    "vvcriCD0BOiogS8ETLOBlTd9BWs2fwMQhE8lBTDw9ncfw69/+PcIomLXuX49LwKXrb8OQVSGS1PA2ZO6uTRFUChh2eXXeg3Qeb4A"
    "QER476m/Q37wFFAQwPJJTALtc7XqVbz39A8ykcmeAJ2DPzuIfAEHn/8vTP/iJQiVP/lCrS0wTaMGkS90ld+/OEQgM0SuAKcT2Dj+"
    "lAomQeTyvgvoXBI4kAxA8tPy5LknhF9P+wA2ib0P4H0A7wN4H8D7AN4H8D6A9wG8D+B9AO8DeB/A+wDeB/ikZOKj5Xi/HqDHfQAG"
    "hFIgGfQc2N4HmCeJ4j27oacne8708T7APFI/SYnWnt149c++iXjPblCoepoE3geYvbGDS2L0jZ6F/tPPgU1i9PrD070PMDcDBAGq"
    "PxvH5CsvQBainmn3vA/wCTSAqU+DrQOFodcAi9EHIClBofRdQDf6ABASJ1y2j3v9P2ffgR3A1BX3CvbejSGf8aglyrBmBiAUKIxA"
    "qcv0h0sBtpkY6VAx6V8GfSLAM2A1AwKQSoCb7yE9oGCTBCJfhigsA6ky4FKwabZJQJ4A3Y8+YFOAJFA+NUD/aol8JY9016OYfC0F"
    "SICCAkQ0hHDoAuRGfwPB0rMzEjjTUdkg6Nrhd7R9RyoBJNr7eX7HHOtcAJxmFE8JMHRegGgwe80SWwY7BkQIgMGmBTP1FszBnYjf"
    "egq5sRsQnbMJIldpZ4POeD1T17kczA5szOEkIAIbA6eTI5LDJa1MmM05xqUanOqP7IfLDKEjiTxnCUvPDDF6jUK0TMBqhtWzX7nA"
    "M60EKMiBchUAQLzrH1H98f0w1XdAYRGd8ph+0T2DnuB0jOLIaSitORMuboGEAAkB22qidMY5GLltM2yzftjjWdg5jG2+F2F54BDY"
    "JCRMvYahq6/H0NU3wNSrICEzIqUaYXkJ1my+L7sbeDYvUsIpF1ew/NIc2LbLwFHF/sxDLNoiUCik+19B9T+2wVbfBQWFjiBB92QA"
    "IjhrEfaXsWbLNpAMwMxgZohQ4dSv3oOxzfehfM7FMM0GRKhgGjUMb7gRo7d/Datu2QirE5CUcDpBfmg51my6D2Ob7kV+aAVc+29W"
    "J1h1629h5Pa7MXzNDTCNWuYsJozyWA7LPpfCxNls48d3edQmoUZ+7ZfQd/kDAAlUn9uelQEhsdBLi7qqBJAQMPUallxwGVbdfAds"
    "swHbamDVzXegcu7FcHELa+/8fQTFEkyjhuLoaRjbuBVpfQIrbrgNy9ZtgKnXwM5izeZ7ocoDUOUBrNl0L9hZmHoNy9ZtwIrrb0Na"
    "n8DYxq0ojq6FbTQQ9OUwdH4OzhjQfJU8AZw2ULzgHpTWfQvRWRtRuvIRmIldaL3+D6Cwb8GzQNdpgGzmr46RL29BNLIG0eo1GPny"
    "Fti4BZu00H/GuRi5fTNMo461v7MNYWUp2KQAgLVbtkGEIQbXX4eha26EaTVgWg0MbbgRg+uvgwhDrN2yLRu1JkVYWYq1v/0N2JQx"
    "dNEgcn0GztL8OjkSYF1DdN6dKJx5B1zrA7hkEjIahiwsQ+uNf4Gt7QbJhZ1ylg/94Tcf7rYOgJ1DUCgiWjmCgYsuR7R6DC7JUjhb"
    "g2j1WhRXj2HZ5dfCJXGW9k2K3NJBFIZXYnD9dQhLZcBaEBGIJKKRNej/3LmonHcpbKuZnSvVKKwcRWFoOYrFXXDxJEgEHxo7hwTA"
    "3C5CgpNp5M+6A8ULt4J1NdMeqh/Nnz+B9MD/AWkTsrgc4fCFgI0XrDUkXfugKw1vZoZUCmDApnrWY2EYIAGpcrBx6/BCzQyRy4ON"
    "AVtzWLtHMgAFQab+Z7eBUgHJPkz+29faxwiALdgmh0A7TNCRBCdTyJ16PUqXPwi2MeAcKF9B89W/QfPlHSBVAusa1Oqr0X/lH4HT"
    "5oIRoGuNoKwr0Id+P7zwMkyr+dGHNRLBJq2shs9tI62Bs+mc+X8GyRDJgTfhdAMURoAzoCCP4kVfhyytQnPn95HufQGk+rMjdBVq"
    "5ZXoW/ctsEuz/88PIH7jnzPww2JGUhHA1d8Hpw2AggUTg9292uHoPdhRn9RJM6n7COf66OIPBiDApgawBZEEmybyp9+Kwpl3IBy+"
    "BP1X/wnUqqvAugZO6wgHz0dp/fa2XZiA8gPQ7/4Y9RcfzQg0k3FIwun6rEziCdBVXjCzgYsnARBK6x+GWvF5iOJylK54GCTzgG2B"
    "chWke19E7fk/zrTDAgLdcyXgswIacBCqnNV2tqAgQrzrnxAOX4Rw6BJwMgkKIvSt+zY4bUDkl4LTOkiVYSZeQ+257WCbgqSapRMA"
    "Zgup+tpdgMNCTRL5DPBxBHAGom9Flr7ZZrXbNFF7djvS/f8Lanv7FEYQxWFw2gCFfbCNPag9tx1O12aBPPe8y9u2sPU+QMe2nFZD"
    "llZB9o+CjW4Lwxw4baD2zANI97+c+f02AUwLCApwyTRqzz4IW98DCqIjAEwAO4SDFwJi4QSgJ8C8+k0LCiPkRr6QLfCAzEazzMOZ"
    "FmrPPoh0/8vZ3H+bCLXntsNMvNF2+o4AvkshokGolVcAJl5QGLrWB/gMGZCJN2cw/e9fh6nuBgX59uyiANsEIiwiOv8uyOIKNHf+"
    "LdJ9L4FU6cipXQTgeALR+XcjOv9ucDK1oFPDngDz4oADhUXo955F9ZkH2nrgcHJk7Vx2Q8pRZ/qEBOs6giWno3zto4AMF1QA+hIw"
    "72EiwGkdatXVKF6wFZxUD7l+WSaQoLAICqJjgB+AdQOisAylzz+YiT9nsdBLxHwbOG8SSLCuoXDWHYCQaLyyI7N4Z5y9QxM6/GGt"
    "p/YYYwsXTyConI7S+u2Q5dEFtX99CTihcsAg1Yd07/+g+fO/yiZ2gKzPF7KdVLPl4exM5gaqUscuCfMEOO7OoA9sNfSvn0Gy+z9h"
    "p96ES6bAVoO6aFGoJ8AJCEOQOLS+z8WTcPFBsGmBSIJUKWsNw76OXhbuCXAyiNAWeYf8fubsplJ/Y8ji6BBmygIbO0cEUsfUek+A"
    "T58J7ezeXe8O9D7AIg9PAE8AH54APjwBfHgC+PAE8OEJ4MMTwMdiIgB5DizaIAHBzr0eKAVm9pNCiySYmQOlAOdeFwC/QWEI9MxL"
    "UHzMhwMUhmDCLgHCK+175TwBFhEBQALE+JkQJJ5jnXhBuMi0H+sYjt0LomEaz6eJ3hfmFHkdsDjqf5hTlCZ6X8zx86JSGZkgEs+Q"
    "ihgg6y9Rz0t/SypiEJ6pVEYmBAA4dt8DM4GdLwM9nwKcADMx43vtus+052D9R7o+/StVKBD3+hsSFnf6d6pQoLQ+/faeg/UfMTMJ"
    "ZsixsbGYiR9DGHkd0OP1H2FEIHpsbGwsBiCJmQkAJicn+0uKXxVBuMLohIm8Rdhroz/I5cjq9P2wz50FLKsDgCAiBiAGBgamnbXb"
    "RS7KHovho9fCCRURO/MQ0WBtBnuaxRBBRC6p7h9XpcoGXZ2yJIT0160XdJ+zqr8ida06nutf9oUZrIFZa5iZWQDgOJ4eC0AvCkLZ"
    "JBokfCnobvDZSRUSg6YM+NJ8vvw2AJohwCFw2ztEoVD5pdXJJhEoQVKyF4VdDT6TlCxDRVYnmwqFyi/bqf9QiT9sdBORZeYgXx56"
    "Kq5Xfzcs9kkSwrFzngTdl/aZpHBhsU+m9erWfHnoKWYOiA43++goijEgIqPrB+4Pi+U/Txs1x87Bdwbdo/hJCITFkkgb03+g+gb/"
    "YgbTuf9LxzhJRoLqgXvCQnGHMxomTQ0R+dvJOht8EygVkAxhWo2tqn/w8aOB/5ESMKccmPHx8UD1Dz6eNKu3QMiDqlQJmNk6XxI6"
    "r8dzjpnZqlIlAImDulm9RfUPPj4+Pn5U8I+ZAT5k1HhAdK1pTe07LcwVviPz+ZugY+gkMW1B4cvCAqd7AE7lcgFUHjaOn06T1rZC"
    "ZfjNY438eROg/SFyRjyY5sRmyPARqaIR6AZ0ksyICkFE5CH5TEBntM06lc9JhEVY3XwXNt0eRAN/PRezEybALJ8AROSmpt5ZEqn+"
    "uwTEVlmIxgAASRNa65kvRu1e0xPi5AE+swmlFCGXPXncJq23nOPHm3rqyUpldHI2TvM5Nx3HlznErH37ftG3tDL6RWftV8jZq4Iw"
    "GILKAc4BaYo01ei2++U7EH6EoQLCEBAC0AlMavazkM8IKX8gJ+pP0/LljU8y6k+IAO0PIgBydn2p1fYO5ym8mB1fBqL1JMSZAK1m"
    "Z+BZcPzoZ6+o4d3s3Gtg/gkJeiHm9KVS6ZR9szs2ALY9r/OJ4v8BGwU5FXzZhtUAAAAASUVORK5CYII="
)
_APP_ICON_PNG = base64.b64decode(_APP_ICON_PNG_B64)

ICON_LINKS = """
    <link rel="icon" type="image/png" href="{{ url_for('app_icon') }}" />"""

FONT_LINKS = """
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
      rel="stylesheet"
    />"""

# Zajednički dizajn-tokeni i bazni stilovi za sva tri Jinja2 templatea.
# Tema "Ledger & Ink": papir/knjigovodstvena traka (greenbar zebra tablice) +
# crveni "pečat" akcenti na primarnim akcijama + poderani rub na masthead traci,
# kao referenca na fizičku traku papirnatog računa koju aplikacija digitalizira.
BASE_STYLE = """
  :root {
    --paper: #F2F4EC;
    --paper-card: #FFFFFF;
    --ink: #1C2620;
    --ink-soft: #4B564A;
    --stamp: #B23A2E;
    --stamp-dark: #8C2C22;
    --ledger-stripe: #E6EFDF;
    --ledger-stripe-hover: #D8E6D2;
    --border: #C9D2C2;
    --success: #2F6B3A;

    --font-display: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
    --font-body: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: var(--font-body);
    background: var(--paper);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }

  .page {
    max-width: 2000px;
    margin: 0 auto;
    padding: 2.25rem 1.75rem 3.5rem;
  }

  /* --- Masthead: stilizirano kao vrh papirnatog računa, s poderanim rubom --- */
  .topbar {
    position: relative;
    background: var(--ink);
    color: var(--paper);
    padding: 1.1rem 1.75rem 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .topbar::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -9px;
    height: 9px;
    background-image:
      linear-gradient(135deg, var(--ink) 50%, transparent 50%),
      linear-gradient(45deg, var(--ink) 50%, transparent 50%);
    background-size: 18px 18px, 18px 18px;
    background-position: 0 0, 9px 0;
    background-repeat: repeat-x;
  }

  .topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.9rem;
  }

  .topbar-logo {
    display: block;
    flex: none;
    width: 46px;
    height: 46px;
    border-radius: 11px;
  }

  .topbar-title {
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .topbar-subtitle {
    font-family: var(--font-display);
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    color: var(--ledger-stripe);
    opacity: 0.8;
    margin-top: 0.2rem;
  }

  .topbar-barcode {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 20px;
    margin-top: 0.4rem;
    opacity: 0.55;
  }

  .topbar-barcode span {
    display: block;
    width: 2px;
    height: 100%;
    background: var(--paper);
  }

  .topbar-barcode span:nth-child(3n) { width: 3px; }
  .topbar-barcode span:nth-child(5n) { height: 65%; }
  .topbar-barcode span:nth-child(7n) { height: 45%; }

  .topbar-actions {
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }

  /* --- Gumbi: potpisni "pečat" element na primarnim akcijama --- */
  a.button,
  button.button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.5rem 1rem;
    font-family: var(--font-display);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border-radius: 3px;
    border: 1.5px solid transparent;
    cursor: pointer;
    text-decoration: none;
    transition: transform 0.1s ease, background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }

  .button-onbar {
    color: var(--paper);
    border-color: rgba(255, 255, 255, 0.55);
    background: transparent;
  }
  .button-onbar:hover {
    background: rgba(255, 255, 255, 0.14);
    border-color: var(--paper);
  }

  .button-primary {
    color: var(--stamp);
    border-color: var(--stamp);
    background: var(--paper-card);
    transform: rotate(-1deg);
  }
  .button-primary:hover {
    background: var(--stamp);
    color: var(--paper-card);
  }
  .button-primary:active {
    transform: rotate(-1deg) translateY(1px);
  }

  .button-secondary {
    color: var(--ink-soft);
    border-color: var(--border);
    background: var(--paper-card);
  }
  .button-secondary:hover {
    border-color: var(--ink-soft);
    color: var(--ink);
    background: var(--paper);
  }

  a.button:focus-visible,
  button.button:focus-visible,
  input:focus-visible,
  select:focus-visible {
    outline: 2px solid var(--stamp);
    outline-offset: 1px;
  }

  h1, h2, h3 {
    font-family: var(--font-display);
    color: var(--ink);
    margin: 1.6rem 0 0.8rem;
    font-weight: 600;
  }

  h1 {
    font-size: 1.25rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .card {
    background: var(--paper-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-top: 1.25rem;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px dashed var(--border);
  }

  .card-header h2,
  .card-header h3 {
    margin: 0;
  }

  .card-header small {
    font-family: var(--font-display);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-soft);
  }

  .alert {
    margin-top: 1.1rem;
    border-radius: 3px;
    padding: 0.8rem 1rem;
    border: 1px solid var(--border);
    border-left: 4px solid var(--ink-soft);
    background: var(--paper-card);
    font-size: 0.88rem;
    color: var(--ink);
  }

  .alert-error {
    border-left-color: var(--stamp);
    color: var(--stamp-dark);
  }

  .alert-success {
    border-left-color: var(--success);
    color: var(--success);
  }

  .alert-info {
    border-left-color: var(--ink-soft);
    color: var(--ink-soft);
  }

  .panel-form {
    margin-top: 1.1rem;
    background: var(--paper-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.85rem 1.1rem;
  }

  form { margin: 0; }

  form .form-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1rem;
    align-items: flex-end;
  }

  label {
    font-family: var(--font-display);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-soft);
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  input[type="text"],
  input[type="number"],
  input[type="file"],
  select {
    border-radius: 2px;
    border: 1px solid var(--border);
    padding: 0.42rem 0.55rem;
    font-size: 0.88rem;
    font-family: var(--font-body);
    background: var(--paper-card);
    color: var(--ink);
  }

  input[type="checkbox"] {
    margin-right: 0.3rem;
  }

  .label-checkbox {
    flex-direction: row;
    align-items: center;
  }

  .cell-link {
    color: inherit;
    text-decoration: none;
  }

  pre {
    font-family: var(--font-display);
    font-size: 0.78rem;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.6rem 0.75rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.6rem;
    background: var(--paper-card);
  }

  th, td {
    padding: 0.5rem 0.6rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: 0.84rem;
  }

  th {
    background: var(--ink);
    color: var(--paper);
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  th a {
    color: inherit;
    text-decoration: none;
  }

  tbody tr:nth-child(even) td {
    background: var(--ledger-stripe);
  }

  tbody tr:hover td {
    background: var(--ledger-stripe-hover);
  }

  .warranty-pill {
    display: inline-block;
    padding: 0.05rem 0.4rem;
    border: 1px solid currentColor;
    border-radius: 2px;
    font-family: var(--font-display);
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .warranty-yes { color: var(--success); }
  .warranty-no { color: var(--ink-soft); opacity: 0.7; }

  .amount-cell {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: var(--font-display);
  }

  .amount-header { text-align: right; }

  .empty {
    margin-top: 2rem;
    font-family: var(--font-display);
    font-size: 0.85rem;
    color: var(--ink-soft);
  }

  .small-text {
    font-size: 0.8rem;
    color: var(--ink-soft);
  }

  .filters-inline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1rem;
  }

  .filters-inline label {
    min-width: 150px;
  }

  /* --- Date picker: "Ledger Stamp Grid" kalendar helper za polja datuma --- */
  .date-field {
    position: relative;
  }

  .date-field-row {
    display: inline-flex;
    gap: 0.4rem;
  }

  .date-picker-toggle {
    padding: 0.42rem 0.55rem;
    line-height: 1;
  }

  .date-picker-popup {
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    z-index: 30;
    width: 272px;
    background: var(--paper-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 8px 20px rgba(28, 38, 32, 0.18);
    overflow: hidden;
  }

  .date-picker-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--ink);
    color: var(--paper);
    padding: 0.5rem 0.6rem;
    font-family: var(--font-display);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.08em;
  }

  .date-picker-nav {
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    color: var(--paper);
    border: 1px solid rgba(255, 255, 255, 0.55);
    border-radius: 2px;
    font-family: var(--font-display);
    cursor: pointer;
  }

  .date-picker-nav:hover {
    background: rgba(255, 255, 255, 0.14);
    border-color: var(--paper);
  }

  .date-picker-weekdays {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    background: var(--ledger-stripe);
    padding: 0.35rem 0;
    font-family: var(--font-display);
    font-size: 0.6rem;
    letter-spacing: 0.03em;
    text-align: center;
    color: var(--ink-soft);
  }

  .date-picker-week {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
  }

  .date-picker-week:nth-child(even) {
    background: var(--ledger-stripe);
  }

  .date-picker-day {
    border: none;
    background: transparent;
    padding: 0.42rem 0;
    font-family: var(--font-display);
    font-size: 0.78rem;
    color: var(--ink);
    cursor: pointer;
    position: relative;
  }

  .date-picker-day:hover:not(:disabled) {
    background: var(--ledger-stripe-hover);
  }

  .date-picker-day-outside {
    color: var(--border);
  }

  .date-picker-day:disabled {
    color: var(--border);
    cursor: not-allowed;
  }

  .date-picker-day-today {
    font-weight: 600;
  }

  .date-picker-day-selected {
    color: var(--stamp-dark);
    font-weight: 700;
  }

  .date-picker-day-selected::after {
    content: "";
    position: absolute;
    inset: 2px 8px;
    border: 1.5px solid var(--stamp);
    border-radius: 50%;
    transform: rotate(-6deg);
    pointer-events: none;
  }

  @media (max-width: 768px) {
    .page { padding: 1.25rem; }
    .topbar { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    .filters-inline label { width: 100%; }
    table { font-size: 0.8rem; display: block; overflow-x: auto; white-space: nowrap; }
    .date-picker-popup { width: calc(100vw - 2.5rem); }
  }
"""


INDEX_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Billing me softly</title>"""
    + FONT_LINKS
    + ICON_LINKS
    + """
<style>"""
    + BASE_STYLE
    + """
</style>
  </head>
  <body>
    <div class="topbar">
      <div class="topbar-brand">
        <img class="topbar-logo" src="{{ url_for('app_icon') }}" alt="" width="46" height="46" />
        <div>
        <div class="topbar-title">Billing me softly</div>
        <div class="topbar-subtitle">Evidencija troškova</div>
        <div class="topbar-barcode">{% for _ in range(28) %}<span></span>{% endfor %}</div>
        </div>
      </div>
      <div class="topbar-actions">
        <a class="button button-onbar" href="/">Osvježi</a>
      </div>
    </div>
    <div class="page">
    {% if error_message %}
      <div class="alert alert-error">
        Dogodila se greška: {{ error_message }}
      </div>
    {% endif %}
    {% if progress %}
      <div class="card" style="margin-top:1rem;">
        <h3>Koraci obrade</h3>
        <pre style="max-height:200px; overflow:auto;">{% for line in progress %}{{ line }}&#10;{% endfor %}</pre>
      </div>
    {% endif %}
    <form action="{{ url_for('upload_receipt') }}" method="post" enctype="multipart/form-data" class="panel-form">
      <label style="align-self:center;">Učitaj jednu ili više (max {{ onedrive_import_limit }}) fotografija računa (PNG/JPG):</label>
      <input type="file" name="image" accept="image/*" multiple />
      <label style="align-self:center;">
        <input type="checkbox" name="manual" /> Ručni unos bez slike
      </label>
      <button class="button button-primary" type="submit">Start</button>
    </form>
    <form action="{{ url_for('import_onedrive') }}" method="post" class="panel-form">
      <label style="align-self:center;">Učitaj slike računa iz foldera:</label>
      <input type="text" name="onedrive_path" value="{{ onedrive_default_path or '' }}"
             placeholder="npr. /home/ituda/OneDrive/Racuni" style="flex:1;" />
      <button class="button button-primary" type="submit">Uvezi</button>
      <p class="small-text" style="flex-basis:100%; margin:0;">
        Učitava maksimalno {{ onedrive_import_limit }} slikovnih datoteka (.png, .jpg, .jpeg, .webp, .heic, .tif, .tiff) iz zadane putanje po jednom kliku.
      </p>
      {% if onedrive_default_path and onedrive_total_files is not none %}
        <p class="small-text" style="flex-basis:100%; margin:0;">
          Trenutno u folderu <code>{{ onedrive_default_path }}</code> ima
          <strong>{{ onedrive_total_files }}</strong> slikovnih datoteka.
          To znači da će biti potrebno
          <strong>{{ (onedrive_total_files // onedrive_import_limit) + (1 if (onedrive_total_files % onedrive_import_limit) > 0 else 0) }}</strong>
          uvoza da obradiš sve račune (ako se broj ne mijenja tijekom obrade).
        </p>
      {% endif %}

    </form>

    {% if category_summary %}
      <h2 id="categories-section">Pregled kategorija po godini
        {% if available_years and available_years|length > 1 %}
          <form method="get"
                action="{{ url_for('index') }}#categories-section"
                style="display:inline;">
            <select name="year" onchange="this.form.submit()">
              {% for year in available_years %}
                <option value="{{ year }}" {% if year == current_year %}selected{% endif %}>{{ year }}</option>
              {% endfor %}
            </select>
            <input type="hidden" name="sort" value="{{ sort_by }}" />
            <input type="hidden" name="dir" value="{{ direction }}" />
          </form>
        {% else %}
          ({{ current_year }})
        {% endif %}
      </h2>
      <table>
        <thead>
          <tr>
            <th>Kategorija</th>
            {% for month in month_names %}
              <th class="amount-header">{{ month }}</th>
            {% endfor %}
            <th class="amount-header">Prosjek</th>
          </tr>
        </thead>
        <tbody>
          {% for category, values in category_summary.items() %}
            <tr>
              <td>{{ category }}</td>
                {% for value in values %}
              <td class="amount-cell">
                <a href="{{ url_for('category_items',
                            category=category,
                            year=current_year,
                            month=loop.index) }}"
                class="cell-link">
                {{ ("%.2f"|format(value)).replace(".", ",") }}
                </a>
              </td>
                {% endfor %}
              <td class="amount-cell">
                {{ ("%.2f"|format((values|sum)/12)).replace(".", ",") }}
              </td>
            </tr>
          {% endfor %}

      {# ZADNJI RED – ZBROJEVI PO MJESECIMA #}
      <tr>
        <th>Zbroj</th>
        {% for total in monthly_totals %}
          <th class="amount-header">
            {{ ("%.2f"|format(total)).replace(".", ",") }}
          </th>
        {% endfor %}
        <th class="amount-header"></th>   {# ova ćelija ostaje prazna #}
      </tr>

          </tr>
        </tbody>
      </table>
    {% endif %}



    <form method="get"
        action="{{ url_for('index') }}#receipts-list"
        class="panel-form">
      <input type="hidden" name="sort" value="{{ sort_by }}" />
      <input type="hidden" name="dir" value="{{ direction }}" />
      {% if available_years and available_years|length > 0 %}
        <input type="hidden" name="year" value="{{ current_year }}" />
      {% endif %}
      <label>Datum od:
        <input type="text" name="date_from" value="{{ filters.date_from or '' }}" placeholder="YYYY-MM-DD ili DD.MM.YYYY" />
      </label>
      <label>Datum do:
        <input type="text" name="date_to" value="{{ filters.date_to or '' }}" placeholder="YYYY-MM-DD ili DD.MM.YYYY" />
      </label>
      <label>Minimalni iznos:
        <input type="text" name="total_min"
               value="{{ ('%.2f'|format(filters.total_min)).replace('.', ',') if filters.total_min is not none else '' }}" />
      </label>
      <label>Maksimalni iznos:
        <input type="text" name="total_max"
               value="{{ ('%.2f'|format(filters.total_max)).replace('.', ',') if filters.total_max is not none else '' }}" />
      </label>
      <label>Pretraži stavke:
        <input type="text" name="item_search" value="{{ filters.item_search or '' }}" placeholder="npr. šunka, mlijeko..." />
      </label>
      <label>Garancija:
        <select name="warranty_filter">
          <option value="" {% if not filters.warranty %}selected{% endif %}>Sve</option>
          <option value="1" {% if filters.warranty == '1' %}selected{% endif %}>Da</option>
          <option value="0" {% if filters.warranty == '0' %}selected{% endif %}>Ne</option>
        </select>
      </label>
      <label>Prikaži zadnjih:
        <select name="limit">
          {% for opt in allowed_limits %}
            <option value="{{ opt }}" {% if opt == limit_value %}selected{% endif %}>
              {% if opt == 'all' %}Sve{% else %}{{ opt }}{% endif %}
            </option>
          {% endfor %}
        </select>
      </label>
      <div style="display:flex; gap:0.5rem;">
        <button class="button button-primary" type="submit">Primijeni filtere</button>
        <a class="button button-secondary" href="/">Resetiraj</a>
      </div>
    </form>
    {% if receipts %}
      <table>
        <thead>
          <tr>
            <th><a href="{{ url_for('index', sort='id', dir='asc' if sort_by != 'id' or direction == 'desc' else 'desc') }}">ID</a></th>
            <th><a href="{{ url_for('index', sort='image_path', dir='asc' if sort_by != 'image_path' or direction == 'desc' else 'desc') }}">Slika</a></th>
            <th><a href="{{ url_for('index', sort='date', dir='asc' if sort_by != 'date' or direction == 'desc' else 'desc') }}">Datum</a></th>
            <th><a href="{{ url_for('index', sort='time', dir='asc' if sort_by != 'time' or direction == 'desc' else 'desc') }}">Vrijeme</a></th>
            <th class="amount-header">
              <a href="{{ url_for('index', sort='total', dir='asc' if sort_by != 'total' or direction == 'desc' else 'desc') }}">
                Ukupno (€)
              </a>
            </th>
            <th><a href="{{ url_for('index', sort='updated_at', dir='asc' if sort_by != 'updated_at' or direction == 'desc' else 'desc') }}">Ažurirano</a></th>
            <th><a href="{{ url_for('index', sort='warranty', dir='asc' if sort_by != 'warranty' or direction == 'desc' else 'desc') }}">Garancija</a></th>
            <th>Akcija</th>
          </tr>
        </thead>
        <tbody>
          {% for receipt in receipts %}
          <tr>
            <td>{{ receipt.id }}</td>
            <td>{{ receipt.image_path }}</td>
            <td>{{ format_date(receipt.date) or "—" }}</td>
            <td>{{ receipt.time or "—" }}</td>
            <td class="amount-cell">
              {{ ("%.2f"|format(receipt.total)).replace(".", ",") if receipt.total is not none else "—" }}
            </td>
            <td>{{ receipt.updated_at }}</td>
            <td>
              {% if receipt.warranty %}
                <span class="warranty-pill warranty-yes">Da</span>
              {% else %}
                <span class="warranty-pill warranty-no">Ne</span>
              {% endif %}
            </td>
            <td><a class="button button-primary" href="{{ url_for('receipt_detail', receipt_id=receipt.id) }}">Uredi</a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <p class="empty">Još nema obrađenih računa. Pokreni OCR kako bi se ovdje pojavili.</p>
    {% endif %}
    </div>
  </body>
</html>
"""
)



DETAIL_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Račun {{ receipt.id if receipt and receipt.id else 'Novi račun' }}</title>"""
    + FONT_LINKS
    + ICON_LINKS
    + """
    <style>"""
    + BASE_STYLE
    + """
      .detail-layout {
        display: flex;
        gap: 1.5rem;
        align-items: flex-start;
        margin-top: 1rem;
        flex-wrap: wrap;
      }

      .detail-image-card {
        flex: 0 0 55%;
        max-width: 640px;
      }

      .detail-form-card {
        flex: 1 1 0;
        min-width: 0;
      }

      .receipt-image {
        width: 100%;
        height: auto;
        max-height: 100vh;
        object-fit: contain;
        border-radius: 4px;
        border: 1px solid var(--border);
        background: var(--paper);
      }

      .actions {
        margin-top: 0.9rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
      }

      #items-sum-wrapper {
        margin-top: 0.5rem;
        text-align: right;
        font-size: 0.85rem;
        font-family: var(--font-display);
        color: var(--ink-soft);
      }

      #items-sum-label {
        font-weight: 600;
        color: var(--ink);
        margin-left: 0.25rem;
      }

      .rotate-group {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }

      @media (max-width: 900px) {
        .detail-layout {
          flex-direction: column;
        }
        .detail-image-card,
        .detail-form-card {
          max-width: 100%;
          flex: 1 1 100%;
        }
      }
    </style>
  </head>
  <body>
    <div class="topbar">
      <div class="topbar-brand">
        <img class="topbar-logo" src="{{ url_for('app_icon') }}" alt="" width="46" height="46" />
        <div>
        <div class="topbar-title">Billing me softly</div>
        <div class="topbar-subtitle">Uređivanje računa</div>
        </div>
      </div>
      <div class="topbar-actions">
        <a class="button button-onbar" href="{{ url_for('index') }}">Natrag na popis</a>
      </div>
    </div>

    <div class="page">
      <h1>Račun {{ receipt.id if receipt and receipt.id else '(novi)' }}</h1>

      {% if form_error %}
        <div class="alert alert-error">
          {{ form_error }}
        </div>
      {% endif %}

      {% if saved %}
        <div class="alert alert-success">
          Promjene su spremljene.
        </div>
      {% endif %}

      {% if progress %}
        <div class="card" style="margin-top:1rem;">
          <div class="card-header">
            <h2>Koraci obrade</h2>
          </div>
          <pre style="max-height:200px; overflow:auto; font-size:0.8rem; margin:0;">
{% for line in progress %}{{ line }}&#10;{% endfor %}
          </pre>
        </div>
      {% endif %}

      {% if pending_count %}
        <div class="alert alert-info">
          Preostalo još {{ pending_count }} računa u redu za obradu nakon spremanja ovog.
        </div>
      {% endif %}

      <div class="detail-layout">
        <!-- LEFT: IMAGE & ROTATE -->
        <div class="card detail-image-card">
          <div class="card-header">
            <h2>Slika računa</h2>
          </div>

          {% set has_real_image = receipt and receipt.id and receipt.image_path and not receipt.image_path.startswith('manual://') %}
          {% if has_real_image %}
            <img class="receipt-image"
                 src="{{ url_for('receipt_image', receipt_id=receipt.id) }}"
                 alt="Račun" />
          {% elif preview_image_path %}
            <img class="receipt-image"
                 src="{{ url_for('preview_image', path=preview_image_path, v=preview_image_mtime or 0) }}"
                 alt="Račun" />
          {% else %}
            <p class="small-text">Nema priložene slike.</p>
          {% endif %}

          {% if receipt and receipt.id and not is_new %}
            {% if not has_real_image %}
              <form method="post" action="{{ url_for('attach_image', receipt_id=receipt.id) }}"
                    enctype="multipart/form-data" style="margin-top:0.75rem;">
                <input type="file" name="image" accept="image/*" required
                       style="margin-bottom:0.5rem;" />
                <button class="button button-primary" type="submit">Pridruži sliku</button>
              </form>
            {% else %}
              <form method="post" action="{{ url_for('attach_image', receipt_id=receipt.id) }}"
                    enctype="multipart/form-data" style="margin-top:0.75rem;">
                <input type="file" name="image" accept="image/*" required
                       style="margin-bottom:0.5rem;" />
                <button class="button button-secondary" type="submit">Zamijeni sliku</button>
              </form>
            {% endif %}
          {% endif %}

          <p class="small-text" style="margin-top:0.5rem;">
            <strong>Datoteka:</strong>
            {{ receipt.image_path if receipt and receipt.image_path else (preview_image_path or 'Ručni unos') }}
          </p>

          {% if rotate_target and is_new %}
            {# New, unsaved receipt – rotate via POST and re-render this page #}
            <form method="post"
                  action="{{ url_for('rotate_image_action') }}"
                  class="rotate-group">
              <input type="hidden" name="path" value="{{ rotate_target }}" />
              <input type="hidden" name="image_path" value="{{ image_path }}" />
              <input type="hidden" name="preview_image_path" value="{{ preview_image_path }}" />
              <input type="hidden" name="base_payload" value='{{ base_payload | tojson }}' />
              <input type="hidden" name="pending_payloads" value='{{ pending_payloads | tojson }}' />
              <input type="hidden" name="default_lang" value="{{ default_lang }}" />
              <input type="hidden" name="default_model" value="{{ default_model }}" />
              <input type="hidden" name="source_path" value="{{ source_path or '' }}" />

              <button class="button button-secondary" type="submit" name="direction" value="left">
                Rotiraj 90° lijevo
              </button>
              <button class="button button-secondary" type="submit" name="direction" value="right">
                Rotiraj 90° desno
              </button>
            </form>
          {% elif rotate_target %}
            {# Existing saved receipt – rotate via GET and redirect back #}
            <div class="rotate-group">
              <form method="get" action="{{ url_for('rotate_image_action') }}">
                <input type="hidden" name="path" value="{{ rotate_target }}" />
                <input type="hidden" name="next" value="{{ current_url }}" />
                <input type="hidden" name="direction" value="left" />
                <button class="button button-secondary" type="submit">
                  Rotiraj 90° lijevo
                </button>
              </form>
              <form method="get" action="{{ url_for('rotate_image_action') }}">
                <input type="hidden" name="path" value="{{ rotate_target }}" />
                <input type="hidden" name="next" value="{{ current_url }}" />
                <input type="hidden" name="direction" value="right" />
                <button class="button button-secondary" type="submit">
                  Rotiraj 90° desno
                </button>
              </form>
            </div>
          {% endif %}
        </div>

        <!-- RIGHT: FORM & ITEMS -->
        <div class="card detail-form-card">
          <div class="card-header">
            <h2>Detalji računa</h2>
            <small>Ručna korekcija podataka</small>
          </div>

          <form method="post"
                action="{% if is_new %}{{ url_for('save_new_receipt') }}{% else %}{{ url_for('receipt_detail', receipt_id=receipt.id) }}{% endif %}">
            {% if is_new %}
              <input type="hidden" name="image_path" value="{{ image_path }}" />
              <input type="hidden" name="base_payload" value='{{ base_payload | tojson }}' />
              <input type="hidden" name="pending_payloads" value='{{ pending_payloads | tojson }}' />
              <input type="hidden" name="preview_image_path" value="{{ preview_image_path }}" />
              <input type="hidden" name="source_path" value="{{ source_path or '' }}" />
            {% endif %}

            <label>Datum:</label>
            <div class="date-field" id="date-field">
              <div class="date-field-row">
                <input type="text"
                       id="date-input"
                       name="date"
                       value="{{ format_date(data.date) }}"
                       placeholder="npr. 01.03.2025"
                       autocomplete="off" />
                <button type="button"
                        class="button button-secondary date-picker-toggle"
                        id="date-picker-toggle"
                        aria-label="Otvori kalendar">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                       stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="4" width="18" height="18" rx="2" />
                    <line x1="16" y1="2" x2="16" y2="6" />
                    <line x1="8" y1="2" x2="8" y2="6" />
                    <line x1="3" y1="10" x2="21" y2="10" />
                  </svg>
                </button>
              </div>
              <div class="date-picker-popup" id="date-picker-popup" hidden>
                <div class="date-picker-header">
                  <button type="button" class="date-picker-nav" id="date-picker-prev" aria-label="Prethodni mjesec">‹</button>
                  <span id="date-picker-title"></span>
                  <button type="button" class="date-picker-nav" id="date-picker-next" aria-label="Sljedeći mjesec">›</button>
                </div>
                <div class="date-picker-weekdays">
                  <span>PON</span><span>UTO</span><span>SRI</span><span>ČET</span><span>PET</span><span>SUB</span><span>NED</span>
                </div>
                <div class="date-picker-grid" id="date-picker-grid"></div>
              </div>
            </div>

            <label>Vrijeme:</label>
            <input type="text"
                   name="time"
                   value="{{ data.time or '' }}"
                   placeholder="HH:MM[:SS]" />

            <label>Ukupno (€):</label>
            <input type="text"
                   name="total"
                   value="{{ ('%.2f'|format(data.total)).replace('.', ',') if data.total is not none else '' }}" />

            <label class="label-checkbox" style="margin-top:0.6rem;">
              <input type="checkbox" name="warranty" {% if data.warranty %}checked{% endif %} />
              Garancija
            </label>

            <h3 style="margin-top:1rem;">Stavke</h3>
            <table id="items-table">
              <thead>
                <tr>
                  <th>Opis</th>
                  <th>Kategorija</th>
                  <th>Količina</th>
                  <th>Jed. cijena</th>
                  <th>Ukupna cijena</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="items-body">
                {% for item in items %}
                <tr>
                  <td>
                    <input type="text"
                           name="item-description"
                           value="{{ item.description }}" />
                  </td>
                  <td>
                    <select name="item-category">
                      {% for option in [
                        "Hrana","Cigarete, alkohol, kave,...","Kućne potrepštine","Kućni ljubimci","Lijekovi, troškovi liječenja",
                        "Odjeća i obuća","Škola i dječje aktivnosti","Sport","Automobili","Osiguranja",
                        "Internet/mobitel/TV","Struja","Voda","Plin","Smeće","Komunalni doprinos",
                        "Vodni doprinos","Putovanja, izleti, ručkovi","Ostalo"
                      ] %}
                        <option value="{{ option }}" {% if item.category == option %}selected{% endif %}>
                          {{ option }}
                        </option>
                      {% endfor %}
                    </select>
                  </td>
                  <td>
                    <input type="text"
                           name="item-quantity"
                           value="{{ ('%.4f'|format(item.quantity)).replace('.', ',') if item.quantity is not none else '' }}" />
                  </td>
                  <td>
                    <input type="text"
                           name="item-unit-price"
                           value="{{ ('%.2f'|format(item.unit_price)).replace('.', ',') if item.unit_price is not none else '' }}" />
                  </td>
                  <td>
                    <input type="text"
                           name="item-total-price"
                           value="{{ ('%.2f'|format(item.total_price)).replace('.', ',') if item.total_price is not none else '' }}" />
                  </td>
                  <td>
                    <button type="button" class="button button-secondary" onclick="removeRow(this)">Obriši</button>
                  </td>
                </tr>
                {% endfor %}
              </tbody>
            </table>

            <div id="items-sum-wrapper">
              <span>Automatski zbroj stavki:</span>
              <span id="items-sum-label">0,00</span> €
            </div>

            <div class="actions">
              <button class="button button-secondary" type="button" id="add-item">Dodaj stavku</button>
              <button class="button button-primary" type="submit">Spremi promjene</button>
              {% if is_new %}
                <button class="button button-secondary" type="submit"
                        formaction="{{ url_for('discard_new_receipt') }}"
                        formnovalidate
                        onclick="return confirm('Odbaci ovaj račun? Neće biti spremljen, a slika će biti trajno obrisana.');">
                  Odbaci ovaj račun
                </button>
              {% endif %}
              <a class="button button-secondary" href="{{ url_for('index') }}">Natrag</a>
            </div>
          </form>
          {% if receipt and receipt.id and not is_new %}
            <form method="post" action="{{ url_for('delete_receipt', receipt_id=receipt.id) }}"
                  onsubmit="return confirm('Trajno obrisati ovaj račun i njegovu sliku? Ova radnja se ne može poništiti.');"
                  style="margin-top:0.6rem;">
              <button class="button button-secondary" type="submit">Obriši račun</button>
            </form>
          {% endif %}
        </div>
      </div>

      <script>
        // Parsiranje hrvatskog formata brojeva (npr. "1.234,56")
        function parseEuro(value) {
          if (!value) return NaN;
          let v = String(value).trim();

          // Ukloni razmake
          v = v.replace(/\\s+/g, '');

          // Zamijeni decimalni zarez točkom
          v = v.replace(',', '.');

          // Zadrži samo znamenke, minus i točku
          v = v.replace(/[^0-9\\.\\-]/g, '');

          const num = parseFloat(v);
          return isNaN(num) ? NaN : num;
        }

        function formatEuro(value) {
          if (!isFinite(value)) return '0,00';
          return value.toFixed(2).replace('.', ',');
        }

        function recalcItemsSum() {
          const inputs = document.querySelectorAll('input[name="item-total-price"]');
          let sum = 0;

          inputs.forEach(function (input) {
            const val = parseEuro(input.value);
            if (isFinite(val)) {
              sum += val;
            }
          });

          const label = document.getElementById('items-sum-label');
          if (label) {
            label.textContent = formatEuro(sum);
          }
        }

        function removeRow(button) {
          const row = button.closest('tr');
          if (row && row.parentNode) {
            row.parentNode.removeChild(row);
            recalcItemsSum();
          }
        }

        document.getElementById('add-item').addEventListener('click', function () {
          const tbody = document.getElementById('items-body');
          const row = document.createElement('tr');
          row.innerHTML = `
            <td><input type="text" name="item-description" /></td>
            <td>
              <select name="item-category">
                {% for option in [
                  "Hrana","Cigarete, alkohol, kave,...","Kućne potrepštine","Kućni ljubimci","Lijekovi, troškovi liječenja",
                  "Odjeća i obuća","Škola i dječje aktivnosti","Sport","Automobili","Osiguranja",
                  "Internet/mobitel/TV","Struja","Voda","Plin","Smeće","Komunalni doprinos",
                  "Vodni doprinos","Putovanja, izleti, ručkovi","Ostalo"
                ] %}
                  <option value="{{ option }}">{{ option }}</option>
                {% endfor %}
              </select>
            </td>
            <td><input type="text" name="item-quantity" /></td>
            <td><input type="text" name="item-unit-price" /></td>
            <td><input type="text" name="item-total-price" /></td>
            <td><button type="button" class="button button-secondary" onclick="removeRow(this)">Obriši</button></td>
          `;
          tbody.appendChild(row);
        });

        document.addEventListener('input', function (event) {
          const target = event.target;
          if (!target) return;

          if (target.name === 'item-total-price' || target.name === 'total') {
            recalcItemsSum();
          }
        });

        document.addEventListener('DOMContentLoaded', recalcItemsSum);

        // --- Kalendar helper za polje "Datum" ---
        (function () {
          const wrapper = document.getElementById('date-field');
          if (!wrapper) return;

          const input = document.getElementById('date-input');
          const toggleBtn = document.getElementById('date-picker-toggle');
          const popup = document.getElementById('date-picker-popup');
          const titleEl = document.getElementById('date-picker-title');
          const gridEl = document.getElementById('date-picker-grid');
          const prevBtn = document.getElementById('date-picker-prev');
          const nextBtn = document.getElementById('date-picker-next');

          const MONTHS = ['Siječanj', 'Veljača', 'Ožujak', 'Travanj', 'Svibanj', 'Lipanj',
            'Srpanj', 'Kolovoz', 'Rujan', 'Listopad', 'Studeni', 'Prosinac'];

          function pad2(n) { return n < 10 ? '0' + n : '' + n; }

          function toDateStr(y, m, d) { return y + '-' + pad2(m + 1) + '-' + pad2(d); }

          function toDisplay(y, m, d) { return pad2(d) + '.' + pad2(m + 1) + '.' + y; }

          function parseDisplay(value) {
            const match = /^(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})$/.exec((value || '').trim());
            if (!match) return null;
            const d = parseInt(match[1], 10);
            const m = parseInt(match[2], 10) - 1;
            const y = parseInt(match[3], 10);
            const dt = new Date(y, m, d);
            if (dt.getFullYear() !== y || dt.getMonth() !== m || dt.getDate() !== d) return null;
            return { y: y, m: m, d: d };
          }

          const now = new Date();
          const todayStr = toDateStr(now.getFullYear(), now.getMonth(), now.getDate());

          let selected = parseDisplay(input.value);
          let viewYear = selected ? selected.y : now.getFullYear();
          let viewMonth = selected ? selected.m : now.getMonth();

          function render() {
            titleEl.textContent = MONTHS[viewMonth].toUpperCase() + ' ' + viewYear;
            gridEl.innerHTML = '';

            const firstOfMonth = new Date(viewYear, viewMonth, 1);
            const startOffset = (firstOfMonth.getDay() + 6) % 7; // ponedjeljak = 0
            const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
            const totalCells = Math.ceil((startOffset + daysInMonth) / 7) * 7;
            const selectedStr = selected ? toDateStr(selected.y, selected.m, selected.d) : null;

            let week = null;
            for (let i = 0; i < totalCells; i++) {
              if (i % 7 === 0) {
                week = document.createElement('div');
                week.className = 'date-picker-week';
                gridEl.appendChild(week);
              }

              const cellDate = new Date(viewYear, viewMonth, i - startOffset + 1);
              const cellStr = toDateStr(cellDate.getFullYear(), cellDate.getMonth(), cellDate.getDate());
              const isOutside = cellDate.getMonth() !== viewMonth;
              const isFuture = cellStr > todayStr;

              const btn = document.createElement('button');
              btn.type = 'button';
              btn.className = 'date-picker-day';
              btn.textContent = String(cellDate.getDate());

              if (isOutside) btn.classList.add('date-picker-day-outside');
              if (cellStr === todayStr) btn.classList.add('date-picker-day-today');
              if (cellStr === selectedStr) btn.classList.add('date-picker-day-selected');

              if (isFuture) {
                btn.disabled = true;
              } else {
                btn.addEventListener('click', function () {
                  selected = { y: cellDate.getFullYear(), m: cellDate.getMonth(), d: cellDate.getDate() };
                  input.value = toDisplay(selected.y, selected.m, selected.d);
                  input.dispatchEvent(new Event('input', { bubbles: true }));
                  input.dispatchEvent(new Event('change', { bubbles: true }));
                  closePopup();
                });
              }

              week.appendChild(btn);
            }
          }

          function openPopup() {
            selected = parseDisplay(input.value);
            viewYear = selected ? selected.y : now.getFullYear();
            viewMonth = selected ? selected.m : now.getMonth();
            render();
            popup.hidden = false;
          }

          function closePopup() {
            popup.hidden = true;
          }

          toggleBtn.addEventListener('click', function (event) {
            event.stopPropagation();
            if (popup.hidden) {
              openPopup();
            } else {
              closePopup();
            }
          });

          prevBtn.addEventListener('click', function () {
            viewMonth -= 1;
            if (viewMonth < 0) { viewMonth = 11; viewYear -= 1; }
            render();
          });

          nextBtn.addEventListener('click', function () {
            viewMonth += 1;
            if (viewMonth > 11) { viewMonth = 0; viewYear += 1; }
            render();
          });

          document.addEventListener('click', function (event) {
            if (!popup.hidden && !wrapper.contains(event.target)) {
              closePopup();
            }
          });

          document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape' && !popup.hidden) {
              closePopup();
              toggleBtn.focus();
            }
          });
        })();
      </script>
    </div>
  </body>
</html>
"""
)

CATEGORY_ITEMS_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Pregled kategorije {{ category }} - {{ month_name }} {{ year }}</title>"""
    + FONT_LINKS
    + ICON_LINKS
    + """
    <style>"""
    + BASE_STYLE
    + """
    </style>
  </head>
  <body>
    <div class="topbar">
      <div class="topbar-brand">
        <img class="topbar-logo" src="{{ url_for('app_icon') }}" alt="" width="46" height="46" />
        <div>
        <div class="topbar-title">Billing me softly</div>
        <div class="topbar-subtitle">Kategorija &middot; mjesečni pregled</div>
        </div>
      </div>
      <div class="topbar-actions">
        <a class="button button-onbar" href="{{ back_url }}">Natrag</a>
      </div>
    </div>

    <div class="page">
      <h2>Pregled kategorije "{{ category }}" za {{ month_name }} {{ year }}</h2>
      <p class="small-text">
        Ukupno: <strong>{{ ("%.2f"|format(total_amount)).replace(".", ",") }} €</strong>
      </p>

      {% if items %}
        <table>
          <thead>
            <tr>
              <th>ID računa</th>
              <th>Datum</th>
              <th>Vrijeme</th>
              <th>Opis</th>
              <th class="amount-cell">Količina</th>
              <th class="amount-cell">Jed. cijena (€)</th>
              <th class="amount-cell">Ukupno (€)</th>
              <th>Akcija</th>
            </tr>
          </thead>
          <tbody>
            {% for item in items %}
              <tr>
                <td>{{ item.receipt_id }}</td>
                <td>{{ format_date(item.raw_date) or "—" }}</td>
                <td>{{ item.time or "—" }}</td>
                <td>{{ item.description }}</td>
                <td class="amount-cell">
                  {{ ('%.4f'|format(item.quantity)).replace('.', ',') if item.quantity is not none else '' }}
                </td>
                <td class="amount-cell">
                  {{ ('%.2f'|format(item.unit_price)).replace('.', ',') if item.unit_price is not none else '' }}
                </td>
                <td class="amount-cell">
                  {{ ('%.2f'|format(item.total_price)).replace('.', ',') if item.total_price is not none else '' }}
                </td>
                <td>
                  <a class="button button-primary"
                     href="{{ url_for('receipt_detail', receipt_id=item.receipt_id) }}">
                    Otvori račun
                  </a>
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <p class="empty">
          Nema stavki za ovu kombinaciju godine, mjeseca i kategorije.
          Ili si stvarno štedljiv, ili filteri lažu.
        </p>
      {% endif %}
    </div>
  </body>
</html>
"""
)


BATCH_STATUS_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Obrada u tijeku &mdash; Billing me softly</title>"""
    + FONT_LINKS
    + ICON_LINKS
    + """
    <style>"""
    + BASE_STYLE
    + """
      .status-queued { color: var(--ink-soft); }
      .status-processing { color: var(--stamp); }
      .status-done { color: var(--success); }
      .status-error { color: var(--stamp-dark); }

      .status-processing::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        margin-right: 0.35rem;
        border-radius: 50%;
        background: currentColor;
        animation: batch-pulse 1s ease-in-out infinite;
      }

      @keyframes batch-pulse {
        0%, 100% { opacity: 0.25; }
        50% { opacity: 1; }
      }

      @media (prefers-reduced-motion: reduce) {
        .status-processing::before { animation: none; }
      }
    </style>
  </head>
  <body>
    <div class="topbar">
      <div class="topbar-brand">
        <img class="topbar-logo" src="{{ url_for('app_icon') }}" alt="" width="46" height="46" />
        <div>
        <div class="topbar-title">Billing me softly</div>
        <div class="topbar-subtitle">Obrada u tijeku</div>
        </div>
      </div>
      <div class="topbar-actions">
        <a class="button button-onbar" href="{{ url_for('index') }}">Natrag</a>
      </div>
    </div>

    <div class="page">
      <h1>Obrada računa</h1>

      <div id="done-banner" class="alert alert-success" style="display:none;"></div>
      <div id="expired-banner" class="alert alert-error" style="display:none;">
        Obrada je istekla ili nije pronađena. <a class="cell-link" href="{{ url_for('index') }}">Natrag na početnu.</a>
      </div>

      {% if duplicates %}
        <div class="card" style="margin-bottom:1rem;">
          <div class="card-header">
            <h2>Preskočeno &mdash; već postoji ({{ duplicates | length }})</h2>
          </div>
          <p class="small-text">
            Ove slike su bajt-identične računu koji je već u bazi (isti sadržaj datoteke),
            pa nisu ponovno slane na OCR. Nisu spremljene ni na koji drugi način.
          </p>
          <table>
            <thead>
              <tr>
                <th>Datoteka</th>
                <th>Već spremljeno kao</th>
              </tr>
            </thead>
            <tbody>
              {% for dup in duplicates %}
              <tr>
                <td>{{ dup.name }}</td>
                <td>
                  <a class="cell-link" href="{{ url_for('receipt_detail', receipt_id=dup.receipt_id) }}">
                    Račun #{{ dup.receipt_id }}{% if dup.date %} &mdash; {{ format_date(dup.date) }}{% endif %}{% if dup.total is not none %} &mdash; {{ ('%.2f'|format(dup.total)).replace('.', ',') }} €{% endif %}
                  </a>
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      {% endif %}

      {% if files %}
      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Datoteka</th>
              <th>Status</th>
              <th>Napomena</th>
            </tr>
          </thead>
          <tbody>
            {% for file in files %}
            <tr data-idx="{{ file.idx }}">
              <td>{{ file.name }}</td>
              <td>
                <span class="warranty-pill status-{{ file.status }}" id="status-{{ file.idx }}">
                  {% if file.status == 'queued' %}Na čekanju
                  {%- elif file.status == 'processing' %}Obrađuje se
                  {%- elif file.status == 'done' %}Gotovo
                  {%- elif file.status == 'error' %}Greška
                  {%- else %}{{ file.status }}{% endif %}
                </span>
              </td>
              <td class="small-text" id="error-{{ file.idx }}">{{ file.error or '' }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}

      <script>
        const batchId = {{ batch_id | tojson }};
        const labels = {queued: "Na čekanju", processing: "Obrađuje se", done: "Gotovo", error: "Greška"};

        function applyStatus(file) {
          const statusEl = document.getElementById("status-" + file.idx);
          const errorEl = document.getElementById("error-" + file.idx);
          if (statusEl) {
            statusEl.className = "warranty-pill status-" + file.status;
            statusEl.textContent = labels[file.status] || file.status;
          }
          if (errorEl) {
            errorEl.textContent = file.error || "";
          }
        }

        function showExpiredNotice() {
          const el = document.getElementById("expired-banner");
          if (el) el.style.display = "block";
        }

        let redirecting = false;

        async function poll() {
          let res;
          try {
            res = await fetch("/batch/" + batchId + "/status.json");
          } catch (err) {
            return;
          }
          if (!res.ok) {
            clearInterval(timer);
            showExpiredNotice();
            return;
          }
          const data = await res.json();
          data.files.forEach(applyStatus);
          if (data.done && !redirecting) {
            redirecting = true;
            clearInterval(timer);
            const banner = document.getElementById("done-banner");
            if (data.files.length > 0) {
              banner.textContent = "Obrada završena — preusmjeravam na pregled...";
              banner.style.display = "block";
              setTimeout(function () {
                window.location.href = "/batch/" + batchId + "/review";
              }, 500);
            } else {
              // Sve slike su bile duplikati (preskočene prije OCR-a) — nema ništa
              // novo za review, ostani na ovoj stranici (vidi popis iznad).
              banner.textContent = "Obrada završena — nema ništa novo za pregled.";
              banner.style.display = "block";
            }
          }
        }

        const timer = setInterval(poll, 1200);
        poll();
      </script>
    </div>
  </body>
</html>
"""
)


def _normalize_date_for_db(value: Optional[str]) -> Optional[str]:
    """
    Pretvara razne tekstualne datume u standardni format 'YYYY-MM-DD'.
    Ako ne uspije parsirati, vraća None.
    """
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

_DATE_CANDIDATE_RE = re.compile(r"(?<!\d)(\d{1,4})([./-])(\d{1,2})\2(\d{1,4})(?!\d)")


def _date_interpretations(text: str) -> set:
    """Sve kalendarski valjane, ne-buduće interpretacije jednog prepisanog datuma.

    Točke su uvijek DD.MM.YY(YY). S kosom crtom/crticom i dvoznamenkastim rubovima
    ("26/09/14") oblik je dvosmislen: DD/MM/YY ili YY/MM/DD, pa vraća oba.
    """
    m = _DATE_CANDIDATE_RE.search(str(text))
    if not m:
        return set()
    a, sep, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
    layouts = []  # (godina, mjesec, dan) kao stringovi
    if len(a) == 4:
        layouts.append((a, b, c))            # YYYY-MM-DD
    elif len(c) == 4:
        layouts.append((c, b, a))            # DD.MM.YYYY
    else:
        layouts.append((c, b, a))            # DD.MM.YY
        if sep != ".":
            layouts.append((a, b, c))        # YY/MM/DD
    now = datetime.now()
    found = set()
    for y, mo, d in layouts:
        year = int(y) + (2000 if len(y) <= 2 else 0)
        try:
            dt = datetime(year, int(mo), int(d))
        except ValueError:
            continue
        if dt <= now:
            found.add(dt.strftime("%Y-%m-%d"))
    return found


def _resolve_receipt_date(candidates, model_date: Optional[str]) -> Optional[str]:
    """Odaberi datum računa iz doslovno prepisanih datuma ("date_candidates").

    Svaki kandidat glasa za sve svoje valjane interpretacije; pobjeđuje datum s
    najviše glasova (računi često nose isti dan dvaput u različitim formatima,
    npr. "14.09.26" + "26/09/14" -> 2026-09-14). Kod izjednačenja prednost ima
    datum koji je model sam vratio, a zatim noviji. Bez upotrebljivih kandidata
    vraća normaliziran model_date.
    """
    fallback = _normalize_date_for_db(model_date)
    if not isinstance(candidates, (list, tuple)):
        return fallback
    votes: Dict[str, int] = {}
    for cand in candidates:
        for iso in _date_interpretations(cand):
            votes[iso] = votes.get(iso, 0) + 1
    if not votes:
        return fallback
    best = max(votes.values())
    tied = [d for d, v in votes.items() if v == best]
    if fallback in tied:
        return fallback
    return max(tied)


def _format_date_for_display(value: Optional[str]) -> str:
    """
    Formatira datum za prikaz kao 'dd.mm.yyyy', bez obzira je li spremljen kao
    'YYYY-MM-DD' ili 'DD.MM.YYYY' ili 'DD.MM.YY'.
    Ako ne uspije parsirati, vrati originalni tekst.
    """
    if not value:
        return ""
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            continue
    return text


def _validate_receipt_payload(payload: dict, require_date: bool = False, require_total: bool = False) -> Optional[str]:
    """
    Validira polja datuma i ukupnog iznosa.
    - Datum ne smije biti u budućnosti.
    - Ako je require_date=True, datum je obavezan.
    - Ako je require_total=True, total je obavezan.
    Vraća string s porukom greške ili None ako je sve u redu.
    """
    today = datetime.today().date()

    # --- Datum ---
    date_str = payload.get("date")
    if require_date and not date_str:
        return "Datum je obavezno polje."

    if date_str:
        try:
            # U payloadu je već normaliziran oblik 'YYYY-MM-DD'
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Datum mora biti u formatu YYYY-MM-DD ili DD.MM.YYYY."
        if dt > today:
            return "Datum ne može biti veći od današnjeg."

    # --- Ukupno ---
    total_val = payload.get("total")
    if require_total and (total_val is None):
        return "Ukupno (€) je obavezno polje."

    return None


def _build_receipt_data_from_payload(
    llm_payload: dict, image_path: str, language: str, image_hash: Optional[str] = None
) -> ReceiptData:
    items: List[ReceiptItem] = []
    for raw in llm_payload.get("items", []):
        raw = raw or {}
        description = _clean_string(raw.get("description") or raw.get("name"))
        if not description:
            continue
        item = ReceiptItem(
            description=description,
            category=_clean_string(raw.get("category")),
            quantity=_to_optional_float(raw.get("quantity")),
            unit_price=_to_optional_float(raw.get("unit_price")),
            total_price=_to_optional_float(raw.get("total_price")),
        )
        items.append(item)

    items_sum = float(
        sum(item.total_price for item in items if item.total_price is not None)
    )
    total_value = _to_optional_float(llm_payload.get("total"))
    raw_date_value = _clean_string(llm_payload.get("date"))
    date_value = _resolve_receipt_date(llm_payload.get("date_candidates"), raw_date_value)
    time_value = _clean_string(llm_payload.get("time"))

    return ReceiptData(
        image=os.path.abspath(image_path),
        language=language,
        items=items,
        items_sum=items_sum,
        total=total_value,
        date=date_value,
        time=time_value,
        image_hash=image_hash,
    )


def build_receipt_data(image_path: str, language: str, image_hash: Optional[str] = None) -> ReceiptData:
    log_progress(f"Pokrećem Gemini Vision pipeline za {image_path}...")
    llm_payload = call_gemini_vision_parser(image_path)
    return _build_receipt_data_from_payload(llm_payload, image_path, language, image_hash=image_hash)



def receipt_from_payload(payload: dict) -> ReceiptData:
    items = []
    for raw in payload.get("items", []):
        items.append(
            ReceiptItem(
                description=raw.get("description", ""),
                total_price=_to_optional_float(raw.get("total_price")),
                quantity=_to_optional_float(raw.get("quantity")),
                unit_price=_to_optional_float(raw.get("unit_price")),
                category=_clean_string(raw.get("category")),
            )
        )
    return ReceiptData(
        image=payload.get("image", ""),
        language=payload.get("language", ""),
        items=items,
        items_sum=float(payload.get("items_sum") or 0.0),
        total=_to_optional_float(payload.get("total")),
        date=_clean_string(payload.get("date")),
        time=_clean_string(payload.get("time")),
        warranty=bool(payload.get("warranty")),
        image_hash=_clean_string(payload.get("image_hash")),
    )


def apply_form_updates(base_payload: dict, form_data) -> dict:
    descriptions = form_data.getlist("item-description")
    categories = form_data.getlist("item-category")
    quantities = form_data.getlist("item-quantity")
    unit_prices = form_data.getlist("item-unit-price")
    total_prices = form_data.getlist("item-total-price")

    updated_items: List[dict] = []
    for idx, desc in enumerate(descriptions):
        desc_value = (desc or "").strip()
        category_value = (categories[idx] if idx < len(categories) else "").strip() or None
        quantity_value = quantities[idx] if idx < len(quantities) else ""
        unit_price_value = unit_prices[idx] if idx < len(unit_prices) else ""
        total_price_value = total_prices[idx] if idx < len(total_prices) else ""

        quantity_float = _to_optional_float(quantity_value)
        unit_price_float = _to_optional_float(unit_price_value)
        total_price_float = _to_optional_float(total_price_value)

        if not desc_value and quantity_float is None and unit_price_float is None and total_price_float is None:
            continue

        updated_items.append(
            {
                "description": desc_value or "N/A",
                "category": category_value,
                "quantity": quantity_float,
                "unit_price": unit_price_float,
                "total_price": total_price_float,
            }
        )

    raw_date_value = (form_data.get("date") or "").strip() or None
    date_value = _normalize_date_for_db(raw_date_value)
    time_value = (form_data.get("time") or "").strip() or None
    total_value = _to_optional_float(form_data.get("total"))
    warranty_value = form_data.get("warranty") == "on"

    items_sum = float(sum((item["total_price"] or 0.0) for item in updated_items if item.get("total_price") is not None))

    payload = json.loads(json.dumps(base_payload))
    payload["items"] = updated_items
    payload["items_sum"] = items_sum
    payload["total"] = total_value
    payload["date"] = date_value
    payload["time"] = time_value
    payload["warranty"] = warranty_value
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Path to receipt image (png, jpg, ...)")
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"Language label saved in output JSON (default: '{DEFAULT_LANG}')",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"SQLite database for reviewed receipts (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the review web server instead of running single-image parsing.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for the web server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web server (default: 5000)",
    )
    return parser.parse_args(argv)


def _migrate_legacy_repo_data(data_dir: str) -> None:
    """
    Jednokratna migracija: ako receipts.db/uploads/ još postoje u korijenu
    repozitorija (stari CWD-relativni default) a u novom home-based data_dir
    još nema baze, premjesti ih. Nakon prvog pokretanja nema efekta.
    """
    legacy_root = os.path.dirname(os.path.abspath(__file__))
    legacy_db = os.path.join(legacy_root, "receipts.db")
    legacy_uploads = os.path.join(legacy_root, "uploads")
    target_db = os.path.join(data_dir, "receipts.db")
    target_uploads = os.path.join(data_dir, "uploads")

    if not os.path.exists(target_db) and os.path.exists(legacy_db):
        os.makedirs(data_dir, exist_ok=True)
        shutil.move(legacy_db, target_db)
        log_progress(f"Migrirana baza iz '{legacy_db}' u '{target_db}'.")

    # "Već migrirano" provjeravamo po sadržaju, ne po pukom postojanju direktorija —
    # bootstrap.py unaprijed kreira prazan uploads/ pa gola isdir() provjera pogrešno
    # preskoči migraciju stvarnih slika.
    target_uploads_has_content = os.path.isdir(target_uploads) and any(os.scandir(target_uploads))
    if os.path.isdir(legacy_uploads) and not target_uploads_has_content:
        os.makedirs(data_dir, exist_ok=True)
        if os.path.isdir(target_uploads):
            os.rmdir(target_uploads)
        shutil.move(legacy_uploads, target_uploads)
        log_progress(f"Migriran uploads/ direktorij iz '{legacy_uploads}' u '{target_uploads}'.")

        # image_path u bazi je apsolutna putanja snimljena pod starim (repo-root) prefiksom —
        # nakon fizičkog premještanja uploads/ mora se prepisati na novi prefiks, inače
        # postojeći računi izgube sliku.
        if os.path.exists(target_db):
            conn = sqlite3.connect(target_db)
            try:
                conn.execute(
                    "UPDATE receipts SET image_path = ? || substr(image_path, ?) "
                    "WHERE image_path LIKE ? || '%'",
                    (target_uploads, len(legacy_uploads) + 1, legacy_uploads),
                )
                conn.commit()
                log_progress(f"Ažurirane putanje slika u bazi ({conn.total_changes} redaka).")
            finally:
                conn.close()


def _repair_broken_image_paths(db_path: str, uploads_dir: str) -> None:
    """
    Popravlja image_path retke koji ne postoje na disku na trenutnoj putanji —
    tipično nakon kopiranja receipts.db + uploads/ s drugog OS-a/lokacije (npr.
    Linux -> Windows), gdje je stara apsolutna putanja (drugi prefiks, drugi
    separator) i dalje zapisana u bazi iako su slike fizički već u uploads_dir.
    Traži datoteku istog imena unutar uploads_dir i, ako postoji, prepisuje
    putanju. Sentinel vrijednosti (manual://..., excel://...) nikad nemaju
    odgovarajuću datoteku pa se tiho preskaču.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, image_path FROM receipts").fetchall()
        fixed = 0
        for receipt_id, image_path in rows:
            if not image_path or os.path.exists(image_path):
                continue
            candidate = os.path.join(uploads_dir, os.path.basename(image_path))
            if candidate != image_path and os.path.isfile(candidate):
                conn.execute(
                    "UPDATE receipts SET image_path = ? WHERE id = ?",
                    (candidate, receipt_id),
                )
                fixed += 1
        if fixed:
            conn.commit()
            log_progress(f"Popravljene putanje slika za {fixed} računa (pronađeno po imenu datoteke u '{uploads_dir}').")
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.db_path == DEFAULT_DB_PATH:
        _migrate_legacy_repo_data(str(DATA_DIR))

    init_db(args.db_path)

    if args.db_path == DEFAULT_DB_PATH:
        _repair_broken_image_paths(args.db_path, UPLOAD_DIR)

    if args.serve:
        log_progress(f"Pokrećem web poslužitelj na http://{args.host}:{args.port}")
        app = create_app(args.db_path, args.lang, GEMINI_MODEL)
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        return 0

    image_path = args.image
    if not image_path:
        print("Greška: morate navesti putanju do slike računa.", file=sys.stderr)
        return 1

    if not os.path.exists(image_path):
        print(f"Greška: ne postoji datoteka '{image_path}'.", file=sys.stderr)
        return 1

    try:
        # Hash izvornih bajtova, prije nego što prepare_image_for_gemini nešto promijeni na disku.
        image_hash = _sha256_of_file(image_path)
        prepared_image = prepare_image_for_gemini(image_path)
        receipt = build_receipt_data(prepared_image, args.lang, image_hash=image_hash)
    except RuntimeError as exc:
        print(f"Gemini obrada nije uspjela: {exc}", file=sys.stderr)
        return 2

    save_receipt_to_db(receipt, args.db_path)

    print(f"Obrada dovršena za '{receipt.image}'.")
    print(f"Prepoznato stavki: {len(receipt.items)}")
    print(f"Zbroj stavki: {_format_decimal(receipt.items_sum)}")
    if receipt.date:
        print(f"Datum računa: {receipt.date}")
    if receipt.time:
        print(f"Vrijeme računa: {receipt.time}")
    if receipt.total is not None:
        print(f"Ukupno (pretpostavljeno): {_format_decimal(receipt.total)}")
        if abs(receipt.items_sum - receipt.total) > TOTAL_TOLERANCE:
            diff = receipt.items_sum - receipt.total
            diff_str = f"{diff:+.2f}".replace(".", ",")
            print(
                (
                    "UPOZORENJE: Zbroj stavki i ukupni iznos se razlikuju "
                    f"({_format_decimal(receipt.items_sum)} vs {_format_decimal(receipt.total)}, razlika {diff_str})."
                ),
                file=sys.stderr,
            )
    else:
        print("Nije pronađen ukupni iznos.")
    print(f"Spremljeno u bazu: {args.db_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
