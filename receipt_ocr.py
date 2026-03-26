#!/usr/bin/env python3
"""Receipt parser powered by Gemini Vision."""

from __future__ import annotations
import argparse
import base64
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
from typing import List, Optional
from uuid import uuid4
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, redirect, url_for, render_template_string, send_file, abort
from werkzeug.utils import secure_filename
from config import (
    MAX_NEW_TOKENS,
    TOTAL_TOLERANCE,
    MAX_UPLOAD_FILES,
    UPLOAD_DIR,
    DEFAULT_LANG,
    DEFAULT_RESIZE_MAX,
    ENABLE_IMAGE_NORMALIZATION,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    GEMINI_ENDPOINT,
    GEMINI_MAX_PARALLEL,
    GEMINI_REQUESTS_PER_MINUTE,
    GEMINI_MAX_429_RETRIES,
    ONEDRIVE_IMPORT_DIR,
)



def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[INFO {timestamp}] {message}"
    print(line, flush=True)


GEMINI_VISION_PROMPT = """
Pročitaj račun sa slike i vrati isključivo valjani JSON bez markdowna.
JSON shema:
{
  "items": [
    {
      "description": string,
      "category": "Hrana | Cigarete, alkohol, kave,... | Kućne potrepštine | Lijekovi, troškovi liječenja | Odjeća i obuća | Škola i dječje aktivnosti | Sport | Automobili | Osiguranja | Internet/mobitel/TV | Struja | Voda | Plin | Smeće | Komunalni doprinos | Vodni doprinos | Putovanja, izleti, ručkovi | Ostalo" | null,
      "quantity": number | null,
      "unit_price": number | null,
      "total_price": number | null
    }
  ],
  "total": number | null,
  "date": "YYYY-MM-DD ili DD.MM.YYYY" | null,
  "time": "HH:MM[:SS]" | null
}
Pravila:
- decimalne zareze pretvori u točku
- nepoznate vrijednosti postavi na null
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


_THREAD_LOCAL = threading.local()
_GEMINI_MAX_OUTPUT_TOKENS = min(MAX_NEW_TOKENS, 2048)
_GEMINI_IMAGE_MAX_SIZE = (1400, 2600)
_GEMINI_RATE_LOCK = threading.Lock()
_GEMINI_NEXT_REQUEST_AT = 0.0
_GEMINI_MIN_INTERVAL_SECONDS = 60.0 / max(1, GEMINI_REQUESTS_PER_MINUTE)


def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE NOT NULL,
                json_path TEXT NOT NULL,
                language TEXT,
                data_json TEXT NOT NULL,
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
        conn.commit()
    finally:
        conn.close()


def save_receipt_to_db(receipt: ReceiptData, json_path: str, db_path: str) -> None:
    payload = json.dumps(serialise_receipt(receipt), ensure_ascii=False)
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO receipts (image_path, json_path, language, data_json, total, items_sum, date, time, warranty, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                json_path=excluded.json_path,
                language=excluded.language,
                data_json=excluded.data_json,
                total=excluded.total,
                items_sum=excluded.items_sum,
                date=excluded.date,
                time=excluded.time,
                warranty=excluded.warranty,
                updated_at=excluded.updated_at
            """,
            (
                receipt.image,
                os.path.abspath(json_path),
                receipt.language,
                payload,
                receipt.total,
                receipt.items_sum,
                receipt.date,
                receipt.time,
                1 if receipt.warranty else 0,
                now,
                now,
            ),
        )
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
        "Hrana","Cigarete, alkohol, kave,...","Kućne potrepštine","Lijekovi, troškovi liječenja",
        "Odjeća i obuća","Škola i dječje aktivnosti","Sport","Automobili","Osiguranja",
        "Internet/mobitel/TV","Struja","Voda","Plin","Smeće","Komunalni doprinos",
        "Vodni doprinos","Putovanja, izleti, ručkovi","Ostalo"
    ]
    summary = {category: [0.0] * 12 for category in categories}

    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT data_json FROM receipts")
        rows = cur.fetchall()
        for row in rows:
            data = json.loads(row["data_json"])
            entry_date = data.get("date")
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
            for item in data.get("items", []):
                category = item.get("category") or "Ostalo"
                total_price = item.get("total_price") or 0.0
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
        cur = conn.execute("SELECT id, data_json FROM receipts")
        for row in cur:
            data = json.loads(row["data_json"])
            entry_date = data.get("date")
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

            for item in data.get("items", []):
                item_category = item.get("category") or "Ostalo"
                if item_category != category:
                    continue

                items.append(
                    {
                        "receipt_id": row["id"],
                        "description": item.get("description") or "",
                        "category": item_category,
                        "quantity": _to_optional_float(item.get("quantity")),
                        "unit_price": _to_optional_float(item.get("unit_price")),
                        "total_price": _to_optional_float(item.get("total_price")),
                        "raw_date": entry_date,
                        "time": data.get("time"),
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


def update_receipt_record(receipt_id: int, data: dict, db_path: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(data, ensure_ascii=False)
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE receipts
            SET data_json = ?, total = ?, items_sum = ?, date = ?, time = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                payload,
                data.get("total"),
                data.get("items_sum"),
                data.get("date"),
                data.get("time"),
                now,
                receipt_id,
            ),
        )
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


def _truncate_broken_items_array(s: str) -> str:
    """
    Ako postoji polje "items": [ ... ] i zadnji element liste je napola
    (ili model nakon toga još nešto nadrobio), pokušaj odrezati sve
    iza zadnje zatvorene vitičaste zagrade '}' unutar te liste.

    Ideja: bolje izgubiti 1 polu-razvijenu stavku nego cijeli JSON.
    """
    key = '"items"'
    key_idx = s.find(key)
    if key_idx == -1:
        return s

    # Nađi početak liste '[' nakon "items"
    bracket_start = s.find("[", key_idx)
    if bracket_start == -1:
        return s

    # Nađi zadnju ']' koja zatvara tu listu (grubo, ali dovoljno dobro za naš slučaj)
    bracket_end = s.rfind("]")
    if bracket_end == -1 or bracket_end < bracket_start:
        return s

    arr_str = s[bracket_start: bracket_end + 1]

    # Zadnja zatvorena '}' unutar liste – tu režemo
    last_obj_end = arr_str.rfind("}")
    if last_obj_end == -1:
        return s

    # Nova lista: od '[' do zadnje '}', pa zatvori s ']'
    new_arr_str = arr_str[: last_obj_end + 1] + "]"

    # Sastavi novi string
    new_s = s[:bracket_start] + new_arr_str + s[bracket_end + 1 :]

    # Za svaki slučaj, opet pobriši trailing comma
    new_s = re.sub(r",(\s*[}\]])", r"\1", new_s)
    return new_s


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
    global _GEMINI_NEXT_REQUEST_AT
    with _GEMINI_RATE_LOCK:
        now = time.monotonic()
        wait_seconds = max(0.0, _GEMINI_NEXT_REQUEST_AT - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        _GEMINI_NEXT_REQUEST_AT = now + _GEMINI_MIN_INTERVAL_SECONDS


def _parse_retry_delay_seconds(response: requests.Response) -> float:
    default_delay = _GEMINI_MIN_INTERVAL_SECONDS
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        details = payload.get("error", {}).get("details", [])
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                retry_value = detail.get("retryDelay")
                if isinstance(retry_value, str) and retry_value.endswith("s"):
                    try:
                        return max(float(retry_value[:-1]), 0.5)
                    except ValueError:
                        continue

    text = response.text or ""
    match = re.search(r"Please retry in ([0-9]+(?:\\.[0-9]+)?)s", text)
    if match:
        try:
            return max(float(match.group(1)), 0.5)
        except ValueError:
            pass
    return default_delay


def call_gemini_vision_parser(image_path: str) -> dict:
    api_key = os.environ.get(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Očekujem varijablu okoline {GEMINI_API_KEY_ENV} s Gemini API ključem."
        )

    mime_type = _guess_mime_type(image_path)

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": GEMINI_VISION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_base64,
                        }
                    },
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

    url = f"{GEMINI_ENDPOINT}/{GEMINI_MODEL}:generateContent?key={api_key}"
    for attempt in range(GEMINI_MAX_429_RETRIES + 1):
        _reserve_gemini_request_slot()
        resp = _gemini_session().post(url, json=payload, timeout=(10, 90))
        if resp.status_code == 200:
            break
        if resp.status_code != 429 or attempt >= GEMINI_MAX_429_RETRIES:
            raise RuntimeError(f"Gemini Vision API error {resp.status_code}: {resp.text}")
        sleep_seconds = _parse_retry_delay_seconds(resp)
        log_progress(
            f"Gemini quota/rate limit (429), čekam {sleep_seconds:.1f}s "
            f"prije ponovnog pokušaja ({attempt + 1}/{GEMINI_MAX_429_RETRIES})."
        )
        time.sleep(sleep_seconds)

    data = resp.json()
    try:
        model_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError(f"Neočekivan odgovor Gemini Vision modela: {data}") from exc

    return parse_llm_json(model_text)



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


def process_single_image(image_path: str, language: str, source_path: Optional[str] = None) -> dict:
    used_path = prepare_image_for_gemini(image_path)
    receipt = build_receipt_data(used_path, language)
    payload = serialise_receipt(receipt)
    payload.setdefault("warranty", False)
    json_output = os.path.splitext(used_path)[0] + "_parsed.json"
    return {
        "image_path": used_path,
        "preview_path": used_path,
        "json_path": json_output,
        "payload": payload,
        "progress": [],
        "warranty": False,
        "source_path": source_path,
    }


def process_images_batch(jobs: List[tuple[str, Optional[str]]], language: str) -> List[dict]:
    if not jobs:
        return []
    workers = min(max(1, GEMINI_MAX_PARALLEL), max(1, len(jobs)))
    results: List[Optional[dict]] = [None] * len(jobs)
    log_progress(f"Paralelna obrada računa: {len(jobs)} datoteka, {workers} radnika.")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_image, image_path, language, source_path): idx
            for idx, (image_path, source_path) in enumerate(jobs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                failed_file = os.path.basename(jobs[idx][0])
                raise RuntimeError(f"{failed_file}: {exc}") from exc
    return [entry for entry in results if entry is not None]


def create_app(db_path: str, default_lang: str, default_model: str) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index() -> str:
        # Defaultno: sortiraj po datumu (noviji prvo) i prikaži zadnjih 100 računa
        sort_by = request.args.get("sort", "date")
        direction = request.args.get("dir", "desc")
        limit_param = request.args.get("limit", "100")
        allowed_limits = ["10", "20", "50", "100", "all"]

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
            default_lang=default_lang,
            default_model=default_model,
            sort_by=sort_by,
            direction=direction,
            filters=filters,
            limit_value=limit_param,
            allowed_limits=allowed_limits,
            category_summary=category_summary,
            current_year=selected_year,
            available_years=available_years,
            monthly_totals=monthly_totals,
            total_sum=sum(monthly_totals),
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
            month=month,
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

        # Ograničenje: max 10 datoteka odjednom
        if upload_files and len(upload_files) > MAX_UPLOAD_FILES:
            return redirect(
                url_for(
                    "index",
                    error=f"Maksimalno je dopušteno učitati 10 datoteka odjednom (pokušali ste {len(upload_files)}).",
                )
            )

        if not upload_files and not manual_entry:
            return redirect(url_for("index"))

        upload_lang = default_lang
        upload_model = default_model

        upload_dir = os.path.join(os.getcwd(), UPLOAD_DIR)
        os.makedirs(upload_dir, exist_ok=True)

        processed_entries = []

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
                json_path=os.path.join(upload_dir, f"manual_{int(datetime.now().timestamp())}_parsed.json"),
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
        upload_jobs: List[tuple[str, Optional[str]]] = []
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
            upload_jobs.append((temp_path, None))

        try:
            processed_entries = process_images_batch(upload_jobs, upload_lang)
        except Exception as exc:
            log_progress(f"Greška pri obradi upload datoteka: {exc}")
            return redirect(url_for("index", error=f"Greška pri obradi upload datoteka: {exc}"))

        if not processed_entries:
            return redirect(url_for("index"))

        first_entry = processed_entries[0]
        pending_entries = processed_entries[1:]

        first_rotate_target = (
            first_entry["preview_path"]
            if first_entry["preview_path"] and not first_entry["preview_path"].startswith("manual://")
            else None
        )
        preview_image_path = first_entry["preview_path"]
        preview_image_mtime = None
        if preview_image_path and os.path.exists(preview_image_path):
            preview_image_mtime = int(os.path.getmtime(preview_image_path))

        return render_template_string(
            DETAIL_TEMPLATE,
            receipt={"id": None, "image_path": first_entry["image_path"]},
            data=first_entry["payload"],
            items=first_entry["payload"]["items"],
            saved=False,
            is_new=True,
            json_path=first_entry["json_path"],
            image_path=first_entry["image_path"],
            preview_image_path=preview_image_path,
            preview_image_mtime=preview_image_mtime,
            base_payload=first_entry["payload"],
            progress=first_entry["progress"],
            default_lang=upload_lang,
            default_model=upload_model,
            pending_payloads=pending_entries,
            pending_count=len(pending_entries),
            rotate_target=first_rotate_target,
            current_url=request.url,
            format_date=_format_date_for_display,
            form_error=None,
            source_path=first_entry.get("source_path"),
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

        upload_dir = os.path.join(os.getcwd(), UPLOAD_DIR)
        os.makedirs(upload_dir, exist_ok=True)

        upload_jobs: List[tuple[str, Optional[str]]] = []

        for src_path in image_paths:
            original_name = secure_filename(os.path.basename(src_path)) or "receipt.png"
            name, ext = os.path.splitext(original_name)
            if not ext:
                ext = ".png"

            unique_suffix = f"{int(datetime.now().timestamp() * 1000)}_{uuid4().hex[:6]}"
            filename = f"{name}_{unique_suffix}{ext}"

            temp_path = os.path.join(upload_dir, filename)
            # Kopiraj iz OneDrive foldera u uploads/
            shutil.copy2(src_path, temp_path)
            upload_jobs.append((temp_path, src_path))

        try:
            processed_entries = process_images_batch(upload_jobs, upload_lang)
        except Exception as exc:
            log_progress(f"Greška pri obradi OneDrive datoteka: {exc}")
            return redirect(url_for("index", error=f"Greška pri obradi OneDrive datoteka: {exc}"))

        if not processed_entries:
            return redirect(url_for("index"))

        first_entry = processed_entries[0]
        pending_entries = processed_entries[1:]

        first_rotate_target = (
            first_entry["preview_path"]
            if first_entry["preview_path"] and not first_entry["preview_path"].startswith("manual://")
            else None
        )
        preview_image_path = first_entry["preview_path"]
        preview_image_mtime = None
        if preview_image_path and os.path.exists(preview_image_path):
            preview_image_mtime = int(os.path.getmtime(preview_image_path))

        return render_template_string(
            DETAIL_TEMPLATE,
            receipt={"id": None, "image_path": first_entry["image_path"]},
            data=first_entry["payload"],
            items=first_entry["payload"]["items"],
            saved=False,
            is_new=True,
            json_path=first_entry["json_path"],
            image_path=first_entry["image_path"],
            preview_image_path=preview_image_path,
            preview_image_mtime=preview_image_mtime,
            base_payload=first_entry["payload"],
            progress=first_entry["progress"],
            default_lang=upload_lang,
            default_model=upload_model,
            pending_payloads=pending_entries,
            pending_count=len(pending_entries),
            rotate_target=first_rotate_target,
            current_url=request.url,
            format_date=_format_date_for_display,
            form_error=None,
            source_path=first_entry.get("source_path"),
        )



    @app.route("/receipt/save_new", methods=["POST"])
    def save_new_receipt() -> str:
        image_path = request.form.get("image_path") or ""
        json_path = request.form.get("json_path") or (os.path.splitext(image_path or "receipt.png")[0] + "_parsed.json")
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
                json_path=json_path,
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

        # --- Ako je sve u redu, spremi u JSON i DB ---
        write_receipt_json_payload(updated_payload, json_path)
        receipt_obj = receipt_from_payload(updated_payload)
        save_receipt_to_db(receipt_obj, json_path, db_path)

        # Ako je ovaj račun uvezen iz OneDrive-a i sad je uspješno spremljen,
        # obriši originalnu datoteku (source_path) iz OneDrive foldera.
        source_path = request.form.get("source_path") or ""
        if source_path:
            abs_source = os.path.abspath(source_path)
            try:
                if os.path.exists(abs_source):
                    os.remove(abs_source)
                    log_progress(f"Obrisan originalni OneDrive fajl: {abs_source}")
            except OSError as exc:
                log_progress(f"Ne mogu obrisati izvorni OneDrive fajl '{abs_source}': {exc}")


        if pending_payloads:
            next_entry = pending_payloads.pop(0)
            resize_image(next_entry["image_path"])
            rotate_target = next_entry["preview_path"] if next_entry.get("preview_path") and not next_entry["preview_path"].startswith("manual://") else None
            preview_image_path = next_entry.get("preview_path", next_entry["image_path"])
            preview_image_mtime = None
            if preview_image_path and os.path.exists(preview_image_path):
                preview_image_mtime = int(os.path.getmtime(preview_image_path))

            return render_template_string(
                DETAIL_TEMPLATE,
                receipt={"id": None, "image_path": next_entry["image_path"]},
                data=next_entry["payload"],
                items=next_entry["payload"]["items"],
                saved=False,
                is_new=True,
                json_path=next_entry["json_path"],
                image_path=next_entry["image_path"],
                preview_image_path=preview_image_path,
                preview_image_mtime=preview_image_mtime,
                base_payload=next_entry["payload"],
                progress=next_entry.get("progress", []),
                default_lang=default_lang,
                default_model=default_model,
                pending_payloads=pending_payloads,
                pending_count=len(pending_payloads),
                rotate_target=rotate_target,
                current_url=request.url,
                format_date=_format_date_for_display,
                form_error=None,
                source_path=next_entry.get("source_path"),
            )

        if image_path and not image_path.startswith("manual://"):
            resize_image(image_path)
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

        upload_dir = os.path.join(os.getcwd(), UPLOAD_DIR)
        os.makedirs(upload_dir, exist_ok=True)
        saved_path = os.path.join(upload_dir, filename)
        upload_file.save(saved_path)

        normalize_image_orientation(saved_path)
        saved_path = ensure_gemini_compatible_image(saved_path)
        resize_image(saved_path)

        abs_path = os.path.abspath(saved_path)

        data = json.loads(row["data_json"])
        data["image"] = abs_path

        conn = get_db_connection(db_path)
        try:
            conn.execute(
                "UPDATE receipts SET image_path = ?, data_json = ?, updated_at = ? WHERE id = ?",
                (abs_path, json.dumps(data, ensure_ascii=False), datetime.now().isoformat(timespec="seconds"), receipt_id),
            )
            conn.commit()
        finally:
            conn.close()

        json_path = row["json_path"]
        if json_path:
            write_receipt_json_payload(data, json_path)

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
        upload_dir = os.path.abspath(os.path.join(os.getcwd(), UPLOAD_DIR))
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
            uploads_dir = os.path.abspath(os.path.join(os.getcwd(), UPLOAD_DIR))
            if not os.path.exists(abs_path) or not abs_path.startswith(uploads_dir):
                return redirect(url_for("index"))

            direction = request.form.get("direction", "right")
            # Rotate the image file on disk
            rotate_image_file(abs_path, direction=direction)

            # Rebuild context for DETAIL_TEMPLATE
            image_path = request.form.get("image_path") or path
            json_path = request.form.get("json_path") or (
                os.path.splitext(image_path or "receipt.png")[0] + "_parsed.json"
            )
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

            default_lang = request.form.get("default_lang") or default_lang
            default_model = request.form.get("default_model") or default_model

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
                json_path=json_path,
                image_path=image_path,
                preview_image_path=preview_image_path,
                preview_image_mtime=preview_image_mtime,
                base_payload=base_payload,
                progress=[],
                default_lang=default_lang,
                default_model=default_model,
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
        uploads_dir = os.path.abspath(os.path.join(os.getcwd(), UPLOAD_DIR))
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
        data = json.loads(row["data_json"])
        data.setdefault("warranty", bool(row["warranty"]))
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
                    json_path=row["json_path"],
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
            write_receipt_json_payload(updated_payload, row["json_path"])
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
            json_path=row["json_path"],
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

    return app


INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Billing me softly</title>
<style>
  :root {
    --color-bg: #F4F7FB;
    --color-card: #FFFFFF;
    --color-border: #D6DFEA;
    --color-text: #111827;
    --color-text-muted: #6B7280;

    --color-primary: #12324A;
    --color-primary-light: #1F4F7F;
    --color-primary-soft: #E3EDF7;

    --color-success: #137333;
    --color-danger: #B00020;
  }

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--color-bg);
    color: var(--color-text);
  }

  .page {
    max-width: 2000px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 3rem;
  }

  .topbar {
    background: var(--color-primary);
    color: #fff;
    padding: 0.75rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
  }

  .topbar-title {
    font-weight: 600;
    letter-spacing: 0.03em;
  }

  .topbar-subtitle {
    font-size: 0.85rem;
    opacity: 0.85;
  }

  .topbar-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .pill {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    background: rgba(255,255,255,0.1);
    font-size: 0.75rem;
  }

  a.button,
  button.button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1.1rem;
    background: var(--color-primary);
    color: #fff;
    border-radius: 9999px;
    border: none;
    font-size: 0.9rem;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.15s ease, transform 0.05s ease;
  }

  a.button:hover,
  button.button:hover {
    background: var(--color-primary-light);
    transform: translateY(-1px);
  }

  a.button:active,
  button.button:active {
    transform: translateY(0);
  }

  h1, h2, h3 {
    margin: 1.5rem 0 0.75rem;
    color: var(--color-primary);
  }

  .card {
    background: var(--color-card);
    border-radius: 12px;
    border: 1px solid var(--color-border);
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.75rem;
  }

  .card-header h2,
  .card-header h3 {
    margin: 0;
  }

  .card-header small {
    color: var(--color-text-muted);
  }

  .alert {
    margin-top: 1rem;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    border: 1px solid #F5C2C7;
    background: #F8D7DA;
    color: #842029;
    font-size: 0.9rem;
  }

  form {
    margin: 0;
  }

  form .form-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1rem;
    align-items: flex-end;
  }

  label {
    font-size: 0.85rem;
    color: var(--color-text-muted);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  input[type="text"],
  input[type="file"],
  select {
    border-radius: 10px;
    border: 1px solid var(--color-border);
    padding: 0.4rem 0.6rem;
    font-size: 0.9rem;
    background: #fff;
  }

  input[type="checkbox"] {
    margin-right: 0.25rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
    background: var(--color-card);
  }

  th, td {
    padding: 0.45rem 0.5rem;
    text-align: left;
    border-bottom: 1px solid #EAECF0;
    font-size: 0.85rem;
  }

  th {
    background: var(--color-primary-soft);
    font-weight: 600;
    color: var(--color-primary);
  }

  tr:nth-child(even) td {
    background: #FAFCFF;
  }

  tbody tr:hover td {
    background: #E9F2FD;
  }

  .warranty-pill {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
  }

  .warranty-yes {
    background: #E6F4EA;
    color: var(--color-success);
  }

  .warranty-no {
    background: #E5E7EB;
    color: var(--color-text-muted);
  }

  .amount-cell {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .amount-header {
    text-align: right;
  }

  .empty {
    margin-top: 2rem;
    font-style: italic;
    color: var(--color-text-muted);
  }


  .filters-inline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1rem;
  }

  .filters-inline label {
    min-width: 150px;
  }

  .small-text {
    font-size: 0.8rem;
    color: var(--color-text-muted);
  }

  @media (max-width: 768px) {
    .page {
      padding: 1rem;
    }
    .topbar {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.35rem;
    }
    .filters-inline label {
      width: 100%;
    }
    table {
      font-size: 0.8rem;
      display: block;
      overflow-x: auto;
      white-space: nowrap;
    }
  }
</style>

  </head>
  <body>
    <div class="topbar">
      <div>
        <div class="topbar-title">Billing me softly</div>
      </div>
      <div class="topbar-actions">
        <a class="button" href="/">Osvježi</a>
      </div>
    </div>
    <div class="page">
    {% if error_message %}
      <div class="alert" style="margin-top:1rem; background:#f8d7da; color:#721c24; padding:0.8rem; border:1px solid #f5c6cb; border-radius:4px;">
        Dogodila se greška: {{ error_message }}
      </div>
    {% endif %}
    {% if progress %}
      <div class="panel" style="margin-top:1rem;">
        <h3>Koraci obrade</h3>
        <pre style="max-height:200px; overflow:auto;">{% for line in progress %}{{ line }}&#10;{% endfor %}</pre>
      </div>
    {% endif %}
    <form action="{{ url_for('upload_receipt') }}" method="post" enctype="multipart/form-data"
          style="margin-top:1rem; background:#fff; padding:1rem; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); display:flex; flex-wrap:wrap; gap:1rem;">
      <label style="align-self:center;">Učitaj jednu ili više (max 10) fotografija računa (PNG/JPG):</label>
      <input type="file" name="image" accept="image/*" multiple />
      <label style="align-self:center;">
        <input type="checkbox" name="manual" /> Ručni unos bez slike
      </label>
      <button class="button" type="submit">Start</button>
    </form>
    <form action="{{ url_for('import_onedrive') }}" method="post"
          style="margin-top:1rem; background:#fff; padding:1rem; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); display:flex; flex-wrap:wrap; gap:1rem;">
      <label style="align-self:center;">Učitaj slike računa iz OneDrive foldera:</label>
      <input type="text" name="onedrive_path" value="{{ onedrive_default_path or '' }}"
             placeholder="npr. /home/ituda/OneDrive/Racuni" style="flex:1;" />
      <button class="button" type="submit">Uvezi iz OneDrive</button>
      <p style="font-size:0.85rem; color:#555; flex-basis:100%;">
        Učitava maksimalno {{ onedrive_import_limit }} slikovnih datoteka (.png, .jpg, .jpeg, .webp, .heic, .tif, .tiff) iz zadane putanje po jednom kliku.
      </p>
      {% if onedrive_default_path and onedrive_total_files is not none %}
        <p style="font-size:0.85rem; color:#333; flex-basis:100%;">
          Trenutno u folderu <code>{{ onedrive_default_path }}</code> ima
          <strong>{{ onedrive_total_files }}</strong> slikovnih datoteka.
          To znači da će biti potrebno
          <strong>{{ (onedrive_total_files // onedrive_import_limit) + (1 if (onedrive_total_files % onedrive_import_limit) > 0 else 0) }}</strong>
          uvoza da obradiš sve račune (ako se broj ne mijenja tijekom obrade).
        </p>
      {% endif %}

    </form>

    {% if category_summary %}
      <h2 id="categories-section">Pregled po kategorijama
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
                style="text-decoration:none; color:inherit;">
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
        style="margin-top:1rem; background:#fff; padding:1rem; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); display:flex; flex-wrap:wrap; gap:1rem;">
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
        <button class="button" type="submit">Primijeni filtere</button>
        <a class="button" href="/" style="background:#6c757d;">Resetiraj</a>
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
            <td>{{ "Da" if receipt.warranty else "Ne" }}</td>
            <td><a class="button" href="{{ url_for('receipt_detail', receipt_id=receipt.id) }}">Uredi</a></td>
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


DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Račun {{ receipt.id if receipt and receipt.id else 'Novi račun' }}</title>
    <style>
      :root {
        --color-bg: #F4F7FB;
        --color-card: #FFFFFF;
        --color-border: #D6DFEA;
        --color-text: #111827;
        --color-text-muted: #6B7280;

        --color-primary: #12324A;
        --color-primary-light: #1F4F7F;
        --color-primary-soft: #E3EDF7;

        --color-success: #137333;
        --color-danger: #B00020;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--color-bg);
        color: var(--color-text);
      }

      .topbar {
        background: var(--color-primary);
        color: #fff;
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      }

      .topbar-title {
        font-weight: 600;
        letter-spacing: 0.03em;
      }

      .topbar-subtitle {
        font-size: 0.85rem;
        opacity: 0.85;
      }

      .topbar-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
      }

      .page {
        max-width: 2000px;
        margin: 0 auto;
        padding: 1.5rem 1.5rem 3rem;
      }

      h1, h2, h3 {
        color: var(--color-primary);
        margin-top: 0;
      }

      h1 {
        margin-bottom: 1rem;
        font-size: 1.4rem;
      }

      .card {
        background: var(--color-card);
        border-radius: 12px;
        border: 1px solid var(--color-border);
        padding: 1rem 1.25rem 1.25rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
      }

      .card-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.75rem;
      }

      .card-header h2 {
        margin: 0;
        font-size: 1.1rem;
      }

      .card-header small {
        color: var(--color-text-muted);
        font-size: 0.8rem;
      }

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
        border-radius: 8px;
        border: 1px solid var(--color-border);
        background: #f9fafb;
      }

      .small-text {
        font-size: 0.8rem;
        color: var(--color-text-muted);
      }

      .alert {
        margin-top: 1rem;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-size: 0.9rem;
      }

      .alert-error {
        border: 1px solid #F5C2C7;
        background: #F8D7DA;
        color: #842029;
      }

      .alert-success {
        border: 1px solid #C3E6CB;
        background: #D4EDDA;
        color: #155724;
      }

      .alert-info {
        border: 1px solid #B3B7FF;
        background: #E2E3FF;
        color: #14135E;
      }

      a.button,
      button.button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1.1rem;
        border-radius: 9999px;
        border: none;
        font-size: 0.9rem;
        cursor: pointer;
        text-decoration: none;
        transition: background 0.15s ease, transform 0.05s ease, box-shadow 0.15s ease;
        font-weight: 500;
      }

      a.button-primary,
      button.button-primary {
        background: var(--color-primary);
        color: #fff;
      }

      a.button-secondary,
      button.button-secondary {
        background: #fff;
        color: var(--color-primary);
        border: 1px solid var(--color-primary-soft);
      }

      a.button:hover,
      button.button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15);
      }

      a.button:active,
      button.button:active {
        transform: translateY(0);
        box-shadow: none;
      }

      form {
        margin: 0;
      }

      label {
        display: block;
        font-size: 0.85rem;
        color: var(--color-text-muted);
        margin-top: 0.5rem;
        margin-bottom: 0.15rem;
      }

      input[type="text"],
      input[type="number"],
      select {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--color-border);
        padding: 0.35rem 0.5rem;
        font-size: 0.9rem;
        background: #fff;
      }

      input[type="checkbox"] {
        margin-right: 0.35rem;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
        background: var(--color-card);
      }

      th, td {
        padding: 0.35rem 0.45rem;
        text-align: left;
        border-bottom: 1px solid #EAECF0;
        font-size: 0.8rem;
      }

      th {
        background: var(--color-primary-soft);
        font-weight: 600;
        color: var(--color-primary);
      }

      tbody tr:nth-child(even) td {
        background: #FAFCFF;
      }

      tbody tr:hover td {
        background: #E9F2FD;
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
        color: var(--color-text-muted);
      }

      #items-sum-label {
        font-weight: 600;
        color: var(--color-primary);
        margin-left: 0.25rem;
      }

      .rotate-group {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }

      @media (max-width: 900px) {
        .page {
          padding: 1rem;
        }
        .detail-layout {
          flex-direction: column;
        }
        .detail-image-card,
        .detail-form-card {
          max-width: 100%;
          flex: 1 1 100%;
        }
        table {
          display: block;
          overflow-x: auto;
          white-space: nowrap;
        }
      }
    </style>
  </head>
  <body>
    <div class="topbar">
      <div>
        <div class="topbar-title">Billing me softly</div>
        <div class="topbar-subtitle">Uređivanje računa</div>
      </div>
      <div class="topbar-actions">
        <a class="button button-secondary" href="{{ url_for('index') }}">Natrag na popis</a>
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
              <input type="hidden" name="json_path" value="{{ json_path }}" />
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
              <input type="hidden" name="json_path" value="{{ json_path }}" />
              <input type="hidden" name="base_payload" value='{{ base_payload | tojson }}' />
              <input type="hidden" name="pending_payloads" value='{{ pending_payloads | tojson }}' />
              <input type="hidden" name="preview_image_path" value="{{ preview_image_path }}" />
              <input type="hidden" name="source_path" value="{{ source_path or '' }}" />
            {% endif %}

            <label>Datum:</label>
            <input type="text"
                   name="date"
                   value="{{ format_date(data.date) }}"
                   placeholder="npr. 01.03.2025" />

            <label>Vrijeme:</label>
            <input type="text"
                   name="time"
                   value="{{ data.time or '' }}"
                   placeholder="HH:MM[:SS]" />

            <label>Ukupno (€):</label>
            <input type="text"
                   name="total"
                   value="{{ ('%.2f'|format(data.total)).replace('.', ',') if data.total is not none else '' }}" />

            <label style="margin-top:0.6rem;">
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
                        "Hrana","Cigarete, alkohol, kave,...","Kućne potrepštine","Lijekovi, troškovi liječenja",
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
              <a class="button button-secondary" href="{{ url_for('index') }}">Natrag</a>
            </div>
          </form>
        </div>
      </div>

      <script>
        // Parsiranje hrvatskog formata brojeva (npr. "1.234,56")
        function parseEuro(value) {
          if (!value) return NaN;
          let v = String(value).trim();

          // Ukloni razmake
          v = v.replace(/\s+/g, '');

          // Zamijeni decimalni zarez točkom
          v = v.replace(',', '.');

          // Zadrži samo znamenke, minus i točku
          v = v.replace(/[^0-9\.\-]/g, '');

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
                  "Hrana","Cigarete, alkohol, kave,...","Kućne potrepštine","Lijekovi, troškovi liječenja",
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
      </script>
    </div>
  </body>
</html>
"""

CATEGORY_ITEMS_TEMPLATE = """
<!DOCTYPE html>
<html lang="hr">
  <head>
    <meta charset="utf-8" />
    <title>Pregled kategorije {{ category }} - {{ month_name }} {{ year }}</title>
    <style>
      :root {
        --color-bg: #F4F7FB;
        --color-card: #FFFFFF;
        --color-border: #D6DFEA;
        --color-text: #111827;
        --color-text-muted: #6B7280;

        --color-primary: #12324A;
        --color-primary-light: #1F4F7F;
        --color-primary-soft: #E3EDF7;

        --color-success: #137333;
        --color-danger: #B00020;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--color-bg);
        color: var(--color-text);
      }

      .page {
        max-width: 2000px;
        margin: 0 auto;
        padding: 1.5rem 1.5rem 3rem;
      }

      .topbar {
        background: var(--color-primary);
        color: #fff;
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      }

      .topbar-title {
        font-weight: 600;
        letter-spacing: 0.03em;
      }

      .topbar-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
      }

      a.button,
      button.button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1.1rem;
        background: var(--color-primary);
        color: #fff;
        border-radius: 9999px;
        border: none;
        font-size: 0.9rem;
        cursor: pointer;
        text-decoration: none;
        transition: background 0.15s ease, transform 0.05s ease;
      }

      a.button:hover,
      button.button:hover {
        background: var(--color-primary-light);
        transform: translateY(-1px);
      }

      h1, h2, h3 {
        margin: 1.5rem 0 0.75rem;
        color: var(--color-primary);
      }

      .card {
        background: var(--color-card);
        border-radius: 12px;
        border: 1px solid var(--color-border);
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
      }

      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
        background: var(--color-card);
      }

      th, td {
        padding: 0.45rem 0.5rem;
        text-align: left;
        border-bottom: 1px solid #EAECF0;
        font-size: 0.85rem;
      }

      th {
        background: var(--color-primary-soft);
        font-weight: 600;
        color: var(--color-primary);
      }

      tbody tr:nth-child(even) td {
        background: #FAFCFF;
      }

      tbody tr:hover td {
        background: #E9F2FD;
      }

      .amount-cell {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }

      .empty {
        margin-top: 2rem;
        font-style: italic;
        color: var(--color-text-muted);
      }

      @media (max-width: 768px) {
        .page {
          padding: 1rem;
        }
        table {
          display: block;
          overflow-x: auto;
          white-space: nowrap;
        }
      }
    </style>
  </head>
  <body>
    <div class="topbar">
      <div>
        <div class="topbar-title">Billing me softly</div>
      </div>
      <div class="topbar-actions">
        <a class="button" href="{{ back_url }}">Natrag</a>
      </div>
    </div>

    <div class="page">
      <h2>Pregled kategorije "{{ category }}" za {{ month_name }} {{ year }}</h2>
      <p style="color:var(--color-text-muted); font-size:0.9rem;">
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
                  <a class="button"
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
          Ili si stvarno štedljiv, ili filteri lažu. 🙂
        </p>
      {% endif %}
    </div>
  </body>
</html>
"""



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


def build_receipt_data(image_path: str, language: str) -> ReceiptData:
    log_progress(f"Pokrećem Gemini Vision pipeline za {image_path}...")
    llm_payload = call_gemini_vision_parser(image_path)

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
    date_value = _normalize_date_for_db(raw_date_value)
    time_value = _clean_string(llm_payload.get("time"))

    return ReceiptData(
        image=os.path.abspath(image_path),
        language=language,
        items=items,
        items_sum=items_sum,
        total=total_value,
        date=date_value,
        time=time_value,
    )



def dump_receipt_json(data: ReceiptData, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(serialise_receipt(data), handle, ensure_ascii=False, indent=2)


def write_receipt_json_payload(payload: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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
        "--output",
        help="Destination JSON path. Defaults to <image>_parsed.json in CWD.",
    )
    parser.add_argument(
        "--db-path",
        default="receipts.db",
        help="SQLite database for reviewed receipts (default: receipts.db)",
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


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    init_db(args.db_path)

    if args.serve:
        log_progress(f"Pokrećem web poslužitelj na http://{args.host}:{args.port}")
        app = create_app(args.db_path, args.lang, GEMINI_MODEL)
        app.run(host=args.host, port=args.port, debug=False)
        return 0

    image_path = args.image
    if not image_path:
        print("Greška: morate navesti putanju do slike računa.", file=sys.stderr)
        return 1

    if not os.path.exists(image_path):
        print(f"Greška: ne postoji datoteka '{image_path}'.", file=sys.stderr)
        return 1

    try:
        prepared_image = prepare_image_for_gemini(image_path)
        receipt = build_receipt_data(prepared_image, args.lang)
    except RuntimeError as exc:
        print(f"Gemini obrada nije uspjela: {exc}", file=sys.stderr)
        return 2

    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_parsed.json"

    dump_receipt_json(receipt, output_path)
    save_receipt_to_db(receipt, output_path, args.db_path)

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
    print(f"JSON spremljen u: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
