#!/usr/bin/env python3
"""
Receipt OCR parser powered by Tesseract CLI.

Given an input image (PNG, JPG, ...), runs Tesseract with Croatian language
data, extracts line items and a likely total, and stores the results as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional


TOTAL_KEYWORDS = [
    "ukupno",
    "u k u p n o",
    "u  k  u  p  n  o",
    "za platiti",
    "za platit",
    "total",
    "ukupni iznos",
    "svega",
]

NON_ITEM_KEYWORDS = [
    "pdv",
    "porez",
    "međuzbroj",
    "meduzbroj",
    "popust",
    "karticom",
    "gotovina",
    "racun",
    "račun",
    "kupac",
    "prodavac",
    "placanje",
    "plaćanje",
    "iznos",
    "datum",
    "vrijeme",
    "fiskal",
    "fisk.",
    "barkod",
    "mjur",
]

DATE_KEYWORDS = [
    "datum",
    "date",
]

TIME_KEYWORDS = [
    "vrijeme",
    "time",
    "sat",
]

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
)
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

TOTAL_TOLERANCE = 0.05  # +/- range to account for rounding in OCR data


@dataclass
class ReceiptItem:
    description: str
    quantity: Optional[int]
    unit_price: Optional[float]
    total_price: float


@dataclass
class ReceiptData:
    image: str
    language: str
    raw_text: str
    items: List[ReceiptItem]
    items_sum: float
    total: Optional[float]
    total_candidates: List[float]
    date: Optional[str]
    time: Optional[str]


class TesseractError(RuntimeError):
    pass


def run_tesseract(image_path: str, lang: str) -> str:
    """Runs Tesseract CLI (stdout mode) and returns the captured text."""
    cmd = ["tesseract", image_path, "stdout", "-l", lang]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise TesseractError(exc.stderr.strip() or str(exc)) from exc
    return proc.stdout


def normalize_line(line: str) -> str:
    """Basic cleanup; strips control chars that often sneak from OCR."""
    return re.sub(r"[\x00-\x1f]+", "", line).strip()


def split_lines(text: str) -> List[str]:
    return [normalize_line(line) for line in text.splitlines() if normalize_line(line)]


def extract_amount_strings(text: str) -> List[str]:
    """
    Returns amount-like substrings such as 12,34 or 45.67. Allows both decimal
    separators so they can be normalised later.
    """
    return re.findall(r"\d+[.,]\d{1,2}", text)


def parse_amount(amount: str) -> Optional[float]:
    """Converts a Croatian/Euro styled number into a float."""
    cleaned = amount.strip()
    cleaned = re.sub(r"[^\d,.-]", "", cleaned)
    if cleaned.count(",") > 1 or cleaned.count(".") > 1:
        # Drop thousands separators by removing the earlier symbol.
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "")
            else:
                cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(".", "").replace(",", "")
    else:
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "")
            else:
                cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def looks_like_total_line(line: str) -> bool:
    lowered = line.lower()
    compact = re.sub(r"\s+", "", lowered)
    for keyword in TOTAL_KEYWORDS:
        kw_lower = keyword.lower()
        if kw_lower in lowered:
            return True
        kw_compact = re.sub(r"\s+", "", kw_lower)
        if kw_compact and kw_compact in compact:
            return True
    return False


def looks_like_non_item(line: str) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in NON_ITEM_KEYWORDS)


def extract_receipt_items(lines: Iterable[str]) -> List[ReceiptItem]:
    """
    Attempt to capture line items by looking for trailing amounts.
    Keeps the last numeric value on the line as the total price.
    """
    items: List[ReceiptItem] = []
    reached_total = False
    for line in lines:
        if reached_total:
            break

        lowered_line = str(line).casefold()

        if looks_like_total_line(line):
            reached_total = True
            continue

        if looks_like_non_item(line):
            continue

        if lowered_line.strip() == "kom":
            continue

        if "kom" in lowered_line and re.search(r"\bx\b", lowered_line):
            continue

        amount_strings = extract_amount_strings(line)
        if not amount_strings:
            continue

        total_amount_raw = amount_strings[-1]
        total_price = parse_amount(total_amount_raw)
        if total_price is None or total_price <= 0:
            continue

        prefix = line.rsplit(total_amount_raw, 1)[0].strip(" .:-\t")
        if not prefix:
            continue

        unit_price: Optional[float] = None
        quantity: Optional[int] = None

        if len(amount_strings) >= 2:
            unit_price_candidate = parse_amount(amount_strings[-2])
            if unit_price_candidate is not None and unit_price_candidate != total_price:
                unit_price = unit_price_candidate

        # Try to guess quantity from start of line/prefix.
        qty_match = re.match(r"(?P<qty>\d+)(?:\s*(?:x|×)\s*(\d+[.,]\d{1,2}))?", prefix, flags=re.IGNORECASE)
        if qty_match:
            qty_value = qty_match.group("qty")
            try:
                quantity = int(qty_value)
            except ValueError:
                quantity = None
            # Remove quantity notation from description.
            prefix = prefix[qty_match.end():].strip(" .:-")

        description = prefix or line

        # Remove known amount strings from description before numeric-only control.
        cleaned_description = description
        for amt in amount_strings[:-1]:
            cleaned_description = cleaned_description.replace(amt, "")
        cleaned_description = re.sub(r"[^\w\s]", " ", cleaned_description)
        if re.search(r"\b\d{3,}\b", cleaned_description):
            # Likely product codes or other numeric noise, skip this line.
            continue

        items.append(
            ReceiptItem(
                description=description,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
            )
        )
    return items


def extract_date_time(lines: Iterable[str]) -> tuple[Optional[str], Optional[str]]:
    date_value: Optional[str] = None
    time_value: Optional[str] = None

    for line in lines:
        lowered = line.lower()
        has_date_keyword = any(keyword in lowered for keyword in DATE_KEYWORDS)
        has_time_keyword = any(keyword in lowered for keyword in TIME_KEYWORDS)

        if has_date_keyword or date_value is None:
            match = DATE_PATTERN.search(line)
            if match:
                captured = match.group(0)
                normalized = captured.replace("/", ".").replace("-", ".")
                date_value = normalized

        if has_time_keyword or (time_value is None and ":" in line):
            match = TIME_PATTERN.search(line)
            if match:
                captured = match.group(0)
                # Normalize to HH:MM[:SS]
                parts = captured.split(":")
                if len(parts) >= 2:
                    hh = parts[0].zfill(2)
                    mm = parts[1].zfill(2)
                    try:
                        hh_int = int(hh)
                        mm_int = int(mm)
                        ss_int = int(parts[2]) if len(parts) == 3 else 0
                    except ValueError:
                        continue
                    if not (0 <= hh_int <= 23 and 0 <= mm_int <= 59 and 0 <= ss_int <= 59):
                        continue
                    if len(parts) == 3:
                        ss = parts[2].zfill(2)
                        time_value = f"{hh}:{mm}:{ss}"
                    else:
                        time_value = f"{hh}:{mm}"

        if date_value and time_value:
            break

    return date_value, time_value


def extract_totals(lines: Iterable[str]) -> List[float]:
    totals: List[float] = []
    for line in lines:
        if looks_like_total_line(line):
            for amount_str in extract_amount_strings(line):
                amount = parse_amount(amount_str)
                if amount is not None:
                    totals.append(amount)
    return totals


def serialise_receipt(data: ReceiptData) -> dict:
    payload = asdict(data)
    payload["items"] = [asdict(item) for item in data.items]
    return payload


def build_receipt_data(image_path: str, lang: str) -> ReceiptData:
    text = run_tesseract(image_path, lang)
    lines = split_lines(text)
    items = extract_receipt_items(lines)
    items_sum = sum(item.total_price for item in items)
    date_value, time_value = extract_date_time(lines)
    total_candidates = extract_totals(lines)
    total = total_candidates[-1] if total_candidates else None

    return ReceiptData(
        image=os.path.abspath(image_path),
        language=lang,
        raw_text=text,
        items=items,
        items_sum=items_sum,
        total=total,
        total_candidates=total_candidates,
        date=date_value,
        time=time_value,
    )


def dump_receipt_json(data: ReceiptData, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(serialise_receipt(data), handle, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to receipt image (png, jpg, ...)")
    parser.add_argument(
        "--lang",
        default="hrv",
        help="Tesseract language code (defaults to 'hrv')",
    )
    parser.add_argument(
        "--output",
        help="Destination JSON path. Defaults to <image>_parsed.json in CWD.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    image_path = args.image

    if not os.path.exists(image_path):
        print(f"Greška: ne postoji datoteka '{image_path}'.", file=sys.stderr)
        return 1

    try:
        receipt = build_receipt_data(image_path, args.lang)
    except TesseractError as exc:
        print(f"Tesseract nije uspio: {exc}", file=sys.stderr)
        return 2

    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_parsed.json"

    dump_receipt_json(receipt, output_path)

    print(f"OCR dovršen za '{image_path}'.")
    print(f"Prepoznato stavki: {len(receipt.items)}")
    print(f"Zbroj stavki: {receipt.items_sum:.2f}")
    if receipt.date:
        print(f"Datum računa: {receipt.date}")
    if receipt.time:
        print(f"Vrijeme računa: {receipt.time}")
    if receipt.total is not None:
        print(f"Ukupno (pretpostavljeno): {receipt.total:.2f}")
        if abs(receipt.items_sum - receipt.total) > TOTAL_TOLERANCE:
            diff = receipt.items_sum - receipt.total
            print(
                (
                    "UPOZORENJE: Zbroj stavki i ukupni iznos se razlikuju "
                    f"({receipt.items_sum:.2f} vs {receipt.total:.2f}, razlika {diff:+.2f})."
                ),
                file=sys.stderr,
            )
    else:
        print("Nije pronađen ukupni iznos.")
    print(f"JSON spremljen u: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
