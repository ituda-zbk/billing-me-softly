#!/usr/bin/env python3
"""
Jednokratna priprema okruženja (Linux/macOS/Windows - identičan postupak):
- kreira data direktorij u Users home folderu (~/BillingMeSoftly/, ili
  BILLING_DATA_DIR ako je postavljen) — uploads/ poddirektorij kreira
  receipt_ocr.py sam po potrebi (i pri migraciji postojećih podataka)
- instalira ovisnosti iz requirements.txt
- ako GEMINI_API_KEY nije postavljen ni u okruženju ni u .env, pita za njega
  i sprema ga u ~/BillingMeSoftly/.env

Pokretanje: python bootstrap.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def resolve_data_dir() -> Path:
    override = os.environ.get("BILLING_DATA_DIR")
    data_dir = Path(override) if override else Path.home() / "BillingMeSoftly"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def install_requirements() -> None:
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


def ensure_api_key(data_dir: Path) -> None:
    env_path = data_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    if os.environ.get("GEMINI_API_KEY") or "GEMINI_API_KEY=" in existing:
        print(f"GEMINI_API_KEY je već postavljen (okruženje ili {env_path}).")
        return

    api_key = input("Unesi GEMINI_API_KEY (Google Gemini API ključ): ").strip()
    if not api_key:
        print(f"Preskočeno — prije pokretanja servera postavi GEMINI_API_KEY ručno (npr. u {env_path}).")
        return

    with env_path.open("a", encoding="utf-8") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
    print(f"Spremljeno u {env_path}.")


def main() -> int:
    system_name = platform.system()  # 'Linux', 'Windows', 'Darwin'
    print(f"Otkriven OS: {system_name}")

    data_dir = resolve_data_dir()
    print(f"Data direktorij: {data_dir}")

    print("Instaliram ovisnosti iz requirements.txt ...")
    install_requirements()

    ensure_api_key(data_dir)

    launcher = "python" if system_name == "Windows" else "python3"
    print("\nGotovo. Pokreni server s:")
    print(f"  {launcher} receipt_ocr.py --serve --host 0.0.0.0 --port 5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
