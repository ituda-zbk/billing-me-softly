# Instalacija i pokretanje

## Preduvjeti

- Linux (trenutno jedina podrzana platforma, WSL na Windowsima takoder radi)
- Python 3.10+
- Google Gemini API kljuc ([generativelanguage.googleapis.com](https://ai.google.dev/))

## Instalacija ovisnosti

```bash
pip install flask werkzeug pillow requests
```

## Konfiguracija

### Obavezne varijable okruzenja

| Varijabla | Opis |
|-----------|------|
| `GEMINI_API_KEY` | API kljuc za Google Gemini Vision |

### Opcionalne varijable okruzenja

| Varijabla | Zadana vrijednost | Opis |
|-----------|-------------------|------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model za OCR parsiranje |
| `GEMINI_MAX_PARALLEL` | `2` | Broj paralelnih poziva prema Gemini API-ju |
| `GEMINI_REQUESTS_PER_MINUTE` | `5` | Maksimalan broj zahtjeva po minuti |
| `GEMINI_MAX_429_RETRIES` | `6` | Broj ponovnih pokusaja kod rate limita (429) |
| `ONEDRIVE_IMPORT_DIR` | *(prazno)* | Putanja do OneDrive foldera za grupni uvoz racuna |

### Konfiguracijski parametri (`config.py`)

| Parametar | Zadana vrijednost | Opis |
|-----------|-------------------|------|
| `MAX_NEW_TOKENS` | `8000` | Maksimalan broj tokena koje Gemini generira |
| `TOTAL_TOLERANCE` | `0.05` | Dopustena razlika (EUR) izmedu zbroja stavki i ukupnog iznosa |
| `UPLOAD_DIR` | `uploads` | Direktorij za uploadane slike i JSON datoteke |
| `MAX_UPLOAD_FILES` | `20` | Maksimalan broj slika po jednom uploadu |
| `DEFAULT_RESIZE_MAX` | `(800, 1800)` | Maksimalna sirina i visina slike za pohranu |
| `ENABLE_IMAGE_NORMALIZATION` | `True` | Automatska korekcija EXIF orijentacije slike |
| `DEFAULT_LANG` | `hrv` | Zadani jezik za OCR |

## Pokretanje

### Web server (primarni nacin koristenja)

```bash
export GEMINI_API_KEY="vas_api_kljuc"
python3 receipt_ocr.py --serve --host 0.0.0.0 --port 5000
```

Aplikacija je dostupna na `http://localhost:5000`.

### Obrada pojedinacne slike (CLI)

```bash
export GEMINI_API_KEY="vas_api_kljuc"
python3 receipt_ocr.py /putanja/do/slike.jpg
```

### Sve CLI opcije

```bash
python3 receipt_ocr.py --help
```

| Opcija | Opis |
|--------|------|
| `image_path` | Putanja do slike racuna (za CLI obradu) |
| `--lang` | Jezik racuna (zadano: `hrv`) |
| `--output` | Putanja za izlazni JSON |
| `--db-path` | Putanja do SQLite baze (zadano: `receipts.db`) |
| `--serve` | Pokreni web server |
| `--host` | Adresa za web server (zadano: `0.0.0.0`) |
| `--port` | Port za web server (zadano: `5000`) |

### Pokretanje u pozadini (tmux)

```bash
tmux new-session -d -s services "python3 receipt_ocr.py --serve"
```

## Baza podataka

SQLite baza (`receipts.db`) se automatski kreira pri prvom pokretanju. Nije potrebna nikakva rucna inicijalizacija.

## Podrzani formati slika

PNG, JPG, JPEG, WebP, HEIC, TIF, TIFF — automatski se konvertiraju u format kompatibilan s Gemini API-jem.
