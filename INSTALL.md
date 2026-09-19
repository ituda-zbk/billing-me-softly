# Instalacija i pokretanje

## Preduvjeti

- Linux, macOS ili Windows (OS-agnostic — iste naredbe posvuda)
- Python 3.10+
- Google Gemini API kljuc ([generativelanguage.googleapis.com](https://ai.google.dev/))

## Postavljanje okruzenja (jednom, po racunalu)

```bash
python bootstrap.py
```

Ovo:
- kreira data direktorij u Users home folderu (`~/BillingMeSoftly/` — na Windowsu `C:\Users\<ime>\BillingMeSoftly\`, na Linuxu/macOS `~/BillingMeSoftly/`) (`uploads/` poddirektorij aplikacija kreira sama po potrebi)
- instalira ovisnosti iz `requirements.txt`
- ako `GEMINI_API_KEY` nije vec postavljen (ni kao env varijabla ni u `.env`), pita za njega i sprema ga u `~/BillingMeSoftly/.env`

Direktorij se moze promijeniti postavljanjem `BILLING_DATA_DIR` prije pokretanja `bootstrap.py`/`receipt_ocr.py`.

### Rucna instalacija ovisnosti (alternativa)

```bash
pip install -r requirements.txt
```

## Konfiguracija

### Obavezne varijable okruzenja

| Varijabla | Opis |
|-----------|------|
| `GEMINI_API_KEY` | API kljuc za Google Gemini Vision |

Moze se postaviti kao prava environment varijabla (nacin se razlikuje po OS-u: `export` u `~/.bashrc` na Linuxu/macOS, `setx` / System Properties na Windowsu), ili jednostavnije — jednom, na isti nacin na oba OS-a — upisati u `~/BillingMeSoftly/.env` (to radi `bootstrap.py` automatski). Prava environment varijabla uvijek ima prednost pred `.env` datotekom.

### Opcionalne varijable okruzenja

| Varijabla | Zadana vrijednost | Opis |
|-----------|-------------------|------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model za OCR parsiranje |
| `GEMINI_MAX_PARALLEL` | `1` | Broj paralelnih poziva prema Gemini API-ju (defaultno sekvencijalna obrada) |
| `GEMINI_REQUESTS_PER_MINUTE` | `2` | Maksimalan broj zahtjeva po minuti (sliding window, 60 s) |
| `GEMINI_MAX_429_RETRIES` | `4` | Broj ponovnih pokusaja kod rate limita (429) |
| `ONEDRIVE_IMPORT_DIR` | *(prazno)* | Putanja do OneDrive foldera za grupni uvoz racuna (npr. `C:\Users\<ime>\OneDrive\...` na Windowsu, ili put do lokalno sinkroniziranog foldera na Linuxu/macOS) |
| `BILLING_DATA_DIR` | `~/BillingMeSoftly` | Gdje se sprema `receipts.db` i `uploads/`; promijeni samo ako zadana lokacija u home folderu ne odgovara |

### Konfiguracijski parametri (`config.py`)

| Parametar | Zadana vrijednost | Opis |
|-----------|-------------------|------|
| `MAX_NEW_TOKENS` | `8000` | Maksimalan broj tokena koje Gemini generira |
| `TOTAL_TOLERANCE` | `0.05` | Dopustena razlika (EUR) izmedu zbroja stavki i ukupnog iznosa |
| `MAX_UPLOAD_FILES` | `20` | Maksimalan broj slika po jednom uploadu |
| `DEFAULT_RESIZE_MAX` | `(800, 1800)` | Maksimalna sirina i visina slike za pohranu |
| `ENABLE_IMAGE_NORMALIZATION` | `True` | Automatska korekcija EXIF orijentacije slike |
| `DEFAULT_LANG` | `hrv` | Zadani jezik za OCR |

## Pokretanje

### Web server (primarni nacin koristenja)

```bash
python receipt_ocr.py --serve --host 0.0.0.0 --port 5000
```

(Na Windowsu koristi `python`, na Linuxu/macOS obicno `python3` — ovisno o instalaciji.)

Aplikacija je dostupna na `http://localhost:5000`. Kod uploada vise slika odjednom (Browse ili OneDrive uvoz), obrada se odvija u pozadini — preglednik odmah preusmjeri na stranicu uzivo statusa (`/batch/<id>`) koja prikazuje napredak po svakoj slici (na cekanju / obraduje se / gotovo / greska) i sama nastavlja na formu za pregled cim je batch gotov.

### Obrada pojedinacne slike (CLI)

```bash
python receipt_ocr.py /putanja/do/slike.jpg
```

### Sve CLI opcije

```bash
python receipt_ocr.py --help
```

| Opcija | Opis |
|--------|------|
| `image_path` | Putanja do slike racuna (za CLI obradu) |
| `--lang` | Jezik racuna (zadano: `hrv`) |
| `--db-path` | Putanja do SQLite baze (zadano: `~/BillingMeSoftly/receipts.db`) |
| `--serve` | Pokreni web server |
| `--host` | Adresa za web server (zadano: `0.0.0.0`) |
| `--port` | Port za web server (zadano: `5000`) |

### Pokretanje u pozadini

Linux/macOS (tmux):

```bash
tmux new-session -d -s services "python3 receipt_ocr.py --serve"
```

Windows (PowerShell, u pozadini):

```powershell
Start-Process python -ArgumentList "receipt_ocr.py --serve" -WindowStyle Hidden
```

## Baza podataka i podaci

SQLite baza (`receipts.db`) i `uploads/` se automatski kreiraju pri prvom pokretanju, u `~/BillingMeSoftly/` (odn. `BILLING_DATA_DIR` ako je postavljen). Nije potrebna nikakva rucna inicijalizacija.

### Prijenos podataka na drugo racunalo / drugi OS

Za premjestanje (npr. Linux -> Windows) kopiraj `receipts.db` i cijeli `uploads/` folder zajedno u data direktorij novog racunala (`~/BillingMeSoftly/`). Baza pamti apsolutne putanje slika sa starog racunala, ali aplikacija pri svakom pokretanju automatski popravi putanje: za svaki racun ciju sliku ne nalazi na zapisanoj putanji potrazi datoteku istog imena u `uploads/` i prepise putanju. Nije potrebna nikakva rucna intervencija — samo se pobrini da se kopiraju i baza i sve slike.

### Neispravne slike

Ako neka uploadana slika stigne prazna ili ostecena (npr. 0 bajtova), aplikacija ju preskoci uz jasnu poruku za tu jednu sliku — ostale slike iz istog uploada se normalno obrade.

### Nadogradnja sa starije verzije

Ako su na ovom racunalu vec postojali `receipts.db`/`uploads/` u korijenu repozitorija (stariji nacin rada, prije ovog OS-agnostic azuriranja), aplikacija ih pri prvom sljedecem pokretanju automatski premjesti u `~/BillingMeSoftly/` — jednokratno, bez gubitka podataka.

## Podrzani formati slika

PNG, JPG, JPEG, WebP, HEIC, TIF, TIFF — automatski se konvertiraju u format kompatibilan s Gemini API-jem. HEIC podrska dolazi iz `pillow-heif` (instaliran preko `requirements.txt`).
