# easy-value-reader

Simple OCR script that extracts `Act` and `Set` temperatures from a control-panel photo.

## Quick Linux setup script

```bash
bash scripts/install_linux.sh
```

## Requirements

- Python 3.10+
- Tesseract OCR installed in OS

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3-venv tesseract-ocr tesseract-ocr-eng
```

Optional language packs:

```bash
sudo apt install -y tesseract-ocr-rus
```

### Windows

Install Tesseract OCR (UB Mannheim build is fine), then reopen terminal.

## Setup

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

Default image:

```bash
python main.py
```

Custom image:

```bash
python main.py --image path/to/image.png
```

Tune fallback thresholds if needed:

```bash
python main.py --act-conf-threshold 35 --set-conf-threshold 65
```

## Notes

- If `tesseract` is not in `PATH`, set `TESSERACT_CMD` to full binary path.
- On Linux it is usually `/usr/bin/tesseract`.
