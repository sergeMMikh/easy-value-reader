#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip tesseract-ocr tesseract-ocr-eng
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y python3 python3-virtualenv python3-pip tesseract tesseract-langpack-eng
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y python3 python3-virtualenv python3-pip tesseract
elif command -v pacman >/dev/null 2>&1; then
  sudo pacman -Sy --noconfirm python python-pip tesseract tesseract-data-eng
else
  echo "Unsupported Linux package manager. Install python3, pip, and tesseract manually."
  exit 1
fi

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete. Run: source venv/bin/activate && python main.py"
