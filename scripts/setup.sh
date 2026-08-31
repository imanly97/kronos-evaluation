#!/usr/bin/env bash
# One-shot environment setup. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

echo "1/4  virtualenv (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip

echo "2/4  dependencies"
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "3/4  vendored Kronos source"
if [ ! -d vendor_kronos ]; then
  git clone --depth 1 https://github.com/shiyu-coder/Kronos.git vendor_kronos
fi

echo "4/4  Kronos weights (cached under hf_cache/)"
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "vendor_kronos")
import os; os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
from model.kronos import Kronos, KronosTokenizer
KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
Kronos.from_pretrained("NeoQuasar/Kronos-small")
print("   weights ready")
PY

echo "done. try:  .venv/bin/python -m pytest -q"
