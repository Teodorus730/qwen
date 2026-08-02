#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/vast_5090_32gb.yaml}"

echo "[setup] root=$ROOT"
echo "[setup] config=$CONFIG"

REQUIRED_FILES=(
  "src/__init__.py"
  "src/config.py"
  "src/data.py"
  "src/runtime.py"
  "configs/vast_5090_32gb.yaml"
)
MISSING_FILES=()
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    MISSING_FILES+=("$required_file")
  fi
done
if [[ "${#MISSING_FILES[@]}" -ne 0 ]]; then
  echo "[setup] ERROR: upload bundle is incomplete." >&2
  echo "[setup] Missing files:" >&2
  printf '  - %s\n' "${MISSING_FILES[@]}" >&2
  echo "[setup] Re-upload the current ZIP and extract it with directories:" >&2
  echo "  python3 -m zipfile -e /workspace/rtx_5090_32gb_upload.zip ." >&2
  exit 5
fi

python3 - <<'PY'
import sys
try:
    import torch
except ImportError as error:
    raise SystemExit(
        "The selected Vast template has no PyTorch. Destroy this instance and "
        "rent a Recommended PyTorch template; do not install a random CPU torch."
    ) from error
print("base python:", sys.version)
print("base torch:", torch.__version__)
print("base CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA in the selected Vast template.")
PY

if [[ ! -x .venv/bin/python ]]; then
  echo "[setup] creating .venv with access to template-provided CUDA PyTorch"
  if ! python3 -m venv --system-site-packages .venv; then
    echo "[setup] python3-venv unavailable; installing virtualenv"
    python3 -m pip install --upgrade virtualenv
    python3 -m virtualenv --system-site-packages .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p artifacts/data outputs results logs exports
python preflight.py --config "$CONFIG"

echo "[setup] complete"
echo "[setup] next: ./run_experiment.sh $CONFIG"
