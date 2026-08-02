#!/usr/bin/env bash
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== GPU ==="
nvidia-smi || true
echo
echo "=== DISK ==="
df -h "$ROOT" || true
echo
echo "=== TRAINING PROCESS ==="
pgrep -af "python.*(train|benchmark|prepare_data|preflight)\\.py" || true
echo
echo "=== LATEST LOG ==="
LATEST_LOG="$(find logs -maxdepth 1 -type f -name 'run_*.log' 2>/dev/null | sort | tail -n 1)"
if [[ -n "$LATEST_LOG" ]]; then
  echo "$LATEST_LOG"
  tail -n 30 "$LATEST_LOG"
else
  echo "No run log yet."
fi
echo
echo "=== SUMMARIES ==="
find outputs -maxdepth 2 -name summary.json -print -exec cat {} \; 2>/dev/null || true

