#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/vast_5090_32gb.yaml}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: ./setup.sh $CONFIG" >&2
  exit 2
fi

# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs results outputs

# Prevent an accidental second training process from sharing the same GPU.
exec 9>".run_experiment.lock"
if ! flock -n 9; then
  echo "Another run_experiment.sh process owns .run_experiment.lock" >&2
  exit 3
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/run_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "[run] UTC start: $(date -u --iso-8601=seconds)"
echo "[run] config: $CONFIG"
echo "[run] log: $LOG"

python preflight.py --config "$CONFIG"
python prepare_data.py --config "$CONFIG"

if [[ "${SKIP_BENCHMARK:-0}" != "1" ]]; then
  python benchmark.py --config "$CONFIG"
else
  echo "[run] benchmark skipped because SKIP_BENCHMARK=1"
fi

TRAIN_ARGS=(--config "$CONFIG" --resume-from auto)
if [[ "${FRESH:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--fresh)
fi
python train.py "${TRAIN_ARGS[@]}"

echo "[run] UTC finish: $(date -u --iso-8601=seconds)"
echo "[run] download outputs/, results/ and logs/ before destroying the instance"
