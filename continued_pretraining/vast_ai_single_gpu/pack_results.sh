#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-metrics}"
mkdir -p exports
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "$MODE" in
  metrics)
    ARCHIVE="exports/qwen_vast_metrics_${STAMP}.tar.gz"
    mapfile -t FILES < <(
      find results logs configs outputs \
        -type f \
        \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' \
           -o -name '*.png' -o -name '*.yaml' -o -name '*.txt' \) \
        -print 2>/dev/null
    )
    if [[ "${#FILES[@]}" -eq 0 ]]; then
      echo "No metrics/config/log files found." >&2
      exit 4
    fi
    tar -czf "$ARCHIVE" "${FILES[@]}"
    ;;
  full)
    ARCHIVE="exports/qwen_vast_full_${STAMP}.tar.gz"
    tar -czf "$ARCHIVE" results logs configs outputs
    ;;
  *)
    echo "Usage: ./pack_results.sh [metrics|full]" >&2
    exit 2
    ;;
esac

du -h "$ARCHIVE"
echo "$ARCHIVE"

