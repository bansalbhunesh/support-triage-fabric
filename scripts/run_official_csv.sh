#!/usr/bin/env bash
# Official batch with 8-column CSV (issue, subject, company + five graded fields).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SUPPORT_AGENT_CSV_EVAL_MINIMAL=1
IN="${1:-support_tickets/support_tickets.csv}"
OUT="${2:-support_tickets/output.csv}"
exec python code/main.py --quiet --csv "$IN" "$OUT"
