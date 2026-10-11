#!/usr/bin/env bash
# Start the ReturnGuard FastAPI backend.
#   bash run_api.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
