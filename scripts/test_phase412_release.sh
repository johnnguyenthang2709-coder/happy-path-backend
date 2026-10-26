#!/bin/sh
# Supported full checkout suite, isolated from the production SQLite database.
set -eu
release_python=${PYTHON_BIN:-python}
release_db=$(mktemp /tmp/happy-path-phase412-XXXXXX.db)
trap 'rm -f "$release_db" "$release_db-journal" "$release_db-wal" "$release_db-shm"' EXIT
export HAPPY_PATH_DB_PATH="$release_db" GOONG_API_KEY= GOONG_REST_KEY=
"$release_python" scripts/seed_db.py --db-path "$release_db"
"$release_python" -m pytest tests -q "$@"
