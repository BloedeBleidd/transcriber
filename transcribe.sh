#!/usr/bin/env bash
# Thin wrapper: checks for python3 and delegates all logic to app/runner.py

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required but was not found in PATH." >&2
  exit 1
fi

exec python3 "${SCRIPT_DIR}/app/runner.py" "$@"
