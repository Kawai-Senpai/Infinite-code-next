#!/usr/bin/env sh
# Open the knowledge explorer for the repository in this directory.
# Falls back to `python -m icn.cli` when the console script is not on PATH.
set -e
if command -v icn-explore >/dev/null 2>&1; then
  exec icn-explore "$@"
fi
exec python3 -m icn.cli "$@"
