#!/usr/bin/env bash
# Open the island in a browser. One command, no arguments.
#
# This exists because the viewer is a STATIC page and opening it by
# double-clicking gives you a `file://` URL, where `fetch` is blocked outright --
# so the replay list cannot load and the page looks broken rather than
# unserved. It has to be served over http, and remembering
# `python -m http.server --directory viewer` is not a thing anybody should have
# to remember.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8777}"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Refuse to start on an empty replay directory rather than opening a page with
# nothing in it -- "no replays" and "the server is broken" look identical.
if ! ls viewer/replays/*.json >/dev/null 2>&1; then
  echo "No replays in viewer/replays/. Record one first, e.g.:"
  echo "  $PY -m sim.society --config config/island4/empire.yaml --ticks 4000 --replay"
  exit 1
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Something is already serving port $PORT — reusing it."
else
  "$PY" -m http.server "$PORT" --directory viewer >/dev/null 2>&1 &
  echo "serving viewer/ on port $PORT (pid $!)"
  sleep 1
fi

URL="http://localhost:$PORT/index.html"
echo
echo "  $URL"
echo
echo "The panel opens on the newest world with a START HERE row at the top;"
echo "everything else is under 'or pick any of the N recorded runs'."
command -v open >/dev/null 2>&1 && open "$URL" || true
