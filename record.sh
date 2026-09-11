#!/usr/bin/env bash
# Record the worlds the viewer's START HERE row points at, from scratch.
#
# Nothing here trains anything -- every one is the scripted utility arbiter, so
# the whole set takes a few minutes rather than a few hours. That is the working
# rule of Island 4.0: mechanics first, scripted and measured; training last.
#
# Usage:  ./record.sh          record any that are missing
#         ./record.sh --all    re-record everything, overwriting
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
FORCE="${1:-}"

# name | config | ticks
WORLDS=(
  "empire|config/island4/empire.yaml|4000"
  "village_frontier|config/island3/village_frontier.yaml|1500"
  "island3_generations|config/island3/village_gen.yaml|6000"
  "island3_village|config/island3/village.yaml|3000"
)

for w in "${WORLDS[@]}"; do
  IFS='|' read -r name cfg ticks <<<"$w"
  out="viewer/replays/${name}.json"
  if [ -f "$out" ] && [ "$FORCE" != "--all" ]; then
    echo "have  $name  (./record.sh --all to re-record)"
    continue
  fi
  echo
  echo "=== recording $name  ($cfg, $ticks ticks) ==="
  # --replay-path so the file matches the name the viewer's START HERE row
  # looks for; the config stem is not always what we want it called.
  "$PY" -m sim.society --config "$cfg" --ticks "$ticks" --replay \
        --replay-path "$out" | tail -3
done

echo
echo "Done. Open them with:  ./watch.sh"
