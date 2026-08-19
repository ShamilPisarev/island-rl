#!/usr/bin/env bash
# Copy the checkpoints git cannot carry.
#
# checkpoints/ and runs/ are gitignored (615MB of intermediates), but the
# milestone chain is load-bearing -- a scarce world is unlearnable from scratch,
# so losing these means re-running ~15 trainings to rebuild the chain. Only the
# latest.pt of each run is needed to continue or evaluate: 53MB in total.
#
#   ./backup_checkpoints.sh /Volumes/stick/rl_game_backup
set -euo pipefail
DEST="${1:?usage: $0 <destination-dir>}"
mkdir -p "$DEST/checkpoints"
for f in checkpoints/*/latest.pt; do
    run=$(basename "$(dirname "$f")")
    mkdir -p "$DEST/checkpoints/$run"
    cp "$f" "$DEST/checkpoints/$run/latest.pt"
done
# metrics CSVs are small and are the only record of the training curves
cp -R runs "$DEST/runs"
echo "copied $(ls checkpoints/*/latest.pt | wc -l | tr -d ' ') checkpoints + runs/ to $DEST"
du -sh "$DEST"
