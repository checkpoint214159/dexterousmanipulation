#!/usr/bin/env bash
# Watch leapsim's runs/ directory and rsync any changes to a Windows-visible
# location (typically /mnt/c/Users/<you>/<somewhere>). Pair with the Windows
# script render_bridge_watch.ps1 which will pick up new run directories and
# auto-relaunch rerun.
#
# Requirements (one-time, on WSL): sudo apt install inotify-tools rsync
#
# Usage:
#   ./sync_runs_to_windows.sh [SRC] [DST]
# Defaults:
#   SRC = $REPO_ROOT/src/LEAP_Hand_Sim/leapsim/runs
#   DST = /mnt/c/Users/$USER/leapsim_runs

set -euo pipefail

# Resolve repo root relative to this script's location, so the defaults work
# regardless of where you run the script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC="${1:-$REPO_ROOT/src/LEAP_Hand_Sim/leapsim/runs}"
DST="${2:-/mnt/e/odyssey/runs}"

if ! command -v inotifywait >/dev/null 2>&1; then
  echo "error: inotifywait not found. Install with:  sudo apt install inotify-tools" >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync not found. Install with:  sudo apt install rsync" >&2
  exit 1
fi
if [[ ! -d "$SRC" ]]; then
  echo "error: source dir does not exist: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"

# Debounce window in seconds. After any file event we wait this long, then
# drain. Keeps us from rsync'ing once per .rrd flush during a busy run.
DEBOUNCE=2

sync_now() {
  # -a archive (perms, times, recursion). Trailing slash on SRC copies contents,
  # not the directory itself. No --delete: keep old runs on Windows.
  rsync -a "$SRC/" "$DST/"
}

echo "[sync] source: $SRC"
echo "[sync] dest:   $DST"
echo "[sync] initial sync..."
sync_now
echo "[sync] watching for changes (Ctrl+C to stop)"

# Persistent inotifywait piped into a read loop. After the first event we drain
# any follow-ups using `read -t` so a burst of file flushes triggers ONE rsync.
inotifywait -m -r -q \
  -e close_write -e create -e moved_to -e delete \
  "$SRC" --format '%w%f' |
while read -r _first; do
  while read -r -t "$DEBOUNCE" _drain; do :; done
  sync_now
  printf '[sync] %s  updated\n' "$(date '+%H:%M:%S')"
done
