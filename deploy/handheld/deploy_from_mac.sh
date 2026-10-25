#!/usr/bin/env bash
# Sync this checkout to the Pi and restart the service. Run from anywhere on the Mac.
#
#   deploy/handheld/deploy_from_mac.sh [--no-restart] [--logs] [--install] [--dry-run]
#                                      [--host user@host] [--dest /abs/path]
#
#   --no-restart   rsync only (e.g. before install_pi.sh on a fresh Pi)
#   --install      run install_pi.sh on the Pi after the sync (then restart unless --no-restart)
#   --logs         follow journalctl --user -u loci-handheld after restarting
#   --dry-run      show what rsync would transfer
#   --host         default ziggy@loci-pi.local   (env PI_HOST)
#   --dest         default /home/ziggy/loci-handheld/src   (env PI_DIR)
#
# Excludes: .git venv .venv __pycache__ *.pyc demo_spatial/.certs .paperclip loci-core/target
# deploy/handheld/models/* (weights live in /home/ziggy/loci-handheld/models) and tool caches.
# ~/loci-db on the Pi is never touched. The restart prefers the systemd user unit and falls back
# to a nohup run_pi.sh when no unit is installed.
set -euo pipefail

PI_HOST="${PI_HOST:-ziggy@loci-pi.local}"
PI_DIR="${PI_DIR:-/home/ziggy/loci-handheld/src}"
RESTART=1; LOGS=0; INSTALL=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-restart) RESTART=0 ;;
    --logs) LOGS=1 ;;
    --install) INSTALL=1 ;;
    --dry-run) DRY=1 ;;
    --host) shift; PI_HOST="$1" ;;
    --dest) shift; PI_DIR="$1" ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RSYNC_OPTS=(-az --delete --delete-excluded --human-readable --info=stats1)
[ "$DRY" = "1" ] && RSYNC_OPTS+=(--dry-run --itemize-changes)
EXCLUDES=(
  --exclude .git
  --exclude venv
  --exclude .venv
  --exclude __pycache__
  --exclude '*.pyc'
  --exclude demo_spatial/.certs
  --exclude .paperclip
  --exclude loci-core/target
  --include deploy/handheld/models/README.md
  --include deploy/handheld/models/.gitignore
  --exclude 'deploy/handheld/models/*'
  --exclude .pytest_cache
  --exclude .hypothesis
  --exclude .ruff_cache
  --exclude .mypy_cache
  --exclude '*.egg-info'
  --exclude node_modules
  --exclude .DS_Store
)

echo "==> rsync $REPO_ROOT/ -> $PI_HOST:$PI_DIR/"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$PI_HOST" "mkdir -p '$PI_DIR'"
rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" "$REPO_ROOT/" "$PI_HOST:$PI_DIR/"
[ "$DRY" = "1" ] && exit 0

if [ "$INSTALL" = "1" ]; then
  echo "==> install_pi.sh on $PI_HOST"
  ssh -t "$PI_HOST" "bash '$PI_DIR/deploy/handheld/install_pi.sh'"
fi

if [ "$RESTART" = "1" ]; then
  echo "==> restarting loci-handheld on $PI_HOST"
  # systemctl --user needs the user bus; export XDG_RUNTIME_DIR for non-login ssh sessions.
  ssh "$PI_HOST" 'export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if systemctl --user restart loci-handheld 2>/dev/null; then
  sleep 1; systemctl --user --no-pager --lines=0 status loci-handheld | head -5
else
  echo "no user unit; falling back to nohup run_pi.sh"
  pkill -f "uvicorn demo_spatial.app.main" 2>/dev/null || true
  mkdir -p /home/ziggy/loci-handheld/data
  nohup '"$PI_DIR"'/deploy/handheld/run_pi.sh >/home/ziggy/loci-handheld/data/server.log 2>&1 &
  sleep 1; echo "started (log: /home/ziggy/loci-handheld/data/server.log)"
fi'
else
  echo "==> not restarted (--no-restart)"
fi

if [ "$LOGS" = "1" ]; then
  exec ssh -t "$PI_HOST" 'export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"; journalctl --user -u loci-handheld -f -n 50'
fi
