#!/usr/bin/env bash
# Install or upgrade the LOCI handheld demo on the Raspberry Pi 5. Run as ziggy, no sudo.
# Idempotent: every step checks before it acts, so re-running after a deploy is the normal path.
#
#   bash /home/ziggy/loci-handheld/src/deploy/handheld/install_pi.sh [options]
#
#   --skip-pip      do not touch the venv (fast re-run for config/service changes)
#   --no-models     skip fetch_models.py / Piper copy
#   --no-service    do not install the systemd user unit
#   --start         (re)start the user service at the end
#   --help          this text
#
# What it does: prints the sudo steps it cannot run itself (usb_max_current_enable, libportaudio2,
# udev rule, linger), creates or upgrades /home/ziggy/loci-handheld/.venv to requirements-pi.txt
# (depthai 3.5.0 -> 3.11.0), installs the repo editable, stages models, writes env + rooms.json
# from the examples when absent, creates data/, runs an import self-test and installs the user unit.
set -euo pipefail

SKIP_PIP=0; NO_MODELS=0; NO_SERVICE=0; START=0
for arg in "$@"; do
  case "$arg" in
    --skip-pip) SKIP_PIP=1 ;;
    --no-models) NO_MODELS=1 ;;
    --no-service) NO_SERVICE=1 ;;
    --start) START=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCI_HOME="${LOCI_HOME:-$HOME/loci-handheld}"
VENV="$LOCI_HOME/.venv"
PY="$VENV/bin/python"
MODELS="$LOCI_HOME/models"
WARNINGS=()

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    ok   %s\n' "$*"; }
warn() { printf '    WARN %s\n' "$*"; WARNINGS+=("$*"); }

if [ "$(id -u)" = "0" ]; then
  echo "Run this as the normal user (ziggy), not root." >&2
  exit 2
fi

say "Layout"
echo "    src:    $SRC_DIR"
echo "    home:   $LOCI_HOME"
echo "    venv:   $VENV"
if [ "$SRC_DIR" != "$LOCI_HOME/src" ]; then
  warn "source tree is not $LOCI_HOME/src; the systemd units expect that path (deploy_from_mac.sh puts it there)"
fi
mkdir -p "$LOCI_HOME" "$LOCI_HOME/data" "$MODELS/zoo" "$MODELS/embeddings" "$MODELS/piper"

# ------------------------------------------------------------------ sudo steps (printed, never run)
say "Steps that need sudo (run them yourself; this script never calls sudo)"
NEED_SUDO=0
if command -v vcgencmd >/dev/null 2>&1; then
  CUR="$(vcgencmd get_config usb_max_current_enable 2>/dev/null || true)"
  if [ "$CUR" = "usb_max_current_enable=1" ]; then
    ok "usb_max_current_enable=1 (camera may draw more than 600 mA)"
  else
    NEED_SUDO=1
    echo "    MANDATORY (once, then reboot) -- the OAK-D Lite needs more than the 600 mA default:"
    echo "      sudo sh -c 'echo usb_max_current_enable=1 >> /boot/firmware/config.txt' && sudo reboot"
  fi
else
  warn "vcgencmd not found (not a Raspberry Pi?) -- skipping the USB current check"
fi
if [ -f /etc/udev/rules.d/80-movidius.rules ]; then
  ok "udev rule /etc/udev/rules.d/80-movidius.rules present"
else
  NEED_SUDO=1
  echo "    RECOMMENDED -- let non-root open the camera (03e7 Movidius):"
  echo "      echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/80-movidius.rules"
  echo "      sudo udevadm control --reload-rules && sudo udevadm trigger"
fi
if "$(command -v ldconfig || echo /sbin/ldconfig)" -p 2>/dev/null | grep -q libportaudio.so.2; then
  ok "libportaudio2 present (sounddevice backend)"
else
  NEED_SUDO=1
  echo "    FOR THE HEADSET VOICE LOOP -- PortAudio for python-sounddevice:"
  echo "      sudo apt install -y libportaudio2"
  echo "      (or set LOCI_AUDIO_BACKEND=cli in env to use pw-record/pw-play instead)"
fi
if command -v loginctl >/dev/null 2>&1 && loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes'; then
  ok "linger enabled for $USER (user service starts at boot)"
else
  NEED_SUDO=1
  echo "    FOR AUTOSTART AT BOOT -- keep the user service manager alive without a login:"
  echo "      sudo loginctl enable-linger $USER"
fi
echo "    OPTIONAL -- system unit instead of the user unit (no headset audio from a system service):"
echo "      sudo cp $SRC_DIR/deploy/handheld/loci-handheld.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now loci-handheld"
echo "    OPTIONAL -- Wi-Fi hotspot when there is no LAN (phone opens http://10.42.0.1:8001/phone):"
echo "      sudo nmcli device wifi hotspot ifname wlan0 ssid loci-pi password <password>"
[ "$NEED_SUDO" = "1" ] && warn "one or more sudo steps above are still pending"

# ------------------------------------------------------------------ venv
say "Python venv"
if [ ! -x "$PY" ]; then
  echo "    creating $VENV"
  python3 -m venv "$VENV"
fi
"$PY" --version
if [ "$SKIP_PIP" = "0" ]; then
  REQ="$SCRIPT_DIR/requirements-pi.txt"
  "$PY" -m pip install --quiet --upgrade pip wheel
  echo "    installing $REQ (this upgrades depthai to 3.11.0 in place)"
  if ! "$PY" -m pip install --upgrade -r "$REQ"; then
    warn "full requirements install failed; retrying without the voice section"
    CORE_REQ="$LOCI_HOME/data/requirements-core.txt"
    sed '/^# --- voice/,/^# --- end voice/d' "$REQ" > "$CORE_REQ"
    "$PY" -m pip install --upgrade -r "$CORE_REQ"
    warn "voice packages (sounddevice, faster-whisper, piper-tts) are NOT installed; typed queries still work"
  fi
  echo "    installing the repo editable (loci + demo_spatial)"
  "$PY" -m pip install --quiet -e "$SRC_DIR" --no-deps
else
  ok "pip skipped (--skip-pip)"
fi
"$PY" - <<'PYEOF'
import importlib
for name in ("depthai", "cv2", "numpy", "pyapriltags", "tokenizers", "safetensors", "fastapi", "uvicorn"):
    try:
        m = importlib.import_module(name)
        print(f"    {name:<12} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"    {name:<12} MISSING ({e})")
PYEOF

# ------------------------------------------------------------------ models
say "Models under $MODELS"
if [ "$NO_MODELS" = "0" ]; then
  OLD_PIPER="$HOME/loci-db/deploy/oak-rpi5/models"
  if [ ! -f "$MODELS/piper/en_US-lessac-medium.onnx" ] && [ -f "$OLD_PIPER/en_US-lessac-medium.onnx" ]; then
    cp "$OLD_PIPER/en_US-lessac-medium.onnx" "$MODELS/piper/"
    [ -f "$OLD_PIPER/en_US-lessac-medium.onnx.json" ] && cp "$OLD_PIPER/en_US-lessac-medium.onnx.json" "$MODELS/piper/"
    ok "copied the Piper voice from $OLD_PIPER"
  fi
  WHISPER_ARG=()
  if "$PY" -c "import faster_whisper" >/dev/null 2>&1; then
    WHISPER_ARG=(--whisper "${WHISPER_LOCAL_MODEL:-tiny.en}")
  fi
  if "$PY" "$SCRIPT_DIR/scripts/fetch_models.py" --models-dir "$MODELS" --layout pi --only-missing ${WHISPER_ARG[@]+"${WHISPER_ARG[@]}"}; then
    ok "models complete"
  else
    warn "fetch_models.py reported missing models (no network?) -- see deploy/handheld/models/README.md for the manual copy"
  fi
else
  ok "models skipped (--no-models)"
fi

# ------------------------------------------------------------------ config files
say "Configuration"
if [ -f "$LOCI_HOME/env" ]; then
  ok "$LOCI_HOME/env exists (not overwritten; diff against env.example after upgrades)"
  MISSING_KEYS="$(grep -oE '^#?[A-Z][A-Z0-9_]*=' "$SCRIPT_DIR/env.example" | sed 's/^#//; s/=$//' | sort -u | while read -r k; do grep -qE "^#?${k}=" "$LOCI_HOME/env" || echo "$k"; done | tr '\n' ' ')"
  [ -n "$MISSING_KEYS" ] && warn "keys in env.example but not in $LOCI_HOME/env: $MISSING_KEYS"
else
  cp "$SCRIPT_DIR/env.example" "$LOCI_HOME/env"
  ok "wrote $LOCI_HOME/env from env.example (edit LOCI_VOICE / LOCI_PTT_DEVICE after pairing the headset)"
fi
if [ -f "$LOCI_HOME/rooms.json" ]; then
  ok "$LOCI_HOME/rooms.json exists"
elif [ -f "$SCRIPT_DIR/config/rooms.example.json" ]; then
  cp "$SCRIPT_DIR/config/rooms.example.json" "$LOCI_HOME/rooms.json"
  ok "wrote $LOCI_HOME/rooms.json from config/rooms.example.json (set tag_size_m to the measured tag)"
else
  warn "config/rooms.example.json not found; the server falls back to the built-in two-room map"
fi
ok "data dir $LOCI_HOME/data"

# ------------------------------------------------------------------ self-test
say "Self-test (imports only; the camera is not opened)"
if (cd "$SRC_DIR" && LOCI_DETECTION_SOURCE=none LOCI_PERSIST=0 LOCI_VOICE=none WHISPER_LOCAL=0 TTS_ENGINE=piper \
      LOCI_LABEL_EMBEDDER=hash HF_HUB_OFFLINE=1 \
      "$PY" -c "import depthai, cv2, pyapriltags, tokenizers, safetensors, loci, demo_spatial.app.main; print('    import OK: depthai', depthai.__version__, 'cv2', cv2.__version__)"); then
  ok "core imports"
else
  warn "self-test import failed (see the traceback above)"
fi
if "$PY" -c "import sounddevice, faster_whisper, piper" >/dev/null 2>&1; then
  ok "voice imports (sounddevice, faster_whisper, piper)"
else
  warn "voice stack incomplete: $("$PY" -c "import sounddevice, faster_whisper, piper" 2>&1 | tail -1) -- typed queries still work; libportaudio2 missing is the usual cause"
fi
if [ -d "$MODELS/embeddings/potion-base-8M" ]; then
  if (cd "$SRC_DIR" && "$PY" -c "import sys; sys.path.insert(0, 'demo_spatial'); from app.label_embedding import StaticLabelEmbedder as S; e = S('$MODELS/embeddings/potion-base-8M'); print('    embedder', e.name, 'dim', e.dim)" 2>/dev/null); then
    ok "static embedder loads"
  else
    warn "static embedder did not load (label_embedding.py missing or model incomplete); LOCI_LABEL_EMBEDDER=hash still works"
  fi
fi

# ------------------------------------------------------------------ systemd user unit
say "systemd user service"
if [ "$NO_SERVICE" = "1" ]; then
  ok "skipped (--no-service)"
elif systemctl --user show-environment >/dev/null 2>&1; then
  UNIT_DIR="$HOME/.config/systemd/user"
  mkdir -p "$UNIT_DIR"
  if ! cmp -s "$SCRIPT_DIR/loci-handheld.user.service" "$UNIT_DIR/loci-handheld.service"; then
    cp "$SCRIPT_DIR/loci-handheld.user.service" "$UNIT_DIR/loci-handheld.service"
    systemctl --user daemon-reload
    ok "installed $UNIT_DIR/loci-handheld.service"
  else
    ok "unit up to date"
  fi
  systemctl --user enable loci-handheld >/dev/null 2>&1 && ok "enabled (starts at boot once linger is on)"
  if [ "$START" = "1" ]; then
    systemctl --user restart loci-handheld && ok "restarted; logs: journalctl --user -u loci-handheld -f"
  else
    echo "    start it with: systemctl --user restart loci-handheld   (or: $SCRIPT_DIR/run_pi.sh)"
  fi
else
  warn "systemctl --user is not available in this session (ssh without a user manager?); run with --start later or use the system unit"
fi

# ------------------------------------------------------------------ summary
say "Done"
if [ "${#WARNINGS[@]}" -gt 0 ]; then
  echo "    ${#WARNINGS[@]} warning(s):"
  for w in "${WARNINGS[@]}"; do echo "      - $w"; done
fi
echo "    next: deploy/handheld/README.md -> 'Bring-up' (step 1: scripts/oak_console.py --seconds 30)"
