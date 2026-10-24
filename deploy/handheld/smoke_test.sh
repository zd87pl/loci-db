#!/usr/bin/env bash
# Step-2 smoke test against a running LOCI handheld server (docs/HANDHELD_DEMO_DESIGN.md section 14).
# Stops at the first failing check with exit code 1. Needs curl and python3.
#
#   deploy/handheld/smoke_test.sh [--url http://localhost:8001] [--host H] [--port P]
#                                 [--expect-state running|any] [--min-fps 8]
#                                 [--expect-object cup] [--query "where is my mug"] [--label cup]
#                                 [--max-latency-ms 300] [--voice] [--snapshot] [--restored]
#
#   --expect-state any    accept whatever runtime.source.state is (default: running)
#   --expect-object cup   require /api/objects to list this label (place a cup in view first)
#   --voice               require /api/handheld/status.voice (LOCI_VOICE=headset)
#   --snapshot            POST /api/handheld/snapshot and require status ok
#   --restored            require /api/handheld/status.restore.records > 0 (after a restart)
set -euo pipefail

URL=""; HOST="localhost"; PORT="${PORT:-8001}"
EXPECT_STATE="running"; MIN_FPS="8"; EXPECT_OBJECT=""; QUERY="where is my mug"; LABEL="cup"
MAX_LAT="300"; VOICE=0; SNAPSHOT=0; RESTORED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --url) shift; URL="$1" ;;
    --host) shift; HOST="$1" ;;
    --port) shift; PORT="$1" ;;
    --expect-state) shift; EXPECT_STATE="$1" ;;
    --min-fps) shift; MIN_FPS="$1" ;;
    --expect-object) shift; EXPECT_OBJECT="$1" ;;
    --query) shift; QUERY="$1" ;;
    --label) shift; LABEL="$1" ;;
    --max-latency-ms) shift; MAX_LAT="$1" ;;
    --voice) VOICE=1 ;;
    --snapshot) SNAPSHOT=1 ;;
    --restored) RESTORED=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -z "$URL" ] && URL="http://$HOST:$PORT"
PY="$(command -v python3)"

PASS=0
pass() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

# jget JSON dotted.path  -> prints the value (None for missing), "true"/"false" lowercased
jget() {
  printf '%s' "$1" | "$PY" -I -c '
import json, sys
path = sys.argv[1].split(".") if sys.argv[1] else []
try:
    cur = json.load(sys.stdin)
except Exception as e:  # noqa: BLE001
    print("INVALID_JSON:", e); sys.exit(0)
for key in path:
    if isinstance(cur, list) and key.lstrip("-").isdigit():
        try:
            cur = cur[int(key)]
        except IndexError:
            cur = None
    elif isinstance(cur, dict):
        cur = cur.get(key)
    else:
        cur = None
    if cur is None:
        break
if isinstance(cur, bool):
    print(str(cur).lower())
elif isinstance(cur, (dict, list)):
    print(json.dumps(cur))
else:
    print(cur)
' "$2"
}
# jlabels JSON -> all "label" values anywhere in the document (one per line)
jlabels() {
  printf '%s' "$1" | "$PY" -I -c '
import json, sys
def walk(o):
    if isinstance(o, dict):
        if "label" in o: print(o["label"])
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(json.load(sys.stdin))'
}
get()  { curl -sS -m 10 -f "$URL$1"; }
post() { curl -sS -m 30 -f -H 'content-type: application/json' -X POST --data "$2" "$URL$1"; }

echo "smoke test against $URL"

H="$(get /health)" || fail "GET /health unreachable"
[ "$(jget "$H" status)" = "ok" ] || fail "/health status != ok: $H"
pass "/health ok (observations=$(jget "$H" observation_count), stt_available=$(jget "$H" pipeline.stt_available), tts=$(jget "$H" pipeline.tts_engine))"

S="$(get /api/handheld/status)" || fail "GET /api/handheld/status"
STATE="$(jget "$S" runtime.source.state)"
FPS="$(jget "$S" runtime.source.fps)"
if [ "$EXPECT_STATE" != "any" ]; then
  [ "$STATE" = "$EXPECT_STATE" ] || fail "runtime.source.state=$STATE (expected $EXPECT_STATE); reconnects=$(jget "$S" runtime.source.reconnects) error=$(jget "$S" runtime.source.error)"
  FPS_OK="$("$PY" -I -c 'import sys
try:
    print("1" if float(sys.argv[1]) >= float(sys.argv[2]) else "0")
except ValueError:
    print("0")' "$FPS" "$MIN_FPS")"
  [ "$FPS_OK" = "1" ] || fail "fps=$FPS below $MIN_FPS"
fi
pass "/api/handheld/status state=$STATE fps=$FPS usb=$(jget "$S" runtime.source.usb_speed) frames=$(jget "$S" runtime.frames) embedder=$(jget "$S" memory.embedder)"

if [ "$RESTORED" = "1" ]; then
  REC="$(jget "$S" restore.records)"
  [ "$REC" != "None" ] && [ "$REC" != "0" ] || fail "restore.records=$REC (expected > 0 after a restart)"
  pass "restore.records=$REC source=$(jget "$S" restore.source)"
fi

A="$(get /api/anchor)" || fail "GET /api/anchor"
pass "/api/anchor room=$(jget "$A" room) frame=$(jget "$A" frame) anchored=$(jget "$A" anchored)"

R="$(get /api/rooms)" || fail "GET /api/rooms"
pass "/api/rooms tag_size_m=$(jget "$R" tag_size_m)"

O="$(get /api/objects)" || fail "GET /api/objects"
if [ -n "$EXPECT_OBJECT" ]; then
  jlabels "$O" | grep -qx "$EXPECT_OBJECT" || fail "/api/objects does not list '$EXPECT_OBJECT': $(jlabels "$O" | sort -u | tr '\n' ' ')"
fi
pass "/api/objects labels: $(jlabels "$O" | sort -u | tr '\n' ' ')"

Q="$(post /api/query "{\"text\": \"$QUERY\"}")" || fail "POST /api/query"
RL="$(jget "$Q" resolution.label)"
LAT="$(jget "$Q" latency_ms)"
[ "$RL" = "$LABEL" ] || fail "resolution.label=$RL (expected $LABEL) for '$QUERY'; answer: $(jget "$Q" answer)"
LAT_OK="$("$PY" -I -c 'import sys
try:
    print("1" if float(sys.argv[1]) < float(sys.argv[2]) else "0")
except ValueError:
    print("0")' "$LAT" "$MAX_LAT")"
[ "$LAT_OK" = "1" ] || fail "latency_ms=$LAT (limit $MAX_LAT)"
pass "/api/query '$QUERY' -> $RL in ${LAT} ms: $(jget "$Q" answer)"

K="$(post /api/query '{"text": "where are my keys"}')" || fail "POST /api/query (unknown label) returned an error"
[ "$(jget "$K" resolution.label)" = "None" ] || fail "keys resolved to $(jget "$K" resolution.label); expected an honest refusal"
pass "/api/query 'where are my keys' -> honest refusal: $(jget "$K" answer)"

C="$(get '/api/objects/changes?window_seconds=600')" || fail "GET /api/objects/changes"
pass "/api/objects/changes count=$(jget "$C" count)"

P="$(get /phone)" || fail "GET /phone"
printf '%s' "$P" | grep -q 'aria-live' || fail "/phone has no aria-live region"
pass "/phone served with aria-live"

if [ "$VOICE" = "1" ]; then
  V="$(jget "$S" voice)"
  [ "$V" != "None" ] || fail "status.voice is null (LOCI_VOICE=headset not active)"
  pass "voice loop reporting: $V"
fi

if [ "$SNAPSHOT" = "1" ]; then
  SN="$(post /api/handheld/snapshot '{}')" || fail "POST /api/handheld/snapshot"
  [ "$(jget "$SN" status)" = "ok" ] || fail "snapshot status: $SN"
  pass "snapshot written (journal_lines=$(jget "$SN" persistence.journal_lines))"
fi

echo "all $PASS checks passed"
