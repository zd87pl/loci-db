#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  LOCI Demo — Automated walkthrough for screen recording
#
#  Usage:
#    pip install -e ".[dev]" && pip install fastapi uvicorn[standard]
#    ./demo/run_demo.sh
#
#  Then open http://localhost:8000 in your browser.
#  The script narrates each step in the terminal while the demo
#  runs in the browser — perfect for a YouTube walkthrough.
# ─────────────────────────────────────────────────────────────────
set -e

PORT=8000
BASE="http://localhost:$PORT"
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

say() { echo -e "\n${CYAN}${BOLD}▸ $1${RESET}"; }
info() { echo -e "  ${GREEN}$1${RESET}"; }
warn() { echo -e "  ${YELLOW}$1${RESET}"; }
pause() {
  echo -e "  ${BOLD}[Press Enter to continue]${RESET}"
  read -r
}

# ── Cleanup on exit ──────────────────────────────────────────────
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Start server ─────────────────────────────────────────────────
say "Starting LOCI demo server on port $PORT..."
uvicorn demo.app.main:app --host 0.0.0.0 --port "$PORT" --log-level warning &
SERVER_PID=$!
sleep 2

# Check server is up
if ! curl -sf "$BASE/health" > /dev/null; then
  echo "ERROR: Server failed to start. Run 'pip install fastapi uvicorn[standard]' first."
  exit 1
fi
info "Server running at $BASE"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  LOCI — Spatiotemporal Memory for Autonomous Systems${RESET}"
echo -e "${BOLD}  Interactive Demo Walkthrough${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  Open ${CYAN}${BOLD}$BASE${RESET} in your browser now."
echo -e "  This script will guide you through each demo step."
echo -e "  The terminal shows what's happening under the hood."
pause

# ── Step 1: Start simulation ────────────────────────────────────
say "Step 1: Start the robot patrol"
echo -e "  Click ${BOLD}▶ Start${RESET} in the browser, or I'll start it for you."
pause

curl -sf -X POST "$BASE/api/simulation/start" > /dev/null
info "Simulation started! The robot is now patrolling the warehouse."
info "Every 500ms it observes its surroundings and stores a memory in LOCI."
echo ""
info "Watch the red dot move through the aisles."
info "Each step creates a WorldState with:"
info "  - (x, y, z) normalized coordinates"
info "  - A 128-dim embedding of what the robot 'sees'"
info "  - A timestamp"
info "  - Causal links to previous observations"
echo ""

say "Let the robot build up some memories (~15 seconds)..."
for i in $(seq 15 -1 1); do
  STATE=$(curl -sf "$BASE/api/simulation/state")
  MEM=$(echo "$STATE" | python3 -c "import json,sys; print(json.load(sys.stdin)['memory_count'])")
  printf "\r  Memories stored: ${CYAN}${BOLD}%s${RESET}  (waiting %ds)  " "$MEM" "$i"
  sleep 1
done
echo ""

STATE=$(curl -sf "$BASE/api/simulation/state")
MEM=$(echo "$STATE" | python3 -c "import json,sys; print(json.load(sys.stdin)['memory_count'])")
EPOCHS=$(curl -sf "$BASE/api/stats" | python3 -c "import json,sys; print(json.load(sys.stdin)['epoch_count'])")
info "Robot has stored $MEM memories across $EPOCHS temporal shards."
pause

# ── Step 2: Spatial + Temporal Query ─────────────────────────────
say "Step 2: Spatial + Temporal Query (Tab 1: 'Where & When')"
echo ""
info "In the browser:"
info "  1. Click the 'Where & When' tab (should be selected)"
info "  2. Click and drag on the warehouse to draw a bounding box"
info "  3. Click 'Search Region'"
echo ""
info "This demonstrates LOCI's Hilbert bucketing — instead of checking"
info "every vector with 3 float-range filters, LOCI maps 4D space to"
info "a single integer via a space-filling curve."
echo ""

warn "Let me also run a query from the API to show what happens:"
pause

RESULT=$(curl -sf -X POST "$BASE/api/query/spatial" \
  -H 'Content-Type: application/json' \
  -d '{"x_min":0,"x_max":10,"y_min":0,"y_max":10,"time_start_s":0,"time_end_s":999}')

FOUND=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['results']))")
SHARDS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['stats']['shards_searched'])")
CANDS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['stats']['total_candidates'])")
TOTAL=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['stats']['total_points'])")
MS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['stats']['elapsed_ms'])")

echo ""
info "Query: Find memories in grid region (0,0)-(10,10), all time"
echo -e "  Results found:      ${CYAN}${BOLD}$FOUND${RESET}"
echo -e "  Shards searched:    ${CYAN}$SHARDS${RESET} (temporal epochs)"
echo -e "  Candidates checked: ${CYAN}$CANDS${RESET} out of ${CYAN}$TOTAL${RESET} total"
echo -e "  Query time:         ${CYAN}${MS}ms${RESET}"
if [ "$TOTAL" -gt 0 ] && [ "$CANDS" -lt "$TOTAL" ]; then
  PCT=$(python3 -c "print(round((1 - $CANDS/$TOTAL) * 100))")
  echo -e "  ${GREEN}${BOLD}→ LOCI skipped ${PCT}% of memories via Hilbert bucketing${RESET}"
fi
pause

# ── Step 3: Similarity Query ─────────────────────────────────────
say "Step 3: Vector Similarity Query (Tab 2: 'What's Similar?')"
echo ""
info "In the browser:"
info "  1. Click the 'What's Similar?' tab"
info "  2. Click any cell on the warehouse grid"
info "  3. Click 'Find Similar'"
echo ""
info "This finds the closest stored memory to your click, then searches"
info "for the top-5 most similar embeddings within a spatial radius."
info "It combines VECTOR SIMILARITY (what the robot saw) with"
info "SPATIAL PROXIMITY (where it was) — something regular vector DBs"
info "can't do efficiently."
echo ""
warn "Running a similarity query from the API:"
pause

RESULT=$(curl -sf -X POST "$BASE/api/query/similar" \
  -H 'Content-Type: application/json' \
  -d '{"x":5,"y":10,"radius":8,"limit":5}')

ANCHOR=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); a=d.get('anchor'); print(f'({a[\"x\"]}, {a[\"y\"]})' if a else 'None')")
FOUND=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['results']))")
MS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['stats']['elapsed_ms'])")

echo ""
info "Query: Find memories similar to what's at (5, 10), radius=8"
echo -e "  Nearest memory:  ${CYAN}${BOLD}$ANCHOR${RESET}"
echo -e "  Similar found:   ${CYAN}$FOUND${RESET}"
echo -e "  Query time:      ${CYAN}${MS}ms${RESET}"
pause

# ── Step 4: Predict + Novelty ────────────────────────────────────
say "Step 4: Predict-then-Retrieve with Novelty Detection (Tab 3)"
echo ""
info "This is LOCI's killer feature."
info ""
info "In ONE atomic call, LOCI:"
info "  1. Runs a world model prediction (where will the robot be?)"
info "  2. Searches for matching memories"
info "  3. Scores how NOVEL the predicted situation is"
echo ""
info "In the browser:"
info "  1. Click the 'Predict & Surprise' tab"
info "  2. Adjust 'Steps ahead' slider"
info "  3. Click 'Predict Future'"
echo ""
warn "Running prediction from the API:"
pause

RESULT=$(curl -sf -X POST "$BASE/api/query/predict" \
  -H 'Content-Type: application/json' \
  -d '{"steps_ahead":10}')

NOVELTY=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['novelty'])")
CURPOS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); p=d['current_position']; print(f'({p[\"x\"]}, {p[\"y\"]})')")
PREDPOS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); p=d['predicted_position']; print(f'({p[\"x\"]}, {p[\"y\"]})')")
PRED_MS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['predictor_ms'])")
RET_MS=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['retrieval_ms'])")

echo ""
info "Prediction: Robot at $CURPOS → expects to be at $PREDPOS in 5 seconds"
echo -e "  Novelty score:    ${CYAN}${BOLD}$NOVELTY${RESET}"
echo -e "  Predictor time:   ${CYAN}${PRED_MS}ms${RESET}"
echo -e "  Retrieval time:   ${CYAN}${RET_MS}ms${RESET}"
echo ""

# Interpret novelty
NFLOAT=$(python3 -c "print(float('$NOVELTY'))")
if python3 -c "exit(0 if $NFLOAT > 0.7 else 1)"; then
  echo -e "  ${RED}${BOLD}HIGH NOVELTY — The robot hasn't seen this area much!${RESET}"
elif python3 -c "exit(0 if $NFLOAT > 0.3 else 1)"; then
  echo -e "  ${YELLOW}MODERATE NOVELTY — Partially familiar territory.${RESET}"
else
  echo -e "  ${GREEN}LOW NOVELTY — The robot knows this area well.${RESET}"
fi
pause

# ── Step 5: Anomaly injection ────────────────────────────────────
say "Step 5: Anomaly Injection — The dramatic moment!"
echo ""
info "Now for the exciting part. We'll place an unexpected object"
info "in the robot's path and watch the novelty score spike."
echo ""
info "In the browser:"
info "  1. Click '🚨 Place Anomaly'"
info "  2. Click an empty cell on the robot's patrol route"
info "  3. Wait for the robot to pass nearby"
info "  4. Run 'Predict Future' again — watch the novelty spike!"
echo ""
warn "Let me place an anomaly at (2, 5) — right on the patrol route:"
pause

curl -sf -X POST "$BASE/api/anomaly/place" \
  -H 'Content-Type: application/json' \
  -d '{"x":2,"y":5}' > /dev/null
info "Anomaly placed at (2, 5)!"
info "A red X now appears on the warehouse grid."
echo ""
info "Wait for the robot to patrol past it..."
info "The embedding will change because the robot now 'sees' an anomaly."
info "When you run Predict again, the novelty score should be higher"
info "because the prediction doesn't match what the robot is experiencing."
pause

# ── Step 6: Heatmap ──────────────────────────────────────────────
say "Step 6: Memory Heatmap"
echo ""
info "Click the 'Heatmap' button to see where the robot has"
info "stored the most memories. Hot spots appear in red,"
info "cold spots in blue."
echo ""
info "This shows how LOCI's temporal sharding distributes"
info "memories across the warehouse floor over time."
pause

# ── Step 7: How It Works ─────────────────────────────────────────
say "Step 7: How It Works (Educational Footer)"
echo ""
info "Scroll down and click 'How It Works' to see the three"
info "core innovations visualized:"
info ""
info "  1. HILBERT CURVES — Map 4D (x,y,z,t) to a single int"
info "  2. TEMPORAL SHARDING — Auto-partition by time epoch"
info "  3. NOVELTY SCORING — Compare predicted vs. stored embeddings"
pause

# ── Wrap up ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Demo Complete!${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo ""
info "Key takeaways for the video:"
echo ""
echo -e "  ${BOLD}1.${RESET} LOCI makes spatiotemporal queries first-class"
echo -e "     (not bolted-on float filters)"
echo ""
echo -e "  ${BOLD}2.${RESET} Hilbert bucketing replaces 3 range filters"
echo -e "     with a single integer-set lookup"
echo ""
echo -e "  ${BOLD}3.${RESET} Temporal sharding auto-partitions by time —"
echo -e "     queries skip irrelevant epochs"
echo ""
echo -e "  ${BOLD}4.${RESET} Predict-then-retrieve is a unique primitive —"
echo -e "     one API call for prediction + search + novelty"
echo ""
echo -e "  ${BOLD}5.${RESET} Zero infrastructure needed — LocalLociClient"
echo -e "     runs entirely in-memory"
echo ""
info "Server is still running at $BASE"
info "Press Ctrl+C to stop."
echo ""

# Keep alive
wait "$SERVER_PID"
