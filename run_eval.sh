#!/usr/bin/env bash
# Run N evaluation games and report win rates.
# Player 0 = NN rollout, Players 1-4 = heuristic.
#
# Usage: ./run_eval.sh [num_games] [num_players] [search_depth] [model_iter] [out_dir]
#   ./run_eval.sh 200              # model_iter_0.onnx -> eval/model_iter_0/
#   ./run_eval.sh 200 5 100 3      # model_iter_3.onnx -> eval/model_iter_3/

set -euo pipefail

NUM_GAMES=${1:-50}
NUM_PLAYERS=${2:-5}
SEARCH_DEPTH=${3:-500}
MODEL_ITER=${4:-0}
EXPLORATION=0.2
OUT_DIR=${5:-"eval/model_iter_${MODEL_ITER}"}
BINARY="./7Wonders"

MODEL_PATH="training/model_iter_${MODEL_ITER}.onnx"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model not found at $MODEL_PATH" >&2
    exit 1
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run 'make' first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Running $NUM_GAMES eval games ($NUM_PLAYERS players, depth $SEARCH_DEPTH)..."
echo "Player 0 = NN rollout ($MODEL_PATH) | Players 1-$((NUM_PLAYERS-1)) = heuristic"
echo ""

for i in $(seq 0 $((NUM_GAMES - 1))); do
    printf "\rGame %3d / %d ..." "$((i+1))" "$NUM_GAMES"
    "$BINARY" "$i" "$NUM_PLAYERS" "$SEARCH_DEPTH" "$EXPLORATION" "$OUT_DIR" "$MODEL_PATH" \
        > "$OUT_DIR/game_$(printf '%03d' $i).log" 2>&1
done
echo ""
echo ""

# ── Tally results from the result lines in each JSONL ─────────────────────────
python3 - "$OUT_DIR" "$NUM_PLAYERS" <<'EOF'
import sys, glob, json

out_dir     = sys.argv[1]
num_players = int(sys.argv[2])

wins   = [0] * num_players
played = 0

for path in sorted(glob.glob(f"{out_dir}/game_*.jsonl")):
    lines = open(path).readlines()
    if not lines:
        continue
    result = json.loads(lines[-1])
    if result.get("type") != "result":
        continue
    played += 1
    scores    = result["final_scores"]
    max_score = result["max_score"]
    for p, s in enumerate(scores):
        if s == max_score:
            wins[p] += 1

print(f"Results over {played} games:")
print(f"{'Player':<10} {'Type':<18} {'Wins':>6} {'Win %':>8}")
print("-" * 46)
for p in range(num_players):
    label = "NN rollout" if p == 0 else "Heuristic"
    pct   = 100.0 * wins[p] / played if played else 0.0
    print(f"  {p:<8} {label:<18} {wins[p]:>6} {pct:>7.1f}%")
print()
# Note: wins can sum to > played if there are ties
total_wins = sum(wins)
if total_wins > played:
    print(f"(Note: {total_wins - played} tied game(s) counted for all tied winners)")
EOF
