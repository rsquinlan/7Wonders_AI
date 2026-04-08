#!/usr/bin/env bash
# AlphaZero-style self-play loop for 7 Wonders.
#
# Each iteration:
#   1. Clear old training games
#   2. Generate GAMES_PER_ITER games (all players use current NN, or heuristic if no model yet)
#   3. Train a new model on that fresh data
#   4. New model becomes the current model for the next iteration
#
# Usage: ./self_play_loop.sh [iterations] [games_per_iter] [num_players] [search_depth]
#   ./self_play_loop.sh 10 1000 5 100

set -euo pipefail

ITERATIONS=${1:-10}
GAMES_PER_ITER=${2:-1000}
NUM_PLAYERS=${3:-5}
SEARCH_DEPTH=${4:-100}
EXPLORATION=0.2
EPOCHS=500
BINARY="./7Wonders"
TRAIN_DIR="training"
PYTHON="python3.11"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run 'make' first." >&2
    exit 1
fi

mkdir -p "$TRAIN_DIR"

# Find the starting iteration by looking for existing models
START_ITER=0
for f in "$TRAIN_DIR"/model_iter_*.onnx; do
    [ -e "$f" ] || break
    NUM=$(basename "$f" .onnx | sed 's/model_iter_//')
    if [ "$NUM" -ge "$START_ITER" ]; then
        START_ITER=$(( NUM + 1 ))
    fi
done

if [ "$START_ITER" -gt 0 ]; then
    echo "Resuming from iteration $START_ITER"
else
    echo "Starting fresh self-play loop"
fi

for (( iter=START_ITER; iter<START_ITER+ITERATIONS; iter++ )); do
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  ITERATION $iter / $((START_ITER+ITERATIONS-1))"
    echo "╚══════════════════════════════════════╝"

    # ── Step 1: Clear old games ────────────────────────────────────────────
    echo "[1/3] Clearing old training games..."
    rm -f "$TRAIN_DIR"/game_*.jsonl

    # ── Step 2: Determine model for this iteration ─────────────────────────
    if [ "$iter" -eq 0 ]; then
        MODEL_PATH=""
        NN_FLAG="0"
        echo "[2/3] No model yet — bootstrapping with heuristic rollout"
    else
        PREV=$(( iter - 1 ))
        MODEL_PATH="$TRAIN_DIR/model_iter_${PREV}.onnx"
        NN_FLAG="1"
        echo "[2/3] Using model: $MODEL_PATH (all players NN rollout)"
    fi

    # ── Step 3: Generate games ─────────────────────────────────────────────
    echo "      Generating $GAMES_PER_ITER games..."
    for (( g=0; g<GAMES_PER_ITER; g++ )); do
        printf "\r      Game %4d / %d ..." "$((g+1))" "$GAMES_PER_ITER"
        if [ -z "$MODEL_PATH" ]; then
            nice -n 19 "$BINARY" "$g" "$NUM_PLAYERS" "$SEARCH_DEPTH" "$EXPLORATION" \
                "$TRAIN_DIR" >> "$TRAIN_DIR/gen_iter_${iter}.log" 2>&1
        else
            nice -n 19 "$BINARY" "$g" "$NUM_PLAYERS" "$SEARCH_DEPTH" "$EXPLORATION" \
                "$TRAIN_DIR" "$MODEL_PATH" "$NN_FLAG" >> "$TRAIN_DIR/gen_iter_${iter}.log" 2>&1
        fi
    done
    echo ""

    TOTAL=$(ls "$TRAIN_DIR"/game_*.jsonl 2>/dev/null | wc -l)
    echo "      Generated $TOTAL games"

    # ── Step 4: Train new model ────────────────────────────────────────────
    OUT_MODEL="$TRAIN_DIR/model_iter_${iter}.onnx"
    echo "[3/3] Training model -> $OUT_MODEL"
    "$PYTHON" training/train.py \
        --data "$TRAIN_DIR/" \
        --out  "$OUT_MODEL"  \
        --epochs "$EPOCHS"

    echo "      Iteration $iter complete. Model saved to $OUT_MODEL"
done

echo ""
echo "Self-play loop complete. Final model: $TRAIN_DIR/model_iter_$((START_ITER+ITERATIONS-1)).onnx"
