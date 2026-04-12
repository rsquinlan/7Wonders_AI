#!/usr/bin/env bash
# AlphaZero-style self-play RL loop for 7 Wonders.
#
# Each iteration:
#   1. Generate games using the latest model (or heuristic rollouts if no model yet)
#   2. Train a new model on a sliding window of recent games (always including heuristic data)
#   3. New model becomes the current model for the next iteration
#
# Iteration 0 generates HEURISTIC_MULTIPLIER * GAMES_PER_ITER games with heuristic rollouts
# to bootstrap a strong initial model. Subsequent iterations use equal search depth with NN.
#
# Usage: ./self_play_loop.sh [iterations] [games_per_iter] [num_players] [search_depth] [workers]
#   ./self_play_loop.sh 3 100 5 100 4

set -euo pipefail

ITERATIONS=${1:-3}
GAMES_PER_ITER=${2:-100}
NUM_PLAYERS=${3:-5}
SEARCH_DEPTH=${4:-100}
WORKERS=${5:-4}
EXPLORATION=0.2
EPOCHS=500
BINARY="./7Wonders"
TRAIN_DIR="training"
PYTHON="python3.11"
HEURISTIC_MULTIPLIER=5   # iter 0 generates 5x more games for a strong bootstrap
TRAIN_WINDOW=5000         # sliding window: train on at most this many recent games + all heuristic

if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run 'make' first." >&2
    exit 1
fi

mkdir -p "$TRAIN_DIR"

# Find starting iteration from existing models
START_ITER=0
for f in "$TRAIN_DIR"/model_iter_*.onnx; do
    [ -e "$f" ] || break
    NUM=$(basename "$f" .onnx | sed 's/model_iter_//')
    if [ "$NUM" -ge "$START_ITER" ]; then
        START_ITER=$(( NUM + 1 ))
    fi
done

# Find starting game ID from existing game files (never overwrite old games)
START_GAME=0
if ls "$TRAIN_DIR"/game_*.jsonl &>/dev/null; then
    LAST=$(ls "$TRAIN_DIR"/game_*.jsonl | sort -V | tail -1)
    LAST_ID=$(basename "$LAST" .jsonl | sed 's/game_//')
    START_GAME=$(( 10#$LAST_ID + 1 ))
fi

if [ "$START_ITER" -gt 0 ]; then
    echo "Resuming from iteration $START_ITER (game IDs start at $START_GAME)"
else
    echo "Starting fresh RL loop"
fi

# ── Helper: run a single game ─────────────────────────────────────────────────
run_game() {
    g=$1; BINARY=$2; NUM_PLAYERS=$3; DEPTH=$4; EXPLORATION=$5
    TRAIN_DIR=$6; MODEL_PATH=$7; NN_FLAG=$8; LOG=$9
    if [ -z "$MODEL_PATH" ]; then
        nice -n 19 "$BINARY" "$g" "$NUM_PLAYERS" "$DEPTH" "$EXPLORATION" \
            "$TRAIN_DIR" >> "$LOG" 2>&1
    else
        nice -n 19 "$BINARY" "$g" "$NUM_PLAYERS" "$DEPTH" "$EXPLORATION" \
            "$TRAIN_DIR" "$MODEL_PATH" "$NN_FLAG" >> "$LOG" 2>&1
    fi
    echo -n "."
}
export -f run_game

# ── Main loop ─────────────────────────────────────────────────────────────────
for (( iter=START_ITER; iter<START_ITER+ITERATIONS; iter++ )); do
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  ITERATION $iter / $((START_ITER+ITERATIONS-1))                         ║"
    echo "╚══════════════════════════════════════════╝"

    LOG="$TRAIN_DIR/gen_iter_${iter}.log"

    # ── Step 1: Determine generation policy ───────────────────────────────────
    if [ "$iter" -eq 0 ]; then
        DEPTH=$SEARCH_DEPTH   # MCTS with heuristic rollouts, no NN
        MODEL_PATH=""
        NN_FLAG="0"
        ITER_GAMES=$(( GAMES_PER_ITER * HEURISTIC_MULTIPLIER ))
        END_GAME=$(( START_GAME + ITER_GAMES ))
        echo "[1/2] Generating $ITER_GAMES heuristic games (${HEURISTIC_MULTIPLIER}x bootstrap, $WORKERS workers)..."
    else
        DEPTH=$SEARCH_DEPTH   # equal search depth — NN value head needs full search
        ITER_GAMES=$GAMES_PER_ITER
        END_GAME=$(( START_GAME + ITER_GAMES ))
        PREV=$(( iter - 1 ))
        MODEL_PATH="$TRAIN_DIR/model_iter_${PREV}.onnx"
        NN_FLAG="1"      # all players use NN
        echo "[1/2] Generating $ITER_GAMES games with model_iter_${PREV} (depth=$DEPTH, $WORKERS workers)..."
    fi

    # ── Step 2: Generate games in parallel ────────────────────────────────────
    seq $START_GAME $((END_GAME - 1)) | xargs -P "$WORKERS" -I{} \
        bash -c 'run_game "$@"' _ {} \
        "$BINARY" "$NUM_PLAYERS" "$DEPTH" "$EXPLORATION" \
        "$TRAIN_DIR" "$MODEL_PATH" "$NN_FLAG" "$LOG"
    echo ""

    TOTAL=$(ls "$TRAIN_DIR"/game_*.jsonl 2>/dev/null | wc -l)
    echo "      $ITER_GAMES new games written (total accumulated: $TOTAL)"

    # ── Step 3: Train on sliding window of recent games + all heuristic data ──
    #   - Always include iter 0 heuristic games (game IDs < HEURISTIC_COUNT)
    #   - Plus the most recent TRAIN_WINDOW games from NN self-play
    HEURISTIC_COUNT=$(( GAMES_PER_ITER * HEURISTIC_MULTIPLIER ))
    TRAIN_LIST="$TRAIN_DIR/train_files.txt"
    # Heuristic games (always included)
    ls "$TRAIN_DIR"/game_*.jsonl 2>/dev/null | sort -V | head -n "$HEURISTIC_COUNT" > "$TRAIN_LIST"
    # Recent NN games (sliding window)
    ls "$TRAIN_DIR"/game_*.jsonl 2>/dev/null | sort -V | tail -n +"$(( HEURISTIC_COUNT + 1 ))" | tail -n "$TRAIN_WINDOW" >> "$TRAIN_LIST"
    TRAIN_TOTAL=$(wc -l < "$TRAIN_LIST")

    OUT_MODEL="$TRAIN_DIR/model_iter_${iter}.onnx"
    echo "[2/2] Training model_iter_${iter} on $TRAIN_TOTAL games (${HEURISTIC_COUNT} heuristic + window) -> $OUT_MODEL"
    "$PYTHON" training/train.py \
        --data        "$TRAIN_DIR/" \
        --out         "$OUT_MODEL"  \
        --epochs      "$EPOCHS"     \
        --hidden      768           \
        --batch       512           \
        --max-samples 100000        \
        --file-list   "$TRAIN_LIST"

    echo "      Iteration $iter complete. Model: $OUT_MODEL"
    START_GAME=$END_GAME
done

echo ""
echo "RL loop complete."
echo "Final model: $TRAIN_DIR/model_iter_$((START_ITER+ITERATIONS-1)).onnx"
echo "Total games: $(ls "$TRAIN_DIR"/game_*.jsonl 2>/dev/null | wc -l)"
