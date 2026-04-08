#!/usr/bin/env bash
# Usage: ./run_training.sh [num_games] [num_players] [search_depth] [exploration_constant]
#   e.g. ./run_training.sh 100 5 500 0.2

NUM_GAMES=${1:-5}
NUM_PLAYERS=${2:-5}
SEARCH_DEPTH=${3:-500}
EXPLORATION=${4:-0.2}

BINARY="./7Wonders"
LOG="training/run.log"

mkdir -p training

# Resume: find the next game ID by looking at existing game_*.jsonl files
START=0
if ls training/game_*.jsonl &>/dev/null; then
    LAST=$(ls training/game_*.jsonl | sort | tail -1)
    LAST_ID=$(basename "$LAST" .jsonl | sed 's/game_//')
    START=$(( 10#$LAST_ID + 1 ))  # 10# forces base-10 parse of zero-padded numbers
fi
END=$(( START + NUM_GAMES ))

echo "Starting $NUM_GAMES games from ID $START to $((END-1)) (players=$NUM_PLAYERS depth=$SEARCH_DEPTH)"
echo "Logs -> $LOG"

for ((i=START; i<END; i++)); do
    echo -n "Game $i/$((END-1)) ... "
    nice -n 19 "$BINARY" "$i" "$NUM_PLAYERS" "$SEARCH_DEPTH" "$EXPLORATION" training >> "$LOG" 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "FAILED (exit $STATUS) — check $LOG"
        exit 1
    fi
    echo "done"
done

TOTAL=$(ls training/game_*.jsonl 2>/dev/null | wc -l)
echo "Done. training/ now has $TOTAL games total."
