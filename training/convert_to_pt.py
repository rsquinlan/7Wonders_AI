"""
Convert JSONL game files to PyTorch binary format (.pt) for faster loading.

Usage:
  python3 training/convert_to_pt.py --data training/ --output training/pt_data/
  python3 training/convert_to_pt.py --data training/ --output training/pt_data/ --workers 8
"""

import argparse
import gc
import glob
import json
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch


def convert_file(args_tuple):
    """Convert a single JSONL file to .pt format (memory-efficient streaming). Returns (filename, num_samples, status)."""
    jsonl_path, output_dir = args_tuple
    filename = Path(jsonl_path).name

    try:
        # First pass: read last line to get result data
        with open(jsonl_path) as f:
            last_line = None
            for last_line in f:
                pass

        if not last_line:
            return (filename, 0, "SKIP")

        result = json.loads(last_line)
        if result.get("type") != "result":
            return (filename, 0, "SKIP")

        final_scores = result["final_scores"]
        max_score = result["max_score"]

        # Second pass: stream and convert samples (don't load all at once)
        samples = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                    if t.get("type") == "result":  # Skip result line
                        continue

                    win = float(final_scores[t["player_index"]] == max_score)
                    samples.append({
                        "state": torch.from_numpy(np.array(t["state_embedding"], dtype=np.float32)),
                        "policy": torch.from_numpy(np.array(t["mcts_policy"], dtype=np.float32)),
                        "value": torch.tensor([win], dtype=torch.float32),
                    })
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        if samples:
            output_path = os.path.join(output_dir, Path(jsonl_path).stem + ".pt")
            torch.save(samples, output_path)
            result = (filename, len(samples), "OK")
            del samples  # Explicitly free memory
            gc.collect()  # Force garbage collection
            return result

        return (filename, 0, "EMPTY")

    except (OSError, json.JSONDecodeError) as e:
        return (filename, 0, "ERROR")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/", help="Directory with game_*.jsonl files")
    parser.add_argument("--output", default="training/pt_data/", help="Output directory for .pt files")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, "game_*.jsonl")))
    if not files:
        print(f"No game_*.jsonl files found in {args.data}")
        return

    # Use fewer workers by default to avoid memory issues (each worker holds a full file in memory)
    # User can override with --workers if they have enough RAM
    default_workers = max(1, min(4, (os.cpu_count() or 1) // 2))
    num_workers = args.workers or default_workers
    print(f"Converting {len(files)} JSONL files to PyTorch binary format ({num_workers} workers)...")
    if not args.workers:
        print(f"  (Use --workers N to adjust. Using conservative default to avoid memory issues.)")

    # Prepare arguments for each worker
    file_args = [(f, args.output) for f in files]

    total_samples = 0
    with Pool(num_workers) as pool:
        results = pool.imap_unordered(convert_file, file_args)
        for filename, n_samples, status in results:
            if status == "OK":
                print(f"  ✓ {filename:30s} → {n_samples:6d} samples")
                total_samples += n_samples
            elif status == "ERROR":
                print(f"  ✗ {filename:30s} → ERROR reading file")
            elif status == "SKIP":
                print(f"  ⊘ {filename:30s} → SKIP (no result)")
            elif status == "EMPTY":
                print(f"  ⊘ {filename:30s} → EMPTY (no samples)")

    print(f"\nDone! Converted {total_samples:,} total samples to {args.output}")


if __name__ == "__main__":
    main()
