"""
Consolidate small per-game .pt files into large shard files with stacked tensors.

This reduces file I/O from 68K+ torch.load() calls to ~70 per epoch, and replaces
lists of Python dicts with contiguous stacked tensors for fast loading.

Output format per shard:
  {"states": tensor(N, 991), "policies": tensor(N, 77), "values": tensor(N, 1)}

Usage:
  python3.11 training/consolidate_shards.py
  python3.11 training/consolidate_shards.py --input training/pt_data --output training/shard_data --files-per-shard 1000
"""

import argparse
import glob
import os
import gc
from multiprocessing import Pool

import torch


def consolidate_shard(args_tuple):
    """Consolidate one shard's worth of files into stacked tensors. Returns (shard_idx, n_samples, size_mb, error)."""
    chunk_files, shard_idx, output_dir = args_tuple

    all_states = []
    all_policies = []
    all_values = []

    for f in chunk_files:
        try:
            samples = torch.load(f, weights_only=False)
            for s in samples:
                all_states.append(s["state"])
                all_policies.append(s["policy"])
                all_values.append(s["value"])
            del samples
        except Exception as e:
            return (shard_idx, 0, 0, f"Failed to load {f}: {e}")

    if not all_states:
        return (shard_idx, 0, 0, "No samples")

    try:
        shard = {
            "states": torch.stack(all_states),
            "policies": torch.stack(all_policies),
            "values": torch.stack(all_values),
        }
        n = shard["states"].size(0)

        out_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(shard, out_path)
        size_mb = os.path.getsize(out_path) / 1e6

        del shard, all_states, all_policies, all_values
        gc.collect()

        return (shard_idx, n, size_mb, None)
    except Exception as e:
        return (shard_idx, 0, 0, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training/pt_data/", help="Directory with game_*.pt files")
    parser.add_argument("--output", default="training/shard_data/", help="Output directory for shard files")
    parser.add_argument("--files-per-shard", type=int, default=1000, help="Number of game files per shard")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(args.input, "game_*.pt")))
    if not pt_files:
        print(f"No game_*.pt files found in {args.input}")
        return

    n_shards = (len(pt_files) + args.files_per_shard - 1) // args.files_per_shard
    num_workers = args.workers or (os.cpu_count() or 1)
    print(f"Consolidating {len(pt_files)} files into ~{n_shards} shards ({args.files_per_shard} files/shard)")
    print(f"Using {num_workers} workers...\n")

    # Prepare shard chunks
    shard_args = []
    for shard_idx, start in enumerate(range(0, len(pt_files), args.files_per_shard)):
        end = min(start + args.files_per_shard, len(pt_files))
        chunk = pt_files[start:end]
        shard_args.append((chunk, shard_idx, args.output))

    total_samples = 0
    with Pool(num_workers) as pool:
        results = pool.imap_unordered(consolidate_shard, shard_args)
        for shard_idx, n, size_mb, error in results:
            if error:
                print(f"  ✗ shard_{shard_idx:04d}.pt  ERROR: {error}")
            else:
                total_samples += n
                print(f"  ✓ shard_{shard_idx:04d}.pt  {n:7,} samples  {size_mb:.0f} MB")

    print(f"\nDone! {len(shard_args)} shards, {total_samples:,} total samples in {args.output}")


if __name__ == "__main__":
    main()
