"""
Train a dual-head AlphaZero-style network on 7 Wonders self-play data.

Inputs:
  state_embedding  — 991 floats (see utils.h stateEmbedSize)

Outputs:
  policy  — 75 floats (softmax), trained against MCTS visit distribution
  value   — 1  float  (logits → sigmoid), trained against win/loss outcome with class weighting

Usage:
  python3.11 training/train.py --out training/model3.onnx --epochs 500
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


# ── Network ───────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Fully-connected residual block with batch norm."""
    def __init__(self, size, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class WondersNet(nn.Module):
    def __init__(self, input_size=991, policy_size=75, hidden=512, num_res_blocks=4, dropout=0.3):
        super().__init__()

        # Project input up to hidden size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # Residual tower (shared trunk)
        self.res_tower = nn.Sequential(
            *[ResBlock(hidden, dropout) for _ in range(num_res_blocks)]
        )

        # Policy head: outputs log-probabilities (use with KLDivLoss directly)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, policy_size),
            nn.LogSoftmax(dim=1),
        )

        # Value head: predicts win logits (sigmoid applied in BCEWithLogitsLoss)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.res_tower(self.input_proj(x))
        return self.policy_head(h), self.value_head(h)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShardDataset(torch.utils.data.IterableDataset):
    """Load consolidated shard files and yield pre-formed batches.

    Each shard contains stacked tensors: {"states": (N, D_s), "policies": (N, D_p), "values": (N, 1)}.
    Yields (states_batch, policies_batch, values_batch) tuples — use with batch_size=None in DataLoader.
    """
    def __init__(self, shard_files, batch_size, max_samples=None):
        self.files = shard_files
        self.batch_size = batch_size
        self.max_samples = max_samples
        print(f"Found {len(self.files)} shard files (batch_size={batch_size}, max_samples={max_samples})")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files.copy()
        random.shuffle(files)
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        emitted = 0
        worker_id = worker_info.id if worker_info else 0
        for i, path in enumerate(files):
            shard_name = os.path.basename(path)
            print(f"  [worker {worker_id}] Loading shard {i+1}/{len(files)}: {shard_name}")
            try:
                data = torch.load(path, weights_only=False)
            except Exception as e:
                print(f"  [worker {worker_id}] WARNING: Failed to load {shard_name}: {e}")
                continue

            states, policies, values = data["states"], data["policies"], data["values"]
            n = states.size(0)
            del data

            # Shuffle within shard
            perm = torch.randperm(n)
            states = states[perm]
            policies = policies[perm]
            values = values[perm]
            del perm

            # Yield pre-formed batches
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                yield states[start:end], policies[start:end], values[start:end]
                emitted += end - start
                if self.max_samples and emitted >= self.max_samples:
                    del states, policies, values
                    return

            # Free shard data before loading next shard
            del states, policies, values


class StreamingBinaryTransitionDataset(torch.utils.data.IterableDataset):
    """Stream pre-converted PyTorch binary files (.pt format) one file at a time — memory efficient."""
    def __init__(self, pt_files, max_samples=None):
        self.files = pt_files
        self.max_samples = max_samples
        print(f"Found {len(self.files)} .pt files (streaming, max_samples={max_samples})")

    def __iter__(self):
        # Partition files across workers so each worker loads a disjoint subset
        worker_info = torch.utils.data.get_worker_info()
        files = self.files.copy()
        random.shuffle(files)
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        emitted = 0
        for path in files:
            try:
                samples = torch.load(path)
            except Exception as e:
                print(f"  WARNING: Failed to load {path}: {e}")
                continue

            for sample in samples:
                yield sample["state"], sample["policy"], sample["value"]
                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return


class TransitionDataset(torch.utils.data.IterableDataset):
    """Streams transitions one file at a time — memory usage is O(one game file)."""
    def __init__(self, data_dir, max_samples=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "game_*.jsonl")))
        if not self.files:
            raise FileNotFoundError(f"No game_*.jsonl files found in {data_dir}")
        self.max_samples = max_samples
        print(f"Found {len(self.files)} game files (streaming, max_samples={max_samples})")

    @staticmethod
    def _parse_file(path):
        """Yield (state, policy, value) tuples from one game file. Returns [] on error."""
        try:
            lines = [json.loads(l) for l in open(path) if l.strip()]
        except (json.JSONDecodeError, OSError):
            return
        if not lines or lines[-1].get("type") != "result":
            return
        final_scores = lines[-1]["final_scores"]
        max_score    = lines[-1]["max_score"]
        for t in lines[:-1]:
            win = float(final_scores[t["player_index"]] == max_score)
            yield (
                np.array(t["state_embedding"], dtype=np.float32),
                np.array(t["mcts_policy"],     dtype=np.float32),
                np.float32(win),
            )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files.copy()
        random.shuffle(files)
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        emitted = 0
        for path in files:
            for state, policy, value in self._parse_file(path):
                yield (
                    torch.from_numpy(state),
                    torch.from_numpy(policy),
                    torch.tensor([value], dtype=torch.float32),
                )
                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir, out_path, epochs, lr, batch_size, val_split, hidden, num_res_blocks, dropout, max_samples=None, file_list=None, use_pt=True, profile_enabled=False, num_workers=None):
    # Auto-detect num_workers if not specified
    if num_workers is None:
        num_workers = 2  # Conservative: each shard worker holds ~450MB in memory
    print(f"Using {num_workers} data loading workers")

    train_files = None  # Track for reporting
    val_files = None
    use_shards = False

    # Priority 1: consolidated shards (fastest — few large files with stacked tensors)
    shard_dir = os.path.join(data_dir, "shard_data")
    if use_pt and os.path.isdir(shard_dir):
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if shard_files:
            print(f"Loading from consolidated shards ({len(shard_files)} shards)...")
            n_val   = max(1, int(len(shard_files) * val_split))
            n_train = len(shard_files) - n_val
            random.shuffle(shard_files)
            train_files = shard_files[:n_train]
            val_files   = shard_files[n_train:]

            train_max = max_samples
            val_max   = max(1000, int((max_samples or 0) * val_split)) if max_samples else None

            train_ds = ShardDataset(train_files, batch_size, max_samples=train_max)
            val_ds   = ShardDataset(val_files,   batch_size, max_samples=val_max)
            # batch_size=None: dataset yields pre-formed batches, skip collation overhead
            train_loader = DataLoader(train_ds, batch_size=None, num_workers=num_workers,
                                     pin_memory=True, prefetch_factor=2,
                                     persistent_workers=num_workers > 0)
            val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=num_workers,
                                     pin_memory=True, prefetch_factor=2,
                                     persistent_workers=num_workers > 0)
            use_shards = True

    # Priority 2: per-game .pt files (slow — 68K+ individual torch.load calls)
    if not use_shards:
        pt_dir = os.path.join(data_dir, "pt_data")
        if use_pt and os.path.isdir(pt_dir):
            pt_files = sorted(glob.glob(os.path.join(pt_dir, "game_*.pt")))
            if pt_files:
                print(f"Loading from per-game .pt files ({len(pt_files)} files, streaming)...")
                print(f"  WARNING: This is slow. Run consolidate_shards.py first for 100x speedup.")
                n_val   = max(1, int(len(pt_files) * val_split))
                n_train = len(pt_files) - n_val
                random.shuffle(pt_files)
                train_files = pt_files[:n_train]
                val_files   = pt_files[n_train:]

                train_max = max_samples
                val_max   = max(1000, int((max_samples or 0) * val_split)) if max_samples else None

                train_ds = StreamingBinaryTransitionDataset(train_files, max_samples=train_max)
                val_ds   = StreamingBinaryTransitionDataset(val_files,   max_samples=val_max)
                train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True, prefetch_factor=2)
                val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True, prefetch_factor=2)
            else:
                print("No .pt files found, falling back to JSONL...")
                use_pt = False

    if not use_shards and (not use_pt or not os.path.isdir(pt_dir)):
        # Fall back to JSONL loading
        if file_list and os.path.isfile(file_list):
            with open(file_list) as f:
                all_files = [line.strip() for line in f if line.strip()]
            print(f"Using file list: {file_list} ({len(all_files)} files)")
        else:
            all_files = sorted(glob.glob(os.path.join(data_dir, "game_*.jsonl")))
        n_val   = max(1, int(len(all_files) * val_split))
        n_train = len(all_files) - n_val
        random.shuffle(all_files)
        train_files = all_files[:n_train]
        val_files   = all_files[n_train:]

        train_max = max_samples
        val_max   = max(1000, int((max_samples or 0) * val_split)) if max_samples else None

        train_ds = TransitionDataset.__new__(TransitionDataset)
        train_ds.files = train_files
        train_ds.max_samples = train_max

        val_ds = TransitionDataset.__new__(TransitionDataset)
        val_ds.files = val_files
        val_ds.max_samples = val_max

        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, prefetch_factor=2)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, prefetch_factor=2)

    # Infer sizes from one sample
    input_size, policy_size = 991, 77   # 75 BUILD_STRUCTURE + 1 BUILD_WONDER + 1 DISCARD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WondersNet(input_size, policy_size, hidden, num_res_blocks, dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: hidden={hidden}, res_blocks={num_res_blocks}, dropout={dropout}, params={total_params:,}")
    file_type = "shards" if use_shards else "files"
    print(f"Train {file_type}: {len(train_files)}, Val {file_type}: {len(val_files)}, device: {device}")

    # Compile model for fused kernels (PyTorch 2.0+, silently skips if unavailable)
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    kl_loss   = nn.KLDivLoss(reduction="batchmean")
    # In 5-player games, ~4 losses for every 1 win: pos_weight = 4
    bce_loss  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=device))

    # Mixed precision: use bfloat16 on Ampere+ (more stable than float16), else float16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(f"AMP enabled: {device.type == 'cuda'} (dtype={amp_dtype})")

    best_val_loss = float("inf")
    best_state_dict = None

    eval_interval = 10  # validate every 10 epochs
    evals_without_improvement = 0
    max_evals_without_improvement = 2  # stop after 2 eval sets without improvement (20 epochs)

    # Profile first batch if enabled
    if profile_enabled:
        print("\n▶ Profiling first epoch (data loading + forward/backward)...")
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=lambda p: print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        )
        prof.__enter__()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_loss = torch.tensor(0.0, device=device)  # Stay on GPU, avoid sync each step
        n_train_batches = 0
        batch_loss_window = torch.tensor(0.0, device=device)  # For frequent printing
        for states, policies, values in train_loader:
            with record_function("forward_backward"):
                states   = states.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True)
                values   = values.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred_policy, pred_value = model(states)
                    loss_p = kl_loss(pred_policy, policies)   # pred_policy is already log-probs
                    loss_v = bce_loss(pred_value, values)
                    loss   = loss_p + 0.5 * loss_v
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.detach()  # No CUDA sync — stays on GPU
            batch_loss_window += loss.detach()
            n_train_batches += 1

            # Print progress every 10 batches
            if n_train_batches % 10 == 0:
                avg_loss_window = (batch_loss_window / 10).item()
                print(f"  Batch {n_train_batches:6d}  loss={avg_loss_window:.4f}")
                batch_loss_window = torch.tensor(0.0, device=device)

            # Exit profiler after first batch if enabled
            if profile_enabled and epoch == 1 and n_train_batches == 1:
                prof.__exit__(None, None, None)
                profile_enabled = False
                print("Profiling complete.\n")

        avg_train_loss = (total_loss / max(n_train_batches, 1)).item()  # Single sync at end of epoch

        # Validate every N epochs
        if epoch % eval_interval == 0 or epoch == 1:
            model.eval()
            val_loss = torch.tensor(0.0, device=device)
            n_val_batches = 0
            with torch.no_grad():
                for states, policies, values in val_loader:
                    states   = states.to(device, non_blocking=True)
                    policies = policies.to(device, non_blocking=True)
                    values   = values.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        pred_policy, pred_value = model(states)
                        loss_p_val = kl_loss(pred_policy, policies)
                        loss_v_val = bce_loss(pred_value, values)
                        val_loss += (loss_p_val + 0.5 * loss_v_val).detach()
                    n_val_batches += 1
            avg_val_loss = (val_loss / max(n_val_batches, 1)).item()

            if avg_val_loss < best_val_loss - 0.005:
                best_val_loss = avg_val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            print(f"Epoch {epoch:4d}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  best={best_val_loss:.4f}  no_improve_evals={evals_without_improvement}")

            if evals_without_improvement >= max_evals_without_improvement:
                print(f"Early stopping at epoch {epoch} ({max_evals_without_improvement * eval_interval} epochs without improvement)")
                break
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d}  train={avg_train_loss:.4f}  (no eval yet)")

    # ── Restore best checkpoint before export ─────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val={best_val_loss:.4f})")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    model.cpu().eval()
    dummy = torch.zeros(1, input_size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["state_embedding"],
        output_names=["policy", "value"],
        dynamic_axes={"state_embedding": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Saved ONNX model to {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="training/",           help="Directory with game_*.jsonl files")
    parser.add_argument("--out",         default="training/model.onnx", help="Output ONNX model path")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch",       type=int,   default=2048,      help="Batch size — larger = better GPU utilization for small FC models")
    parser.add_argument("--val",         type=float, default=0.1,       help="Validation fraction")
    parser.add_argument("--hidden",      type=int,   default=768,       help="Hidden layer width")
    parser.add_argument("--res-blocks",  type=int,   default=8,         help="Number of residual blocks")
    parser.add_argument("--dropout",     type=float, default=0.5,       help="Dropout rate")
    parser.add_argument("--max-samples", type=int,   default=None,      help="Cap training set to this many randomly sampled transitions")
    parser.add_argument("--file-list",   default=None,                 help="Path to text file listing game files to train on (overrides --data glob)")
    parser.add_argument("--workers",     type=int,   default=None,      help="Number of data loading workers (auto-detect if not specified)")
    parser.add_argument("--profile",     action="store_true",           help="Profile first epoch to identify bottlenecks")
    parser.add_argument("--no-pt",       action="store_true",           help="Disable .pt file loading (use JSONL fallback)")
    args = parser.parse_args()

    train(args.data, args.out, args.epochs, args.lr, args.batch, args.val,
          args.hidden, args.res_blocks, args.dropout, args.max_samples, args.file_list,
          use_pt=not args.no_pt, profile_enabled=args.profile, num_workers=args.workers)
