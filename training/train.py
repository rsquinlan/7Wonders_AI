"""
Train a dual-head AlphaZero-style network on 7 Wonders self-play data.

Inputs:
  state_embedding  — 991 floats (see utils.h stateEmbedSize)

Outputs:
  policy  — 75 floats (softmax), trained against MCTS visit distribution
  value   — 1  float  (sigmoid), trained against win/loss outcome

Usage:
  python3.11 training/train.py --out training/model3.onnx --epochs 500
"""

import argparse
import glob
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


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

        # Policy head: predicts a distribution over cards
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, policy_size),
            nn.Softmax(dim=1),
        )

        # Value head: predicts win probability
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.res_tower(self.input_proj(x))
        return self.policy_head(h), self.value_head(h)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        files = sorted(glob.glob(os.path.join(data_dir, "game_*.jsonl")))
        if not files:
            raise FileNotFoundError(f"No game_*.jsonl files found in {data_dir}")

        skipped = 0
        for path in files:
            try:
                lines = []
                for raw in open(path):
                    raw = raw.strip()
                    if raw:
                        lines.append(json.loads(raw))
            except json.JSONDecodeError:
                skipped += 1
                continue  # truncated file — skip entirely

            if not lines:
                skipped += 1
                continue

            result = lines[-1]
            if result.get("type") != "result":
                skipped += 1
                continue  # game didn't finish cleanly

            final_scores = result["final_scores"]
            max_score    = result["max_score"]

            for t in lines[:-1]:
                p   = t["player_index"]
                win = float(final_scores[p] == max_score)
                self.samples.append({
                    "state":  torch.tensor(t["state_embedding"], dtype=torch.float32),
                    "policy": torch.tensor(t["mcts_policy"],     dtype=torch.float32),
                    "value":  torch.tensor([win],                dtype=torch.float32),
                })

        if skipped:
            print(f"Skipped {skipped} incomplete/corrupt game file(s)")
        print(f"Loaded {len(self.samples)} transitions from {len(files) - skipped} games")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["state"], s["policy"], s["value"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir, out_path, epochs, lr, batch_size, val_split, hidden, num_res_blocks, dropout):
    dataset     = TransitionDataset(data_dir)
    input_size  = dataset.samples[0]["state"].shape[0]   # 991
    policy_size = dataset.samples[0]["policy"].shape[0]  # 75

    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = WondersNet(input_size, policy_size, hidden, num_res_blocks, dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: hidden={hidden}, res_blocks={num_res_blocks}, dropout={dropout}, params={total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    kl_loss   = nn.KLDivLoss(reduction="batchmean")
    bce_loss  = nn.BCELoss()

    recent_losses = []  # track last 30 epoch train losses for early stopping

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for states, policies, values in train_loader:
            optimizer.zero_grad()
            pred_policy, pred_value = model(states)
            loss_p = kl_loss(torch.log(pred_policy + 1e-8), policies)
            loss_v = bce_loss(pred_value, values)
            loss   = loss_p + loss_v
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        recent_losses.append(avg_loss)
        if len(recent_losses) > 30:
            recent_losses.pop(0)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for states, policies, values in val_loader:
                    pred_policy, pred_value = model(states)
                    val_loss += (kl_loss(torch.log(pred_policy + 1e-8), policies)
                                 + bce_loss(pred_value, values)).item()
            print(f"Epoch {epoch:4d}  train={avg_loss:.4f}  val={val_loss/len(val_loader):.4f}")

        # Early stopping: if loss hasn't changed more than 0.003 over last 30 epochs
        if len(recent_losses) == 30:
            if max(recent_losses) - min(recent_losses) < 0.003:
                print(f"Early stopping at epoch {epoch} (loss plateau over 30 epochs)")
                break

    # ── Export to ONNX ────────────────────────────────────────────────────────
    model.eval()
    dummy = torch.zeros(1, input_size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["state_embedding"],
        output_names=["policy", "value"],
        dynamic_axes={"state_embedding": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved ONNX model to {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="training/",           help="Directory with game_*.jsonl files")
    parser.add_argument("--out",        default="training/model.onnx", help="Output ONNX model path")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--val",        type=float, default=0.1,  help="Validation fraction")
    parser.add_argument("--hidden",     type=int,   default=512,  help="Hidden layer width")
    parser.add_argument("--res-blocks", type=int,   default=4,    help="Number of residual blocks")
    parser.add_argument("--dropout",    type=float, default=0.3,  help="Dropout rate")
    args = parser.parse_args()

    train(args.data, args.out, args.epochs, args.lr, args.batch, args.val,
          args.hidden, args.res_blocks, args.dropout)
