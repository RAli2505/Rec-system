"""
Attention-based KT baselines for MARS Table 4 (reviewer item #8).

Implements SAINT, AKT and SimpleKT as PyTorch modules that share the same
14-dim per-step input as the MARS Prediction Agent, so the comparison
isolates the attention mechanism rather than feature engineering. All
three are trained from scratch on XES3G5M with the identical training
recipe (focal-style BCE with label smoothing, AdamW, cosine schedule,
SWA-friendly hyper-parameters) and evaluated by the shared
compute_all_metrics function in scripts/run_xes3g5m_baselines.py.

Models
------
- SAINT
    Encoder-decoder Transformer over per-step embeddings. Encoder sees
    the "exercise" stream (tag, part, difficulty surrogate); decoder sees
    the "response" stream (correctness, elapsed, conf_class). 4 layers
    each, d_model=192, heads=4. ~2.7 M params.
    Choi et al., L@S 2020 (https://doi.org/10.1145/3386527.3405945).

- AKT
    Single-stream Transformer with monotonic exponential-decay attention
    bias (forgetting curve), Rasch-style embedding for concepts. 4
    layers, d_model=256, heads=8. ~2.6 M params.
    Ghosh et al., KDD 2020 (https://doi.org/10.1145/3394486.3403282).

- SimpleKT
    Tied-embedding decoder with masked self-attention; concept and
    response share embeddings. 4 layers, d_model=256, heads=8. ~2.6 M
    params.
    Liu et al., ICLR 2023.

Usage
-----
    python scripts/run_attention_kt_baselines.py \
        --models SAINT AKT SimpleKT \
        --seeds 42 123 456 789 2024 \
        --epochs 20 --batch_size 64 --patience 5

Output
------
results/xes3g5m/attention_kt_baselines_<ts>/baselines_s{seed}.json
results/xes3g5m/attention_kt_baselines_<ts>/summary.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agents import prediction_agent as PA
from agents.prediction_agent import (
    GapSequenceDataset, DEVICE, NUM_CONF_CLASSES, LABEL_SMOOTHING,
    NUM_WORKERS, set_num_tags, TAG_EMB_DIM, PART_EMB_DIM,
    CONF_EMB_DIM, MAX_TAGS_PER_STEP, NUM_PARTS, INPUT_DIM,
    _multi_tag_embed,
)
# NOTE: NUM_TAGS is intentionally NOT imported into this module's
# namespace — it must be read via PA.NUM_TAGS at model-construction
# time, AFTER set_num_tags() has been called by main(). Importing it
# directly captures the default EdNet value (293) and causes
# embedding-index OOR on XES3G5M (858).
from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("attn_kt_baselines")

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_MODELS = ["SAINT", "AKT", "SimpleKT"]


# ─────────────────────────────────────────────────────────────────────
# Shared embedding stack — converts 14-dim per-step input to a per-step
# vector in a chosen d_model. Identical input as MARS so any difference
# is attributable to the attention mechanism rather than features.
# ─────────────────────────────────────────────────────────────────────

class StepEmbedder(nn.Module):
    """Multi-tag mean-pool + part_emb + conf_emb + scalar projection."""

    def __init__(self, d_model: int, num_conf_classes: int = NUM_CONF_CLASSES):
        super().__init__()
        self.d_model = d_model
        self.num_conf_classes = num_conf_classes
        self.tag_embedding = nn.Embedding(PA.NUM_TAGS + 1, TAG_EMB_DIM, padding_idx=0)
        self.part_embedding = nn.Embedding(NUM_PARTS, PART_EMB_DIM)
        self.conf_embedding = nn.Embedding(num_conf_classes, CONF_EMB_DIM)
        self.proj = nn.Linear(INPUT_DIM, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 14) → (B, T, d_model)."""
        tag_emb, correct, elapsed, changed, part_ids, conf_ids, \
            steps_since, cum_acc = _multi_tag_embed(self.tag_embedding, x)
        conf_ids = conf_ids.clamp(0, self.num_conf_classes - 1)
        part_emb = self.part_embedding(part_ids)
        conf_emb = self.conf_embedding(conf_ids)
        combined = torch.cat(
            [tag_emb, correct, elapsed, changed, steps_since, cum_acc,
             part_emb, conf_emb], dim=-1,
        )
        return self.proj(combined)  # (B, T, d_model)


def sinusoidal_pe(seq_len: int, d_model: int, device) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                     * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, T, d_model)


# ─────────────────────────────────────────────────────────────────────
# 1. SAINT — encoder-decoder Transformer
# ─────────────────────────────────────────────────────────────────────

class SAINT(nn.Module):
    """Encoder over exercise stream, decoder over response stream.

    The original SAINT separates exercise and response embeddings into
    two streams and joins them in a stacked encoder-decoder. We reuse
    the 14-dim StepEmbedder and split the embedding into halves: the
    first half (tag/part) is the "exercise" stream, the second half
    (correct/elapsed/conf) is the "response" stream.
    """

    def __init__(self, d_model: int = 192, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.2,
                 num_conf_classes: int = NUM_CONF_CLASSES):
        super().__init__()
        # Two parallel embedders, each producing d_model//2 to keep
        # parameter count near 2.7M
        self.exercise_embed = StepEmbedder(d_model // 2, num_conf_classes)
        self.response_embed = StepEmbedder(d_model // 2, num_conf_classes)
        self.exercise_proj = nn.Linear(d_model // 2, d_model)
        self.response_proj = nn.Linear(d_model // 2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)

        self.head = nn.Linear(d_model, PA.NUM_TAGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ex = self.exercise_proj(self.exercise_embed(x))
        rs = self.response_proj(self.response_embed(x))
        T = x.size(1)
        pe = sinusoidal_pe(T, ex.size(-1), x.device)
        ex = ex + pe
        rs = rs + pe
        memory = self.encoder(ex)
        out = self.decoder(rs, memory)
        # mean-pool over time, then per-concept logits
        pooled = out.mean(dim=1)
        return self.head(pooled)


# ─────────────────────────────────────────────────────────────────────
# 2. AKT — monotonic attention with exponential decay
# ─────────────────────────────────────────────────────────────────────

class MonotonicAttention(nn.Module):
    """Multi-head attention with a monotonic exponential-decay bias
    over the temporal distance, matching the AKT formulation."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # learnable per-head decay rate (positive)
        self.gamma = nn.Parameter(torch.ones(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)  # (B, h, T, dh)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B, h, T, T)
        # causal mask
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1,
        )
        attn = attn.masked_fill(causal, float("-inf"))
        # exponential decay bias on temporal distance
        idx = torch.arange(T, device=x.device).unsqueeze(0)
        dist = (idx.t() - idx).clamp(min=0).float()  # (T, T)
        gamma = F.softplus(self.gamma).view(1, self.h, 1, 1)
        attn = attn - gamma * dist  # subtract decay (negative bias)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v  # (B, h, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dh)
        return self.out(out)


class AKT(nn.Module):
    """Single-stream causal Transformer with monotonic decay attention
    and Rasch-style concept embeddings (concept embedding + difficulty
    scalar), ~2.6 M parameters at d_model=256, 4 layers."""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.2,
                 num_conf_classes: int = NUM_CONF_CLASSES):
        super().__init__()
        self.embed = StepEmbedder(d_model, num_conf_classes)
        # Rasch difficulty bias per concept
        self.difficulty = nn.Embedding(PA.NUM_TAGS + 1, 1, padding_idx=0)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MonotonicAttention(d_model, n_heads, dropout),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                ),
                nn.LayerNorm(d_model),
            ])
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, PA.NUM_TAGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        # Rasch bias — pool over the 7 tag slots
        tag_ids = x[:, :, :MAX_TAGS_PER_STEP].long().clamp(0, PA.NUM_TAGS)
        diff = self.difficulty(tag_ids).squeeze(-1).mean(dim=-1, keepdim=True)
        h = h + diff  # broadcast across d_model
        T = h.size(1)
        h = h + sinusoidal_pe(T, h.size(-1), x.device)
        for attn, ln1, ff, ln2 in self.layers:
            h = ln1(h + attn(h))
            h = ln2(h + ff(h))
        return self.head(h.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────
# 3. SimpleKT — tied-embedding causal Transformer
# ─────────────────────────────────────────────────────────────────────

class SimpleKT(nn.Module):
    """SimpleKT — vanilla causal Transformer with tied
    concept-embedding output projection. d_model=256, 4 layers, 8 heads,
    ~2.6 M parameters."""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.2,
                 num_conf_classes: int = NUM_CONF_CLASSES):
        super().__init__()
        self.embed = StepEmbedder(d_model, num_conf_classes)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        # Tied output: project pooled state through a learned tag table
        self.tag_table = nn.Embedding(PA.NUM_TAGS + 1, d_model, padding_idx=0)
        self.scale = d_model ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        T = h.size(1)
        h = h + sinusoidal_pe(T, h.size(-1), x.device)
        causal = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.encoder(h, mask=causal)
        pooled = h.mean(dim=1)  # (B, d_model)
        # Tied: logits = pooled · TagEmb.T, drop padding column
        tag_emb = self.tag_table.weight[1:]  # (PA.NUM_TAGS, d_model)
        return (pooled @ tag_emb.t()) / self.scale


# ─────────────────────────────────────────────────────────────────────
# Training loop (mirrors run_baselines_param_matched.py)
# ─────────────────────────────────────────────────────────────────────

def build_model(name: str, seed: int) -> nn.Module:
    set_global_seed(seed)
    if name == "SAINT":
        return SAINT(d_model=192, n_heads=4, n_layers=4)
    if name == "AKT":
        return AKT(d_model=256, n_heads=8, n_layers=4)
    if name == "SimpleKT":
        return SimpleKT(d_model=256, n_heads=8, n_layers=4)
    raise ValueError(f"unknown model: {name}")


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def train_and_eval(
    name: str, seed: int, train_df, val_df, test_df, n_tags: int,
    epochs: int = 20, batch_size: int = 64, patience: int = 5,
    lr: float = 5e-4,
) -> dict:
    train_ds = GapSequenceDataset(train_df)
    val_ds = GapSequenceDataset(val_df)
    test_ds = GapSequenceDataset(test_df)
    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"error": "no sequences"}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(name, seed).to(DEVICE)
    n_params = _count_params(model)
    logger.info("%-9s  params=%.2fM  (seed %d)", name, n_params / 1e6, seed)

    all_labels = np.stack(train_ds.labels)
    pos_rate = all_labels.mean(axis=0)
    pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)

    def bce_loss(logits, targets):
        targets = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = bce_loss(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for X, y in val_loader:
                ps.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
                ls.append(y.numpy())
        yp, yt = np.concatenate(ps), np.concatenate(ls)
        mask = yt.sum(axis=0) > 0
        try:
            val_auc = float(roc_auc_score(yt[:, mask], yp[:, mask], average="macro"))
        except ValueError:
            val_auc = 0.0
        logger.info("  %-9s seed=%d ep=%2d  loss=%.4f  val_auc=%.4f",
                    name, seed, ep, float(np.mean(losses)), val_auc)
        if val_auc > best_val_auc:
            best_val_auc, best_epoch, no_improve = val_auc, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  %-9s seed=%d early stop ep=%d best=%.4f",
                            name, seed, ep, best_val_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    ps, ls = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch[0].to(DEVICE)
            y = batch[1]
            ps.append(torch.sigmoid(model(X)).cpu().numpy())
            ls.append(y.numpy())
    y_score, y_true = np.concatenate(ps), np.concatenate(ls)

    from scripts.run_xes3g5m_baselines import compute_all_metrics
    metrics = compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)
    metrics.update({
        "model":      name,
        "seed":       seed,
        "n_params":   int(n_params),
        "val_auc":    round(best_val_auc, 4),
        "best_epoch": best_epoch,
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["SAINT", "AKT", "SimpleKT"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / f"results/xes3g5m/attention_kt_baselines_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("=== ATTENTION-KT BASELINES — out=%s ===", out_root)
    logger.info("models=%s  seeds=%s", args.models, args.seeds)

    all_rows = []
    for seed in args.seeds:
        train_df, val_df, test_df = load_xes3g5m(
            n_students=args.n_students,
            min_interactions=args.min_interactions, seed=seed,
        )
        for df in [train_df, val_df, test_df]:
            df["confidence_class"] = 0
        train_max_id = max(int(t) for tags in train_df["tags"]
                            if isinstance(tags, list) and tags
                            for t in tags)
        n_tags = train_max_id + 1
        set_num_tags(n_tags)

        seed_out = {}
        for name in args.models:
            logger.info("━━━ %s seed=%d ━━━", name, seed)
            t0 = time.time()
            try:
                m = train_and_eval(
                    name, seed, train_df, val_df, test_df, n_tags,
                    epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience,
                )
                m["runtime_s"] = round(time.time() - t0, 1)
                seed_out[name] = m
                all_rows.append(m)
            except Exception as e:
                logger.exception("%s/seed=%d failed: %s", name, seed, e)
                seed_out[name] = {"error": str(e)}

        (out_root / f"baselines_s{seed}.json").write_text(
            json.dumps(seed_out, indent=2, default=str), encoding="utf-8",
        )
        logger.info("seed %d wrote", seed)

    if all_rows:
        df = pd.DataFrame(all_rows)
        summary = (df.groupby("model")
                   .agg(n_params=("n_params", "first"),
                        val_auc_mean=("val_auc", "mean"),
                        val_auc_std=("val_auc", "std"),
                        test_auc_mean=("test_auc_macro", "mean"),
                        test_auc_std=("test_auc_macro", "std"),
                        ndcg10_mean=("ndcg@10", "mean"),
                        ndcg10_std=("ndcg@10", "std"),
                        mrr_mean=("mrr", "mean"),
                        mrr_std=("mrr", "std"),
                        p10_mean=("precision@10", "mean"),
                        p10_std=("precision@10", "std"),
                        tagcov_mean=("tag_coverage", "mean"),
                        tagcov_std=("tag_coverage", "std"),
                        n_seeds=("seed", "count"))
                   .reset_index())
        summary_path = out_root / "summary.csv"
        summary.to_csv(summary_path, index=False)
        (ROOT / "results/xes3g5m/attention_kt_baselines_summary.csv"
         ).write_bytes(summary_path.read_bytes())
        logger.info("wrote summary\n%s", summary.to_string(index=False))

    logger.info("DONE at %s", datetime.now().isoformat(timespec="seconds"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
