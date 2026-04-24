"""
Prediction Agent for MARS.

Predicts future knowledge gaps per tag using a sequence encoder trained on
student interactions.  Each interaction is encoded as a 51-dimensional vector:

    tag_emb(32) + correct(1) + elapsed_norm(1) + changed(1)
    + part_emb(8) + confidence_emb(8) = 51

Supports three encoder architectures (configurable via ``model_type``):
  - **LSTM** (default): O(1) incremental update, ideal for continuous pipeline
  - **GRU**: lighter alternative, similar properties to LSTM
  - **Transformer**: SOTA baseline for comparison, requires full re-computation

The model consumes the last ``SEQ_LEN`` (50) interactions and outputs
a (293,) sigmoid vector — the probability that the student will fail
each tag within the next ``HORIZON`` (10) questions.

Loss: BCEWithLogitsLoss | Optimizer: Adam lr=1e-3 | Early stopping: patience 5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .base_agent import BaseAgent
from .confidence_agent import DEFAULT_CONFIDENCE_N_CLASSES

logger = logging.getLogger("mars.agent.prediction")

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

NUM_TAGS = 293          # default = EdNet TOEIC; override via set_num_tags() for other datasets
NUM_PARTS = 7           # TOEIC parts 1-7
NUM_CONF_CLASSES = DEFAULT_CONFIDENCE_N_CLASSES


def set_num_tags(n: int) -> None:
    """Override the global NUM_TAGS used by all subsequently-built datasets
    and models in this module.

    Call this BEFORE constructing any ``GapSequenceDataset`` or ``create_model``
    when training on a dataset whose concept-space differs from EdNet
    (e.g. XES3G5M has 858 concepts, not 293). Existing instances are not
    retroactively updated.

    Parameters
    ----------
    n : int
        New concept-space size. Must be >= 1.
    """
    global NUM_TAGS
    if n < 1:
        raise ValueError(f"num_tags must be >= 1, got {n}")
    NUM_TAGS = int(n)
    logger.info("NUM_TAGS set to %d", NUM_TAGS)
SEQ_LEN = 100           # window of recent interactions (was 50 → more context)
HORIZON = 20            # predict failures within next N questions (wider = more signal)
MAX_TAGS_PER_STEP = 7   # max tags per question in EdNet

TAG_EMB_DIM = 48        # increased for multi-tag (was 32)
PART_EMB_DIM = 8
CONF_EMB_DIM = 8
SCALAR_FEATURES = 5     # correct, elapsed_norm, changed, steps_since_last_tag, cumulative_accuracy
INPUT_DIM = TAG_EMB_DIM + SCALAR_FEATURES + PART_EMB_DIM + CONF_EMB_DIM  # 69
HIDDEN_DIM = 256        # increased (was 128)
NUM_LAYERS = 2
DROPOUT = 0.25          # slightly lower for larger model
ATTN_HEADS = 4          # multi-head attention over LSTM outputs
LABEL_SMOOTHING = 0.05  # smooth binary labels to reduce overconfidence

LEARNING_RATE = 5e-4    # lower base LR, cosine scheduler will handle warmup
BATCH_SIZE = 256         # smaller batch for larger model (memory)
EPOCHS = 50
PATIENCE = 8            # more patience — let scheduler find better minima
# Feature layout per timestep: [tag0..tag6, correct, elapsed, changed, part_id, conf_class,
#                                steps_since_last_tag, cumulative_accuracy]
# = 7 + 7 = 14 columns
FEATURES_PER_STEP = MAX_TAGS_PER_STEP + 7  # 14
NUM_WORKERS = int(__import__("os").getenv(
    "MARS_NUM_WORKERS",
    "2" if __import__("sys").platform == "win32" else "4",
))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class PredictedGap:
    """A single predicted knowledge gap."""
    tag_id: int
    probability: float
    tag_name: str = ""


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

class GapSequenceDataset(Dataset):
    """
    Converts a chronologically-sorted interaction DataFrame into
    (sequence, label) pairs for the LSTM.

    Each sample:
      X : (SEQ_LEN, FEATURES_PER_STEP) — [tag0..tag6, correct, elapsed, changed, part_id, conf_class]
          Multi-tag: up to MAX_TAGS_PER_STEP tag IDs per timestep (0-padded).
      y : (NUM_TAGS,) — binary vector of failed tags in next HORIZON questions
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        seq_len: int = SEQ_LEN,
        horizon: int = HORIZON,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.sequences: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

        self._build(interactions)

    # ── helpers ──

    @staticmethod
    def _primary_tag(tags) -> int:
        """Extract the first (primary) tag from a tags field."""
        if isinstance(tags, list) and len(tags) > 0:
            return int(tags[0])
        if isinstance(tags, str):
            parts = tags.replace(";", ",").split(",")
            for p in parts:
                p = p.strip()
                if p.isdigit():
                    return int(p)
        if isinstance(tags, (int, float)) and not np.isnan(tags):
            return int(tags)
        return 0

    @staticmethod
    def _all_tags(tags) -> list[int]:
        """Extract all tags from a tags field."""
        if isinstance(tags, list):
            return [int(t) for t in tags]
        if isinstance(tags, str):
            parts = tags.replace(";", ",").split(",")
            return [int(p.strip()) for p in parts if p.strip().isdigit()]
        if isinstance(tags, (int, float)) and not np.isnan(tags):
            return [int(tags)]
        return []

    @staticmethod
    def _padded_tags(tags, max_tags: int = MAX_TAGS_PER_STEP) -> np.ndarray:
        """Extract all tags, pad/truncate to max_tags, clamp to valid range."""
        all_t = GapSequenceDataset._all_tags(tags)
        result = np.zeros(max_tags, dtype=np.float32)
        for i, t in enumerate(all_t[:max_tags]):
            result[i] = float(np.clip(t, 0, NUM_TAGS - 1))
        return result

    def _build(self, df: pd.DataFrame) -> None:
        """Build sequences per user."""
        required = {"user_id", "timestamp", "correct", "elapsed_time", "tags"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Normalise elapsed_time globally (ms → [0, 1] via clipped division)
        median_elapsed = df["elapsed_time"].median()
        if median_elapsed == 0 or pd.isna(median_elapsed):
            median_elapsed = 15_000.0

        for _uid, grp in df.groupby("user_id"):
            if len(grp) < self.seq_len + self.horizon:
                continue

            rows = grp.reset_index(drop=True)

            # Pre-compute per-row arrays — multi-tag: (N, MAX_TAGS_PER_STEP)
            tag_matrix = np.stack(rows["tags"].apply(self._padded_tags).values)  # (N, 7)
            correct = rows["correct"].astype(float).values
            elapsed_norm = (rows["elapsed_time"] / median_elapsed).clip(0, 5).fillna(1.0).values
            changed = rows["changed_answer"].astype(float).values if "changed_answer" in rows.columns else np.zeros(len(rows))
            part_ids = (rows["part_id"].fillna(1).astype(int).values - 1).clip(0, NUM_PARTS - 1) if "part_id" in rows.columns else np.zeros(len(rows), dtype=int)
            conf_cls = rows["confidence_class"].values.astype(int) if "confidence_class" in rows.columns else np.zeros(len(rows), dtype=int)

            # All tags per row (for labels)
            all_tags_per_row = rows["tags"].apply(self._all_tags).tolist()

            # === NEW FEATURES ===
            # 1. steps_since_last_tag: how many steps since the primary tag was last seen
            #    Captures forgetting effect — larger gaps → more likely to fail
            steps_since = np.zeros(len(rows), dtype=np.float32)
            tag_last_seen: dict[int, int] = {}
            for idx in range(len(rows)):
                primary = int(tag_matrix[idx, 0])
                if primary in tag_last_seen:
                    steps_since[idx] = min((idx - tag_last_seen[primary]) / 50.0, 5.0)
                else:
                    steps_since[idx] = 5.0  # never seen before = max
                tag_last_seen[primary] = idx

            # 2. cumulative_accuracy: running accuracy up to this point
            #    Captures overall student performance trend
            cum_correct = np.cumsum(correct)
            cum_count = np.arange(1, len(rows) + 1, dtype=np.float32)
            cumulative_acc = (cum_correct / cum_count).astype(np.float32)

            n = len(rows)
            # Stride = horizon//2: overlapping windows for more training data
            # Label leakage is minimal since labels are computed from FUTURE data
            stride = max(1, self.horizon // 2)
            for i in range(0, n - self.seq_len - self.horizon + 1, stride):
                sl = slice(i, i + self.seq_len)
                # Input: [tag0..tag6, correct, elapsed, changed, part_id, conf_class,
                #         steps_since_last_tag, cumulative_accuracy] = 14 cols
                seq = np.column_stack([
                    tag_matrix[sl],                                  # (seq_len, 7)
                    correct[sl].reshape(-1, 1),                      # (seq_len, 1)
                    elapsed_norm[sl].reshape(-1, 1),                 # (seq_len, 1)
                    changed[sl].reshape(-1, 1),                      # (seq_len, 1)
                    part_ids[sl].reshape(-1, 1).astype(np.float32),  # (seq_len, 1)
                    conf_cls[sl].reshape(-1, 1).astype(np.float32),  # (seq_len, 1)
                    steps_since[sl].reshape(-1, 1),                  # (seq_len, 1) NEW
                    cumulative_acc[sl].reshape(-1, 1),               # (seq_len, 1) NEW
                ]).astype(np.float32)  # (seq_len, 14)

                # Label: tags failed in next HORIZON questions
                label = np.zeros(NUM_TAGS, dtype=np.float32)
                future_start = i + self.seq_len
                future_end = future_start + self.horizon
                for j in range(future_start, min(future_end, n)):
                    if not correct[j]:  # failed
                        for t in all_tags_per_row[j]:
                            if 0 <= t < NUM_TAGS:
                                label[t] = 1.0

                self.sequences.append(seq)
                self.labels.append(label)

        logger.info(
            "Built %d sequences (seq_len=%d, horizon=%d, features=%d) from %d users",
            len(self.sequences), self.seq_len, self.horizon,
            FEATURES_PER_STEP, df["user_id"].nunique(),
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.from_numpy(self.labels[idx]),
        )


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────

class GapPredictionLSTM(nn.Module):
    """
    LSTM with multi-tag input + attention pooling for per-tag failure prediction.

    Input per timestep (FEATURES_PER_STEP=12 values):
        tag0..tag6 (int) → sum of Embedding(293, 48) for non-zero tags
        correct (float, 0/1)
        elapsed_norm (float)
        changed (float, 0/1)
        part_id (int 0-6) → Embedding(7, 8)
        conf_class (int 0-5) → Embedding(6, 8)

    After embedding: 48 + 5 + 8 + 8 = 69 → BiLSTM(69, 256, 2) → 512-dim
    Attention pooling → MLP head → Linear(256, 293)
    """

    def __init__(self, num_conf_classes: int = NUM_CONF_CLASSES) -> None:
        super().__init__()
        self.num_conf_classes = num_conf_classes

        self.tag_embedding = nn.Embedding(NUM_TAGS + 1, TAG_EMB_DIM, padding_idx=0)
        self.part_embedding = nn.Embedding(NUM_PARTS, PART_EMB_DIM)
        self.confidence_embedding = nn.Embedding(num_conf_classes, CONF_EMB_DIM)

        self.input_dropout = nn.Dropout(0.1)

        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = HIDDEN_DIM * 2  # bidirectional doubles output

        # Attention pooling: learn which timesteps matter most for prediction
        self.attn_w = nn.Linear(lstm_out_dim, 1, bias=False)

        # Deeper MLP head with residual connection
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_TAGS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, FEATURES_PER_STEP)
            Columns: [tag0..tag6, correct, elapsed, changed, part_id, conf_class]

        Returns
        -------
        (batch, NUM_TAGS) logits
        """
        # Multi-tag: columns 0..6 are tag IDs (0-padded)
        tag_ids = x[:, :, :MAX_TAGS_PER_STEP].long().clamp(0, NUM_TAGS)  # (B, T, 7)
        correct = x[:, :, MAX_TAGS_PER_STEP:MAX_TAGS_PER_STEP + 1]      # (B, T, 1)
        elapsed = x[:, :, MAX_TAGS_PER_STEP + 1:MAX_TAGS_PER_STEP + 2]  # (B, T, 1)
        changed = x[:, :, MAX_TAGS_PER_STEP + 2:MAX_TAGS_PER_STEP + 3]  # (B, T, 1)
        part_ids = x[:, :, MAX_TAGS_PER_STEP + 3].long().clamp(0, NUM_PARTS - 1)
        conf_ids = x[:, :, MAX_TAGS_PER_STEP + 4].long().clamp(0, self.num_conf_classes - 1)
        steps_since = x[:, :, MAX_TAGS_PER_STEP + 5:MAX_TAGS_PER_STEP + 6]
        cumulative_acc = x[:, :, MAX_TAGS_PER_STEP + 6:MAX_TAGS_PER_STEP + 7]

        # Sum-of-embeddings for all tags (masked: ignore padding=0)
        tag_emb_all = self.tag_embedding(tag_ids)     # (B, T, 7, 48)
        tag_mask = (tag_ids > 0).unsqueeze(-1).float()  # (B, T, 7, 1)
        n_tags = tag_mask.sum(dim=2).clamp(min=1)     # (B, T, 1)
        tag_emb = (tag_emb_all * tag_mask).sum(dim=2) / n_tags  # (B, T, 48) mean pooling

        part_emb = self.part_embedding(part_ids)       # (B, T, 8)
        conf_emb = self.confidence_embedding(conf_ids) # (B, T, 8)

        # Concatenate: (B, T, 48+5+8+8) = (B, T, 69)
        combined = torch.cat(
            [tag_emb, correct, elapsed, changed, steps_since, cumulative_acc, part_emb, conf_emb],
            dim=-1,
        )
        combined = self.input_dropout(combined)

        lstm_out, _ = self.lstm(combined)              # (B, T, 512) bidirectional

        # Attention pooling: weighted combination of all timesteps
        attn_scores = self.attn_w(lstm_out).squeeze(-1)   # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, 256)

        logits = self.fc(context)                      # (B, 293)
        return logits  # raw logits; use BCEWithLogitsLoss for training


def _multi_tag_embed(tag_embedding, x):
    """Shared multi-tag embedding: mean-pool non-zero tag embeddings."""
    tag_ids = x[:, :, :MAX_TAGS_PER_STEP].long().clamp(0, NUM_TAGS)
    correct = x[:, :, MAX_TAGS_PER_STEP:MAX_TAGS_PER_STEP + 1]
    elapsed = x[:, :, MAX_TAGS_PER_STEP + 1:MAX_TAGS_PER_STEP + 2]
    changed = x[:, :, MAX_TAGS_PER_STEP + 2:MAX_TAGS_PER_STEP + 3]
    part_ids = x[:, :, MAX_TAGS_PER_STEP + 3].long().clamp(0, NUM_PARTS - 1)
    conf_ids = x[:, :, MAX_TAGS_PER_STEP + 4].long()
    steps_since = x[:, :, MAX_TAGS_PER_STEP + 5:MAX_TAGS_PER_STEP + 6]
    cumulative_acc = x[:, :, MAX_TAGS_PER_STEP + 6:MAX_TAGS_PER_STEP + 7]

    tag_emb_all = tag_embedding(tag_ids)
    tag_mask = (tag_ids > 0).unsqueeze(-1).float()
    n_tags = tag_mask.sum(dim=2).clamp(min=1)
    tag_emb = (tag_emb_all * tag_mask).sum(dim=2) / n_tags

    return tag_emb, correct, elapsed, changed, part_ids, conf_ids, steps_since, cumulative_acc


class GapPredictionGRU(nn.Module):
    """
    GRU variant — same architecture as LSTM but with GRU cells.

    Fewer parameters (no cell state), slightly faster training.
    Same input/output format for fair comparison.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_conf_classes: int = NUM_CONF_CLASSES,
    ) -> None:
        super().__init__()
        self.num_conf_classes = num_conf_classes

        self.tag_embedding = nn.Embedding(NUM_TAGS + 1, TAG_EMB_DIM, padding_idx=0)
        self.part_embedding = nn.Embedding(NUM_PARTS, PART_EMB_DIM)
        self.confidence_embedding = nn.Embedding(num_conf_classes, CONF_EMB_DIM)

        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, NUM_TAGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tag_emb, correct, elapsed, changed, part_ids, conf_ids, steps_since, cumulative_acc = _multi_tag_embed(self.tag_embedding, x)
        conf_ids = conf_ids.clamp(0, self.num_conf_classes - 1)
        part_emb = self.part_embedding(part_ids)
        conf_emb = self.confidence_embedding(conf_ids)
        combined = torch.cat(
            [tag_emb, correct, elapsed, changed, steps_since, cumulative_acc, part_emb, conf_emb],
            dim=-1,
        )

        gru_out, _ = self.gru(combined)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)


class GapPredictionTransformer(nn.Module):
    """
    SAINT-inspired Transformer for seq-to-set knowledge gap prediction.

    Architecture draws from SAINT (Choi et al., 2020) and AKT (Ghosh et al., 2020):
      - Separate exercise stream (tags, part, difficulty) and response stream
        (correct, elapsed, changed, confidence) merged via gated fusion
      - Deep encoder (4 layers, 256 dim) with pre-norm for stable training
      - Sinusoidal + learnable positional encoding
      - Multi-query attention pooling over all timesteps (not just last)
      - MLP head with residual connection

    Same input/output format as LSTM/GRU for fair comparison.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = SEQ_LEN,
        num_conf_classes: int = NUM_CONF_CLASSES,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_conf_classes = num_conf_classes

        # ── Exercise stream embeddings ──
        self.tag_embedding = nn.Embedding(NUM_TAGS + 1, TAG_EMB_DIM, padding_idx=0)
        self.part_embedding = nn.Embedding(NUM_PARTS, PART_EMB_DIM)

        # ── Response stream embeddings ──
        self.confidence_embedding = nn.Embedding(num_conf_classes, CONF_EMB_DIM)
        self.correct_embedding = nn.Embedding(2, 16)  # correct/incorrect

        # ── Projections to d_model ──
        exercise_dim = TAG_EMB_DIM + PART_EMB_DIM + 1  # tag + part + steps_since
        response_dim = 16 + CONF_EMB_DIM + 3  # correct_emb + conf + elapsed + changed + cum_acc
        self.exercise_proj = nn.Linear(exercise_dim, d_model)
        self.response_proj = nn.Linear(response_dim, d_model)

        # ── Gated fusion of exercise + response ──
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # ── Positional encoding (sinusoidal + learnable) ──
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self._init_sinusoidal_pe(seq_len, d_model)
        self.pos_learnable = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(d_model)

        # ── Transformer encoder (pre-norm for stable deep training) ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for deeper models
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Multi-query attention pooling ──
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True,
        )

        # ── MLP head ──
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(HIDDEN_DIM, NUM_TAGS),
        )

    def _init_sinusoidal_pe(self, max_len: int, d_model: int) -> None:
        """Initialize sinusoidal positional encoding (not learned)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.pos_encoding.data.copy_(pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tag_emb, correct, elapsed, changed, part_ids, conf_ids, steps_since, cumulative_acc = _multi_tag_embed(self.tag_embedding, x)
        conf_ids = conf_ids.clamp(0, self.num_conf_classes - 1)
        part_emb = self.part_embedding(part_ids)
        conf_emb = self.confidence_embedding(conf_ids)
        correct_emb = self.correct_embedding(correct.squeeze(-1).long().clamp(0, 1))

        # ── Exercise stream: what was asked ──
        exercise = torch.cat([tag_emb, part_emb, steps_since], dim=-1)
        exercise = self.exercise_proj(exercise)  # (B, T, d_model)

        # ── Response stream: how the student answered ──
        response = torch.cat([correct_emb, conf_emb, elapsed, changed, cumulative_acc], dim=-1)
        response = self.response_proj(response)  # (B, T, d_model)

        # ── Gated fusion ──
        gate_input = torch.cat([exercise, response], dim=-1)
        gate_weight = self.gate(gate_input)
        h = gate_weight * exercise + (1 - gate_weight) * response

        # ── Add positional encoding ──
        seq_len = h.size(1)
        h = h + self.pos_encoding[:, :seq_len, :] + self.pos_learnable[:, :seq_len, :]
        h = self.input_norm(h)
        h = self.input_dropout(h)

        # ── Causal mask ──
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(h.device)
        h = self.transformer(h, mask=causal_mask)  # (B, T, d_model)

        # ── Multi-query attention pooling ──
        query = self.pool_query.expand(h.size(0), -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pool_attn(query, h, h)  # (B, 1, d_model)
        pooled = pooled.squeeze(1)  # (B, d_model)

        return self.fc(pooled)  # (B, NUM_TAGS)


# ──────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "lstm": GapPredictionLSTM,
    "gru": GapPredictionGRU,
    "transformer": GapPredictionTransformer,
}


def create_model(model_type: str = "lstm", **kwargs) -> nn.Module:
    """Create a gap prediction model by name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)


# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class PredictionAgent(BaseAgent):
    """
    Sequence-based knowledge gap prediction agent.

    Supports LSTM (default), GRU, and Transformer encoders.
    Predicts which tags a student is likely to fail in the near future,
    enabling proactive recommendations.
    """

    name = "prediction"
    REQUIRED_COLUMNS = {
        "train": ["user_id", "timestamp", "tags", "correct",
                  "elapsed_time", "changed_answer", "part_id", "confidence_class"],
        "predict": ["tags", "correct", "elapsed_time", "changed_answer",
                    "part_id", "confidence_class"],
    }

    def __init__(self, model_type: str | None = None) -> None:
        super().__init__()
        # Seed fixing
        from .utils import set_global_seed
        set_global_seed(self.global_seed)

        # Model type from argument or config
        self.model_type = model_type or self._config.get("model", "lstm")
        self._num_conf_classes = int(
            self._config.get("n_confidence_classes", NUM_CONF_CLASSES)
        )

        # Config-driven parameters (use model-specific section if available)
        model_cfg = self._config.get(self.model_type, self._config.get("lstm", {}))
        self._batch_size = model_cfg.get("batch_size", BATCH_SIZE)
        self._epochs = model_cfg.get("epochs", EPOCHS)
        self._lr = model_cfg.get("learning_rate", LEARNING_RATE)
        self._patience = model_cfg.get("patience", PATIENCE)
        self._gap_threshold = self._config.get("gap_threshold", 0.5)

        self.model: nn.Module | None = None
        self._models_dir = Path("models")
        self._user_sequences: dict[str, list[np.ndarray]] = {}  # user → recent interactions
        self._training_metrics: dict[str, Any] = {}
        self._tag_names: dict[int, str] = {}  # tag_id → human-readable name

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def set_confidence_schema(self, n_classes: int) -> None:
        """Update the confidence embedding cardinality for this agent."""
        self._num_conf_classes = int(max(2, n_classes))

    def initialize(self, **kwargs: Any) -> None:
        """Load a pre-trained model or train from scratch."""
        model_path = self._models_dir / f"gap_{self.model_type}.pt"
        if model_path.exists():
            self._load_model(model_path)
            logger.info("Loaded pre-trained LSTM from %s", model_path)
        elif "interactions_df" in kwargs:
            self.train(kwargs["interactions_df"])

    def receive_message(self, message):
        super().receive_message(message)
        action = message.data.get("action")
        if action == "predict_gaps":
            return self.predict_gaps(
                user_id=message.data.get("user_id"),
                recent=message.data.get("recent"),
            )
        if action == "update_state":
            return self.update_state(
                user_id=message.data.get("user_id"),
                interaction=message.data.get("interaction"),
            )
        return None

    # ──────────────────────────────────────────────────────
    # 1. Training
    # ──────────────────────────────────────────────────────

    def train(
        self,
        interactions_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        patience: int | None = None,
    ) -> dict[str, float]:
        """
        Train the GapPredictionLSTM on interaction sequences.

        Sets global seed for reproducibility before training.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Training interactions (chronologically sorted per user).
        val_df : pd.DataFrame, optional
            Validation set. If None, uses last 15% of training users.
        epochs : int
        batch_size : int
        lr : float
        patience : int
            Early stopping patience (epochs without val loss improvement).

        Returns
        -------
        dict with training metrics: train_loss, val_loss, val_auc, best_epoch
        """
        # Resolve parameters from config
        if epochs is None:
            epochs = self._epochs
        if batch_size is None:
            batch_size = self._batch_size
        if lr is None:
            lr = self._lr
        if patience is None:
            patience = self._patience

        from .utils import set_global_seed
        set_global_seed(self.global_seed)

        self._set_processing()
        self._validate_dataframe(interactions_df, "train")
        logger.info("Building training sequences...")

        # Split into train/val if no val_df provided
        if val_df is None:
            users = interactions_df["user_id"].unique()
            n_val = max(1, int(len(users) * 0.15))
            rng = np.random.RandomState(self.global_seed)
            val_users = set(rng.choice(users, size=n_val, replace=False))
            val_df = interactions_df[interactions_df["user_id"].isin(val_users)]
            train_df = interactions_df[~interactions_df["user_id"].isin(val_users)]
        else:
            train_df = interactions_df

        train_dataset = GapSequenceDataset(train_df)
        val_dataset = GapSequenceDataset(val_df)

        if len(train_dataset) == 0:
            logger.warning("No training sequences built — not enough data")
            self._set_idle()
            return {"train_loss": float("nan"), "val_loss": float("nan")}

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        ) if len(val_dataset) > 0 else None

        logger.info(
            "Train: %d sequences, Val: %d sequences",
            len(train_dataset), len(val_dataset) if val_dataset else 0,
        )

        # ── Model, loss, optimizer ──
        # Pass architecture params from config to model constructor
        model_cfg = self._config.get(self.model_type, self._config.get("lstm", {}))
        model_arch_kwargs = {
            k: v for k, v in model_cfg.items()
            if k in ("d_model", "nhead", "num_layers", "dim_feedforward",
                     "hidden_dim", "dropout")
        }
        model = create_model(
            self.model_type,
            num_conf_classes=self._num_conf_classes,
            **model_arch_kwargs,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Using %s encoder (%s) — %.2fM params",
            self.model_type, model.__class__.__name__, n_params / 1e6,
        )

        # Focal Loss for extreme class imbalance (~1% positive per tag)
        # Focuses learning on hard-to-classify examples, reduces easy-negative dominance
        all_labels = np.stack(train_dataset.labels)  # (N, NUM_TAGS)
        pos_rate = all_labels.mean(axis=0)            # (NUM_TAGS,)
        pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
        pw = np.clip(pw, 1.0, 50.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)
        logger.info(
            "pos_weight: min=%.1f, median=%.1f, max=%.1f",
            pw.min(), np.median(pw), pw.max(),
        )
        focal_gamma = 2.0  # focus parameter: higher = more focus on hard examples

        def focal_bce_loss(logits, targets):
            """Focal BCE loss: down-weights easy examples, emphasizes hard ones."""
            # Label smoothing: soften 0/1 targets to reduce overconfidence
            targets = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none",
            )
            probs = torch.sigmoid(logits)
            # p_t = probability assigned to the correct class
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** focal_gamma
            return (focal_weight * bce).mean()

        criterion = focal_bce_loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # CosineAnnealing with warm restarts — better exploration of loss landscape
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )

        # ── Training loop ──
        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            # --- Train ---
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history["train_loss"].append(avg_train_loss)

            # --- Validate ---
            avg_val_loss = float("nan")
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_losses.append(loss.item())
                avg_val_loss = float(np.mean(val_losses))

            history["val_loss"].append(avg_val_loss)

            # Step the cosine scheduler per epoch
            scheduler.step(epoch)

            current_lr = optimizer.param_groups[0]["lr"]
            if epoch % 5 == 1 or epoch == epochs:
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                    epoch, epochs, avg_train_loss, avg_val_loss, current_lr,
                )

            # --- Early stopping ---
            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                # Save best model weights
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best=%d, val_loss=%.4f)",
                        epoch, best_epoch, best_val_loss,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Stochastic Weight Averaging (SWA) phase ──
        # Fine-tune from best checkpoint with high LR for 5 epochs,
        # average weights for flatter minima → better generalization
        swa_epochs = 5
        swa_lr = lr * 0.5
        logger.info("Starting SWA phase (%d epochs, lr=%.2e)", swa_epochs, swa_lr)
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_optimizer = torch.optim.SGD(model.parameters(), lr=swa_lr, momentum=0.9)
        swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            swa_optimizer, T_max=swa_epochs, eta_min=swa_lr * 0.1,
        )
        for swa_ep in range(swa_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                swa_optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                swa_optimizer.step()
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Use SWA averaged weights
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
        model = swa_model.module
        model.eval()
        self.model = model

        # ── Compute validation AUC ──
        val_auc = self._compute_val_auc(model, val_loader) if val_loader else float("nan")

        # ── Find optimal threshold on validation set (maximize F1-micro) ──
        best_thr, best_f1 = 0.5, 0.0
        if val_loader is not None:
            from sklearn.metrics import f1_score
            model.eval()
            all_preds_v, all_labels_v = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(DEVICE)
                    p = torch.sigmoid(model(X_b)).cpu().numpy()
                    all_preds_v.append(p)
                    all_labels_v.append(y_b.numpy())
            y_pred_v = np.concatenate(all_preds_v)
            y_true_v = np.concatenate(all_labels_v)
            # Fine-grained threshold search: 0.005 to 0.50 in 40 steps
            for thr in np.concatenate([
                np.arange(0.005, 0.05, 0.005),   # very fine at low end
                np.arange(0.05, 0.15, 0.01),      # fine in typical range
                np.arange(0.15, 0.51, 0.05),      # coarser at high end
            ]):
                y_bin = (y_pred_v >= thr).astype(int)
                if y_bin.sum() == 0:
                    continue
                f1 = f1_score(y_true_v.ravel(), y_bin.ravel(), zero_division=0.0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            logger.info("Optimal threshold=%.3f → val F1-micro=%.4f", best_thr, best_f1)
        self._optimal_threshold = best_thr

        # Save model
        self._models_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"gap_{self.model_type}.pt"
        torch.save(model.state_dict(), self._models_dir / model_filename)

        self._training_metrics = {
            "best_epoch": best_epoch,
            "train_loss": round(history["train_loss"][best_epoch - 1], 4) if best_epoch > 0 else round(avg_train_loss, 4),
            "val_loss": round(best_val_loss, 4),
            "val_auc": round(val_auc, 4),
            "optimal_threshold": best_thr,
            "val_f1_at_threshold": round(best_f1, 4),
            "model_type": self.model_type,
            "n_train_sequences": len(train_dataset),
            "n_val_sequences": len(val_dataset) if val_dataset else 0,
            "total_epochs": len(history["train_loss"]),
            "history": history,
        }

        logger.info(
            "Training complete: best_epoch=%d, val_loss=%.4f, val_auc=%.4f",
            best_epoch, best_val_loss, val_auc,
        )

        self._set_idle()
        return self._training_metrics

    @staticmethod
    def _compute_val_auc(
        model: GapPredictionLSTM,
        val_loader: DataLoader,
    ) -> float:
        """Compute macro AUC-ROC on validation set."""
        from sklearn.metrics import roc_auc_score

        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                preds = torch.sigmoid(model(X_batch)).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(y_batch.numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_score = np.concatenate(all_preds, axis=0)

        # Only evaluate tags that appear in labels (avoid undefined AUC)
        tag_mask = y_true.sum(axis=0) > 0
        if tag_mask.sum() == 0:
            return 0.0

        try:
            auc = roc_auc_score(
                y_true[:, tag_mask],
                y_score[:, tag_mask],
                average="macro",
            )
            return float(auc)
        except ValueError:
            return 0.0

    # ──────────────────────────────────────────────────────
    # 2. Inference: predict_gaps
    # ──────────────────────────────────────────────────────

    def predict_gaps(
        self,
        user_id: str,
        recent: pd.DataFrame | list[dict] | None = None,
        diagnostic: dict | None = None,
        kg_profile: dict | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Predict knowledge gaps for a user from their recent interactions.

        Called by the Orchestrator in assessment_pipeline (step 4).

        Parameters
        ----------
        user_id : str
        recent : DataFrame or list of interaction dicts
            Last SEQ_LEN interactions. If shorter, left-padded with zeros.
        diagnostic : dict, optional
            Output from DiagnosticAgent (unused here, available for context).
        kg_profile : dict, optional
            Output from KGAgent (unused here, available for context).
        threshold : float, optional
            Probability threshold for flagging a gap. If None, uses the
            optimal threshold found during training (default 0.5 fallback).

        Returns
        -------
        dict with keys: user_id, gaps (list of PredictedGap dicts),
              gap_probabilities (full 293-dim array), n_gaps
        """
        if threshold is None:
            threshold = getattr(self, "_optimal_threshold", 0.5)
        self._set_processing()

        if self.model is None:
            self._set_idle()
            return {
                "user_id": user_id,
                "gaps": [],
                "gap_probabilities": np.zeros(NUM_TAGS).tolist(),
                "n_gaps": 0,
            }

        # Build input sequence
        seq = self._build_sequence(user_id, recent)
        if seq is None:
            self._set_idle()
            return {
                "user_id": user_id,
                "gaps": [],
                "gap_probabilities": np.zeros(NUM_TAGS).tolist(),
                "n_gaps": 0,
            }

        # Predict
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, SEQ_LEN, FEATURES_PER_STEP)
            probs = torch.sigmoid(self.model(x)).squeeze(0).cpu().numpy()  # (293,)

        # Build gap list
        gaps = []
        for tag_id in np.where(probs >= threshold)[0]:
            gaps.append({
                "tag_id": int(tag_id),
                "probability": round(float(probs[tag_id]), 4),
                "tag_name": self._tag_names.get(tag_id, f"tag_{tag_id}"),
            })

        # Sort by probability descending
        gaps.sort(key=lambda g: g["probability"], reverse=True)

        self._set_idle()
        return {
            "user_id": user_id,
            "gaps": gaps,
            "gap_probabilities": probs.tolist(),
            "n_gaps": len(gaps),
        }

    def get_at_risk_tags(
        self,
        user_id: str,
        recent: pd.DataFrame | list[dict] | None = None,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Convenience method: return only tags above threshold, sorted by risk.

        Returns list of {"tag_id": int, "probability": float, "tag_name": str}.
        """
        result = self.predict_gaps(user_id, recent=recent, threshold=threshold)
        return result["gaps"]

    # ──────────────────────────────────────────────────────
    # 3. Continuous pipeline: update_state
    # ──────────────────────────────────────────────────────

    def update_state(
        self,
        user_id: str,
        interaction: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Append a new interaction to the user's sequence buffer
        and return updated gap predictions.

        Called by the Orchestrator in continuous_pipeline (step 3).
        """
        if interaction is None:
            return {"user_id": user_id, "gaps": [], "n_gaps": 0}

        # Build interaction vector with the same feature layout used in training.
        tags = interaction.get("tags", [])
        padded_tags = GapSequenceDataset._padded_tags(tags)
        primary_tag = int(padded_tags[0]) if len(padded_tags) > 0 else 0

        correct = float(interaction.get("correct", 0))
        elapsed = float(interaction.get("elapsed_time", 15000)) / 15000.0
        elapsed = min(max(elapsed, 0), 5.0)
        changed = float(interaction.get("changed_answer", 0))
        part_id = int(interaction.get("part_id", 1)) - 1
        part_id = min(max(part_id, 0), NUM_PARTS - 1)
        conf_class = int(interaction.get("confidence_class", 0))
        conf_class = min(max(conf_class, 0), self._num_conf_classes - 1)

        history = self._user_sequences.get(user_id, [])
        prev_primary_tags = [int(vec[0]) for vec in history if len(vec) > 0]
        if primary_tag in prev_primary_tags:
            last_idx = max(i for i, tag in enumerate(prev_primary_tags) if tag == primary_tag)
            steps_since = min((len(history) - last_idx) / 50.0, 5.0)
        else:
            steps_since = 5.0

        prev_correct = [float(vec[MAX_TAGS_PER_STEP]) for vec in history if len(vec) > MAX_TAGS_PER_STEP]
        cumulative_acc = float((sum(prev_correct) + correct) / max(len(prev_correct) + 1, 1))

        vec = np.concatenate([
            padded_tags,
            np.array(
                [correct, elapsed, changed, part_id, conf_class, steps_since, cumulative_acc],
                dtype=np.float32,
            ),
        ])

        # Append to user buffer
        if user_id not in self._user_sequences:
            self._user_sequences[user_id] = []
        self._user_sequences[user_id].append(vec)

        # Keep only last SEQ_LEN
        if len(self._user_sequences[user_id]) > SEQ_LEN:
            self._user_sequences[user_id] = self._user_sequences[user_id][-SEQ_LEN:]

        # Predict if we have enough history
        if len(self._user_sequences[user_id]) >= SEQ_LEN and self.model is not None:
            seq = np.stack(self._user_sequences[user_id][-SEQ_LEN:])
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
                probs = torch.sigmoid(self.model(x)).squeeze(0).cpu().numpy()

            n_at_risk = int((probs >= 0.5).sum())
            top_risks = [
                {"tag_id": int(t), "probability": round(float(probs[t]), 4)}
                for t in np.argsort(probs)[-5:][::-1]
            ]
            return {
                "user_id": user_id,
                "n_at_risk": n_at_risk,
                "top_risks": top_risks,
                "buffer_size": len(self._user_sequences[user_id]),
            }

        return {
            "user_id": user_id,
            "n_at_risk": 0,
            "top_risks": [],
            "buffer_size": len(self._user_sequences[user_id]),
        }

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _build_sequence(
        self,
        user_id: str,
        recent: pd.DataFrame | list[dict] | None,
    ) -> np.ndarray | None:
        """
        Build a (SEQ_LEN, FEATURES_PER_STEP) input array from recent interactions.

        If fewer than SEQ_LEN interactions, left-pad with zeros.
        If recent is None, try the user's internal buffer.
        """
        if recent is None:
            # Fall back to internal buffer
            buf = self._user_sequences.get(user_id, [])
            if len(buf) == 0:
                return None
            if len(buf) >= SEQ_LEN:
                return np.stack(buf[-SEQ_LEN:])
            pad = np.zeros((SEQ_LEN - len(buf), FEATURES_PER_STEP), dtype=np.float32)
            return np.vstack([pad, np.stack(buf)])

        if isinstance(recent, list):
            recent = pd.DataFrame(recent)

        if len(recent) == 0:
            return None

        recent = recent.sort_values("timestamp").reset_index(drop=True)

        median_elapsed = recent["elapsed_time"].median() if "elapsed_time" in recent.columns else 15_000.0
        if median_elapsed == 0 or pd.isna(median_elapsed):
            median_elapsed = 15_000.0

        tag_matrix = np.stack(recent["tags"].apply(GapSequenceDataset._padded_tags).values)
        correct_series = (
            recent["correct"].astype(float).values
            if "correct" in recent.columns
            else np.zeros(len(recent), dtype=np.float32)
        )

        steps_since = np.zeros(len(recent), dtype=np.float32)
        tag_last_seen: dict[int, int] = {}
        for idx in range(len(recent)):
            primary = int(tag_matrix[idx, 0])
            if primary in tag_last_seen:
                steps_since[idx] = min((idx - tag_last_seen[primary]) / 50.0, 5.0)
            else:
                steps_since[idx] = 5.0
            tag_last_seen[primary] = idx

        cum_correct = np.cumsum(correct_series)
        cum_count = np.arange(1, len(recent) + 1, dtype=np.float32)
        cumulative_acc = (cum_correct / cum_count).astype(np.float32)

        # Build vectors from DataFrame
        vecs = []
        for idx, row in recent.iterrows():
            padded_tags = tag_matrix[idx]

            correct = float(row.get("correct", 0))
            elapsed = float(row.get("elapsed_time", median_elapsed)) / float(median_elapsed)
            elapsed = min(max(elapsed, 0), 5.0)
            changed = float(row.get("changed_answer", 0))

            part_id = int(row.get("part_id", 1)) - 1
            part_id = min(max(part_id, 0), NUM_PARTS - 1)

            conf_class = int(row.get("confidence_class", 0))
            conf_class = min(max(conf_class, 0), self._num_conf_classes - 1)

            vec = np.concatenate([
                padded_tags,
                np.array(
                    [correct, elapsed, changed, part_id, conf_class, steps_since[idx], cumulative_acc[idx]],
                    dtype=np.float32,
                ),
            ])
            vecs.append(vec)

        # Update user buffer
        self._user_sequences[user_id] = vecs[-SEQ_LEN:]

        # Pad or truncate to SEQ_LEN
        if len(vecs) >= SEQ_LEN:
            return np.stack(vecs[-SEQ_LEN:])

        pad = np.zeros((SEQ_LEN - len(vecs), FEATURES_PER_STEP), dtype=np.float32)
        return np.vstack([pad, np.stack(vecs)])

    def _load_model(self, path: Path) -> None:
        """Load model weights from disk."""
        model_cfg = self._config.get(self.model_type, self._config.get("lstm", {}))
        model_arch_kwargs = {
            k: v for k, v in model_cfg.items()
            if k in ("d_model", "nhead", "num_layers", "dim_feedforward",
                     "hidden_dim", "dropout")
        }
        model = create_model(
            self.model_type,
            num_conf_classes=self._num_conf_classes,
            **model_arch_kwargs,
        ).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        self.model = model

    @property
    def training_metrics(self) -> dict[str, Any]:
        return self._training_metrics
