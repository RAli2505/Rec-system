"""
Prediction Agent for MARS.

Predicts future knowledge gaps per tag using an LSTM trained on
sequences of student interactions.  Each interaction is encoded as a
51-dimensional vector:

    tag_emb(32) + correct(1) + elapsed_norm(1) + changed(1)
    + part_emb(8) + confidence_emb(8) = 51

The model consumes the last ``SEQ_LEN`` (50) interactions and outputs
a (293,) sigmoid vector — the probability that the student will fail
each tag within the next ``HORIZON`` (10) questions.

Training label: binary vector marking tags the student actually failed
in the next 10 questions after the window.

Architecture
------------
- tag_embedding  : Embedding(NUM_TAGS, 32)   — one tag per interaction
- part_embedding : Embedding(NUM_PARTS, 8)
- conf_embedding : Embedding(NUM_CONF_CLASSES, 8)
- LSTM(51, 128, num_layers=2, dropout=0.3, batch_first=True)
- Linear(128, NUM_TAGS) → sigmoid

Loss: BCELoss | Optimizer: Adam lr=1e-3 | Early stopping: patience 5
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

logger = logging.getLogger("mars.agent.prediction")

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

NUM_TAGS = 293
NUM_PARTS = 7           # TOEIC parts 1-7
NUM_CONF_CLASSES = 6    # from ConfidenceAgent
SEQ_LEN = 50            # window of recent interactions
HORIZON = 10            # predict failures within next N questions

TAG_EMB_DIM = 32
PART_EMB_DIM = 8
CONF_EMB_DIM = 8
SCALAR_FEATURES = 3     # correct, elapsed_norm, changed
INPUT_DIM = TAG_EMB_DIM + SCALAR_FEATURES + PART_EMB_DIM + CONF_EMB_DIM  # 51
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 5

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
      X : (SEQ_LEN, 5) — tag_id, correct, elapsed_norm, changed, part_id, conf_class
          (raw ints/floats; embeddings applied inside the model)
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

            # Pre-compute per-row arrays
            tag_ids = rows["tags"].apply(self._primary_tag).values
            correct = rows["correct"].astype(float).values
            elapsed_norm = (rows["elapsed_time"] / median_elapsed).clip(0, 5).fillna(1.0).values
            changed = rows["changed_answer"].astype(float).values if "changed_answer" in rows.columns else np.zeros(len(rows))
            part_ids = (rows["part_id"].fillna(1).astype(int).values - 1).clip(0, NUM_PARTS - 1) if "part_id" in rows.columns else np.zeros(len(rows), dtype=int)
            conf_cls = rows["confidence_class"].values.astype(int) if "confidence_class" in rows.columns else np.zeros(len(rows), dtype=int)

            # Clamp tag_ids to valid range
            tag_ids = np.clip(tag_ids, 0, NUM_TAGS - 1)

            # All tags per row (for labels)
            all_tags_per_row = rows["tags"].apply(self._all_tags).tolist()

            n = len(rows)
            for i in range(n - self.seq_len - self.horizon + 1):
                # Input sequence: rows [i : i+seq_len]
                seq = np.stack([
                    tag_ids[i:i + self.seq_len],
                    correct[i:i + self.seq_len],
                    elapsed_norm[i:i + self.seq_len],
                    changed[i:i + self.seq_len],
                    part_ids[i:i + self.seq_len],
                    conf_cls[i:i + self.seq_len],
                ], axis=1).astype(np.float32)  # (seq_len, 6)

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
            "Built %d sequences (seq_len=%d, horizon=%d) from %d users",
            len(self.sequences), self.seq_len, self.horizon,
            df["user_id"].nunique(),
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
    LSTM that predicts per-tag failure probability from a sequence
    of student interactions.

    Input per timestep (6 raw values):
        tag_id (int) → Embedding(293, 32)
        correct (float, 0/1)
        elapsed_norm (float)
        changed (float, 0/1)
        part_id (int 0-6) → Embedding(7, 8)
        conf_class (int 0-5) → Embedding(6, 8)

    After embedding: 32 + 1 + 1 + 1 + 8 + 8 = 51 → LSTM(51, 128, 2)
    Output: Linear(128, 293) → sigmoid
    """

    def __init__(self) -> None:
        super().__init__()

        self.tag_embedding = nn.Embedding(NUM_TAGS, TAG_EMB_DIM, padding_idx=0)
        self.part_embedding = nn.Embedding(NUM_PARTS, PART_EMB_DIM)
        self.confidence_embedding = nn.Embedding(NUM_CONF_CLASSES, CONF_EMB_DIM)

        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.fc = nn.Linear(HIDDEN_DIM, NUM_TAGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 6)
            Columns: tag_id, correct, elapsed_norm, changed, part_id, conf_class

        Returns
        -------
        (batch, NUM_TAGS) probabilities in [0, 1]
        """
        tag_ids = x[:, :, 0].long().clamp(0, NUM_TAGS - 1)
        correct = x[:, :, 1:2]          # (B, T, 1)
        elapsed = x[:, :, 2:3]          # (B, T, 1)
        changed = x[:, :, 3:4]          # (B, T, 1)
        part_ids = x[:, :, 4].long().clamp(0, NUM_PARTS - 1)
        conf_ids = x[:, :, 5].long().clamp(0, NUM_CONF_CLASSES - 1)

        tag_emb = self.tag_embedding(tag_ids)       # (B, T, 32)
        part_emb = self.part_embedding(part_ids)     # (B, T, 8)
        conf_emb = self.confidence_embedding(conf_ids)  # (B, T, 8)

        # Concatenate: (B, T, 32+1+1+1+8+8) = (B, T, 51)
        combined = torch.cat([tag_emb, correct, elapsed, changed, part_emb, conf_emb], dim=-1)

        lstm_out, _ = self.lstm(combined)            # (B, T, 128)
        last_hidden = lstm_out[:, -1, :]             # (B, 128)
        logits = self.fc(last_hidden)                # (B, 293)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class PredictionAgent(BaseAgent):
    """
    LSTM-based knowledge gap prediction agent.

    Predicts which tags a student is likely to fail in the near future,
    enabling proactive recommendations.
    """

    name = "prediction"

    def __init__(self) -> None:
        super().__init__()
        self.model: GapPredictionLSTM | None = None
        self._models_dir = Path("models")
        self._user_sequences: dict[str, list[np.ndarray]] = {}  # user → recent interactions
        self._training_metrics: dict[str, Any] = {}
        self._tag_names: dict[int, str] = {}  # tag_id → human-readable name

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        """Load a pre-trained model or train from scratch."""
        model_path = self._models_dir / "gap_lstm.pt"
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
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        patience: int = PATIENCE,
    ) -> dict[str, float]:
        """
        Train the GapPredictionLSTM on interaction sequences.

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
        self._set_processing()
        logger.info("Building training sequences...")

        # Split into train/val if no val_df provided
        if val_df is None:
            users = interactions_df["user_id"].unique()
            n_val = max(1, int(len(users) * 0.15))
            rng = np.random.RandomState(42)
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
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        ) if len(val_dataset) > 0 else None

        logger.info(
            "Train: %d sequences, Val: %d sequences",
            len(train_dataset), len(val_dataset) if val_dataset else 0,
        )

        # ── Model, loss, optimizer ──
        model = GapPredictionLSTM().to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            if epoch % 5 == 1 or epoch == epochs:
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, epochs, avg_train_loss, avg_val_loss,
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
        model.eval()
        self.model = model

        # ── Compute validation AUC ──
        val_auc = self._compute_val_auc(model, val_loader) if val_loader else float("nan")

        # Save model
        self._models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self._models_dir / "gap_lstm.pt")

        self._training_metrics = {
            "best_epoch": best_epoch,
            "train_loss": round(history["train_loss"][best_epoch - 1], 4) if best_epoch > 0 else round(avg_train_loss, 4),
            "val_loss": round(best_val_loss, 4),
            "val_auc": round(val_auc, 4),
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
                preds = model(X_batch).cpu().numpy()
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
        threshold: float = 0.5,
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
        threshold : float
            Probability threshold for flagging a gap.

        Returns
        -------
        dict with keys: user_id, gaps (list of PredictedGap dicts),
              gap_probabilities (full 293-dim array), n_gaps
        """
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
            x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, SEQ_LEN, 6)
            probs = self.model(x).squeeze(0).cpu().numpy()      # (293,)

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

        # Build interaction vector: [tag_id, correct, elapsed_norm, changed, part_id, conf_class]
        tags = interaction.get("tags", [])
        if isinstance(tags, str):
            parts = tags.replace(";", ",").split(",")
            tag_id = int(parts[0].strip()) if parts and parts[0].strip().isdigit() else 0
        elif isinstance(tags, list) and len(tags) > 0:
            tag_id = int(tags[0])
        else:
            tag_id = 0

        tag_id = min(max(tag_id, 0), NUM_TAGS - 1)
        correct = float(interaction.get("correct", 0))
        elapsed = float(interaction.get("elapsed_time", 15000)) / 15000.0
        elapsed = min(max(elapsed, 0), 5.0)
        changed = float(interaction.get("changed_answer", 0))
        part_id = int(interaction.get("part_id", 1)) - 1
        part_id = min(max(part_id, 0), NUM_PARTS - 1)
        conf_class = int(interaction.get("confidence_class", 0))
        conf_class = min(max(conf_class, 0), NUM_CONF_CLASSES - 1)

        vec = np.array([tag_id, correct, elapsed, changed, part_id, conf_class],
                       dtype=np.float32)

        # Append to user buffer
        if user_id not in self._user_sequences:
            self._user_sequences[user_id] = []
        self._user_sequences[user_id].append(vec)

        # Keep only last SEQ_LEN
        if len(self._user_sequences[user_id]) > SEQ_LEN:
            self._user_sequences[user_id] = self._user_sequences[user_id][-SEQ_LEN:]

        # Predict if we have enough history
        if len(self._user_sequences[user_id]) >= SEQ_LEN and self.model is not None:
            seq = np.stack(self._user_sequences[user_id][-SEQ_LEN:])  # (SEQ_LEN, 6)
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
                probs = self.model(x).squeeze(0).cpu().numpy()

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
        Build a (SEQ_LEN, 6) input array from recent interactions.

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
            pad = np.zeros((SEQ_LEN - len(buf), 6), dtype=np.float32)
            return np.vstack([pad, np.stack(buf)])

        if isinstance(recent, list):
            recent = pd.DataFrame(recent)

        if len(recent) == 0:
            return None

        # Build vectors from DataFrame
        vecs = []
        for _, row in recent.iterrows():
            tags = row.get("tags", [])
            tag_id = GapSequenceDataset._primary_tag(tags)
            tag_id = min(max(tag_id, 0), NUM_TAGS - 1)

            correct = float(row.get("correct", 0))
            elapsed = float(row.get("elapsed_time", 15000)) / 15000.0
            elapsed = min(max(elapsed, 0), 5.0)
            changed = float(row.get("changed_answer", 0))

            part_id = int(row.get("part_id", 1)) - 1
            part_id = min(max(part_id, 0), NUM_PARTS - 1)

            conf_class = int(row.get("confidence_class", 0))
            conf_class = min(max(conf_class, 0), NUM_CONF_CLASSES - 1)

            vecs.append(np.array([tag_id, correct, elapsed, changed, part_id, conf_class],
                                 dtype=np.float32))

        # Update user buffer
        self._user_sequences[user_id] = vecs[-SEQ_LEN:]

        # Pad or truncate to SEQ_LEN
        if len(vecs) >= SEQ_LEN:
            return np.stack(vecs[-SEQ_LEN:])

        pad = np.zeros((SEQ_LEN - len(vecs), 6), dtype=np.float32)
        return np.vstack([pad, np.stack(vecs)])

    def _load_model(self, path: Path) -> None:
        """Load model weights from disk."""
        model = GapPredictionLSTM().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        self.model = model

    @property
    def training_metrics(self) -> dict[str, Any]:
        return self._training_metrics
