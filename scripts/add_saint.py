"""Add SAINT-simplified baseline to 08_evaluation.ipynb."""
import json

with open('notebooks/08_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Add FAST_SAINT_EPOCHS to cell 1
src1 = ''.join(nb['cells'][1]['source'])
if 'FAST_SAINT_EPOCHS' not in src1:
    src1 = src1.replace(
        'FAST_MARS_PRED_EPOCHS = 5',
        'FAST_MARS_PRED_EPOCHS = 5\nFAST_SAINT_EPOCHS = 5'
    )
    nb['cells'][1]['source'] = [line + '\n' for line in src1.rstrip('\n').split('\n')]

# 2. Insert SAINT cell after cell 11 (BPR) — becomes new cell 12
saint_source = '''# ═══════════════════════════════════════════════════════════
# Baseline: SAINT-simplified (Choi et al., 2020 — Transformer encoder)
# ═══════════════════════════════════════════════════════════

class SAINTSimplified(nn.Module):
    """
    Simplified SAINT: Transformer encoder for Knowledge Tracing.
    Input: sequence of (tag_id, part_id, correct, elapsed) -> P(fail) per tag.
    Same seq-to-set formulation as DKT/SAKT for fair comparison.
    Differs from SAKT: uses part_id + elapsed_time features, 8-head attention.
    """
    def __init__(self, n_tags=NUM_TAGS, d_model=128, nhead=8, num_layers=2,
                 dim_ff=256, dropout=0.1, seq_len=50, tag_emb_dim=32, part_emb_dim=8):
        super().__init__()
        self.tag_emb = nn.Embedding(n_tags, tag_emb_dim, padding_idx=0)
        self.part_emb = nn.Embedding(8, part_emb_dim)  # 0-7
        input_dim = tag_emb_dim + part_emb_dim + 2  # +correct +elapsed

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, n_tags)

    def forward(self, tag_ids, part_ids, correct, elapsed, mask=None):
        te = self.tag_emb(tag_ids)
        pe = self.part_emb(part_ids)
        x = torch.cat([te, pe, correct.unsqueeze(-1), elapsed.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        # Causal mask
        T = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        if mask is not None:
            x = self.encoder(x, mask=causal, src_key_padding_mask=mask)
        else:
            x = self.encoder(x, mask=causal)
        return torch.sigmoid(self.output(x[:, -1, :]))


def _normalize_elapsed(elapsed_arr):
    """Log-normalize elapsed_time to [0, 1] range."""
    e = np.log1p(np.clip(elapsed_arr, 0, 300000).astype(np.float64))
    emax = e.max()
    if emax > 0:
        e = e / emax
    return e.astype(np.float32)


def train_saint(train_df, val_df, seed=42, epochs=FAST_SAINT_EPOCHS, seq_len=50):
    """Train SAINT-simplified for KT baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def build_seqs(df, sl=seq_len):
        seqs_tag, seqs_part, seqs_corr, seqs_elapsed, labels = [], [], [], [], []
        for _, grp in df.groupby("user_id"):
            grp = grp.sort_values("timestamp")
            tags = grp["tags"].apply(lambda x: parse_tags(x)[0] if parse_tags(x) else 0).values
            parts = grp["part_id"].fillna(0).astype(int).clip(0, 7).values
            corr = grp["correct"].astype(float).values
            elapsed = _normalize_elapsed(grp["elapsed_time"].fillna(0).values)

            for i in range(len(grp) - sl - 1):
                seqs_tag.append(tags[i:i+sl])
                seqs_part.append(parts[i:i+sl])
                seqs_corr.append(corr[i:i+sl])
                seqs_elapsed.append(elapsed[i:i+sl])
                # Label: failure vector for next interaction
                lbl = np.zeros(NUM_TAGS, dtype=np.float32)
                next_tags = parse_tags(grp.iloc[i+sl].get("tags", []))
                if not corr[i+sl]:
                    for t in next_tags:
                        if 0 <= t < NUM_TAGS:
                            lbl[t] = 1.0
                labels.append(lbl)

        if not seqs_tag:
            return None, None, None, None, None
        return (
            torch.tensor(np.array(seqs_tag), dtype=torch.long),
            torch.tensor(np.array(seqs_part), dtype=torch.long),
            torch.tensor(np.array(seqs_corr), dtype=torch.float),
            torch.tensor(np.array(seqs_elapsed), dtype=torch.float),
            torch.tensor(np.array(labels), dtype=torch.float),
        )

    result = build_seqs(train_df)
    if result[0] is None:
        return None
    t_tag, t_part, t_corr, t_elapsed, t_labels = result

    model = SAINTSimplified()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    ds = torch.utils.data.TensorDataset(t_tag, t_part, t_corr, t_elapsed, t_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        for bt, bp, bc, be, bl in dl:
            optimizer.zero_grad()
            pred = model(bt, bp, bc, be)
            loss = criterion(pred, bl)
            loss.backward()
            optimizer.step()

        # Quick validation
        model.eval()
        with torch.no_grad():
            vr = build_seqs(val_df)
            if vr[0] is not None and len(vr[0]) > 0:
                v_pred = model(vr[0], vr[1], vr[2], vr[3])
                val_loss = criterion(v_pred, vr[4]).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= 3:
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def baseline_saint(eval_pairs, saint_model, seq_len=50):
    """Run SAINT predictions on evaluation pairs."""
    preds, scores_list, gt_list, gta_list = [], [], [], []
    for uid, ctx, gt, gt_all, _ in eval_pairs:
        tags = ctx["tags"].apply(lambda x: parse_tags(x)[0] if parse_tags(x) else 0).values
        parts = ctx["part_id"].fillna(0).astype(int).clip(0, 7).values
        corr = ctx["correct"].astype(float).values
        elapsed = _normalize_elapsed(ctx["elapsed_time"].fillna(0).values)

        def pad_trunc(arr, sl=seq_len):
            if len(arr) >= sl:
                return arr[-sl:]
            return np.pad(arr, (sl - len(arr), 0), constant_values=0)

        with torch.no_grad():
            t_tag = torch.tensor(pad_trunc(tags), dtype=torch.long).unsqueeze(0)
            t_part = torch.tensor(pad_trunc(parts), dtype=torch.long).unsqueeze(0)
            t_corr = torch.tensor(pad_trunc(corr), dtype=torch.float).unsqueeze(0)
            t_elapsed = torch.tensor(pad_trunc(elapsed), dtype=torch.float).unsqueeze(0)
            scores = saint_model(t_tag, t_part, t_corr, t_elapsed).squeeze(0).numpy()

        ranked = np.argsort(scores)[::-1].tolist()
        preds.append(ranked)
        scores_list.append(scores)
        gt_list.append(gt)
        gta_list.append(gt_all)

    return preds, scores_list, gt_list, gta_list


print("SAINT-simplified baseline defined (Choi et al., 2020).")'''

saint_lines = saint_source.strip().split('\n')
saint_cell_source = [line + '\n' for line in saint_lines[:-1]] + [saint_lines[-1]]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": saint_cell_source,
}

# Insert after cell 11 (BPR), before cell 12 (MARS)
nb['cells'].insert(12, new_cell)
# Now MARS is cell 13, main loop is cell 15

# 3. Update main loop (now cell 15) to include SAINT in Phase 2
main_src = ''.join(nb['cells'][15]['source'])

# Add _train_eval_saint function and add to thread pool
saint_fn = """
    def _train_eval_saint(seed):
        m = load_cached_metrics("main", seed, "SAINT")
        if m is None:
            saint_model = train_saint(train_df, val_df, seed=seed, epochs=FAST_SAINT_EPOCHS)
            if saint_model is not None:
                p, s, g, ga = baseline_saint(eval_pairs, saint_model)
                m = compute_metrics(p, g, s, ga)
                save_cached_metrics("main", seed, "SAINT", m)
        return "SAINT", m

"""

# Insert before the ThreadPoolExecutor block for heavy models
old_pool = '    with ThreadPoolExecutor(max_workers=3) as pool:'
new_pool = saint_fn + '    with ThreadPoolExecutor(max_workers=4) as pool:'
main_src = main_src.replace(old_pool, new_pool, 1)

# Add SAINT to the futures list
main_src = main_src.replace(
    '            pool.submit(_train_eval_sakt, seed),\n        ]',
    '            pool.submit(_train_eval_sakt, seed),\n            pool.submit(_train_eval_saint, seed),\n        ]'
)

main_lines = main_src.rstrip('\n').split('\n')
nb['cells'][15]['source'] = [line + '\n' for line in main_lines[:-1]] + [main_lines[-1]]
nb['cells'][15]['outputs'] = []

# 4. Update Table 1 method_order (now cell 17) to include SAINT
tbl_src = ''.join(nb['cells'][17]['source'])
tbl_src = tbl_src.replace(
    '"SAKT", "Binary-conf"',
    '"SAKT", "SAINT", "Binary-conf"'
)
tbl_lines = tbl_src.rstrip('\n').split('\n')
nb['cells'][17]['source'] = [line + '\n' for line in tbl_lines[:-1]] + [tbl_lines[-1]]

with open('notebooks/08_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("SAINT baseline added successfully:")
print(f"  - FAST_SAINT_EPOCHS=5 added to cell 1")
print(f"  - SAINTSimplified model + train_saint + baseline_saint in new cell 12")
print(f"  - SAINT added to parallel Phase 2 in main loop (cell 15)")
print(f"  - SAINT added to method_order in Table 1 (cell 17)")
