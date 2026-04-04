"""Add Moving Average baseline to NB08."""
import json

with open('notebooks/08_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Find Monolithic cell
mono_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'def baseline_monolithic(' in src:
        mono_idx = i
        break

print(f"Monolithic: cell {mono_idx}")

# 2. Insert Moving Avg cell after it
ma_src = """# ═══════════════════════════════════════════════════════════
# Baseline: Moving Average (naive — no ML)
# ═══════════════════════════════════════════════════════════
def baseline_moving_average(eval_pairs, train_df, window=50):
    \"\"\"
    Naive baseline: predict gap based on recent accuracy per tag.
    Lower accuracy on tag -> higher gap probability. Unseen tags -> 0.5.
    \"\"\"
    preds, scores_list, gt_list, gta_list = [], [], [], []
    for uid, ctx, gt, gt_all, _ in eval_pairs:
        recent = ctx.tail(window)
        tag_accuracy = {}
        for _, row in recent.iterrows():
            for tag in parse_tags(row.get("tags", [])):
                if tag not in tag_accuracy:
                    tag_accuracy[tag] = []
                tag_accuracy[tag].append(float(row["correct"]))

        gap_scores = np.zeros(NUM_TAGS, dtype=np.float32)
        for tag, acc_list in tag_accuracy.items():
            if 0 <= tag < NUM_TAGS:
                gap_scores[tag] = 1.0 - np.mean(acc_list)

        unseen_mask = gap_scores == 0
        gap_scores[unseen_mask] = 0.5

        ranked = np.argsort(gap_scores)[::-1].tolist()
        preds.append(ranked)
        scores_list.append(gap_scores)
        gt_list.append(gt)
        gta_list.append(gt_all)
    return preds, scores_list, gt_list, gta_list


print("Moving Average baseline defined.")"""

lines = ma_src.strip().split('\n')
ma_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + '\n' for l in lines[:-1]] + [lines[-1]],
}
nb['cells'].insert(mono_idx + 1, ma_cell)
print(f"Inserted Moving Avg at cell {mono_idx + 1}")

# 3. Find main loop (simple_tasks) — indices shifted by 1
main_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'simple_tasks' in src and 'baseline_random' in src:
        main_idx = i
        break

if main_idx:
    src = ''.join(nb['cells'][main_idx]['source'])
    # Add Moving Avg to simple_tasks dict
    src = src.replace(
        '"Monolithic":   lambda s=seed: baseline_monolithic(eval_pairs, train_df, s),',
        '"Monolithic":   lambda s=seed: baseline_monolithic(eval_pairs, train_df, s),\n        "Moving Avg":   lambda: baseline_moving_average(eval_pairs, train_df),'
    )
    lines = src.rstrip('\n').split('\n')
    nb['cells'][main_idx]['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    print(f"Added to main loop simple_tasks (cell {main_idx})")

# 4. Add to method_order in Table 1
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'method_order' in src and '"Monolithic"' in src and '"MARS (ours)"' in src:
        src = src.replace(
            '"Monolithic", "MARS (ours)"',
            '"Monolithic", "Moving Avg", "MARS (ours)"'
        )
        lines = src.rstrip('\n').split('\n')
        nb['cells'][i]['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        print(f"Added to method_order (cell {i})")
        break

with open('notebooks/08_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done.")
