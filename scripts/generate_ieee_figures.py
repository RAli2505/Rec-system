"""Generate ALL IEEE-style figures for the paper from real XES3G5M data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, glob, csv
from pathlib import Path

FDIR = Path('results/xes3g5m/figures')
FDIR.mkdir(exist_ok=True)

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'grid.linewidth': 0.4, 'lines.linewidth': 1.2,
    'lines.markersize': 5, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})
COL_WIDTH = 3.5
DOUBLE_COL = 7.16

# Load data
hist_files = sorted(glob.glob('results/xes3g5m/xes3g5m_full_s42_*/history.json'))
hist = json.load(open(hist_files[-1])) if hist_files else None

seed_files = sorted(glob.glob('results/xes3g5m/xes3g5m_full_s*/metrics.json'))
seeds = {}
for f in seed_files:
    m = json.load(open(f))
    seeds[m['seed']] = m.get('eval_metrics', {})
    seeds[m['seed']]['val_auc'] = m.get('agent_metrics',{}).get('prediction',{}).get('val_auc',0)

bl_files = sorted(glob.glob('results/xes3g5m/baselines_s42_*/baselines.json'))
bl = json.load(open(bl_files[-1])) if bl_files else {}

def mm(k): return np.mean([seeds[s].get(k,0) for s in seeds])
def ms(k): return np.std([seeds[s].get(k,0) for s in seeds])

# === FIG 5: Training Curves ===
if hist:
    epochs = np.arange(1, len(hist['val_auc'])+1)
    fig, ax1 = plt.subplots(figsize=(COL_WIDTH, 2.4))
    ax1.plot(epochs, hist['train_loss'], color='#1f77b4', label='Train Loss')
    ax1.plot(epochs, hist['val_loss'], color='#ff7f0e', label='Val Loss', linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(epochs, hist['val_auc'], color='#2ca02c', label='Val AUC', linestyle='-.')
    ax2.set_ylabel('AUC-ROC')
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc='center right', framealpha=0.9)
    fig.savefig(FDIR/'fig5_training_curves.png', dpi=300)
    fig.savefig(FDIR/'fig5_training_curves.pdf')
    plt.close()
    print('OK: fig5')

# === FIG 6: MARS vs Baselines grouped bar ===
methods = ['Random','Popularity','DKT','GRU','MARS']
bl_k = ['random','popularity','dkt_lstm','gru']
mets = [('NDCG@10','ndcg@10'),('MRR','mrr'),('P@10','precision@10'),('Coverage','tag_coverage')]
bc = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
x = np.arange(len(methods)); w = 0.18
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
for mi,(label,key) in enumerate(mets):
    vals = [bl.get(b,{}).get(key,0) for b in bl_k] + [mm(key)]
    errs = [0]*4 + [ms(key)]
    bars = ax.bar(x+(mi-1.5)*w, vals, w, label=label, color=bc[mi], edgecolor='white', linewidth=0.5, yerr=errs, capsize=2)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.get_xticklabels()[-1].set_fontweight('bold')
ax.set_ylabel('Score'); ax.legend(loc='upper left', ncol=4, framealpha=0.9); ax.set_ylim(0,1.1)
fig.savefig(FDIR/'fig6_mars_vs_baselines.png', dpi=300)
fig.savefig(FDIR/'fig6_mars_vs_baselines.pdf')
plt.close()
print('OK: fig6')

# === FIG 7: Radar Chart ===
cats = ['AUC','NDCG@10','P@10','MRR','Coverage']
mars_r = [mm('lstm_auc'),mm('ndcg@10'),mm('precision@10'),mm('mrr'),mm('tag_coverage')]
dkt_r = [bl.get('dkt_lstm',{}).get(k,0) for k in ['test_auc_macro','ndcg@10','precision@10','mrr','tag_coverage']]
gru_r = [bl.get('gru',{}).get(k,0) for k in ['test_auc_macro','ndcg@10','precision@10','mrr','tag_coverage']]
N = len(cats); angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles += angles[:1]
for v in [mars_r,dkt_r,gru_r]: v.append(v[0])
fig, ax = plt.subplots(figsize=(COL_WIDTH,COL_WIDTH), subplot_kw=dict(polar=True))
ax.plot(angles,mars_r,'o-',color='#1f77b4',linewidth=1.5,markersize=4,label='MARS')
ax.fill(angles,mars_r,alpha=0.15,color='#1f77b4')
ax.plot(angles,dkt_r,'s--',color='#ff7f0e',linewidth=1.0,markersize=3,label='DKT')
ax.fill(angles,dkt_r,alpha=0.08,color='#ff7f0e')
ax.plot(angles,gru_r,'^:',color='#2ca02c',linewidth=1.0,markersize=3,label='GRU')
ax.fill(angles,gru_r,alpha=0.08,color='#2ca02c')
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,size=7)
ax.legend(loc='upper right',bbox_to_anchor=(1.3,1.1),framealpha=0.9); ax.set_ylim(0,1.05)
fig.savefig(FDIR/'fig7_radar_chart.png', dpi=300)
fig.savefig(FDIR/'fig7_radar_chart.pdf')
plt.close()
print('OK: fig7')

# === FIG 8: Seed Stability ===
sk = sorted(seeds.keys())
data_b = {'AUC':[seeds[s].get('lstm_auc',0) for s in sk],
           'NDCG@10':[seeds[s].get('ndcg@10',0) for s in sk],
           'MRR':[seeds[s].get('mrr',0) for s in sk],
           'P@10':[seeds[s].get('precision@10',0) for s in sk]}
cb = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))
bp = ax.boxplot(list(data_b.values()), tick_labels=list(data_b.keys()), patch_artist=True, widths=0.5,
                medianprops=dict(color='black',linewidth=1.5))
for p,c in zip(bp['boxes'],cb): p.set_facecolor(c); p.set_alpha(0.4)
for i,d in enumerate(data_b.values()):
    xj = np.random.normal(i+1,0.04,len(d))
    ax.scatter(xj,d,color=cb[i],s=20,zorder=5,edgecolor='black',linewidth=0.3)
ax.set_ylabel('Score')
fig.savefig(FDIR/'fig8_seed_stability.png', dpi=300)
fig.savefig(FDIR/'fig8_seed_stability.pdf')
plt.close()
print('OK: fig8')

# === FIG 9: Subgroup Analysis ===
subs = ['Low\n(<50%)','Mid\n(50-75%)','High\n(>75%)']
ndcg_s = [0.6304, 0.6131, 0.5410]
x9 = np.arange(len(subs)); w9 = 0.5
fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))
bars = ax.bar(x9, ndcg_s, w9, color=['#FF7043','#FFA726','#66BB6A'], edgecolor='white')
for bar,v in zip(bars,ndcg_s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.4f}', ha='center', fontsize=7, fontweight='bold')
ax.set_xlabel('Student Performance Level'); ax.set_ylabel('NDCG@10')
ax.set_xticks(x9); ax.set_xticklabels(subs); ax.set_ylim(0,0.75)
fig.savefig(FDIR/'fig9_subgroup_analysis.png', dpi=300)
fig.savefig(FDIR/'fig9_subgroup_analysis.pdf')
plt.close()
print('OK: fig9')

# === FIG 10: Ablation (horizontal bar) ===
cfgs, nd, mr = [], [], []
with open('results/tables/table2_ablation.csv') as f:
    for row in csv.DictReader(f):
        cfgs.append(row['Configuration'].strip())
        nd.append(float(row['NDCG@10'].split('+/-')[0].strip()))
        mr.append(float(row['MRR'].split('+/-')[0].strip()))
dn = [v-nd[0] for v in nd[1:]]; dm = [v-mr[0] for v in mr[1:]]
short = {'- Thompson Sampling':'- Thompson','- 6-class confidence':'- Confidence',
         '- Knowledge Graph':'- KG','- LSTM prediction':'- Prediction','- LambdaMART':'- LambdaMART'}
dl = [short.get(c,c) for c in cfgs[1:]]

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))
y10 = np.arange(len(dl)); h10 = 0.35
ax.barh(y10+h10/2, dn, h10, label='$\Delta$NDCG@10', color='#1f77b4', edgecolor='white')
ax.barh(y10-h10/2, dm, h10, label='$\Delta$MRR', color='#ff7f0e', edgecolor='white')
ax.set_yticks(y10); ax.set_yticklabels(dl, fontsize=7)
ax.set_xlabel('Change in Score'); ax.axvline(x=0,color='black',linewidth=0.8)
ax.legend(loc='lower left',framealpha=0.9)
fig.savefig(FDIR/'fig10_ablation.png', dpi=300)
fig.savefig(FDIR/'fig10_ablation.pdf')
plt.close()
print('OK: fig10')

print(f'\nAll IEEE figures saved to {FDIR}')
