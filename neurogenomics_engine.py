"""
NeurogenomicsEngine: Brain Cell-Type Transcriptomics and Neurodegeneration Analysis
- Cell-type deconvolution (7 brain cell types)
- Synaptic gene module scoring
- Neurodegeneration risk score (AD/PD/ALS gene sets)
- Brain region-specific expression analysis (5 regions)
- Neuron-glia communication network
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Parameters ───────────────────────────────────────────────────────────────
N_SAMPLES = 150
N_GENES = 10000
N_REGIONS = 5
N_CELL_TYPES = 7
CELL_TYPES = ['Neurons', 'Astrocytes', 'Oligodendrocytes', 'Microglia',
              'OPC', 'Endothelial', 'Pericytes']
REGIONS = ['Frontal Cortex', 'Hippocampus', 'Cerebellum', 'Basal Ganglia', 'Brainstem']
N_PER_REGION = 30  # 30 samples per region
N_AD = 75          # AD samples
N_CTRL = 75        # Control samples

# Sample metadata
region_labels = np.repeat(np.arange(N_REGIONS), N_PER_REGION)
diagnosis = np.array([0]*N_CTRL + [1]*N_AD)  # 0=control, 1=AD

# ─── 1. Simulate Gene Expression Data ────────────────────────────────────────
print("[Data] Simulating 150 × 10,000 gene expression matrix...")

# Cell-type marker genes (50 per type)
N_MARKERS = 50
marker_genes = {}
all_marker_idx = []
for i, ct in enumerate(CELL_TYPES):
    start = i * N_MARKERS
    end = start + N_MARKERS
    marker_genes[ct] = list(range(start, end))
    all_marker_idx.extend(range(start, end))

# True cell-type proportions (Dirichlet)
alpha = np.array([3.0, 1.5, 1.5, 0.5, 0.5, 0.3, 0.2])
true_props = np.random.dirichlet(alpha, N_SAMPLES)
# AD samples have fewer neurons, more microglia
true_props[diagnosis == 1, 0] *= 0.7   # fewer neurons
true_props[diagnosis == 1, 3] *= 1.8   # more microglia
# Renormalize
true_props = true_props / true_props.sum(axis=1, keepdims=True)

# Reference profiles (N_GENES × N_CELL_TYPES)
ref_profiles = np.random.exponential(0.5, (N_GENES, N_CELL_TYPES))
for i, ct in enumerate(CELL_TYPES):
    ref_profiles[marker_genes[ct], i] *= 20  # high expression for markers

# Expression = proportions @ ref_profiles.T + noise
expr = true_props @ ref_profiles.T + np.random.normal(0, 0.5, (N_SAMPLES, N_GENES))
expr = np.maximum(expr, 0)
print(f"[Data] Expression matrix shape: {expr.shape}")

# ─── 2. Cell-Type Deconvolution (NNLS) ───────────────────────────────────────
print("[Deconvolution] Running NNLS deconvolution...")
from scipy.optimize import nnls

est_props = np.zeros((N_SAMPLES, N_CELL_TYPES))
for s in range(N_SAMPLES):
    # Use marker genes only
    y = expr[s, all_marker_idx]
    A = ref_profiles[all_marker_idx, :]
    coeffs, _ = nnls(A, y)
    if coeffs.sum() > 0:
        est_props[s] = coeffs / coeffs.sum()
    else:
        est_props[s] = np.ones(N_CELL_TYPES) / N_CELL_TYPES

# Deconvolution accuracy
deconv_corr = np.array([np.corrcoef(true_props[:, i], est_props[:, i])[0, 1]
                         for i in range(N_CELL_TYPES)])
print(f"[Deconvolution] Mean correlation: {deconv_corr.mean():.4f}")

# ─── 3. Synaptic Gene Module Scoring ─────────────────────────────────────────
SYNAPTIC_GENES_NAMES = ['SNAP25', 'SYN1', 'GRIN2A', 'GRIA1', 'GRIA2', 'GRIN1',
                         'DLG4', 'SHANK3', 'HOMER1', 'NRXN1', 'NLGN1', 'CAMK2A',
                         'SYP', 'VAMP2', 'STX1A', 'STXBP1', 'CPLX1', 'CPLX2',
                         'SYT1', 'RIMS1', 'BASSOON', 'PICCOLO', 'ELKS', 'MUNC13',
                         'MUNC18', 'NSF', 'SNAP23', 'VAMP1', 'VAMP3', 'VAMP4',
                         'GRIN2B', 'GRIN2C', 'GRIN2D', 'GRIA3', 'GRIA4', 'GRIP1',
                         'PICK1', 'SHANK1', 'SHANK2', 'SYNGAP1', 'DLGAP1', 'DLGAP2',
                         'DLGAP3', 'DLGAP4', 'LRRTM1', 'LRRTM2', 'NRXN2', 'NRXN3',
                         'NLGN2', 'NLGN3']
N_SYN = len(SYNAPTIC_GENES_NAMES)
# Assign synaptic genes to specific gene indices
syn_gene_idx = np.random.choice(N_GENES, N_SYN, replace=False)
# Neurons have high synaptic expression
for idx in syn_gene_idx:
    expr[:, idx] += true_props[:, 0] * 10  # neuron proportion drives synaptic expression

synaptic_score = expr[:, syn_gene_idx].mean(axis=1)
print(f"[Synaptic] Mean synaptic score: {synaptic_score.mean():.4f}")

# ─── 4. Neurodegeneration Risk Score ─────────────────────────────────────────
AD_GENES = ['APOE', 'TREM2', 'BIN1', 'CLU', 'PICALM', 'CR1', 'ABCA7',
            'MS4A6A', 'EPHA1', 'CD33', 'CD2AP', 'SORL1', 'FERMT2', 'SLC24A4',
            'ZCWPW1', 'CELF1', 'NME8', 'IGHV1-67', 'INPP5D', 'MEF2C']
N_AD_GENES = len(AD_GENES)
ad_gene_idx = np.random.choice(N_GENES, N_AD_GENES, replace=False)
# AD samples have higher AD gene expression
for idx in ad_gene_idx:
    expr[diagnosis == 1, idx] += np.random.exponential(2, N_AD)

# Weights based on known effect sizes (simulated)
ad_weights = np.random.exponential(1, N_AD_GENES)
ad_weights /= ad_weights.sum()
neuro_risk = expr[:, ad_gene_idx] @ ad_weights
print(f"[NeuroDegen] AD risk score: AD={neuro_risk[diagnosis==1].mean():.4f}, "
      f"Ctrl={neuro_risk[diagnosis==0].mean():.4f}")

# ─── 5. Region-Specific Expression Analysis (ANOVA) ──────────────────────────
print("[Region] Running ANOVA across 5 brain regions...")
# Select 200 genes for ANOVA (faster)
gene_subset = np.random.choice(N_GENES, 200, replace=False)
f_stats = np.zeros(200)
p_vals = np.zeros(200)
for gi, g in enumerate(gene_subset):
    groups = [expr[region_labels == r, g] for r in range(N_REGIONS)]
    f_stats[gi], p_vals[gi] = stats.f_oneway(*groups)

region_specific = gene_subset[p_vals < 0.05]
print(f"[Region] Region-specific genes (p<0.05): {len(region_specific)}/200")

# Region mean expression (top 20 region-specific genes)
top_region_genes = gene_subset[np.argsort(f_stats)[-20:]]
region_expr = np.zeros((N_REGIONS, 20))
for r in range(N_REGIONS):
    region_expr[r] = expr[region_labels == r][:, top_region_genes].mean(axis=0)

# ─── 6. Neuron-Glia Communication Network ────────────────────────────────────
N_LR_PAIRS = 20
# Simulate ligand-receptor pairs between cell types
lr_pairs = []
for _ in range(N_LR_PAIRS):
    sender = np.random.randint(N_CELL_TYPES)
    receiver = np.random.randint(N_CELL_TYPES)
    strength = np.random.exponential(1)
    lr_pairs.append((sender, receiver, strength))

# Communication matrix
comm_matrix = np.zeros((N_CELL_TYPES, N_CELL_TYPES))
for s, r, strength in lr_pairs:
    comm_matrix[s, r] += strength

# ─── 7. AD Risk Gene Expression Heatmap ──────────────────────────────────────
# Sample 30 AD and 30 control for heatmap
ad_idx = np.where(diagnosis == 1)[0][:30]
ctrl_idx = np.where(diagnosis == 0)[0][:30]
hm_samples = np.concatenate([ctrl_idx, ad_idx])
ad_hm_expr = expr[hm_samples][:, ad_gene_idx]
# Normalize per gene
ad_hm_norm = (ad_hm_expr - ad_hm_expr.mean(axis=0)) / (ad_hm_expr.std(axis=0) + 1e-8)

# ─── Dashboard ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

COLORS = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8',
          '#ff922b', '#74c0fc', '#f783ac', '#a9e34b']
TEXT_COLOR = 'white'
GRID_COLOR = '#333333'

def style_ax(ax, title):
    ax.set_facecolor('#111111')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, fontweight='bold', pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)

# Panel 1: Cell-type composition heatmap
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Cell-Type Composition (Samples × 7 Types)')
# Show first 50 samples
im = ax1.imshow(est_props[:50].T, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR, labelsize=7)
ax1.set_yticks(range(N_CELL_TYPES))
ax1.set_yticklabels([ct[:8] for ct in CELL_TYPES], fontsize=7)
ax1.set_xlabel('Sample Index', fontsize=8)
ax1.grid(False)

# Panel 2: AD vs control cell-type proportions
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'AD vs Control Cell-Type Proportions')
x = np.arange(N_CELL_TYPES)
width = 0.35
ad_mean = est_props[diagnosis == 1].mean(axis=0)
ctrl_mean = est_props[diagnosis == 0].mean(axis=0)
ax2.bar(x - width/2, ctrl_mean, width, label='Control', color=COLORS[0], alpha=0.8)
ax2.bar(x + width/2, ad_mean, width, label='AD', color=COLORS[1], alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([ct[:6] for ct in CELL_TYPES], rotation=45, ha='right', fontsize=7)
ax2.set_ylabel('Mean Proportion', fontsize=8)
ax2.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)

# Panel 3: Synaptic score by region and diagnosis
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Synaptic Score by Region & Diagnosis')
positions = []
labels_3 = []
data_3 = []
for r in range(N_REGIONS):
    for d, dlabel in enumerate(['Ctrl', 'AD']):
        mask = (region_labels == r) & (diagnosis == d)
        data_3.append(synaptic_score[mask])
        positions.append(r * 2.5 + d * 1.0)
        labels_3.append(f'{REGIONS[r][:5]}\n{dlabel}')
bp = ax3.boxplot(data_3, positions=positions, widths=0.7, patch_artist=True,
                  medianprops=dict(color='white', lw=2))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(COLORS[0] if i % 2 == 0 else COLORS[1])
    patch.set_alpha(0.7)
ax3.set_xticks(positions[::2])
ax3.set_xticklabels([REGIONS[r][:5] for r in range(N_REGIONS)], fontsize=7)
ax3.set_ylabel('Synaptic Score', fontsize=8)

# Panel 4: Neurodegeneration risk score distribution
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, 'Neurodegeneration Risk Score Distribution')
ax4.hist(neuro_risk[diagnosis == 0], bins=20, color=COLORS[0], alpha=0.7, label='Control', density=True)
ax4.hist(neuro_risk[diagnosis == 1], bins=20, color=COLORS[1], alpha=0.7, label='AD', density=True)
ax4.set_xlabel('AD Risk Score', fontsize=8)
ax4.set_ylabel('Density', fontsize=8)
ax4.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)

# Panel 5: Region-specific gene expression heatmap
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, 'Region-Specific Gene Expression (Top 20)')
# Normalize
region_expr_norm = (region_expr - region_expr.mean(axis=0)) / (region_expr.std(axis=0) + 1e-8)
im = ax5.imshow(region_expr_norm, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR, labelsize=7)
ax5.set_yticks(range(N_REGIONS))
ax5.set_yticklabels([r[:8] for r in REGIONS], fontsize=7)
ax5.set_xlabel('Gene Index', fontsize=8)
ax5.grid(False)

# Panel 6: Neuron-glia communication network (matrix)
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, 'Neuron-Glia Communication Network')
im = ax6.imshow(comm_matrix, cmap='hot', aspect='auto')
plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR, labelsize=7)
ax6.set_xticks(range(N_CELL_TYPES))
ax6.set_yticks(range(N_CELL_TYPES))
ax6.set_xticklabels([ct[:5] for ct in CELL_TYPES], rotation=45, ha='right', fontsize=7)
ax6.set_yticklabels([ct[:5] for ct in CELL_TYPES], fontsize=7)
ax6.set_xlabel('Receiver', fontsize=8)
ax6.set_ylabel('Sender', fontsize=8)
ax6.grid(False)

# Panel 7: AD risk gene expression heatmap
ax7 = fig.add_subplot(gs[2, 0])
style_ax(ax7, 'AD Risk Gene Expression Heatmap')
im = ax7.imshow(ad_hm_norm.T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR, labelsize=7)
ax7.axvline(29.5, color='yellow', lw=2, linestyle='--', label='AD|Ctrl')
ax7.set_yticks(range(N_AD_GENES))
ax7.set_yticklabels(AD_GENES, fontsize=5)
ax7.set_xlabel('Sample (Ctrl | AD)', fontsize=8)
ax7.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
ax7.grid(False)

# Panel 8: Deconvolution accuracy
ax8 = fig.add_subplot(gs[2, 1])
style_ax(ax8, 'Cell-Type Deconvolution Accuracy')
x = np.arange(N_CELL_TYPES)
bars = ax8.bar(x, deconv_corr, color=COLORS[:N_CELL_TYPES], alpha=0.8)
ax8.set_xticks(x)
ax8.set_xticklabels([ct[:8] for ct in CELL_TYPES], rotation=45, ha='right', fontsize=7)
ax8.set_ylabel('Pearson r (True vs Est)', fontsize=8)
ax8.set_ylim(0, 1)
ax8.axhline(deconv_corr.mean(), color='white', lw=1.5, linestyle='--',
            label=f'Mean r={deconv_corr.mean():.3f}')
ax8.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)

# Panel 9: Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#111111')
ax9.axis('off')
for spine in ax9.spines.values():
    spine.set_edgecolor('#444444')
t_stat, t_p = stats.ttest_ind(neuro_risk[diagnosis == 1], neuro_risk[diagnosis == 0])
summary_lines = [
    '══ NeurogenomicsEngine Summary ══',
    '',
    f'  Samples: {N_SAMPLES} ({N_AD} AD, {N_CTRL} Ctrl)',
    f'  Genes: {N_GENES}',
    f'  Brain regions: {N_REGIONS}',
    f'  Cell types: {N_CELL_TYPES}',
    '',
    f'  Deconv mean r: {deconv_corr.mean():.4f}',
    f'  Best cell type: {CELL_TYPES[deconv_corr.argmax()]}',
    f'  Best r: {deconv_corr.max():.4f}',
    '',
    f'  Synaptic score (AD): {synaptic_score[diagnosis==1].mean():.4f}',
    f'  Synaptic score (Ctrl): {synaptic_score[diagnosis==0].mean():.4f}',
    '',
    f'  AD risk score (AD): {neuro_risk[diagnosis==1].mean():.4f}',
    f'  AD risk score (Ctrl): {neuro_risk[diagnosis==0].mean():.4f}',
    f'  t-test p-value: {t_p:.2e}',
    '',
    f'  Region-specific genes: {len(region_specific)}/200',
    f'  LR pairs simulated: {N_LR_PAIRS}',
]
ax9.text(0.05, 0.97, '\n'.join(summary_lines), transform=ax9.transAxes,
         color=TEXT_COLOR, fontsize=7.5, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8, edgecolor='#444'))

fig.suptitle('NeurogenomicsEngine: Brain Cell-Type Transcriptomics Dashboard',
             color=TEXT_COLOR, fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/workspace/subagents/70405644/neurogenomics_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("[Dashboard] Saved: neurogenomics_dashboard.png")

# ─── Structured Summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  NEUROGENOMICS ENGINE — STRUCTURED SUMMARY")
print("="*60)
print(f"  Samples:                   {N_SAMPLES} ({N_AD} AD, {N_CTRL} Ctrl)")
print(f"  Genes:                     {N_GENES}")
print(f"  Brain regions:             {N_REGIONS}")
print(f"  Cell types:                {N_CELL_TYPES}")
print(f"  Deconv mean r:             {deconv_corr.mean():.6f}")
print(f"  Best cell type:            {CELL_TYPES[deconv_corr.argmax()]} (r={deconv_corr.max():.4f})")
print(f"  Synaptic score AD:         {synaptic_score[diagnosis==1].mean():.6f}")
print(f"  Synaptic score Ctrl:       {synaptic_score[diagnosis==0].mean():.6f}")
print(f"  AD risk score AD:          {neuro_risk[diagnosis==1].mean():.6f}")
print(f"  AD risk score Ctrl:        {neuro_risk[diagnosis==0].mean():.6f}")
print(f"  AD risk t-test p:          {t_p:.4e}")
print(f"  Region-specific genes:     {len(region_specific)}/200")
print(f"  LR pairs simulated:        {N_LR_PAIRS}")
print("="*60)
