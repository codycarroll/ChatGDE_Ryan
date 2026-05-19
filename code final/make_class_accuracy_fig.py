"""
Per-class recall (overall + by GSP) and confusion matrices (count + fraction).
Saves:
  images/class_accuracy_overall.png
  images/class_accuracy_by_gsp.png
  images/binary_accuracy_overall.png
  images/binary_accuracy_by_gsp.png
  images/confusion_matrices.png
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUBRIC_DIR = os.path.expanduser('~/Desktop/gsps_all/ChatGDE_Draft_Scoring_Rubrics_CSV')
no_test    = [2, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 23, 26, 27, 35, 38, 39, 69]

TRIAL_GSPS = [
    ('BigValley',           'Big Valley',      '1_BigValley_DraftGSP_ScoringRubric.csv'),
    ('EastContraCosta',     'E. Contra Costa', '14_EastContraCosta_DraftGSP_ScoringRubric.csv'),
    ('Fillmore',            'Fillmore',         '15_Fillmore_DraftGSP_ScoringRubric.csv'),
    ('SonomaValley',        'Sonoma',           '30_SonomaValley_DraftGSP_ScoringRubric.csv'),
    ('SanLuisObispoValley', 'San Luis Obispo',  '50_SanLuisObispoValley_DraftGSP_ScoringRubric.csv'),
]

CLASSES    = ['Yes', 'Somewhat', 'No']
GSP_CNAMES = [g[0] for g in TRIAL_GSPS]
GSP_LABELS = [g[1] for g in TRIAL_GSPS]

os.makedirs('images', exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def HumanRubric(path):
    df = pd.read_csv(path)
    df = df.iloc[10:, 3:].reset_index().drop('index', axis=1)
    df.columns = df.iloc[0]
    return df[1:]

def load_rubric_answers(rubric_filename):
    rubric  = HumanRubric(os.path.join(RUBRIC_DIR, rubric_filename))
    answers = rubric['Answer'].drop(no_test, errors='ignore')
    return [str(v).strip() for v in answers]

def parse_answer(response):
    for line in str(response).split('\n'):
        if line.strip().upper().startswith('ANSWER:'):
            return line.strip()[7:].strip().split(',')[0].strip()
    return 'Unknown'

def per_class_recall(true_list, pred_list):
    """Returns {cls: recall} for Yes/Somewhat/No, ignoring NotApplicable rows."""
    out = {}
    for cls in CLASSES:
        idx = [i for i, t in enumerate(true_list) if t == cls]
        out[cls] = sum(pred_list[i] == cls for i in idx) / len(idx) if idx else np.nan
    return out

def confusion_matrix_3class(true_list, pred_list):
    """Returns 3×3 count matrix, rows=true, cols=predicted (Yes/Somewhat/No)."""
    mat = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_list, pred_list):
        if t in CLASSES and p in CLASSES:
            mat[CLASSES.index(t), CLASSES.index(p)] += 1
    return mat

# ── Load rubric answers per GSP ───────────────────────────────────────────────
rubric_by_gsp = {cname: load_rubric_answers(rf) for cname, _, rf in TRIAL_GSPS}

# ── Load predictions per model per GSP ───────────────────────────────────────
def load_ckpt_dict(ckpt_file, key_map):
    with open(ckpt_file) as f:
        ckpt = json.load(f)
    return {cname: [parse_answer(r) for r in ckpt[ckpt_key]]
            for cname, ckpt_key in key_map.items()}

def load_opus_preds():
    gsp_files = {
        'BigValley':           'bigvalley',
        'EastContraCosta':     'eastcontracosta',
        'Fillmore':            'fillmore',
        'SonomaValley':        'sonoma',
        'SanLuisObispoValley': 'sanluisobispovalley',
    }
    out = {}
    for cname, slug in gsp_files.items():
        fpath = sorted(glob.glob(f'results/checkpoint_opus47_vision_{slug}_*.json'))[-1]
        with open(fpath) as f:
            out[cname] = [parse_answer(r) for r in json.load(f)]
    return out

std_keys = {c: c for c in GSP_CNAMES}
o3_keys  = {
    'BigValley': 'BigValley', 'EastContraCosta': 'EastContraCosta',
    'Fillmore': 'Fillmore', 'SonomaValley': 'Sonoma', 'SanLuisObispoValley': 'SLO',
}

MODELS = [
    ('GPT-3.5 FT',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_gpt35ftv4_trial5_*.json'))[-1], std_keys),
     '#E69F00'),
    ('GPT-4o',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_gpt4o_trial5_*.json'))[-1], std_keys),
     '#56B4E9'),
    ('GPT-4o FT',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_gpt4oftv4_trial5_*.json'))[-1], std_keys),
     '#0072B2'),
    ('GPT-4.1',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_gpt41_trial5_*.json'))[-1], std_keys),
     '#009E73'),
    ('GPT-4.1 FT',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_gpt41ftv4_trial5_*.json'))[-1], std_keys),
     '#D55E00'),
    ('GPT-5.5',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_gpt55_trial5_*.json'))[-1], std_keys),
     '#CC79A7'),
    ('o3',
     load_ckpt_dict('results/checkpoint_o3_finetuned_20260312_174405.json', o3_keys),
     '#000000'),
    ('Sonnet 4.6',
     load_ckpt_dict(sorted(glob.glob('results/checkpoint_sonnet46_trial5_*.json'))[-1], std_keys),
     '#C9A800'),
    ('Opus 4.7\n(vision)',
     load_opus_preds(),
     '#666666'),
]

n_models = len(MODELS)
x        = np.arange(len(CLASSES))
width    = 0.08
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

def draw_class_accuracy_panel(ax, true_list, pred_by_model, title,
                              ylabel=True, include_overall=False, show_labels=False,
                              tick_fs=12, label_fs=13):
    n_per_cls = {cls: sum(t == cls for t in true_list) for cls in CLASSES}
    n_total   = sum(n_per_cls.values())

    x_pos = np.arange(len(CLASSES) + (1 if include_overall else 0))

    for mi, (name, _, color) in enumerate(MODELS):
        rec  = per_class_recall(true_list, pred_by_model[name])
        vals = [rec[cls] for cls in CLASSES]

        if include_overall:
            preds = pred_by_model[name]
            n_correct = sum(t == p for t, p in zip(true_list, preds) if t in CLASSES)
            vals_plot = vals + [n_correct / n_total]
        else:
            vals_plot = vals

        bars = ax.bar(x_pos + offsets[mi], vals_plot, width, color=color, alpha=0.88,
                      edgecolor='white', linewidth=0.4, label=name)
        if show_labels:
            for bar, val in zip(bars, vals_plot):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                            f'{val:.0%}', ha='center', va='bottom', fontsize=8,
                            fontweight='bold', color=color)

    tick_labels = [f'{cls}\n(n={n_per_cls[cls]})' for cls in CLASSES]
    if include_overall:
        tick_labels += [f'Overall\n(n={n_total})']
        ax.axvline(len(CLASSES) - 0.5, color='#aaaaaa', lw=1.2, linestyle='--', zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_xlim(x_pos[0] - 0.55, x_pos[-1] + 0.55)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    if ylabel:
        ax.set_ylabel('Accuracy', fontsize=label_fs)

# ── Overall: all 5 GSPs pooled ────────────────────────────────────────────────
true_overall = sum((rubric_by_gsp[c] for c in GSP_CNAMES), [])
pred_overall = {name: sum((preds[c] for c in GSP_CNAMES), []) for name, preds, _ in MODELS}

fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
draw_class_accuracy_panel(ax, true_overall, pred_overall, 'Overall (all 5 Trial GSPs)',
                          include_overall=True, show_labels=True)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=11, loc='upper right', framealpha=0.92, ncol=2)
plt.tight_layout()
plt.savefig('images/class_accuracy_overall.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: images/class_accuracy_overall.png')

# ── Per-GSP: 2×3 grid (5 panels + 1 legend slot) ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 9), dpi=150)
axes = axes.flatten()

for panel_idx, (cname, label) in enumerate(zip(GSP_CNAMES, GSP_LABELS)):
    pred_gsp = {name: preds[cname] for name, preds, _ in MODELS}
    draw_class_accuracy_panel(axes[panel_idx], rubric_by_gsp[cname], pred_gsp,
                              label, ylabel=(panel_idx % 3 == 0), show_labels=False,
                              tick_fs=16, label_fs=17)
    axes[panel_idx].set_title(f'{chr(ord("a") + panel_idx)}.  {label}',
                              fontsize=16, fontweight='bold')

axes[5].axis('off')
handles, labels = axes[0].get_legend_handles_labels()
axes[5].legend(handles, labels, loc='center', fontsize=15, framealpha=0.92,
               title='Model', title_fontsize=15)

plt.tight_layout()
plt.savefig('images/class_accuracy_by_gsp.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: images/class_accuracy_by_gsp.png')

# ════════════════════════════════════════════════════════════════════════════
# Figures 3 & 4 — Binary (Yes vs. No+Somewhat) accuracy
# ════════════════════════════════════════════════════════════════════════════
BIN_CLASSES = ['Yes', 'No+Somewhat']

def binarize(label):
    return 'Yes' if label == 'Yes' else ('No+Somewhat' if label in ('No', 'Somewhat') else None)

def draw_binary_panel(ax, true_list, pred_by_model, title, ylabel=True,
                      include_overall=False, show_labels=False,
                      tick_fs=12, label_fs=13):
    true_bin  = [binarize(t) for t in true_list]
    n_per_cls = {cls: sum(t == cls for t in true_bin if t) for cls in BIN_CLASSES}
    n_total   = sum(n_per_cls.values())

    x_pos = np.arange(len(BIN_CLASSES) + (1 if include_overall else 0))
    for mi, (name, _, color) in enumerate(MODELS):
        pred_bin = [binarize(p) for p in pred_by_model[name]]
        vals = [
            sum(p == cls for t, p in zip(true_bin, pred_bin) if t == cls) /
            n_per_cls[cls] if n_per_cls[cls] else np.nan
            for cls in BIN_CLASSES
        ]
        if include_overall:
            n_correct = sum(t == p for t, p in zip(true_bin, pred_bin) if t)
            vals = vals + [n_correct / n_total]
        bars = ax.bar(x_pos + offsets[mi], vals, width, color=color, alpha=0.88,
                      edgecolor='white', linewidth=0.4, label=name)
        if show_labels:
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                            f'{val:.0%}', ha='center', va='bottom', fontsize=8,
                            fontweight='bold', color=color)

    tick_labels = [f'{cls}\n(n={n_per_cls.get(cls, 0)})' for cls in BIN_CLASSES]
    if include_overall:
        tick_labels += [f'Overall\n(n={n_total})']
        ax.axvline(len(BIN_CLASSES) - 0.5, color='#aaaaaa', lw=1.2, linestyle='--', zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_xlim(x_pos[0] - 0.55, x_pos[-1] + 0.55)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    if ylabel:
        ax.set_ylabel('Accuracy', fontsize=label_fs)

# Overall binary
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
draw_binary_panel(ax, true_overall, pred_overall, 'Overall — Binary (Yes vs. No+Somewhat)',
                  include_overall=True, show_labels=True)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=11, loc='upper right', framealpha=0.92, ncol=2)
plt.tight_layout()
plt.savefig('images/binary_accuracy_overall.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: images/binary_accuracy_overall.png')

# Per-GSP binary
fig, axes = plt.subplots(2, 3, figsize=(20, 9), dpi=150)
axes = axes.flatten()
for panel_idx, (cname, label) in enumerate(zip(GSP_CNAMES, GSP_LABELS)):
    pred_gsp = {name: preds[cname] for name, preds, _ in MODELS}
    draw_binary_panel(axes[panel_idx], rubric_by_gsp[cname], pred_gsp,
                      label, ylabel=(panel_idx % 3 == 0), show_labels=False,
                      tick_fs=16, label_fs=17)
    axes[panel_idx].set_title(f'{chr(ord("a") + panel_idx)}.  {label}',
                              fontsize=16, fontweight='bold')
axes[5].axis('off')
handles, labels = axes[0].get_legend_handles_labels()
axes[5].legend(handles, labels, loc='center', fontsize=15, framealpha=0.92,
               title='Model', title_fontsize=15)
plt.tight_layout()
plt.savefig('images/binary_accuracy_by_gsp.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: images/binary_accuracy_by_gsp.png')

# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Confusion matrices (count top, row-fraction bottom) — 2×9 grid
# ════════════════════════════════════════════════════════════════════════════

# Overall true + pred for each model
true_all = sum((rubric_by_gsp[c] for c in GSP_CNAMES), [])
# Only rows where true is Yes/Somewhat/No
active_idx = [i for i, t in enumerate(true_all) if t in CLASSES]
true_active = [true_all[i] for i in active_idx]

fig, axes = plt.subplots(2, len(MODELS), figsize=(20, 6), dpi=150,
                         constrained_layout=True)

for mi, (name, preds_by_gsp, color) in enumerate(MODELS):
    pred_all    = sum((preds_by_gsp[c] for c in GSP_CNAMES), [])
    pred_active = [pred_all[i] for i in active_idx]
    mat_count   = confusion_matrix_3class(true_active, pred_active)
    # Row-normalize (true-class recall fractions)
    row_sums    = mat_count.sum(axis=1, keepdims=True).astype(float)
    mat_frac    = np.where(row_sums > 0, mat_count / row_sums, 0.0)

    for row_idx, (mat, fmt, title_suffix) in enumerate([
        (mat_count, 'd',    'Count'),
        (mat_frac,  '.2f',  'Fraction'),
    ]):
        ax = axes[row_idx, mi]
        vmax = mat.max() if row_idx == 0 else 1.0
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=vmax, aspect='equal')

        for r in range(3):
            for c in range(3):
                val     = mat[r, c]
                txt     = f'{val:{fmt}}' if fmt == 'd' else f'{val:.2f}'
                bg_dark = val > vmax * 0.55
                ax.text(c, r, txt, ha='center', va='center', fontsize=12,
                        fontweight='bold',
                        color='white' if bg_dark else '#222222')

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        if row_idx == 0:
            ax.set_title(name.replace('\n', ' '), fontsize=12, fontweight='bold',
                         color=color, pad=8)
        if row_idx == 1:
            ax.set_xticklabels(CLASSES, fontsize=11, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
        if mi == 0:
            ax.set_yticklabels(CLASSES, fontsize=11)
            ax.set_ylabel(title_suffix, fontsize=12, fontweight='bold', labelpad=6)
        else:
            ax.set_yticklabels([])

        # Diagonal highlight
        for d in range(3):
            ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                                       fill=False, edgecolor='#FF6B35', lw=1.8))

fig.supxlabel('Predicted Response', fontsize=13, fontweight='bold')
fig.supylabel('True Response', fontsize=13, fontweight='bold')

plt.savefig('images/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: images/confusion_matrices.png')
