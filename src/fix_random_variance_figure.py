#!/usr/bin/env python3
"""Generate fig_random_variance.png — retention box-plots, muted colours, 1x3.

Retention = pruned_score / defult_score * 100.
"""

import os, json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── paths ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_DIR / "Beyond_Redundancy__Diagnosing_Information_Distribution_in_Text_Embeddings_via_Task_Aware_Dimension_Selection"
OUTPUT = PAPER_DIR / 'figures' / 'fig_random_variance.png'
RV_JSON = PROJECT_DIR / 'data' / 'experiment_results' / 'random_variance_tail_risk.json'
ANALYZE_DIR = PROJECT_DIR / 'data' / 'analyze'

# ── model metadata ─────────────────────────────────────────────────
MODEL_DISPLAY = {
    "gte-large-en-v1.5":     "GTE-Large",
    "stella_en_400M_v5":      "Stella EN 400M",
    "roberta-large-InBedder": "RoBERTa-InBedder",
    "bge-m3":                 "BGE-M3",
    "instructor-large":       "Instructor-Large",
    "mxbai-embed-large-v1":   "MxBai-Embed-Large",
    "gte-base":               "GTE-Base",
    "gtr-t5-large":           "GTR-T5-Large",
    "bart-base":              "BART-Base",
    "roberta-large":          "RoBERTa-Large",
    "Qwen3-Embedding-0.6B":   "Qwen-Embed.-0.6B",
    "inbedder-roberta-large": "RoBERTa-InBedder",
}
# ── paper classification ────────────────────────────────────────────
RETRIEVAL_OPT = {"GTE-Large", "BGE-M3", "MxBai-Embed-Large", "GTE-Base",
                 "GTR-T5-Large"}
INSTRUCTION   = {"Stella EN 400M", "RoBERTa-InBedder",
                 "Instructor-Large", "Qwen-Embed.-0.6B"}
BACKBONE      = {"RoBERTa-Large", "BART-Base"}

# ── muted colour system ────────────────────────────────────────────
C_P5    = '#C2702E'   # muted orange
C_CVAR  = '#7B5EA7'   # muted purple
C_REF100 = '#5A9E6F'  # muted green
C_REF90  = '#C0564B'  # muted red
C_TEXT  = '#1A1A1A'

# Blue family – Retrieval-optimized
RETR_HUES = ['#3B6FA0', '#5A94B8', '#7BB4D0', '#2E5E8C', '#4A88AF']
# Green-teal family – Instruction-conditioned
INST_HUES = ['#2E7D5B', '#4A9E78', '#6DB896', '#3D8B69']
# Warm family – General-purpose backbones
BACK_HUES = ['#B85B45', '#CC7E57']


def model_color(display_name, idx=0):
    if display_name in RETRIEVAL_OPT:
        return RETR_HUES[idx % len(RETR_HUES)]
    if display_name in INSTRUCTION:
        return INST_HUES[(idx - len(RETRIEVAL_OPT)) % len(INST_HUES)]
    return BACK_HUES[(idx - len(RETRIEVAL_OPT) - len(INSTRUCTION)) % len(BACK_HUES)]


# ── load defult_score from analyze JSONs ────────────────────────────
def load_full_dim_scores():
    scores = {}
    if not os.path.isdir(ANALYZE_DIR):
        return scores
    for fname in os.listdir(ANALYZE_DIR):
        if not fname.endswith('.json'):
            continue
        model_name = fname[:-5]
        try:
            d = json.load(open(os.path.join(ANALYZE_DIR, fname)))
            task_map = d.get('task_name', {})
            scores[model_name] = {}
            for tname, td in task_map.items():
                ds = td.get('defult_score')
                if ds is not None and ds > 0:
                    scores[model_name][tname] = ds
        except Exception:
            pass
    return scores


# ── compute retention table ────────────────────────────────────────
def build_retention_table(rv_data, full_dim_scores):
    budgets = rv_data['config']['budgets']
    models  = list(rv_data['per_model'].keys())
    table   = {}

    for model in models:
        pm    = rv_data['per_model'][model]
        tasks = pm.get('tasks', {})
        table[model] = {}

        ref_map = full_dim_scores.get(model, {})
        if not ref_map:
            for alt in (model.lower(), model.replace('_', '-'),
                        model.replace('-', '_')):
                if alt in full_dim_scores:
                    ref_map = full_dim_scores[alt]
                    break

        for budget in budgets:
            per_task = {k: [] for k in
                        ('mean_ret', 'p5_ret', 'cvar_ret', 'cv',
                         'median_ret', 'p25_ret', 'p75_ret')}

            for tname, tdata in tasks.items():
                bdata = tdata.get('budgets', {}).get(str(budget))
                if not bdata:
                    continue
                ref = ref_map.get(tname)
                if not ref or ref <= 0:
                    continue

                per_task['mean_ret'].append(bdata.get('mean', 0) / ref * 100)
                per_task['p5_ret'].append(bdata.get('p5', 0) / ref * 100)
                per_task['cvar_ret'].append(bdata.get('cvar_5', 0) / ref * 100)
                per_task['median_ret'].append(bdata.get('median', 0) / ref * 100)
                per_task['p25_ret'].append(bdata.get('p25', 0) / ref * 100)
                per_task['p75_ret'].append(bdata.get('p75', 0) / ref * 100)
                cv_raw = bdata.get('cv')
                if cv_raw is not None:
                    per_task['cv'].append(cv_raw * 100)

            table[model][budget] = per_task

    return table, budgets, models


# ── style ──────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset':  'stix',
        'font.weight':       'bold',
        'font.size':         18,
        'text.color':        C_TEXT,
        'figure.facecolor':  'white',
        'axes.facecolor':    '#FEFEFE',
        'axes.edgecolor':    '#888888',
        'axes.labelcolor':   C_TEXT,
        'axes.labelsize':    20,
        'axes.titlesize':    22,
        'axes.titleweight':  'bold',
        'xtick.labelsize':   16,
        'ytick.labelsize':   16,
        'legend.fontsize':   16,
        'figure.dpi':        300,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.facecolor': 'white',
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.color':        '#D0D0D0',
        'grid.linewidth':    0.5,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.7,
    })


# ── draw ───────────────────────────────────────────────────────────
def draw_figure(table, budgets, models):
    apply_style()
    show_budgets = [b for b in (64, 128, 256) if b in budgets]

    fig, axes = plt.subplots(1, 3, figsize=(22, 10), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.26,
                        wspace=0.10)

    for col, budget in enumerate(show_budgets):
        ax = axes[col]
        vals_list, labels, face_c, edge_c, p5_avg, cvar_avg = (
            [], [], [], [], [], [])

        for mi, model in enumerate(models):
            display = MODEL_DISPLAY.get(model, model)
            bd = table.get(model, {}).get(budget, {})
            mv = bd.get('mean_ret', [])
            pv = bd.get('p5_ret', [])
            cv = bd.get('cvar_ret', [])
            if not mv:
                continue
            vals_list.append(mv)
            labels.append(display)
            c = model_color(display, mi)
            face_c.append(c + '44')
            edge_c.append(c)
            p5_avg.append(np.mean(pv) if pv else np.nan)
            cvar_avg.append(np.mean(cv) if cv else np.nan)

        if not vals_list:
            continue
        positions = np.arange(len(vals_list))

        bp = ax.boxplot(
            vals_list, positions=positions, patch_artist=True,
            widths=0.55, showfliers=False,
            medianprops=dict(color=C_TEXT, linewidth=1.8),
            whiskerprops=dict(color='#777', linewidth=1, linestyle='--'),
            capprops=dict(color='#777', linewidth=0.8),
            boxprops=dict(linewidth=1.2))

        for patch, fc, ec in zip(bp['boxes'], face_c, edge_c):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)

        for i, (p5, cvar) in enumerate(zip(p5_avg, cvar_avg)):
            if not np.isnan(p5):
                ax.plot(i, p5, marker='v', color=C_P5, markersize=10,
                        zorder=5, markeredgecolor='white',
                        markeredgewidth=0.8)
            if not np.isnan(cvar):
                ax.plot(i, cvar, marker='D', color=C_CVAR, markersize=8,
                        zorder=5, markeredgecolor='white',
                        markeredgewidth=0.6)

        ax.axhline(100, color=C_REF100, ls='--', lw=0.9, alpha=0.5, zorder=1)
        ax.axhline(90, color=C_REF90, ls=':', lw=0.8, alpha=0.35, zorder=1)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=50, ha='right')
        ax.set_title(f'dim = {budget}', pad=12)
        if col == 0:
            ax.set_ylabel('Retention (%)')

    # single shared legend at bottom
    legend_els = [
        Line2D([0], [0], marker='v', color='w', markerfacecolor=C_P5,
               markersize=10, markeredgecolor='white', markeredgewidth=0.8,
               label='P5 Retention'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_CVAR,
               markersize=8, markeredgecolor='white', markeredgewidth=0.6,
               label='CVaR-5 Retention'),
        Patch(facecolor=RETR_HUES[0] + '44', edgecolor=RETR_HUES[0],
              linewidth=1.2, label='Retrieval-optimized'),
        Patch(facecolor=INST_HUES[0] + '44', edgecolor=INST_HUES[0],
              linewidth=1.2, label='Instruction-conditioned'),
        Patch(facecolor=BACK_HUES[0] + '44', edgecolor=BACK_HUES[0],
              linewidth=1.2, label='General-purpose Backbone'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=5,
               fontsize=16, framealpha=0.95, edgecolor='#CCC',
               fancybox=True, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        'Random Dimension Selection: Retention Distribution Across Tasks',
        fontsize=24, fontweight='bold', y=0.97, color=C_TEXT)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    fig.savefig(OUTPUT.with_suffix('.pdf'))
    plt.close(fig)
    print(f'Saved: {OUTPUT}')


# ── main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading full-dim scores ...')
    full_dim = load_full_dim_scores()
    print(f'  {len(full_dim)} models')

    print('Loading random variance data...')
    rv = json.load(open(RV_JSON))

    print('Computing retention table...')
    table, budgets, models = build_retention_table(rv, full_dim)

    for m in models:
        for b in budgets:
            bd = table.get(m, {}).get(b, {})
            mr = bd.get('mean_ret', [])
            p5r = bd.get('p5_ret', [])
            if mr:
                suffix = f'  p5={np.mean(p5r):.1f}%' if p5r else ''
                print(f'  {MODEL_DISPLAY.get(m,m):12s} dim={b:>3d}: '
                      f'mean={np.mean(mr):.1f}%{suffix}')

    print('Drawing figure...')
    draw_figure(table, budgets, models)
