#!/usr/bin/env python3
"""Figures for report.tex.

Regenerates figures/{menu_selection,mvp_results,app_entropy,app_severity,app_frontier}.pdf
from the measured Phase-1 profile and the Phase-3 analysis CSVs. Every figure is authored at
its final printed width (6.5in = \\textwidth) so font sizes are true-size in the PDF.

Usage: python scripts/make_report_figures.py
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=0.78)
OUT = Path("/project/6104653/ehghaghi/proteus/figures")
d = json.load(open("/home/ehghaghi/scratch/ehghaghi/proteus/mvp/profile/2002/profile.json"))
SEL = ["qwen2.5-3b-instruct", "qwen3-14b", "qwen3-4b", "qwen3-8b"]; TAU = 0.7
qids = d["qids"]; pids = sorted(d["j_matrix"][qids[0]])
J = np.array([[d["j_matrix"][q][p] for p in pids] for q in qids], float)
jb = J.mean(1); helps = np.array([d["help_rate"][q] for q in qids])

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45),
                         gridspec_kw={"width_ratios": [1.0, 1.05]})

ax = axes[0]
sel = np.array([q in SEL for q in qids])
ax.axvline(TAU, color="crimson", ls="--", lw=1.0)
ax.axvspan(0.55, TAU, color="crimson", alpha=0.06)
ax.scatter(helps[~sel], jb[~sel], s=26, c="0.55", edgecolor="0.3", lw=.4,
           label="excluded", zorder=3)
ax.scatter(helps[sel], jb[sel], s=44, c="#2b6cb0", marker="D", edgecolor="black",
           lw=0.4, label="selected", zorder=4)
offs = {"qwen3-4b-saferl": (.004, .030), "gemma3-12b": (-.062, .019),
        "qwen3.5-9b": (.004, -.050), "gemma3-4b": (.004, .019),
        "qwen3-14b": (.006, -.032), "qwen2.5-3b-instruct": (-.056, -.058),
        "qwen3-8b": (.005, .022), "qwen3-4b": (.005, .021),
        "gemma3-12b-abliterated": (-.118, .023), "qwen2.5-1.5b-instruct": (.004, -.052)}
for q, h, b in zip(qids, helps, jb):
    dx, dy = offs.get(q, (.004, .015)); ax.annotate(q, (h+dx, b+dy), fontsize=5.4)
ax.text(TAU-.004, .53, r"$\tau=0.7$", rotation=90, va="top", ha="right", fontsize=6.5,
        color="crimson")
ax.set_xlabel(r"Helpfulness $\mathrm{Help}(q)$", fontsize=8)
ax.set_ylabel(r"Jailbreak rate $\mathrm{JB}(q)$", fontsize=8)
ax.set_xlim(.545, .835); ax.set_ylim(-.06, .58)
ax.tick_params(labelsize=7)
ax.legend(loc="upper right", fontsize=6.5, handletextpad=.3, borderpad=.3)
ax.set_title("(a) Risk vs. helpfulness for the 10 members", fontsize=8)

blind = [set(np.where(J[i] > 0)[0]) for i in range(len(qids))]
n = len(qids); Jac = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        u = blind[i] | blind[j]
        Jac[i, j] = len(blind[i] & blind[j])/len(u) if u else 1.0
order = sorted(range(n), key=lambda i: (qids[i] not in SEL, qids[i]))
Jac = Jac[np.ix_(order, order)]; labs = [qids[i] for i in order]
ax = axes[1]
hm = sns.heatmap(Jac, cmap="rocket_r", vmin=0, vmax=1, xticklabels=labs, yticklabels=labs,
                 ax=ax, square=True, linewidths=.3, linecolor="white",
                 cbar_kws={"label": "blind-spot Jaccard", "shrink": .82, "pad": .02})
ax.add_patch(plt.Rectangle((0, 0), 4, 4, fill=False, edgecolor="#2b6cb0", lw=1.6))
ax.tick_params(labelsize=5.4, length=0)
cb = ax.figure.axes[-1]
cb.tick_params(labelsize=6); cb.yaxis.label.set_size(7)
plt.setp(ax.get_xticklabels(), rotation=42, ha="right")
ax.set_title("(b) Pairwise blind-spot overlap", fontsize=8)

fig.tight_layout(pad=0.3, w_pad=1.1)
fig.savefig(OUT / "menu_selection.pdf", bbox_inches="tight")
print("ok")

# ============================================================================

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=0.78)
OUT = Path("/project/6104653/ehghaghi/proteus/figures")
m = pd.read_csv("/home/ehghaghi/scratch/ehghaghi/proteus/analysis/metrics_mvp.csv")

ks = [1, 2, 3, 5, 8]
styles = {"deterministic": ("o", "-", "#c0392b"), "minimax": ("s", "--", "#8e44ad"),
          "uniform": ("^", "-.", "#2b6cb0"), "validation": ("v", ":", "#27ae60"),
          "reasoner": ("D", "--", "#e67e22")}
order = ["deterministic", "minimax", "reasoner", "validation", "uniform"]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0))

ax = axes[0]
for sel in order:
    g = m[m.selector == sel]
    mk, ls, c = styles[sel]
    ax.plot(ks, [g[f"asr_at_{k}"].mean() for k in ks], marker=mk, ls=ls, color=c,
            lw=1.9 if sel == "deterministic" else 1.2, ms=3.6, label=sel)
ax.set_xlabel("Attacker budget $k$ (PAIR steps)", fontsize=8)
ax.set_ylabel("Cumulative ASR@$k$", fontsize=8)
ax.set_xticks(ks); ax.tick_params(labelsize=7)
ax.legend(fontsize=6.2, loc="upper left", handlelength=1.8, handletextpad=.4,
          borderpad=.3, labelspacing=.25)
ax.set_title("(a) Attack success vs. budget", fontsize=8)

ax = axes[1]
vals = [m[m.selector == s]["flops_to_first_jb"].mean() / 1e12 for s in order]
bars = ax.bar(range(len(order)), vals, color=[styles[s][2] for s in order], alpha=.85,
              edgecolor="0.25", linewidth=.4)
base = vals[0]
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(b.get_x() + b.get_width()/2, v + 1.0,
            "baseline" if i == 0 else f"+{100*(v/base-1):.0f}%", ha="center", fontsize=6.4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=18, ha="right", fontsize=7)
ax.set_ylabel("TFLOPs to first jailbreak", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylim(0, max(vals)*1.20)
ax.set_title("(b) Cost of a successful adaptive attack", fontsize=8)

fig.tight_layout(pad=0.3, w_pad=1.3)
fig.savefig(OUT / "mvp_results.pdf", bbox_inches="tight")
print("ok", [round(v, 1) for v in vals])

# ============================================================================

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=0.78)
OUT = Path("/project/6104653/ehghaghi/proteus/figures")
AN = Path("/home/ehghaghi/scratch/ehghaghi/proteus/analysis")
m = pd.read_csv(AN / "metrics_mvp.csv")
sev = pd.read_csv(AN / "severity_by_config.csv")

order = ["deterministic", "minimax", "reasoner", "validation", "uniform"]
cols = {"deterministic": "#c0392b", "minimax": "#8e44ad", "uniform": "#2b6cb0",
        "validation": "#27ae60", "reasoner": "#e67e22"}

# ---- entropy + support ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.95))
for ax, col, lab, hi in [(axes[0], "entropy", r"$H(c^\star)$ [nats]", 1.386),
                         (axes[1], "support", r"$|\mathrm{supp}(c^\star)|$", None)]:
    v = [m[m.selector == s][col].mean() for s in order]
    ax.bar(range(len(order)), v, color=[cols[s] for s in order], alpha=.85,
           edgecolor="0.25", linewidth=.4)
    if hi:
        ax.axhline(hi, color="0.3", ls="--", lw=.9)
        ax.text(4.45, hi + .03, r"$\ln 4$", fontsize=6.4, ha="right", color="0.3")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel(lab, fontsize=8); ax.tick_params(axis="y", labelsize=7)
axes[0].set_title("(a) Coverage entropy", fontsize=8)
axes[1].set_title("(b) Support size", fontsize=8)
fig.tight_layout(pad=.3, w_pad=1.3)
fig.savefig(OUT / "app_entropy.pdf", bbox_inches="tight"); plt.close(fig)

# ---- frontier ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.15, 2.15))
for s in order:
    g = m[m.selector == s]
    ax.scatter(g["helpfulness"], g["asr"], s=42, color=cols[s], label=s,
               edgecolor="black", linewidth=.4, zorder=3)
ax.set_xlabel("Helpfulness on benign traffic", fontsize=8)
ax.set_ylabel("Adaptive ASR", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.2, loc="center left", handletextpad=.3, borderpad=.3,
          labelspacing=.25)
fig.tight_layout(pad=.3)
fig.savefig(OUT / "app_frontier.pdf", bbox_inches="tight"); plt.close(fig)

# ---- severity ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0))
v = [m[m.selector == s]["mean_jb_severity"].mean() for s in order]
axes[0].bar(range(len(order)), v, color=[cols[s] for s in order], alpha=.85,
            edgecolor="0.25", linewidth=.4)
axes[0].set_xticks(range(len(order)))
axes[0].set_xticklabels(order, rotation=18, ha="right", fontsize=7)
axes[0].set_ylabel("Mean jailbreak severity (0-10)", fontsize=8)
axes[0].set_ylim(0, 10); axes[0].tick_params(axis="y", labelsize=7)
axes[0].set_title("(a) By selector", fontsize=8)

g = (sev.groupby("qid")
        .apply(lambda d: (d.n_jailbreaks * d.mean_severity).sum() / d.n_jailbreaks.sum())
        .sort_values())
axes[1].barh(range(len(g)), g.values, color="#7e57a6", alpha=.85, edgecolor="0.25",
             linewidth=.4)
axes[1].set_yticks(range(len(g)))
axes[1].set_yticklabels(g.index, fontsize=7)
axes[1].set_xlabel("Mean jailbreak severity (0-10)", fontsize=8)
axes[1].set_xlim(0, 10); axes[1].tick_params(axis="x", labelsize=7)
axes[1].set_title("(b) By served configuration", fontsize=8)
fig.tight_layout(pad=.3, w_pad=1.3)
fig.savefig(OUT / "app_severity.pdf", bbox_inches="tight"); plt.close(fig)
print("ok")
