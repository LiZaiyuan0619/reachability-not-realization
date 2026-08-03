#!/usr/bin/env python3
"""Fig 4: one MLP block suppresses an answer the model already holds.
(a) census -- generation against recognition for every clean checkpoint cell.
(b) necessity -- which part of the layer is the locus, six cases.
(c) selectivity -- repairs the sick items, mostly leaves healthy ones alone.
(d) minimality -- the write is distributed, top-k neurons do not carry it.
Inputs mined by mine_mechanism.py from the battery JSONs and the census.
"""
import os, csv
from collections import defaultdict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════ TUNABLE PARAMETERS ══════════════════════════════════════════
FIGSIZE  = (7.16, 2.08)
DPI      = 400
WRATIO   = [1.06, 1.30, 1.16, 1.02]
WSPACE   = 0.40
FS_TITLE = 8.4
FS_LABEL = 7.4
FS_TICK  = 6.8
FS_ANNOT = 6.5
BAR_W    = 0.26

C_MLP    = "#0F4D92"     # the locus
C_ATTN   = "#B64342"     # the control that does nothing
C_LAYER  = "#8FB3D9"     # whole-layer reference
C_SICK   = "#0F4D92"
C_HEAL   = "#C8860D"
C_TOPK   = "#0F4D92"
C_RANDK  = "#767676"
C_GRID   = "#DCDCDC"
C_TEXT   = "#1A1A1A"
C_MUTE   = "#6E6E6E"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7.5, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
rd   = lambda n: list(csv.DictReader(open(os.path.join(HERE, n))))
nec, sel, mini, cen = rd("necessity.csv"), rd("selectivity.csv"), \
                      rd("minimality.csv"), rd("census.csv")
ORDER = ["prl318", "g431", "g4qat", "g412", "q2505", "l3b"]
SHORT = {"prl318": "L8B-pr", "g431": "G31B", "g4qat": "G12B-q",
         "g412": "G12B", "q2505": "Q0.5B", "l3b": "L3B"}

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
gs  = fig.add_gridspec(1, 4, width_ratios=WRATIO, wspace=WSPACE,
                       left=0.052, right=0.995, top=0.815, bottom=0.205)

# ══════════════ (a) census: recognition above generation ════════════════════
axa = fig.add_subplot(gs[0, 0])
axa.plot([0, 1], [0, 1], color=C_MUTE, lw=0.6, ls=(0, (3, 2)), zorder=2)
gen = [c for c in cen if c["genuine"] == "True"]
oth = [c for c in cen if c["genuine"] != "True"]
axa.scatter([float(c["G"]) for c in oth], [float(c["R"]) for c in oth],
            s=3.2, c="#BFBFBF", lw=0, alpha=0.75, zorder=3)
axa.scatter([float(c["G"]) for c in gen], [float(c["R"]) for c in gen],
            s=5.2, c=C_MLP, lw=0, alpha=0.9, zorder=4)
axa.set_xlim(-0.03, 1.03); axa.set_ylim(-0.03, 1.03)
axa.set_xticks([0, 0.5, 1.0]); axa.set_yticks([0, 0.5, 1.0])
axa.set_xlabel("produced", fontsize=FS_LABEL, labelpad=1.0)
axa.set_ylabel("recognized", fontsize=FS_LABEL, labelpad=1.0)
axa.tick_params(labelsize=FS_TICK, length=2.0, pad=1.2)
axa.set_title("(a)  census", fontsize=FS_TITLE, loc="left", pad=3.0,
              color=C_TEXT)
axa.text(0.055, 0.945, f"{len(gen)} of {len(cen)} cells\nrecognize but\ndo not produce",
         fontsize=FS_ANNOT - 0.3, color=C_MLP, va="top", ha="left",
         linespacing=1.22)

# ══════════════ (b) necessity: the MLP block is the locus ═══════════════════
axb = fig.add_subplot(gs[0, 1])
by = defaultdict(dict)
for r in nec:
    by[r["case"]][r["key"]] = float(r["repair"])
x = np.arange(len(ORDER))
for off, key, col, lab in [(-BAR_W, "zero_mlp", C_MLP, "silence the MLP block"),
                           (0.0, "zero_attn", C_ATTN, "silence attention"),
                           (BAR_W, "zero_layer", C_LAYER, "silence the layer")]:
    axb.bar(x + off, [by[c].get(key, 0) * 100 for c in ORDER], BAR_W,
            color=col, edgecolor="white", lw=0.35, zorder=3, label=lab)
axb.set_xticks(x)
axb.set_xticklabels([SHORT[c] for c in ORDER], fontsize=FS_TICK, rotation=32,
                    ha="right", rotation_mode="anchor")
axb.set_ylim(0, 104); axb.set_yticks([0, 50, 100])
axb.set_ylabel("items repaired (%)", fontsize=FS_LABEL, labelpad=1.0)
axb.tick_params(labelsize=FS_TICK, length=2.0, pad=1.2)
axb.set_title("(b)  which part carries it", fontsize=FS_TITLE, loc="left",
              pad=3.0, color=C_TEXT)
axb.legend(fontsize=FS_ANNOT - 0.5, loc="upper center",
           bbox_to_anchor=(0.47, -0.335), ncol=1, handlelength=0.85,
           handleheight=0.7, labelspacing=0.22, borderpad=0.0,
           handletextpad=0.38)

# ══════════════ (c) selectivity: targeted, not a blanket edit ═══════════════
axc = fig.add_subplot(gs[0, 2])
sm = {r["case"]: r for r in sel}
rep = [float(sm[c]["repair_sick"]) * 100 for c in ORDER]
brk = [float(sm[c]["break_healthy"]) * 100 for c in ORDER]
axc.bar(x - BAR_W / 2, rep, BAR_W, color=C_SICK, edgecolor="white", lw=0.35,
        zorder=3, label="repairs a suppressed item")
axc.bar(x + BAR_W / 2, brk, BAR_W, color=C_HEAL, edgecolor="white", lw=0.35,
        zorder=3, label="breaks a healthy item")
axc.set_xticks(x)
axc.set_xticklabels([SHORT[c] for c in ORDER], fontsize=FS_TICK, rotation=32,
                    ha="right", rotation_mode="anchor")
axc.set_ylim(0, 104); axc.set_yticks([0, 50, 100])
axc.set_ylabel("share of items (%)", fontsize=FS_LABEL, labelpad=1.0)
axc.tick_params(labelsize=FS_TICK, length=2.0, pad=1.2)
axc.set_title("(c)  it is targeted", fontsize=FS_TITLE, loc="left", pad=3.0,
              color=C_TEXT)
axc.legend(fontsize=FS_ANNOT - 0.5, loc="upper center",
           bbox_to_anchor=(0.47, -0.335), ncol=1, handlelength=0.85,
           handleheight=0.7, labelspacing=0.22, borderpad=0.0,
           handletextpad=0.38)

# ══════════════ (d) minimality: the write is distributed ════════════════════
axd = fig.add_subplot(gs[0, 3])
mm = defaultdict(list)
for r in mini:
    if r["topk"] and r["randk"]:
        mm[r["case"]].append((int(r["k"]), float(r["topk"]) * 100,
                              float(r["randk"]) * 100))
for c in ORDER:
    pts = sorted(mm.get(c, []))
    if not pts:
        continue
    k = [p[0] for p in pts]
    axd.plot(k, [p[1] for p in pts], color=C_TOPK, lw=0.9, alpha=0.62,
             marker="o", ms=1.9, zorder=4)
    axd.plot(k, [p[2] for p in pts], color=C_RANDK, lw=0.7, alpha=0.5,
             ls=(0, (2.5, 1.5)), zorder=3)
full = [float(r["full_zero"]) * 100 for r in mini if r["full_zero"]]
if full:
    axd.axhline(float(np.mean(full)), color=C_MLP, lw=0.85,
                ls=(0, (4, 2)), zorder=5)
    axd.text(1.15, float(np.mean(full)) + 3.5, "whole block",
             fontsize=FS_ANNOT - 0.5, color=C_MLP, ha="left")
axd.set_xscale("log", base=2)
axd.set_xticks([1, 8, 128]); axd.set_xticklabels(["1", "8", "128"])
axd.set_ylim(-4, 104); axd.set_yticks([0, 50, 100])
axd.set_xlabel("neurons silenced", fontsize=FS_LABEL, labelpad=1.0)
axd.set_ylabel("items repaired (%)", fontsize=FS_LABEL, labelpad=1.0)
axd.tick_params(labelsize=FS_TICK, length=2.0, pad=1.2)
axd.set_title("(d)  a minority carries it", fontsize=FS_TITLE, loc="left",
              pad=3.0, color=C_TEXT)
axd.legend([plt.Line2D([], [], color=C_TOPK, lw=1.0, marker="o", ms=2.2),
            plt.Line2D([], [], color=C_RANDK, lw=0.8, ls=(0, (2.5, 1.5)))],
           ["strongest $k$", "random $k$"], fontsize=FS_ANNOT - 0.5,
           loc="upper center", bbox_to_anchor=(0.47, -0.335), ncol=1,
           handlelength=1.15, handleheight=0.7, labelspacing=0.22,
           borderpad=0.0, handletextpad=0.38)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig4_mechanism.{ext}"),
                bbox_inches="tight", pad_inches=0.012)
print("wrote fig4_mechanism.pdf / .png")
