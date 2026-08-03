#!/usr/bin/env python3
"""Fig 5: where each item goes after training.
(a) the deployed gain is paid for out of the base model's own reachable pool --
    every newly produced item was already reachable by sampling, none came from
    outside it.
(b) the deployed score rises while the reachable set stays flat or contracts.
Inputs mined by mine_fate.py from the paired per-item reroll files.
"""
import os, csv
from collections import defaultdict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════ TUNABLE PARAMETERS ══════════════════════════════════════════
FIGSIZE  = (7.16, 2.62)
DPI      = 400
WRATIO   = [1.30, 1.0]
WSPACE   = 0.255
FS_TITLE = 8.8
FS_LABEL = 7.8
FS_TICK  = 7.2
FS_ANNOT = 6.9
BAR_W    = 0.60

C_GAIN   = "#0F4D92"   # became produced, out of the base's own reachable pool
C_CREATE = "#C8860D"   # became produced from outside that pool
C_LOSS   = "#B64342"   # was produced, is not any more
C_DEP    = "#0F4D92"
C_ORC    = "#B64342"
C_GRID   = "#DCDCDC"
C_TEXT   = "#1A1A1A"
C_MUTE   = "#6E6E6E"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7.8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.75, "legend.frameon": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
fate = list(csv.DictReader(open(os.path.join(HERE, "fate.csv"))))

CELLS = [("SimpleRL", "gsm8k"), ("SimpleRL", "dart"),
         ("DAPO", "gsm8k"), ("DAPO", "dart"),
         ("OR1", "gsm8k"), ("OR1", "dart")]
TNAME = {"gsm8k": "GSM8K", "dart": "DART"}

agg = defaultdict(dict)
for r in fate:
    key = (r["pair"], r["task"])
    d = agg[key]
    d.setdefault("n", int(r["n"]))
    d.setdefault("flows", {})
    d["flows"][(r["base_state"], r["trained_state"])] = int(r["count"])
    for f in ("base_deployed", "trained_deployed", "base_oracle",
              "trained_oracle"):
        d[f] = float(r[f])
    d["delta_reachable"] = int(r["delta_reachable"])

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
gs  = fig.add_gridspec(1, 2, width_ratios=WRATIO, wspace=WSPACE,
                       left=0.072, right=0.988, top=0.845, bottom=0.365)

# ══════════════ (a) who paid for the deployed gain ══════════════════════════
axa = fig.add_subplot(gs[0, 0])
axa.axhline(0, color=C_TEXT, lw=0.85, zorder=4)
x = np.arange(len(CELLS))
gain, create, loss = [], [], []
for key in CELLS:
    fl = agg[key]["flows"]
    gain.append(fl.get(("latent", "solid"), 0))
    create.append(fl.get(("absent", "solid"), 0))
    loss.append(-(fl.get(("solid", "latent"), 0) + fl.get(("solid", "absent"), 0)))

axa.bar(x, gain, BAR_W, color=C_GAIN, edgecolor="white", lw=0.4, zorder=3,
        label="produced now, already reachable before")
axa.bar(x, create, BAR_W, bottom=gain, color=C_CREATE, edgecolor="white",
        lw=0.4, zorder=3, label="produced now, was not reachable before")
axa.bar(x, loss, BAR_W, color=C_LOSS, edgecolor="white", lw=0.4, zorder=3,
        label="was produced, is not any more")

for i, key in enumerate(CELLS):
    net = gain[i] + create[i] + loss[i]
    axa.text(i, gain[i] + create[i] + 3.2, f"net {net:+d}", fontsize=FS_ANNOT - 0.4,
             ha="center", va="bottom", color=C_TEXT)
# The middle band is zero in all six cells, which is the point of the panel.
axa.text(-0.42, -52, "the middle band is zero\nin all six cells",
         fontsize=FS_ANNOT - 0.3, color=C_CREATE, ha="left", va="top",
         linespacing=1.22)

axa.set_xticks(x)
axa.set_xticklabels([f"{p} {TNAME[t]}" for p, t in CELLS], fontsize=FS_TICK,
                    rotation=26, ha="right", rotation_mode="anchor")
axa.set_ylim(-88, 96)
axa.set_yticks([-75, -50, -25, 0, 25, 50, 75])
axa.set_ylabel("items (of 300)", fontsize=FS_LABEL, labelpad=1.5)
axa.tick_params(labelsize=FS_TICK, length=2.2, pad=1.3)
axa.set_title("(a)  the gain is paid out of the base model's own pool",
              fontsize=FS_TITLE, loc="left", pad=4.0, color=C_TEXT)
axa.legend(fontsize=FS_ANNOT - 0.4, loc="lower center",
           bbox_to_anchor=(0.47, -0.66), ncol=1, handlelength=0.9,
           handleheight=0.7, labelspacing=0.24, borderpad=0.0,
           handletextpad=0.40)

# ══════════════ (b) deployed rises, the ceiling does not ════════════════════
axb = fig.add_subplot(gs[0, 1])
axb.axhline(0, color=C_TEXT, lw=0.85, zorder=4)
d_dep = [(agg[k]["trained_deployed"] - agg[k]["base_deployed"]) * 100
         for k in CELLS]
d_orc = [(agg[k]["trained_oracle"] - agg[k]["base_oracle"]) * 100 for k in CELLS]
w = 0.34
axb.bar(x - w / 2, d_dep, w, color=C_DEP, edgecolor="white", lw=0.4, zorder=3,
        label="deployed score")
axb.bar(x + w / 2, d_orc, w, color=C_ORC, edgecolor="white", lw=0.4, zorder=3,
        label="reachable ceiling")
axb.set_xticks(x)
axb.set_xticklabels([f"{p} {TNAME[t]}" for p, t in CELLS], fontsize=FS_TICK,
                    rotation=26, ha="right", rotation_mode="anchor")
axb.set_ylim(-22, 22); axb.set_yticks([-20, -10, 0, 10, 20])
axb.set_ylabel("change after training (pp)", fontsize=FS_LABEL, labelpad=1.5)
axb.tick_params(labelsize=FS_TICK, length=2.2, pad=1.3)
axb.set_title("(b)  the ceiling does not follow", fontsize=FS_TITLE,
              loc="left", pad=4.0, color=C_TEXT)
axb.legend(fontsize=FS_ANNOT - 0.4, loc="lower center",
           bbox_to_anchor=(0.47, -0.66), ncol=1, handlelength=0.9,
           handleheight=0.7, labelspacing=0.24, borderpad=0.0,
           handletextpad=0.40)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig5_fate.{ext}"), bbox_inches="tight",
                pad_inches=0.012)
print("wrote fig5_fate.pdf / .png")
print("created-from-outside counts:", create)
