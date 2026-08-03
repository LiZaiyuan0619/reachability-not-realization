"""Cross-model reachable-set join (C3 deep evidence): 3B fate buckets vs 8B outcomes
on the SAME 300 arc_challenge samples.

Question: are the 3B "path-reachable-only" items (greedy wrong, K temperature samples
never hit, but some layer-program hits) exactly the items the bigger 8B model already
answers directly (G=1)? If yes, path perturbation unlocks items at the capability
boundary -- cross-scale evidence.

Panels (both n=300, eval_drllm.load_task('arc_challenge') limit=300, dataset order --
allenai/ai2_arc ARC-Challenge test split, deterministic, no shuffle; both fairbudget
audits are seed0):
  3B: results/ledger/capability_ledger.jsonl rows model=Llama-3.2-3B task=arc_challenge
      (tools/build_ledger.py output; audit budget K=200, fate P arm = mcts)
  8B: results/ledger/ledger_8b_arc_challenge.csv
      (tools/build_ledger_8b.py K=200 upgrade; fate P arm = rand)

Alignment gates (hard-fail):
  - idx sequences are 0..299 on both sides
  - per-index gold answers identical 300/300 (3B gold comes from w15_bench_l3b
    baseline 'correct'; 8B gold from w11_gap_prl318bs0m baseline 'correct' --
    independent runs on two different models, so identity of the sequences is
    strong evidence the panels index the same questions in the same order)
  (within-panel gates already enforced upstream: 3B audit orig == arm12 baseline
   300/300 in build_ledger.py; 8B fairv2 orig vs w11 greedy >=90% + k200-vs-k64
   greedy 300/300 in build_ledger_8b.py)

CPU-only, read-only except the markdown report. Run from dr-llm/:
    python tools/crossmodel_join.py
"""
import csv
import json
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
LEDGER = os.path.join(ROOT, "results", "ledger")
OUT_MD = "outputs/07-10-crossmodel-join.md"

B3 = ["deployed_G1", "scar_repairable", "recognition_gap",
      "sampling_reachable", "path_reachable_only", "unreached"]
B8 = ["deployed_G1", "scar_repairable", "recognition_gap",
      "sampling_reachable", "path_reachable_only", "unreached"]
C8 = ["8B_G1", "8B_G0_reachable", "8B_unreachable"]
SMALL_N = 10  # buckets below this get an explicit "limited statistical meaning" flag


def fisher_one_sided(a, b, c, d):
    """P(X <= a) under hypergeometric with margins (a+b, c+d, a+c) -- one-sided
    Fisher exact for 'group1 hit-rate lower than group2'. Table: [[a,b],[c,d]]
    = [[bucket 8B_G1, bucket 8B_G0], [base 8B_G1, base 8B_G0]]."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    denom = math.comb(n, k)
    p = 0.0
    for x in range(0, a + 1):
        if k - x <= n2 and k - x >= 0 and x <= n1:
            p += math.comb(n1, x) * math.comb(n2, k - x) / denom
    return p


def load_panels():
    rows3 = [json.loads(l) for l in open(os.path.join(LEDGER, "capability_ledger.jsonl"))]
    arc3 = sorted([r for r in rows3 if r["model"] == "Llama-3.2-3B"
                   and r["task"] == "arc_challenge"], key=lambda r: r["idx"])
    arc8 = list(csv.DictReader(open(os.path.join(LEDGER, "ledger_8b_arc_challenge.csv"))))
    assert len(arc3) == len(arc8) == 300, f"panel sizes {len(arc3)}/{len(arc8)} != 300"
    assert [r["idx"] for r in arc3] == list(range(300)), "3B idx not 0..299"
    assert [int(r["idx"]) for r in arc8] == list(range(300)), "8B idx not 0..299"
    gold3 = [r["gold"] for r in arc3]
    gold8 = [r["gold"] for r in arc8]
    mism = [i for i in range(300) if gold3[i] != gold8[i]]
    assert not mism, f"gold sequences differ at idx {mism[:10]} -- refusing to join"
    align = {"n": 300, "gold_agree": 300 - len(mism),
             "gold_dist": dict(Counter(gold3)),
             "gold_source_3b": "w15_bench_l3b/arc_challenge_baseline.json records[i].correct",
             "gold_source_8b": "w11_gap_prl318bs0m/arc_challenge_baseline.json records[i].correct"}
    return arc3, arc8, align


def cat8(r):
    if int(r["G"]) == 1:
        return "8B_G1"
    if r["fate"] == "unreached":
        return "8B_unreachable"
    return "8B_G0_reachable"


def main():
    arc3, arc8, align = load_panels()
    print(f"[align] idx 0..299 both panels; gold identical "
          f"{align['gold_agree']}/300; dist {align['gold_dist']}")

    tab = defaultdict(Counter)      # 3B fate -> 8B 3-way category
    tab66 = defaultdict(Counter)    # 3B fate -> 8B fate
    detail_pathonly = []
    for a, b in zip(arc3, arc8):
        tab[a["fate"]][cat8(b)] += 1
        tab66[a["fate"]][b["fate"]] += 1
        if a["fate"] == "path_reachable_only":
            detail_pathonly.append((a, b))

    base_n = sum(tab["deployed_G1"].values())
    base_g1 = tab["deployed_G1"]["8B_G1"]

    # console + markdown simultaneously
    L = []
    def emit(s=""):
        print(s)
        L.append(s)

    emit("# 07-10 跨模型可达集 join：3B fate 桶 x 8B 直答/可达 (arc_challenge n=300)")
    emit()
    emit("**科学问题**：3B 上 path-reachable-only（直答错、200 次温度采样碰不到、"
         "但换层程序能碰到）的题，是否恰好是 8B 直接就会（G=1）的题？"
         "若是，“路径扰动解锁的是能力边界上的题”有跨尺度证据。")
    emit()
    emit("**答案（先说结论）：否——方向相反。** 3B path-only 的 7 题里 8B 直答对的只有 "
         f"{tab['path_reachable_only']['8B_G1']}/7，远低于基线"
         f"（3B production 桶的 8B G=1 比例 {base_g1}/{base_n}"
         f" = {base_g1/base_n:.1%}）；其中 3/7 在 8B 上依然是 path-reachable-only。"
         "路径可达性标记的不是“下一级规模即将直答”的边界题，而是跨尺度持续存在的"
         "路径依赖难题。反倒是 3B scar_repairable 桶（28 题）在 8B 上 28/28 全部直答对——"
         "“疤痕修复=已具备但被损伤的能力”得到跨尺度支持。")
    emit()
    emit("## 数据与对齐")
    emit()
    emit("- 3B 面板：`results/ledger/capability_ledger.jsonl`（build_ledger.py，"
         "K=200，fate P 臂=mcts，fairbudget 源 w8_fairbudget/fair_arc_challenge_seed0）")
    emit("- 8B 面板：`results/ledger/ledger_8b_arc_challenge.csv`（build_ledger_8b.py "
         "今日升级为 kmax=200，fate P 臂=rand，源 w8_fairbudget_v2/l318bk200/"
         "fairv2_arc_challenge_seed0）")
    emit("- 两边均为 `eval_drllm.load_task('arc_challenge')` 前 300 题：ai2_arc "
         "ARC-Challenge test split 原始顺序，无 shuffle（load_task 默认 "
         "shuffle_seed=-1），故 limit=300 前缀确定且两边一致；两边 fairbudget 审计均为 seed0。")
    emit("- **对齐验证（本脚本 hard-fail 断言）**：两边 idx 均为 0..299；逐题 gold 完全一致 "
         f"{align['gold_agree']}/300（gold 分布 {align['gold_dist']}）。注意两边 gold 来自"
         "两次彼此独立的跑（3B 侧 w15_bench_l3b baseline，8B 侧 w11_gap_prl318bs0m "
         "baseline，模型都不同），序列逐题一致本身就是同题同序的强证据。")
    emit("- 面板内部对齐（上游 builder 已断言）：3B 审计 orig == arm12 baseline 300/300；"
         "8B fairv2 orig vs w11 greedy 299/300=99.67%（唯一分歧 idx=292，两次独立 greedy "
         "跑的边缘波动）；8B k200 重跑 vs 旧 k64 审计 greedy 300/300 完全一致。")
    emit()
    emit("## 主表：3B fate 桶 x 8B {G=1, G=0 但可达, 8B 也不可达}（计数）")
    emit()
    emit("| 3B fate | n | 8B G=1 | 8B G=0 但可达 | 8B 也不可达 | 8B G=1 占比 |")
    emit("|---|---|---|---|---|---|")
    rows_stats = {}
    for b in B3:
        n = sum(tab[b].values())
        g1 = tab[b]["8B_G1"]
        rows_stats[b] = (g1, n)
        flag = " *(n<10)*" if 0 < n < SMALL_N else ""
        pct = f"{g1/n:.0%}" if n else "-"
        emit(f"| {b}{flag} | {n} | {g1} | {tab[b]['8B_G0_reachable']} | "
             f"{tab[b]['8B_unreachable']} | {pct} |")
    emit(f"| **合计** | 300 | {sum(tab[b]['8B_G1'] for b in B3)} | "
         f"{sum(tab[b]['8B_G0_reachable'] for b in B3)} | "
         f"{sum(tab[b]['8B_unreachable'] for b in B3)} | "
         f"{sum(tab[b]['8B_G1'] for b in B3)/300:.0%} |")
    emit()
    emit("注：**“8B 也不可达”列在 K=200 口径下结构性为空**——8B rand 臂 "
         "P@200=1.0（300/300 可达），即 200 个随机层程序内每题都至少命中一次，"
         "所以该列全 0 不是巧合而是 8B 在此预算下无 unreached。n<10 的桶"
         "（recognition_gap、path_reachable_only）只报计数，百分比仅作方向参考，"
         "统计意义有限。")
    emit()

    # Fisher one-sided contrasts vs production baseline
    emit("## 关键对比（对照基线 = 3B production 桶的 8B G=1 比例）")
    emit()
    for b in ["path_reachable_only", "sampling_reachable", "unreached", "scar_repairable"]:
        g1, n = rows_stats[b]
        p = fisher_one_sided(g1, n - g1, base_g1, base_n - base_g1)
        note = "（n<10，仅作方向参考）" if n < SMALL_N else ""
        emit(f"- 3B **{b}** (n={n})：8B G=1 = {g1}/{n}，基线 {base_g1}/{base_n}"
             f"={base_g1/base_n:.1%}；单侧 Fisher exact（该桶更低）p={p:.4g} {note}")
    emit()
    emit("## 附表：3B fate x 8B fate（6x6 全交叉，计数）")
    emit()
    emit("| 3B fate \\ 8B fate | " + " | ".join(B8) + " |")
    emit("|---|" + "---|" * len(B8))
    for b in B3:
        emit(f"| {b} | " + " | ".join(str(tab66[b][c]) for c in B8) + " |")
    emit()
    emit("## 3B path-only 7 题在 8B 侧的逐题命运（核心证据）")
    emit()
    emit("| idx | gold | 3B Pmcts_rank / Prand_rank | 8B G | 8B fate | 8B S_rank | 8B Prand_rank | 8B 直答答案 |")
    emit("|---|---|---|---|---|---|---|---|")
    for a, b in detail_pathonly:
        emit(f"| {a['idx']} | {a['gold']} | {a['P_rank']} / {a['Prand_rank']} | "
             f"{b['G']} | {b['fate']} | {b['S_rank']} | {b['P_rank']} | "
             f"{b['orig_answer']} |")
    emit()
    n_po_po = tab66["path_reachable_only"]["path_reachable_only"]
    emit(f"7 题中：8B 直答对 {tab['path_reachable_only']['8B_G1']} 题；"
         f"{n_po_po} 题在 8B 上仍是 path-reachable-only（8B 200 次采样也碰不到、"
         f"仅层程序可达）；其余为 recognition_gap "
         f"{tab66['path_reachable_only']['recognition_gap']} 题、sampling_reachable "
         f"{tab66['path_reachable_only']['sampling_reachable']} 题。")
    emit()
    emit("## 解读（3 句）")
    emit()
    p_po = fisher_one_sided(rows_stats["path_reachable_only"][0],
                            rows_stats["path_reachable_only"][1]
                            - rows_stats["path_reachable_only"][0],
                            base_g1, base_n - base_g1)
    emit(f"1. **假设被数据反驳**：3B path-only 题不是 8B 轻松直答的“能力边界题”——"
         f"8B 直答仅 {rows_stats['path_reachable_only'][0]}/"
         f"{rows_stats['path_reachable_only'][1]}，显著低于 {base_g1/base_n:.1%} 的基线"
         f"（单侧 Fisher p={p_po:.2g}，尽管 n=7 仍需谨慎），"
         "且 3/7 在 8B 上同样只有换层程序才碰得到：**路径可达性刻画的是一类跨尺度稳定的"
         "路径依赖难题，而非规模即将消化的题**。")
    emit("2. 反而 **scar_repairable 是真正的“边界即将消化”桶**：3B 上疤痕修复"
         "可救的 28 题，8B（无疤痕模型）28/28 全部直答——支持 C3 里“修复解锁的是"
         "已具备但被单层损伤压制的能力”的解释，且给出跨尺度佐证。")
    emit("3. 3B 可达性层级对 8B 直答率呈单调梯度（production 90% > recognition 6/7 > "
         "sampling 56% > unreached 45% > path-only 2/7）——除 path-only 落到最底部外，"
         "3B 的审计桶序基本预测 8B 难度；8B 侧在 K=200 预算下无 unreached"
         "（rand P@200=1.0），故“8B 也不可达”一列结构性为空。")
    emit()
    emit("## 文件")
    emit()
    emit("- 3B 面板：`" + os.path.join(LEDGER, "capability_ledger.jsonl") + "`")
    emit("- 8B 面板（K=200 升级后）：`" + os.path.join(LEDGER, "ledger_8b_arc_challenge.csv")
         + "`，summary：`" + os.path.join(LEDGER, "ledger_8b_summary.json") + "`")
    emit("- 本脚本：`" + os.path.abspath(__file__) + "`（纯 CPU、只读分析，"
         "输出仅本报告）")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nwrote: {OUT_MD}")


if __name__ == "__main__":
    main()
