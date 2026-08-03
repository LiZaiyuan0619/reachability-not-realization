"""W8 fair-budget reroll audit -- GPU runner **v2**: adds the SELECTION column.

What v2 adds over fair_budget_oracle.py (Goal A)
------------------------------------------------
v1 recorded, per generator, only the 1-indexed RANK of the first correct candidate.
That yields the ORACLE ceiling (pass@k = any-correct) but NOT the deployable "selection"
column, because the candidate ANSWER STRINGS were thrown away -- so majority-vote /
confidence could not be computed. v2 stores, per sample and per generator (mcts/rand/samp,
plus the 07-10 hillclimb arm whose record prefix is hc_*):
  {gen}_answers : the extracted ANSWER STRING of each candidate (index 0 = full greedy)
  {gen}_correct : per-candidate 0/1 grade (unified verifier == eval_drllm.grade)
  {gen}_conf    : per-candidate mean transition logprob (path_sc.py convention)
From these, fair_budget_analyze.py computes, per k:
  oracle@k     = any-correct ceiling (== v1 rank<=k; needs a verifier)
  majority@k   = self-consistency vote (most-common answer string; verifier-free)
  confidence@k = grade the highest mean-logprob candidate (BoN-by-conf; verifier-free)
  selection gap = oracle@k - majority@k = reachable-but-unrecoverable answers.

Difference vs v1 that matters for cost: majority/confidence need the FULL candidate set,
so v2 does NOT early-stop -- it always generates kmax candidates for rand/samp (mcts uses
its natural distinct set). rank/oracle numbers are unchanged (derived from the same first
hit). Candidate #1 is ALWAYS the full/default greedy path (shared across arms), so
pass@1 == baseline and every selector == baseline at k=1.

Same generators / verifier / decode length / samples as v1. Needs the data_models drop-in
(modeling_llama_FULL.py -> model.model.layer_indices). Run via rjob/w8_fairbudget_v2.sh.
Assumes an INSTRUCT model (Llama-3.2-3B-Instruct / Llama-3.1-8B-Instruct): boxed-answer
emission is native, so no base-model boxed hack is applied.

07-10 adds a FOURTH generator `hillclimb` (--gens hillclimb; record keys hc_rank /
hc_nevals / hc_answers / hc_correct / hc_conf / hc_meta): greedy hill-climb over layer
programs by single skip/repeat edits, steered ONLY by the model's own mean-logprob
confidence -- NEVER by the gold grade (see hillclimb_candidates). It is the audit's
targeted STRONG-searcher arm: the last attack surface on "random >= MCTS" is "you only
showed MCTS is weak; a stronger searcher might beat random". If even a climber designed
for this space fails to beat the random-program control at matched budget, the
"search-immune action space" conclusion closes. OUTPUT SAFETY: fairv2_* filenames do NOT
encode --gens, so a hillclimb-only rerun of an existing task/seed/tag would OVERWRITE the
landed 3-arm file -- run it with a fresh tag AND --out_suffix _hc (both done by
rjob/submit_hillclimb_0710.sh).

07-10 also adds a FIFTH generator `genetic` (--gens genetic; record keys ga_rank /
ga_nevals / ga_answers / ga_correct / ga_conf / ga_meta): an evolutionary search over
layer programs -- population P=8, 50%-elitism (top-4 by conf), refilled each generation by
MUTATION (one skip/repeat edit of an elite, the hillclimb edit family) and CROSSOVER
(prefix/suffix splice of two elites) -- steered ONLY by the model's own mean-logprob, again
NEVER by the gold grade (see genetic_candidates / ga_fitness). It is the audit's SECOND
strong-searcher arm: where hillclimb tests local search, genetic tests population +
RECOMBINATION, the capability a local climber lacks. If neither beats the random-program
control at matched budget, "random >= structured" survives both search families. Same
OUTPUT SAFETY caveat: run it with --out_suffix _ga (rjob/submit_genetic_0710.sh) so the
landed 3-arm / hillclimb files are untouched.

Self-test (no GPU): python fair_budget_oracle_v2.py --selftest
"""
import argparse
import json
import os
import random
import re
import time

# pure-python selector core lives in fair_budget_analyze.py (single source of truth,
# no torch/mathruler) so both the GPU driver and the CPU analyzer agree bit-for-bit.
from fair_budget_analyze import oracle_hit, majority_hit, confidence_hit


# ----------------------------------------------------------------- pure-python core
def first_hit_rank(reward_seq):
    """1-indexed rank of first reward>0.5, else -1. Pure python (CPU-testable)."""
    for i, r in enumerate(reward_seq):
        if float(r) > 0.5:
            return i + 1
    return -1


_MC_RE = re.compile(r"^(?:Answer:\s*)?([A-Ea-e])\.?$")   # == eval_drllm.ARC_RE
_NUM_RE = re.compile(r"-?\d*\.?\d+")

# tasks eval_drllm.load_task supports AND that are cached offline (CPU-verified 07-09):
# arc + dart(+pool) + the 7 MC tasks (all reuse the ARC letter protocol, A-E fits _MC_RE).
ALLOWED_TASKS = ["arc_easy", "arc_challenge", "dart", "dart_pool", "mmlu", "hellaswag",
                 "winogrande", "piqa", "boolq", "openbookqa", "commonsense_qa"]


def parse_tasks(tasks_str, task_str=None):
    """Resolve the task list for ONE model-load. Prefer --tasks (comma list, run serially
    so the 2-5 min model load is amortized across all of them); else fall back to a single
    --task. De-dups preserving order; validates every entry against ALLOWED_TASKS. Pure-
    python -> the multi-task DISPATCH logic is CPU-testable via --selftest (no GPU)."""
    raw = [t.strip() for t in (tasks_str or "").split(",") if t.strip()]
    if not raw and task_str:
        raw = [task_str]
    seen, out = set(), []
    for t in raw:
        if t not in ALLOWED_TASKS:
            raise ValueError(f"unknown task {t!r} (allowed: {ALLOWED_TASKS})")
        if t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise ValueError("no task specified (need --task or --tasks)")
    return out


ALLOWED_GENS = ["mcts", "rand", "samp", "hillclimb", "genetic"]
# (gen name in --gens, record-key prefix). hillclimb stores hc_*, genetic stores ga_* --
# short and distinct from the 3 legacy arms, exactly the keys fair_budget_analyze-style
# consumers expect.
GEN_PREFIX = [("mcts", "mcts"), ("rand", "rand"), ("samp", "samp"), ("hillclimb", "hc"),
              ("genetic", "ga")]


def parse_gens(gens_str):
    """Validate/resolve --gens (comma list, subset of ALLOWED_GENS, de-dup preserving
    order). A typo'd generator name would otherwise silently no-op the whole GPU job
    (every arm is `if g in gens`-guarded), so unknown names FAIL FAST here instead.
    Pure-python -> CPU-testable via --selftest."""
    seen, out = set(), []
    for g in [x.strip() for x in (gens_str or "").split(",") if x.strip()]:
        if g not in ALLOWED_GENS:
            raise ValueError(f"unknown generator {g!r} (allowed: {ALLOWED_GENS})")
        if g not in seen:
            seen.add(g)
            out.append(g)
    if not out:
        raise ValueError("no generator specified (--gens)")
    return out


def extract_answer(task, text, protocol="short"):
    """Canonical answer STRING used as the self-consistency vote key. MIRRORS
    eval_drllm.grade's extraction so candidates with the same string share a grade:
      dart*  -> \\boxed{...} content   (mathruler; imported lazily -> GPU-only)
      gsm8k  -> last number
      MC     -> single letter A-E (uppercased), else '' (unparseable = abstain bucket)
    dart voting is on the RAW boxed string, so '1/2' and '0.5' are distinct buckets --
    a documented conservative choice that only WIDENS the selection gap (majority is an
    even weaker selector), while correctness is still looked up from the exact grade()."""
    t = (text or "").strip()
    if task.startswith("dart"):
        from mathruler.grader import extract_boxed_content
        return (extract_boxed_content(t) or "").strip()
    if task == "gsm8k":
        nums = _NUM_RE.findall(t.replace(",", "").replace("$", ""))
        return nums[-1] if nums else ""
    m = _MC_RE.match(t)
    return m.group(1).upper() if m else ""


# ------------------------------------------------ PoLar action-space constraint
# PoLar restricts its layer PROGRAMS to contiguous segments of length <= POLAR_MAX_SEG
# with AT MOST ONE repeat (one segment run twice) per program -- a different program
# family than Dr.LLM's MCTS space (skip 1-2 contiguous + unbounded repeats). Pinning
# BOTH the structured (MCTS) and the random arm to *this* family makes the audit's
# "random >= structured" claim hold inside PoLar's OWN action space, not just Dr.LLM's.
# These helpers are pure-python (no torch) so the constraint logic is CPU-testable via
# --selftest; the MCTS node subclass that consumes them (PolarNode) is defined in main()
# after the torch-backed data_generation import. "segment length <= 4" is read at the
# ACTION granularity (each skip/repeat edits a contiguous block of <=4 layers), the same
# knob oracle_v3.py parametrizes; "<=1 repeat" = at most one repeated segment per program.
POLAR_MAX_SEG = 4


def _has_repeat(layers):
    """True iff the layer program already contains a duplicated layer (a repeat)."""
    return len(layers) != len(set(layers))


def polar_actions(layers, num_layers, max_seg=POLAR_MAX_SEG):
    """PoLar-constrained action list for a node whose current path is `layers`:
      - SKIP a contiguous block of size 1..max_seg (always allowed);
      - REPEAT a contiguous block of size 1..max_seg, but ONLY while the path is still
        repeat-free -> enforces <=1 repeat per program.
    Pure list ops (no torch). Mirrors data_generation.MCTSNode._generate_possible_actions
    but with PoLar's segment/repeat limits (skip<=4 vs Dr.LLM's <=2; repeat<=4 x1 and at
    most once vs Dr.LLM's 1-layer repeats that can stack up the tree)."""
    path_len = len(layers)
    acts = []
    for start in range(path_len):
        for seg in range(1, max_seg + 1):
            if start + seg <= path_len:
                acts.append({"type": "skip", "start": start, "size": seg})
    if not _has_repeat(layers):  # <=1 repeat: offer repeats only on a repeat-free path
        for start in range(path_len):
            for seg in range(1, max_seg + 1):
                if start + seg <= path_len and path_len + seg <= num_layers * 2:
                    acts.append({"type": "repeat", "start": start, "size": seg, "count": 1})
    random.shuffle(acts)
    return acts


def polar_random_layers(rng, nl, max_seg=POLAR_MAX_SEG):
    """A random PoLar-constrained program (the PoLar-space analogue of
    null_oracle.random_edit_path): 1-3 contiguous SKIP edits (size 1..max_seg) then
    optionally ONE contiguous REPEAT (size 1..max_seg). Guarantees every edit-segment
    <= max_seg and <=1 repeat. Returns a layer-id list (wrap in LayerPath at call site).
    Pure-python -> CPU-testable."""
    path = list(range(nl))
    for _ in range(rng.choice([1, 2, 3])):
        if len(path) <= max_seg + 1:
            break
        k = rng.randint(1, max_seg)
        s = rng.randrange(0, len(path) - k + 1)
        path = path[:s] + path[s + k:]
    if rng.random() < 0.5 and len(path) + max_seg <= 2 * nl:  # <=1 repeat
        k = rng.randint(1, min(max_seg, len(path)))
        s = rng.randrange(0, len(path) - k + 1)
        path = path[:s + k] + path[s:s + k] + path[s + k:]  # double the segment in place
    return path


def skiponly_random_layers(rng, nl):
    """A random program in the SKIP-ONLY family (test-time depth-adaptation space):
    Dr.LLM's random_edit_path with the repeat op removed -- 1-3 contiguous skips of
    size 1-2, never a repeat, so every program is a pure depth REDUCTION. Third audited
    action space: if random >= MCTS holds here too, the result does not depend on the
    compute-adding repeat dimension. Pure-python -> CPU-testable."""
    path = list(range(nl))
    for _ in range(rng.choice([1, 2, 3])):
        if len(path) <= 3:
            break
        k = rng.choice([1, 2])
        s = rng.randrange(0, len(path) - k + 1)
        path = path[:s] + path[s + k:]
    return path


# --------------------------------------------- hillclimb arm (4th generator, pure core)
# The audit's targeted STRONG-searcher arm. Design: first-improvement greedy hill-climb
# over layer programs (one random single edit per step), with eps-acceptance of worse
# moves and restart-at-full on stalls. The climb signal is the model's OWN per-candidate
# confidence = mean transition logprob -- the exact signal the confidence@k selector uses,
# i.e. the strongest steering a DEPLOYABLE searcher could legitimately have.
HC_EPS = 0.3    # prob. of accepting a NON-improving edit as the new climb start
                # (stochastic hill-climb; escapes local conf maxima without gold)
HC_STALL = 12   # consecutive already-seen/illegal edits before restarting the climb at FULL


def hc_single_edit(rng, layers, nl, space="drllm"):
    """ONE random single edit of `layers` in the given action space; returns a NEW list
    (pure list ops -> CPU-testable). Edit family per space:
      drllm    : the per-edit body of null_oracle.random_edit_path -- p=0.6 skip 1-2
                 contiguous (needs len>3), else repeat ONE position (needs len+1 <= 2*nl);
      skiponly : the skip branch only (pure depth reduction, len>3 guard);
      polar    : one action drawn from polar_actions (segments <=4, <=1 repeat/program).
    May return an UNCHANGED copy when no edit is legal (the caller's dedup treats that
    as a stall and eventually restarts)."""
    path = list(layers)
    if space == "polar":
        acts = polar_actions(path, nl)
        if not acts:
            return path
        a = rng.choice(acts)
        if a["type"] == "skip":
            return path[:a["start"]] + path[a["start"] + a["size"]:]
        seg = path[a["start"]:a["start"] + a["size"]]  # == LayerPath.repeat_layers
        return path[:a["start"] + a["size"]] + seg * a["count"] + path[a["start"] + a["size"]:]
    if space == "skiponly":
        if len(path) > 3:
            k = rng.choice([1, 2])
            s = rng.randrange(0, len(path) - k + 1)
            return path[:s] + path[s + k:]
        return path
    # drllm (default) == one iteration of null_oracle.random_edit_path's edit loop
    if rng.random() < 0.6 and len(path) > 3:   # skip 1-2 contiguous
        k = rng.choice([1, 2])
        s = rng.randrange(0, len(path) - k + 1)
        return path[:s] + path[s + k:]
    if len(path) + 1 <= 2 * nl:                # repeat one position
        s = rng.randrange(0, len(path))
        return path[:s + 1] + [path[s]] + path[s + 1:]
    return path


def hillclimb_candidates(eval_fn, rng, nl, kmax, first, space="drllm",
                         eps=HC_EPS, stall_limit=HC_STALL):
    """CORE of the hillclimb arm (pure-python; the GPU enters only through eval_fn).

    eval_fn(layers_list) -> (answer_str, correct01, conf). `first` = (a0, r0, f0) of the
    shared candidate #1 = FULL greedy path (already evaluated by the caller, never re-run
    here: the full path is pre-seeded into the dedup set). Loop: ONE random single edit of
    the current path (hc_single_edit); dedup against every path ever evaluated; evaluate
    ONCE (same decode_and_score as every other arm); ACCEPT as the new climb start iff
    conf strictly improves, else accept anyway with prob `eps` (stochastic escape from
    local maxima); `stall_limit` consecutive duplicate/illegal edits -> restart at FULL.
    NO early stop: runs to kmax distinct candidates (majority/confidence need the full
    set); the rand-arm's tries cap (kmax*50) bounds the loop on tiny/exhausted spaces.

    !! NO-GOLD INVARIANT (audit fairness, do not weaken): the accept rule reads ONLY
    `conf` (mean transition logprob, the model's own signal). The gold grade `correct`
    is appended to the audit record -- oracle@k/majority@k need it -- but is NEVER read
    by the climb. A searcher steered by gold correctness would be grading itself with
    the answer key (an oracle selector, not a deployable policy) and the "even a
    targeted strong searcher fails to beat random" conclusion would be circular. The
    selftest proves the visited-path sequence is invariant to flipping every gold bit.

    Returns (ans, cor, cnf, meta); meta counts accepts / eps_accepts / rejects / restarts
    so the analysis can show the climber really climbed."""
    a0, r0, f0 = first
    ans, cor, cnf = [a0], [int(r0)], [f0]
    full_layers = list(range(nl))
    cur, cur_conf = full_layers, f0
    seen = {tuple(full_layers)}
    meta = {"accepts": 0, "eps_accepts": 0, "rejects": 0, "restarts": 0}
    tries = stall = 0
    while len(ans) < kmax and tries < kmax * 50:
        tries += 1
        nxt = hc_single_edit(rng, cur, nl, space)
        key = tuple(nxt)
        if key in seen:  # duplicate (or no legal edit): stall; restart at FULL if stuck
            stall += 1
            if stall >= stall_limit:
                cur, cur_conf, stall = full_layers, f0, 0
                meta["restarts"] += 1
            continue
        stall = 0
        seen.add(key)
        a, c, f = eval_fn(nxt)
        ans.append(a); cor.append(int(c)); cnf.append(f)
        if f > cur_conf:            # greedy accept: conf ONLY -- never the gold grade c
            cur, cur_conf = nxt, f
            meta["accepts"] += 1
        elif rng.random() < eps:    # eps-accept a worse candidate (escape local maxima)
            cur, cur_conf = nxt, f
            meta["eps_accepts"] += 1
        else:                       # reject: keep climbing from the current start
            meta["rejects"] += 1
    return ans, cor, cnf, meta


# --------------------------------------------- genetic arm (5th generator, pure core)
# The audit's SECOND targeted strong-searcher arm (companion to hillclimb): an
# evolutionary / GENETIC search over layer programs. Where hillclimb is pure single-edit
# LOCAL search, the GA adds a POPULATION and RECOMBINATION (crossover) -- the one
# capability a local climber structurally lacks -- so if the random-program control still
# holds >= GA at matched budget, the "search-immune action space" conclusion survives
# population-based recombination search too, not just local hill-climbing. Same NO-GOLD
# rule as hillclimb: selection is steered ONLY by the model's own conf (mean transition
# logprob); the gold grade is recorded for the audit columns but NEVER read by the search
# (see ga_fitness -- gold is not even in the selection signature -- and the selftest's
# flip-invariance proof).
GA_POP = 8       # population size P
GA_ELITE = 4     # elites kept each generation (top-half by conf == 50% truncation select)
GA_N_MUT = 2     # mutation offspring / generation \  together refill the P-ELITE=4 non-
GA_N_CROSS = 2   # crossover offspring / generation /  elite slots (1:1 mutate:recombine)


def ga_fitness(individual):
    """Fitness of a GA individual = its stored conf (mean transition logprob, the model's
    OWN signal). An individual is a (layers, conf) 2-TUPLE -- there is NO gold/correctness
    field in this signature, so selection CANNOT read the answer key. This is the no-gold
    audit rule enforced STRUCTURALLY (not just by convention); the selftest asserts the
    signature and proves the search is invariant to flipping every gold bit."""
    return individual[1]


def ga_select_elites(pop, elite):
    """Truncation selection: sort the population [(layers, conf), ...] by conf DESCENDING
    (ga_fitness -- gold-blind) and return the top-`elite` (all of it if fewer). Pure list
    ops -> CPU-testable; this is where 'keep the best half' lives."""
    return sorted(pop, key=ga_fitness, reverse=True)[:elite]


def ga_legal(child, nl):
    """Offspring legality for the GA (crossover can splice arbitrary programs): NON-EMPTY
    and length <= 2*num_layers -- the same length bound the drllm/polar edit families and
    decode_and_score respect. Layer-id MONOTONICITY is NOT required: a recombined program
    may reorder/revisit layers, exactly the freedom that makes the GA a strict superset of
    the single-edit climber. Pure -> CPU-testable."""
    return bool(child) and len(child) <= 2 * nl


def ga_crossover(rng, p1, p2, nl):
    """One-point (per-parent) crossover: child = prefix(p1) ++ suffix(p2) with an
    INDEPENDENT random cut in each parent (variable-length safe -- the two elite parents
    usually differ in length). Returns the raw child list; the caller runs ga_legal, so
    empty (both cuts drop everything) / over-long splices are discarded. Pure list ops ->
    CPU-testable."""
    c1 = rng.randint(0, len(p1))
    c2 = rng.randint(0, len(p2))
    return list(p1[:c1]) + list(p2[c2:])


def genetic_candidates(eval_fn, rng, nl, kmax, first, space="drllm",
                       pop_size=GA_POP, elite=GA_ELITE,
                       n_mut=GA_N_MUT, n_cross=GA_N_CROSS):
    """CORE of the genetic arm (pure-python; the GPU enters only through eval_fn).

    eval_fn(layers_list) -> (answer_str, correct01, conf). `first` = (a0, r0, f0) of the
    shared candidate #1 = FULL greedy path (already evaluated by the caller, pre-seeded
    into the dedup set AND the population, never re-run here). Algorithm:
      * initial population = FULL + up to pop_size-1 random single-edit variants of FULL
        (each a hc_single_edit mutation), every one evaluated ONCE and deduped;
      * each GENERATION: rank the population by conf (ga_select_elites -> keep top `elite`),
        then refill the pop_size-elite non-elite slots with n_mut MUTATION offspring (one
        skip/repeat edit of a random elite -- the SAME hc_single_edit family the climber
        uses) and n_cross CROSSOVER offspring (ga_crossover splice of two distinct elites,
        kept only if ga_legal). Every NEW distinct offspring is evaluated ONCE (same
        decode_and_score as every arm) and deduped GLOBALLY on tuple(layers).
    NO early stop: runs to kmax distinct candidates (majority/confidence need the full set);
    the rand-arm's tries cap (kmax*50) bounds the loop on tiny/exhausted spaces (a slot
    that can't be filled is simply skipped, so a stalled generation still terminates).

    !! NO-GOLD INVARIANT (audit fairness, do not weaken): selection reads ONLY conf, via
    ga_fitness on (layers, conf) 2-tuples -- the gold grade `correct` is appended to the
    audit record (oracle@k/majority@k need it) but is NEVER read by the search. A GA
    steered by gold would be an oracle selector, not a deployable policy, and the "even a
    targeted strong searcher fails to beat random" conclusion would be circular. The
    selftest proves the visited-path sequence is invariant to flipping every gold bit.

    Returns (ans, cor, cnf, meta); meta counts generations / init / mutations / crossovers
    / cross_illegal / dups so the analysis can show the GA really evolved (recombined)."""
    a0, r0, f0 = first
    ans, cor, cnf = [a0], [int(r0)], [f0]
    full_layers = list(range(nl))
    seen = {tuple(full_layers)}
    meta = {"generations": 0, "init": 0, "mutations": 0, "crossovers": 0,
            "cross_illegal": 0, "dups": 0}
    tries, cap = 0, kmax * 50

    def _birth(child):
        """Admit ONE candidate: legality + GLOBAL dedup, then evaluate ONCE (appending to
        the audit columns) and return (layers, conf); else None (dup/illegal). Reads no
        gold -- only eval_fn's conf ever feeds selection."""
        key = tuple(child)
        if (not ga_legal(child, nl)) or key in seen:
            return None
        seen.add(key)
        a, c, f = eval_fn(child)
        ans.append(a); cor.append(int(c)); cnf.append(f)
        return (list(child), f)

    # ---- initial population: FULL (shared #1, conf f0, never re-evaluated) + up to
    #      pop_size-1 single-edit variants of FULL ----
    pop = [(full_layers, f0)]
    while len(pop) < pop_size and len(ans) < kmax and tries < cap:
        tries += 1
        born = _birth(hc_single_edit(rng, full_layers, nl, space))
        if born is None:
            meta["dups"] += 1
        else:
            meta["init"] += 1
            pop.append(born)

    # ---- generational loop: rank by conf, keep elites, refill by mutation + crossover ----
    while len(ans) < kmax and tries < cap:
        meta["generations"] += 1
        elites = ga_select_elites(pop, elite)
        new_pop = list(elites)
        for op in (["mut"] * n_mut + ["cross"] * n_cross):
            if len(ans) >= kmax or tries >= cap:
                break
            tries += 1
            if op == "cross" and len(elites) >= 2:
                l1, l2 = (e[0] for e in rng.sample(elites, 2))
                child = ga_crossover(rng, l1, l2, nl)
                if not ga_legal(child, nl):     # empty / over-long splice -> skip this slot
                    meta["cross_illegal"] += 1
                    continue
                kind = "crossovers"
            else:                                # mutation (or crossover with <2 elites)
                child = hc_single_edit(rng, rng.choice(elites)[0], nl, space)
                kind = "mutations"
            born = _birth(child)
            if born is None:                     # duplicate / no-op edit -> skip this slot
                meta["dups"] += 1
                continue
            meta[kind] += 1
            new_pop.append(born)
        pop = new_pop
    return ans, cor, cnf, meta


def _selftest():
    """CPU-only: extract_answer (MC/gsm8k pure-python) + rank + selection gap on a
    synthetic 3-candidate sample (== Case A of fair_budget_analyze._selftest)."""
    # extract_answer mirrors grade's ARC_RE (anchored single letter, optional 'Answer:')
    assert extract_answer("arc_challenge", "Answer: B") == "B"
    assert extract_answer("arc_challenge", "C") == "C"
    assert extract_answer("arc_challenge", "  d. ") == "D"
    assert extract_answer("arc_easy", "Answer: e") == "E"
    assert extract_answer("arc_challenge", "The answer is B because") == ""  # not clean -> abstain
    assert extract_answer("gsm8k", "The final answer is 42.") == "42"
    # rank derivation from a per-candidate correctness list
    assert first_hit_rank([0, 0, 1, 0]) == 3 and first_hit_rank([0, 0]) == -1
    # end-to-end synthetic: 3 candidates, gold=B. answers=[B,A,A] correct=[1,0,0]
    #   conf=[-0.5,-0.1,-0.2]. oracle sees B (rank 1); majority picks A (2>1, WRONG);
    #   confidence picks max-logprob A@-0.1 (WRONG). selection gap = 1.
    ans, cor, cnf = ["B", "A", "A"], [1, 0, 0], [-0.5, -0.1, -0.2]
    assert first_hit_rank(cor) == 1
    assert oracle_hit(cor, 3) == 1
    assert majority_hit(ans, cor, 3) == 0
    assert confidence_hit(cor, cnf, 3) == 0
    assert oracle_hit(cor, 3) - majority_hit(ans, cor, 3) == 1
    # ---- PoLar action-space constraint (pure-python; segment<=4, <=1 repeat) ----
    import random as _r
    rng = _r.Random(0)
    acts = polar_actions(list(range(28)), 28)                 # full repeat-free 28-path
    assert acts and all(1 <= a["size"] <= 4 for a in acts), "segment size must be <=4"
    assert all(a["type"] in ("skip", "repeat") for a in acts)
    assert any(a["type"] == "repeat" for a in acts), "repeat offered on a repeat-free path"
    acts2 = polar_actions([0, 0, 1, 2, 3], 28)                # path already has a repeat
    assert acts2 and all(a["type"] == "skip" for a in acts2), "no 2nd repeat once one exists"
    for _ in range(3000):                                     # random PoLar programs
        for nl in (28, 32):                                   # 3B (28) and 8B (32)
            ly = polar_random_layers(rng, nl)
            assert ly, "non-empty program"
            assert len(ly) - len(set(ly)) <= 4, f"<=1 repeat of <=4 layers: {ly}"
            assert len(ly) <= 2 * nl, "length bounded by 2*num_layers"
    for _ in range(3000):                                     # random SKIP-ONLY programs
        for nl in (28, 32, 36, 48):
            ly = skiponly_random_layers(rng, nl)
            assert ly, "non-empty program"
            assert len(ly) == len(set(ly)), f"skip-only must never repeat: {ly}"
            assert len(ly) <= nl, "skip-only can only shorten the path"
            assert nl - len(ly) <= 6, "at most 3 skips of size <=2"
    # ---- multi-task dispatch (the once-loaded serial loop's task resolver) ----
    assert parse_tasks("arc_challenge,mmlu,piqa") == ["arc_challenge", "mmlu", "piqa"]
    assert parse_tasks("mmlu, mmlu ,piqa") == ["mmlu", "piqa"]       # de-dup + whitespace
    assert parse_tasks("", "dart") == ["dart"]                       # single --task fallback
    assert parse_tasks("dart_pool") == ["dart_pool"]
    assert parse_tasks("boolq,winogrande", "arc_easy") == ["boolq", "winogrande"]  # --tasks wins
    for bad in [("not_a_task", None), ("", None), ("arc_challenge,nope", None)]:
        try:
            parse_tasks(*bad); assert False, f"should reject {bad}"
        except ValueError:
            pass
    # ---- --gens validation (a typo'd generator must FAIL FAST, not no-op a GPU job) ----
    assert parse_gens("mcts,rand,samp") == ["mcts", "rand", "samp"]
    assert parse_gens("hillclimb") == ["hillclimb"]
    assert parse_gens("genetic") == ["genetic"]
    assert parse_gens(" hillclimb , rand ,hillclimb") == ["hillclimb", "rand"]
    assert parse_gens("hillclimb,genetic") == ["hillclimb", "genetic"]
    for bad in ["hc", "ga", "", "mcts,hillclimbb", "genetics"]:
        try:
            parse_gens(bad); assert False, f"should reject --gens {bad!r}"
        except ValueError:
            pass
    # ---- hillclimb arm CORE under a SYNTHETIC scorer (no GPU, no model) ----
    nl28, fullL = 28, list(range(28))

    def _mk_eval(conf_of, correct_of, calls):
        """Synthetic eval_fn; logs every evaluated layer-tuple into `calls`."""
        def _ev(layers):
            calls.append(tuple(layers))
            return "A", correct_of(layers), conf_of(layers)
        return _ev

    # (1) shared #1 == greedy triple; dedup; kmax truncation; full path never re-evaluated;
    #     with a conf gradient (shorter->better toward len 20) the climb actually CLIMBS.
    callsA = []
    ans, cor, cnf, meta = hillclimb_candidates(
        _mk_eval(lambda L: -abs(len(L) - 20) / 10.0, lambda L: 0, callsA),
        _r.Random(11), nl28, 16, ("B", 1, -0.7))
    assert (ans[0], cor[0], cnf[0]) == ("B", 1, -0.7), "candidate #1 must be the greedy triple"
    assert len(ans) == len(cor) == len(cnf) == 16, "kmax truncation, no early stop"
    assert len(callsA) == 15 == len(set(callsA)), "exactly one eval per DISTINCT candidate"
    assert tuple(fullL) not in callsA, "full path = shared #1, never re-evaluated"
    assert meta["accepts"] >= 1, "improving edits must be accepted"
    assert any(len(t) <= nl28 - 3 for t in callsA), \
        "multi-step descent: must reach beyond a single edit of FULL"
    # (2) pure REJECTION (eps=0, every edit strictly worse): the climb start stays FULL,
    #     so every candidate is exactly ONE drllm edit away (len in {26,27,29}).
    callsB = []
    ans, cor, cnf, meta = hillclimb_candidates(
        _mk_eval(lambda L: -5.0, lambda L: 0, callsB),
        _r.Random(12), nl28, 12, ("B", 1, -0.7), eps=0.0)
    assert meta == {"accepts": 0, "eps_accepts": 0, "rejects": 11, "restarts": 0}, meta
    assert all(len(t) in (nl28 - 2, nl28 - 1, nl28 + 1) for t in callsB), \
        "eps=0 + no improvement -> every candidate is a single edit of FULL"
    # (3) eps=1: every candidate accepted -> a random WALK that drifts multiple edits away.
    callsC = []
    ans, cor, cnf, meta = hillclimb_candidates(
        _mk_eval(lambda L: -5.0, lambda L: 0, callsC),
        _r.Random(13), nl28, 12, ("B", 1, -0.7), eps=1.0)
    assert meta["eps_accepts"] == 11 and meta["rejects"] == 0, meta
    assert any(len(t) not in (nl28 - 2, nl28 - 1, nl28 + 1) for t in callsC), \
        "eps=1 must walk beyond distance-1 of FULL"
    # (4) NO-GOLD INVARIANT: flipping EVERY gold bit must not move a single step of the
    #     climb -- the visited-path sequence only depends on conf (and the rng seed).
    seqs = []
    for flip in (0, 1):
        calls = []
        hillclimb_candidates(
            _mk_eval(lambda L: -abs(len(L) - 20) / 10.0, lambda L, _f=flip: _f, calls),
            _r.Random(7), nl28, 16, ("B", 1 - flip, -0.7))
        seqs.append(calls)
    assert seqs[0] == seqs[1], "climb must be BLIND to gold correctness (audit fairness)"
    # (5) tiny skiponly space (nl=4): reachable set = 7 single-skip programs of FULL
    #     (4 skip-1 + 3 skip-2; len<=3 paths admit no further skip). The loop must
    #     RESTART on stalls and terminate gracefully via the tries cap with 1+7 < kmax.
    callsE = []
    ans, cor, cnf, meta = hillclimb_candidates(
        _mk_eval(lambda L: -1.0, lambda L: 0, callsE),
        _r.Random(5), 4, 16, ("B", 0, -0.5), space="skiponly")
    assert len(ans) == 8 and len(set(callsE)) == 7, (len(ans), len(set(callsE)))
    assert meta["restarts"] >= 1, "exhausted neighborhood must trigger restart-at-FULL"
    # (6) single-edit families stay inside their action space (incl. degenerate lengths)
    erng = _r.Random(3)
    for _ in range(2000):
        assert len(hc_single_edit(erng, fullL, nl28, "drllm")) in (26, 27, 29)
        sk = hc_single_edit(erng, fullL, nl28, "skiponly")
        assert len(sk) in (26, 27) and len(sk) == len(set(sk))
        pl = hc_single_edit(erng, fullL, nl28, "polar")
        assert len(pl) != nl28 and 24 <= len(pl) <= 32 and len(pl) - len(set(pl)) <= 4
    assert len(hc_single_edit(erng, [0, 1, 2], nl28, "drllm")) == 4      # skip blocked -> repeat
    assert hc_single_edit(erng, [0, 1, 2], nl28, "skiponly") == [0, 1, 2]  # no legal edit
    withrep = [0, 0, 1, 2, 3, 4, 5, 6, 7]                                # already has a repeat
    for _ in range(200):  # polar <=1 repeat: only skips remain -> the edit always shortens
        assert len(hc_single_edit(erng, withrep, nl28, "polar")) < len(withrep)
    # ---- genetic arm CORE under a SYNTHETIC scorer (no GPU, no model) ----
    # pure operators first: gold-blind fitness signature, elite truncation, crossover.
    import inspect as _inspect
    assert list(_inspect.signature(ga_fitness).parameters) == ["individual"], \
        "fitness takes ONE individual = (layers, conf); there is no gold field to read"
    assert ga_fitness(([9, 8, 7], -0.3)) == -0.3
    _gpop = [([0], -0.5), ([1], -0.1), ([2], -0.9), ([3], -0.2), ([4], -0.05)]
    assert all(len(ind) == 2 for ind in _gpop), "individual is a (layers, conf) 2-tuple: no gold slot"
    assert ga_select_elites(_gpop, 2) == [([4], -0.05), ([1], -0.1)], "keep the top-conf half"
    assert ga_select_elites(_gpop, 99) == sorted(_gpop, key=lambda x: x[1], reverse=True)
    assert GA_N_MUT + GA_N_CROSS == GA_POP - GA_ELITE, "operators refill exactly the non-elite slots"
    # crossover legality: non-empty, len<=2*nl, monotonicity NOT required
    assert ga_legal([5, 4, 3, 2, 1, 0], 28) is True, "non-monotonic layer ids are allowed"
    assert ga_legal([7], 28) is True and ga_legal([], 28) is False, "empty program rejected"
    assert ga_legal(list(range(56)), 28) is True, "len == 2*num_layers allowed"
    assert ga_legal(list(range(57)), 28) is False, "len > 2*num_layers rejected"
    # crossover SHAPE: child == prefix(p1) ++ suffix(p2) for SOME per-parent cuts (p1,p2
    # value-disjoint so the split is well-defined); length bounded by len(p1)+len(p2)
    p1x, p2x, xr = [0, 1, 2, 3, 4], [10, 11, 12, 13, 14], _r.Random(1)
    for _ in range(3000):
        ch = ga_crossover(xr, p1x, p2x, 28)
        assert any(ch == p1x[:c1] + p2x[c2:] for c1 in range(6) for c2 in range(6)), ch
        assert len(ch) <= len(p1x) + len(p2x)

    def _mkga_eval(conf_of, correct_of, calls):
        """Synthetic eval_fn; logs every evaluated layer-tuple into `calls`."""
        def _ev(layers):
            calls.append(tuple(layers))
            return "A", correct_of(layers), conf_of(layers)
        return _ev

    # (1) shared #1 == greedy triple; dedup; kmax cut; FULL never re-evaluated; init fills
    #     the pop; BOTH operators fire and generations advance (conf gradient -> it climbs).
    gcallsA = []
    ans, cor, cnf, gmeta = genetic_candidates(
        _mkga_eval(lambda L: -abs(len(L) - 20) / 10.0, lambda L: 0, gcallsA),
        _r.Random(11), nl28, 16, ("B", 1, -0.7))
    assert (ans[0], cor[0], cnf[0]) == ("B", 1, -0.7), "candidate #1 must be the greedy triple"
    assert len(ans) == len(cor) == len(cnf) == 16, "kmax truncation, no early stop"
    assert len(gcallsA) == 15 == len(set(gcallsA)), "exactly one eval per DISTINCT candidate"
    assert tuple(fullL) not in gcallsA, "full path = shared #1, never re-evaluated"
    assert gmeta["init"] == GA_POP - 1, "initial population = FULL + (P-1) single-edit variants"
    assert gmeta["generations"] >= 1 and gmeta["mutations"] >= 1 and gmeta["crossovers"] >= 1, \
        ("both operators must fire and the GA must run generations", gmeta)
    # (2) kmax < pop_size: budget truncates DURING init; the generational loop never runs.
    gcallsB = []
    ans, cor, cnf, gmeta = genetic_candidates(
        _mkga_eval(lambda L: -1.0, lambda L: 0, gcallsB),
        _r.Random(12), nl28, 4, ("B", 1, -0.7))
    assert len(ans) == 4 and len(gcallsB) == 3 == len(set(gcallsB)), (len(ans), gcallsB)
    assert gmeta["generations"] == 0 and gmeta["init"] == 3, gmeta
    # (3) NO-GOLD INVARIANT: flipping EVERY gold bit must not move one step of the search --
    #     the visited-path sequence only depends on conf (and the rng seed), never on gold.
    gseqs = []
    for flip in (0, 1):
        calls = []
        genetic_candidates(
            _mkga_eval(lambda L: -abs(len(L) - 20) / 10.0, lambda L, _f=flip: _f, calls),
            _r.Random(7), nl28, 20, ("B", 1 - flip, -0.7))
        gseqs.append(calls)
    assert gseqs[0] == gseqs[1], "GA must be BLIND to gold correctness (audit fairness)"
    # (4) tiny skiponly space (nl=4): mutation of FULL reaches only the 7 short skip programs
    #     and short elites admit no further skip, so CROSSOVER (splicing elites into new,
    #     non-monotonic combos) is the SOLE progress source. The loop must recombine, dedup
    #     globally, and terminate gracefully (tries cap) -- proving no-early-stop never hangs.
    gcallsD = []
    ans, cor, cnf, gmeta = genetic_candidates(
        _mkga_eval(lambda L: -float(len(L)), lambda L: 0, gcallsD),
        _r.Random(5), 4, 64, ("B", 0, -0.5), space="skiponly")
    assert len(ans) == 1 + len(gcallsD) and len(gcallsD) == len(set(gcallsD)), "global dedup"
    assert tuple([0, 1, 2, 3]) not in gcallsD, "full path never re-evaluated"
    assert len(ans) <= 64, "kmax cap respected"
    assert gmeta["generations"] >= 1 and gmeta["crossovers"] >= 1, \
        ("crossover must drive progress in the tiny mutation-exhausted space", gmeta)
    print("SELFTEST OK (extract_answer MC/gsm8k + rank + selection gap + PoLar constraint "
          "[skip<=4, <=1 repeat, random program bounded, 3B/8B] + multi-task dispatch + "
          "gens validation + hillclimb core [greedy #1, dedup, kmax cut, accept/reject/"
          "eps-walk/restart, gold-blind climb, per-space single edits] + genetic core "
          "[greedy #1, dedup, kmax cut, init pop, elite truncation, crossover legality/"
          "shape, gold-blind fitness+search, tiny-space recombination+graceful stop])")


# ----------------------------------------------------------------------- GPU driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="CPU-only logic check, no model")
    ap.add_argument("--model_path")
    ap.add_argument("--task", choices=ALLOWED_TASKS,
                    help="single task (kept for back-compat); prefer --tasks for a serial batch")
    ap.add_argument("--tasks", default="",
                    help="comma list of tasks (subset of ALLOWED_TASKS) run SERIALLY under ONE "
                         "model load -> amortizes the 2-5 min load across all of them. Each task "
                         "writes its own fairv2_<task>_seed<seed>[_polar].json. Overrides --task.")
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=0,
                    help="override the decode length for EVERY arm (0 = keep the historical "
                         "per-task default: 15 for dart*, 10 otherwise). Needed to read this "
                         "path arm at the SAME decode length as the act-3 reroll arm: the "
                         "landed *path dart runs sit at 15 tokens, where reasoning models are "
                         "truncated before they emit an answer (orig_acc 0.000-0.010 against "
                         "0.550-0.803 for the same checkpoints under reroll_sc.py's 1024). Any "
                         "pass@k computed on top of a truncated orig_acc is an artifact, so a "
                         "cross-arm comparison MUST pin this explicitly.")
    ap.add_argument("--kmax", type=int, default=64, help="candidate budget for rand/samp arms")
    ap.add_argument("--num_simulations", type=int, default=200, help="MCTS sim budget")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", default="mcts,rand,samp",
                    help="comma list subset of {mcts,rand,samp,hillclimb,genetic} to run. "
                         "hillclimb (hc_*) = the 07-10 local strong-searcher arm; genetic "
                         "(ga_*) = the 07-10 population/recombination strong-searcher arm. "
                         "run either ALONE (e.g. --gens genetic) against already-landed "
                         "3-arm baselines to save GPU -- but then set --out_suffix (below).")
    ap.add_argument("--action_space", choices=["drllm", "polar", "skiponly"], default="drllm",
                    help="drllm = Dr.LLM's MCTS space (skip 1-2 contiguous + repeats, the "
                         "default -- unchanged from before). polar = PoLar-constrained layer "
                         "PROGRAMS (contiguous segments of length <=4, at most ONE repeat per "
                         "program) applied to BOTH the mcts AND rand arms, so 'random >= "
                         "structured' is pinned to PoLar's OWN action space, not Dr.LLM's.")
    ap.add_argument("--out_dir")
    ap.add_argument("--out_suffix", default="",
                    help="extra output-filename suffix, e.g. _hc for hillclimb-only runs. "
                         "The stock name fairv2_<task>_seed<N>[_<action_space>].json is "
                         "BLIND to --gens, so re-running a subset of arms on an existing "
                         "task/seed/tag would OVERWRITE the landed file. Default '' keeps "
                         "every pre-existing filename byte-identical.")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    assert args.model_path and args.out_dir, "need --model_path and --out_dir"
    tasks = parse_tasks(args.tasks, args.task)  # >=1 task, run serially under ONE model load

    import torch
    from data_generation import MCTS, MCTSConfig, MCTSModel, MCTSNode, LayerPath
    from eval_drllm import load_task, grade
    from null_oracle import random_edit_path

    class PolarNode(MCTSNode):
        """MCTS node whose action space is PoLar-constrained (contiguous segments <=4,
        <=1 repeat per program). Overrides action generation (polar_actions) and expand
        (so every DESCENDANT stays a PolarNode -> the constraint holds down the whole
        tree, not just at the root). Everything else (UCB/select/backprop) is inherited."""
        def _generate_possible_actions(self):
            return polar_actions(self.path.layers, self.num_layers)

        def expand(self):
            if not self.untried_actions:
                return self
            a = self.untried_actions.pop()
            if a["type"] == "skip":
                new_path = self.path.skip_layers(a["start"], a["size"])
                desc = f"skip_{a['size']}_at_{a['start']}"
            else:
                new_path = self.path.repeat_layers(a["start"], a["size"], a["count"])
                desc = f"repeat_{a['size']}x{a['count']}_at_{a['start']}"
            child = type(self)(new_path, parent=self, action=desc, num_layers=self.num_layers)
            self.children.append(child)
            return child

    class SkipOnlyNode(PolarNode):
        """MCTS node in the SKIP-ONLY space: Dr.LLM's stock action list filtered to its
        skip branch (skip 1-2 contiguous; the repeat branch removed) = pure test-time
        depth reduction. expand() inherited from PolarNode (type(self) keeps descendants
        SkipOnly; its repeat branch is unreachable since no repeat action is offered)."""
        def _generate_possible_actions(self):
            return [a for a in MCTSNode._generate_possible_actions(self) if a["type"] == "skip"]

    polar = args.action_space == "polar"
    skiponly = args.action_space == "skiponly"
    root_node_cls = PolarNode if polar else (SkipOnlyNode if skiponly else MCTSNode)

    gens = parse_gens(args.gens)  # fail fast on typos (else the GPU job silently no-ops)

    # ---- load the model (data_models shadow already on PYTHONPATH) ONCE, then run every
    #      task in `tasks` serially against it -> amortizes the 2-5 min model load across
    #      the whole list (the point of --tasks). RNG is reset PER TASK in run_task so each
    #      task's stream == a solo run (task-order-independent, reproducible). ----
    cfg = MCTSConfig(num_simulations=args.num_simulations, model_name=args.model_path,
                     dataset="generic")  # dataset field is UNUSED by v2 (grade() via eval_drllm)
    model = MCTSModel(args.model_path, rank=0)
    mcts = MCTS(model, cfg)
    nl = model.num_layers
    full = LayerPath(list(range(nl)))
    state = {"task": tasks[0], "max_new": 10}  # per-task context, updated by run_task()
    print(f"fair-budget-v2: model loaded ({nl} layers); tasks={tasks} kmax={args.kmax} "
          f"sims={args.num_simulations} seed={args.seed} gens={gens} "
          f"action_space={args.action_space}", flush=True)

    # attribute layout of the MCTSModel wrapper (see data_generation.py):
    #   model            = MCTSModel wrapper
    #   model.model      = LlamaForCausalLM (.generate / .compute_transition_scores / .device)
    #   model.model.model= LlamaModel: carries the data_models drop-in .layer_indices
    # (matches generate_with_path's self.model.model.layer_indices -- NOT model.model.*)
    lm = model.model
    inner = lm.model

    def decode_and_score(inp, path, do_sample, temperature):
        """Decode along `path` (greedy or temperature-sampled); return (text, conf).
        conf = mean transition logprob of generated tokens (path_sc.py convention:
        compute_transition_scores(normalize_logits=True), finite-mean, -1e9 fallback).
        Sets/restores inner.layer_indices (the drop-in's active-layer list). Same decode
        as generate_with_path (min_length=n_in+2, cache off for REPEAT paths) + logprob."""
        layers = list(path.layers if hasattr(path, "layers") else path)
        if not layers:  # degenerate all-skip path (reachable by any skip-based search, incl.
            return "", -1e9  # stock Dr.LLM's): no layers -> abstain (empty answer, worst conf).
        orig = inner.layer_indices
        inner.layer_indices = layers
        try:
            prompt = model.prepare_prompt(inp, model.tokenizer)
            ids = model.tokenizer(prompt, return_tensors="pt").to(lm.device)
            n_in = ids.input_ids.shape[1]
            dup = len(layers) != len(set(layers))  # REPEAT -> KV-cache mask overflow, disable
            kw = dict(max_new_tokens=state["max_new"], pad_token_id=model.tokenizer.eos_token_id,
                      min_length=n_in + 2, use_cache=not dup,
                      return_dict_in_generate=True, output_scores=True)
            if do_sample:
                kw.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None)
            with torch.no_grad():
                out = lm.generate(**ids, **kw)
            text = model.tokenizer.decode(out.sequences[0][n_in:], skip_special_tokens=True).strip()
            ts = lm.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True)[0]
            conf = ts[torch.isfinite(ts)].mean().item() if torch.isfinite(ts).any() else -1e9
            return text, conf
        finally:
            inner.layer_indices = orig

    def cand(inp, gf, path, do_sample, temperature):
        """One candidate -> (answer_string, correct 0/1, conf)."""
        text, conf = decode_and_score(inp, path, do_sample, temperature)
        return extract_answer(state["task"], text), int(gf(text) > 0.5), conf

    def mcts_arm(inp, gf, a0, r0, c0):
        """Recording full-sweep (oracle_v2 semantics) but capturing answer+conf per
        DISTINCT path, in eval order (elem0 = full/default path). Re-graded with the
        unified verifier so every arm shares grade()."""
        root = root_node_cls(model.default_path, num_layers=nl)
        evaluated, ans, cor, cnf = {}, [], [], []
        for sim in range(args.num_simulations):
            node = mcts.select_node(root, sim + 1)
            if not node.is_fully_expanded() and node.visits > 0:
                node = node.expand()
            if node.path in evaluated:
                reward = evaluated[node.path]
            else:
                a, c, f = cand(inp, gf, node.path, False, None)
                reward = float(c)
                evaluated[node.path] = reward
                ans.append(a); cor.append(c); cnf.append(f)
            node.backpropagate(reward)
        if ans:  # enforce shared candidate #1 == full greedy exactly (elem0 is the root)
            ans[0], cor[0], cnf[0] = a0, int(r0 > 0.5), c0
        return ans, cor, cnf

    def rand_arm(inp, gf, a0, r0, c0):
        """full greedy (shared #1), then distinct random skip/repeat edits to kmax. NO
        early stop (majority/confidence need the full set). THE control."""
        ans, cor, cnf = [a0], [int(r0 > 0.5)], [c0]
        seen = {tuple(full.layers)}
        tries = 0
        while len(ans) < args.kmax and tries < args.kmax * 50:
            tries += 1
            p = (LayerPath(polar_random_layers(random, nl)) if polar
                 else LayerPath(skiponly_random_layers(random, nl)) if skiponly
                 else random_edit_path(random, random.choice([1, 2, 3]), nl))
            key = tuple(p.layers)
            if key in seen:
                continue
            seen.add(key)
            a, c, f = cand(inp, gf, p, False, None)
            ans.append(a); cor.append(c); cnf.append(f)
        return ans, cor, cnf

    def samp_arm(inp, gf, a0, r0, c0):
        """full greedy (shared #1), then temperature samples on the FIXED full path to
        kmax (output reroll only). Isolates 'more samples' from 'different path'."""
        ans, cor, cnf = [a0], [int(r0 > 0.5)], [c0]
        while len(ans) < args.kmax:
            a, c, f = cand(inp, gf, full, True, args.temperature)
            ans.append(a); cor.append(c); cnf.append(f)
        return ans, cor, cnf

    def hc_arm(inp, gf, a0, r0, c0):
        """full greedy (shared #1), then GREEDY HILL-CLIMB with eps-acceptance over single
        skip/repeat edits (hillclimb_candidates) -- the targeted strong-searcher arm, same
        decode_and_score / grade / kmax / no-early-stop protocol as every other arm. The
        climb is steered ONLY by conf (mean transition logprob); the gold grade is stored
        for the audit columns but never read by the climb (NO-GOLD INVARIANT in the core).
        Returns (ans, cor, cnf, meta); meta = accept/reject/restart counters."""
        return hillclimb_candidates(lambda layers: cand(inp, gf, layers, False, None),
                                    random, nl, args.kmax, (a0, int(r0 > 0.5), c0),
                                    space=args.action_space)

    def ga_arm(inp, gf, a0, r0, c0):
        """full greedy (shared #1), then a GENETIC search (population P=8, top-4 elitism,
        mutation = single skip/repeat edit + crossover = prefix/suffix splice) over layer
        programs (genetic_candidates) -- the population/recombination strong-searcher arm,
        same decode_and_score / grade / kmax / no-early-stop protocol as every other arm.
        Selection is steered ONLY by conf (mean transition logprob); the gold grade is
        stored for the audit columns but never read by the search (NO-GOLD INVARIANT in the
        core). Returns (ans, cor, cnf, meta); meta = generation/operator/dedup counters."""
        return genetic_candidates(lambda layers: cand(inp, gf, layers, False, None),
                                  random, nl, args.kmax, (a0, int(r0 > 0.5), c0),
                                  space=args.action_space)

    def _round(xs):
        return [round(float(x), 4) for x in xs]

    def run_task(task):
        """Full budget-matched audit for ONE task on the ALREADY-LOADED model; writes
        fairv2_<task>_seed<seed>[_polar].json into out_dir. Called once per task in `tasks`;
        the model/shadow are loaded only once by the caller. RNG is reset here so each task's
        stream == a solo run (task-order-independent & reproducible)."""
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        state["task"] = task
        if task.startswith("dart"):
            samples = load_task(task, "short", args.per_level if task == "dart" else 0)
            if task == "dart_pool":
                by = {}
                for s in samples:
                    by.setdefault(s["level"], []).append(s)
                samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
        else:
            samples = load_task(task)[: args.limit]
        state["max_new"] = args.max_new or (15 if task.startswith("dart") else 10)
        grade_fn_of = lambda correct: (lambda text: grade(task, text, correct))
        print(f"[task {task}] n={len(samples)} kmax={args.kmax} sims={args.num_simulations} "
              f"seed={args.seed} gens={gens} action_space={args.action_space}", flush=True)

        recs = []
        t0 = time.time()
        for i, s in enumerate(samples):
            inp, correct = s["input"], s["correct"]
            gf = grade_fn_of(correct)
            # candidate #1 shared by ALL arms: full/default greedy (+answer, +confidence)
            full_text, c0 = decode_and_score(inp, full, False, None)
            r0 = 1.0 if gf(full_text) > 0.5 else 0.0
            a0 = extract_answer(task, full_text)
            rec = {"level": s.get("level"), "orig": r0, "kmax": args.kmax,
                   "orig_answer": a0, "orig_conf": round(float(c0), 4)}
            if "mcts" in gens:
                ans, cor, cnf = mcts_arm(inp, gf, a0, r0, c0)
                rec["mcts_rank"] = first_hit_rank(cor)
                rec["mcts_ndistinct"] = len(cor)
                rec["mcts_answers"], rec["mcts_correct"], rec["mcts_conf"] = ans, cor, _round(cnf)
            if "rand" in gens:
                ans, cor, cnf = rand_arm(inp, gf, a0, r0, c0)
                rec["rand_rank"] = first_hit_rank(cor)
                rec["rand_nevals"] = len(cor)
                rec["rand_answers"], rec["rand_correct"], rec["rand_conf"] = ans, cor, _round(cnf)
            if "samp" in gens:
                ans, cor, cnf = samp_arm(inp, gf, a0, r0, c0)
                rec["samp_rank"] = first_hit_rank(cor)
                rec["samp_nevals"] = len(cor)
                rec["samp_answers"], rec["samp_correct"], rec["samp_conf"] = ans, cor, _round(cnf)
            if "hillclimb" in gens:
                ans, cor, cnf, hmeta = hc_arm(inp, gf, a0, r0, c0)
                rec["hc_rank"] = first_hit_rank(cor)
                rec["hc_nevals"] = len(cor)
                rec["hc_answers"], rec["hc_correct"], rec["hc_conf"] = ans, cor, _round(cnf)
                rec["hc_meta"] = hmeta  # accept/eps/reject/restart counts: proves it climbed
            if "genetic" in gens:
                ans, cor, cnf, gmeta = ga_arm(inp, gf, a0, r0, c0)
                rec["ga_rank"] = first_hit_rank(cor)
                rec["ga_nevals"] = len(cor)
                rec["ga_answers"], rec["ga_correct"], rec["ga_conf"] = ans, cor, _round(cnf)
                rec["ga_meta"] = gmeta  # generation/operator/dedup counts: proves it evolved
            recs.append(rec)
            if (i + 1) % 20 == 0:
                def orc(gen, k):
                    v = [r for r in recs if f"{gen}_correct" in r]
                    return sum(oracle_hit(r[f"{gen}_correct"], k) for r in v) / max(len(v), 1)
                def maj(gen, k):
                    v = [r for r in recs if f"{gen}_answers" in r]
                    return sum(majority_hit(r[f"{gen}_answers"], r[f"{gen}_correct"], k) for r in v) / max(len(v), 1)
                msg = []
                for g, px in GEN_PREFIX:
                    if g in gens:
                        msg.append(f"{px}: orac@{args.kmax}={orc(px, args.kmax):.3f} maj@{args.kmax}={maj(px, args.kmax):.3f}")
                print(f"[{task}] {i+1}/{len(samples)} orig={sum(r['orig'] for r in recs)/len(recs):.3f} "
                      f"{' | '.join(msg)} ({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

        n = len(recs)
        KS = [1, 2, 4, 5, 8, 16, 32, args.kmax]
        summary = {"task": task, "n": n, "kmax": args.kmax, "seed": args.seed,
                   "num_simulations": args.num_simulations, "temperature": args.temperature,
                   "action_space": args.action_space, "model_path": args.model_path,
                   "max_new": state["max_new"],
                   "orig_acc": sum(r["orig"] for r in recs) / n}
        for g, px in GEN_PREFIX:  # px = record-key prefix (hillclimb stores hc_*)
            if g in gens:
                rows = [r for r in recs if f"{px}_answers" in r]
                # oracle kept rank-based for exact backward-compat with v1's {gen}_passk key
                summary[f"{px}_passk"] = {k: sum(1 for r in recs if 1 <= r[f"{px}_rank"] <= k) / n for k in KS}
                summary[f"{px}_majority"] = {
                    k: sum(majority_hit(r[f"{px}_answers"], r[f"{px}_correct"], k) for r in rows) / n for k in KS}
                summary[f"{px}_confidence"] = {
                    k: sum(confidence_hit(r[f"{px}_correct"], r[f"{px}_conf"], k) for r in rows) / n for k in KS}
                summary[f"{px}_selgap"] = {
                    k: summary[f"{px}_passk"][k] - summary[f"{px}_majority"][k] for k in KS}
        os.makedirs(args.out_dir, exist_ok=True)
        as_tag = "" if args.action_space == "drllm" else f"_{args.action_space}"
        out = os.path.join(args.out_dir, f"fairv2_{task}_seed{args.seed}{as_tag}{args.out_suffix}.json")
        json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
        print("SUMMARY", json.dumps(summary, indent=1))
        print("saved", out)

    # run each task serially against the one loaded model. A per-task failure is isolated
    # (traceback + continue) so one bad task never discards the whole job's other outputs;
    # the job still exits non-zero if ANY task failed (so the watcher flags it).
    failed = []
    for task in tasks:
        try:
            run_task(task)
        except Exception as e:  # noqa: BLE001 -- want to salvage the other tasks' results
            import traceback
            traceback.print_exc()
            failed.append(task)
            print(f"[task {task}] FAILED: {type(e).__name__}: {e}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)}/{len(tasks)} task(s) FAILED: {failed}")


if __name__ == "__main__":
    main()
