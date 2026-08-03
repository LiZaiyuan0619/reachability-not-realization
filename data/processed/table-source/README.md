# Main-Paper Table Source Data

The eight CSV files in this directory are machine-readable versions of every
numeric table in the main paper. Run:

```bash
python src/tables/build_table_source.py
```

The builder first checks the direct JSON sources for the ordinary-sampling,
MLP localization, directional intervention, checkpoint-fate, and OLMo lineage
tables. It then rewrites all eight CSVs and their hash contract.

Two tables require a frozen analysis selection in addition to direct summary
reads:

- `tab-krobust.csv` records the matched 150-item regrade used to compare
  `K=200` and `K=400`, rather than comparing unmatched full summaries.
- `tab-promptcontrol.csv` records the fate counts for base-reachable but
  initially unproduced questions under checkpoint-native and shared prompt
  formats. The native endpoint summaries and the same-weight shared-format
  per-question outputs are included in `data/raw/rlvr/`.

`tab-reroll.csv` combines four experiment families. The learned-router, MCTS
oracle, and random-oracle rows are directly recoverable from the listed JSON
summaries. The random-vote row preserves the final selected paper values;
the corresponding endpoint and multi-seed fair-budget outputs are included
for audit.

Contracts and provenance are stored in
`data/processed/contracts/table-*.csv`.
