# HDD-LOC BugsInPy benchmark suite

Run the complete configured suite from this directory with:

```bash
python3 main_pipeline.py
```

The command runs every case in `benchmark_cases.py` with three minimizers:

- `baseline_hdd`: non-weighted, count-based HDD partitioning and unit spectrum evidence.
- `weighted_hdd`: subtree-size HDD partitioning and `1 / weight` spectrum evidence.
- `ddmin_loc`: the original sibling DDMIN-LOC character-level reducer with
  unit spectrum evidence. Structured seeds are serialized before DDMIN and
  candidates are decoded by the existing semantic adapters.

All three runs are analyzed at execution budgets 5, 10, 20, 30, 50, and full with
raw SBFL and deduplicated raw SBFL. The available formulas are Ochiai, Jaccard,
Tarantula, and DStar.

Outputs are written under `benchmark_results/`:

- `all_bugs_status.csv`: execution counts and PASS/FAIL/INVALID balance.
- `all_bugs_metrics.csv`: per-case rank, tie-aware EXAM score, and Inspect@k.
- `aggregate_metrics.csv`: mean/median EXAM and Inspect@k rates across cases.
- `all_bugs_rankings.csv`: top-ranked and ground-truth lines with raw spectra.
- `execution_records/*.jsonl`: every original HDD/DDMin execution before deduplication.
- `run_config.json`: partitions, budgets, formulas, and cases.
- `graphs/*.png` and `graphs/*.pdf`: automatically generated comparison figures.

The generated figures include mean EXAM Score and median tie-aware fault rank
over execution budgets, full-budget Inspect@K rates, EXAM distributions,
per-bug Ochiai heatmaps, and PASS/FAIL/INVALID balance. `TIME_LIMIT` DDMin runs
are visibly noted and use only candidates completed before the cutoff. Aggregate
plots compare only cases present for every displayed approach, preventing a
missing run from making one approach appear artificially better.

To regenerate graphs from existing CSV reports without rerunning any candidate:

```bash
python3 benchmark_visualization.py benchmark_results
```

Use `--reuse-execution-records` to regenerate analytical reports from the
preserved JSONL data without rerunning Docker candidates. Use `--cases` for a
comma-separated subset, for example:

```bash
python3 main_pipeline.py --cases black_2,youtube-dl_2
```

Use `--partitions` to run or regenerate a specific comparison subset, for
example `--partitions baseline_hdd,weighted_hdd` or `--partitions ddmin_loc`.
The original character-level DDMIN can require far more executions than HDD
for long structured serializations, particularly XML.

DDMin-LOC follows the paper's evaluation protocol and has a 15-minute
(900-second) wall-clock limit per bug. If it reaches that limit, all candidate
executions completed before the cutoff are retained and used for SBFL, and the
case is marked `TIME_LIMIT` in `all_bugs_status.csv`. The original DDMin
algorithm is not modified to implement this experiment-level limit. For an
explicit alternative experiment, use `--ddmin-time-limit SECONDS`.

The suite uses the actual BugsInPy patches as authoritative ground truth. A
changed continuation line in a multi-line Python statement is mapped
statically to the executable statement line recorded by coverage. This mapping
happens only after candidate generation and ranking.
