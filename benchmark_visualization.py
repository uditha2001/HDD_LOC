"""Generate static research graphs from HDD-LOC benchmark CSV reports."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence


PARTITION_ORDER = ("baseline_hdd", "weighted_hdd", "ddmin_loc")
PARTITION_LABELS = {
    "baseline_hdd": "Baseline HDD",
    "weighted_hdd": "Weighted HDD-LOC",
    "ddmin_loc": "DDMin-LOC",
}
PARTITION_COLORS = {
    "baseline_hdd": "#0072B2",
    "weighted_hdd": "#D55E00",
    "ddmin_loc": "#009E73",
}
PARTITION_MARKERS = {
    "baseline_hdd": "o",
    "weighted_hdd": "s",
    "ddmin_loc": "^",
}
PARTITION_LINESTYLES = {
    "baseline_hdd": "-",
    "weighted_hdd": "--",
    "ddmin_loc": ":",
}
FORMULA_ORDER = ("ochiai", "jaccard", "tarantula", "dstar2")
METHOD_ORDER = ("raw_sbfl", "deduplicated_raw_sbfl")
METHOD_LABELS = {
    "raw_sbfl": "Raw SBFL",
    "deduplicated_raw_sbfl": "Deduplicated raw SBFL",
}
INSPECT_FIELDS = (
    ("inspect_at_1", "Inspect@1"),
    ("inspect_at_3", "Inspect@3"),
    ("inspect_at_5", "Inspect@5"),
    ("inspect_at_10", "Inspect@10"),
)
OUTPUT_FORMATS = ("png", "pdf")


def _pyplot():
    cache = Path(tempfile.gettempdir()) / "hddloc_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark report not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    present = set(values)
    return [value for value in preferred if value in present] + sorted(
        present.difference(preferred)
    )


def _budget_sort_key(value: str) -> tuple[int, float]:
    if value.lower() == "full":
        return (1, math.inf)
    try:
        return (0, float(value))
    except ValueError:
        return (0, math.inf)


def _budget_labels(rows: Sequence[dict[str, str]]) -> list[str]:
    return sorted({row["budget"] for row in rows}, key=_budget_sort_key)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else math.nan


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.median(materialized) if materialized else math.nan


def _common_cases(
    rows: Sequence[dict[str, str]], partitions: Sequence[str]
) -> set[str]:
    cases_by_partition = {
        partition: {row["case"] for row in rows if row["partition"] == partition}
        for partition in partitions
    }
    if not cases_by_partition:
        return set()
    return set.intersection(*cases_by_partition.values())


def _status_note(status_rows: Sequence[dict[str, str]]) -> str:
    limited = [
        row for row in status_rows if row.get("status", "").upper() == "TIME_LIMIT"
    ]
    if not limited:
        return ""
    return (
        f"{len(limited)} partition run(s) reached the time limit; "
        "their metrics use completed candidates available at cutoff."
    )


def _style_axes(axis, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color="#D0D0D0", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(
    figure,
    output_stem: Path,
    formats: Sequence[str],
    status_note: str = "",
) -> list[Path]:
    if status_note:
        figure.text(0.5, 0.005, status_note, ha="center", va="bottom", fontsize=8)
    paths = []
    for output_format in formats:
        path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    return paths


def _formula_axes(plt, formulas: Sequence[str], title: str, ylabel: str):
    columns = 2
    rows = math.ceil(len(formulas) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, 4.1 * rows),
        squeeze=False,
        sharey=False,
    )
    figure.suptitle(title, fontsize=15, y=0.99)
    flat = list(axes.flat)
    for axis, formula in zip(flat, formulas):
        axis.set_title(formula.upper())
        axis.set_ylabel(ylabel)
    for axis in flat[len(formulas) :]:
        axis.set_visible(False)
    return figure, flat[: len(formulas)]


def _plot_budget_metric(
    plt,
    metric_rows: Sequence[dict[str, str]],
    method: str,
    field: str,
    reducer: Callable[[Iterable[float]], float],
    ylabel: str,
    title: str,
    output_dir: Path,
    formats: Sequence[str],
    status_note: str,
) -> list[Path]:
    rows = [row for row in metric_rows if row["method"] == method]
    formulas = _ordered((row["formula"] for row in rows), FORMULA_ORDER)
    if not formulas:
        return []
    budgets = _budget_labels(rows)
    partitions = _ordered((row["partition"] for row in rows), PARTITION_ORDER)
    figure, axes = _formula_axes(plt, formulas, title, ylabel)
    for axis, formula in zip(axes, formulas):
        formula_rows = [row for row in rows if row["formula"] == formula]
        for partition in partitions:
            points = []
            for budget in budgets:
                budget_rows = [
                    row for row in formula_rows if row["budget"] == budget
                ]
                common_cases = _common_cases(budget_rows, partitions)
                values = (
                    float(row[field])
                    for row in budget_rows
                    if row["partition"] == partition and row["case"] in common_cases
                )
                points.append(reducer(values))
            axis.plot(
                range(len(budgets)),
                points,
                color=PARTITION_COLORS.get(partition),
                marker=PARTITION_MARKERS.get(partition, "o"),
                linestyle=PARTITION_LINESTYLES.get(partition, "-"),
                linewidth=2,
                markersize=5,
                label=PARTITION_LABELS.get(partition, partition),
            )
        axis.set_xticks(range(len(budgets)), budgets)
        axis.set_xlabel("Execution budget")
        if field == "exam_score":
            axis.set_ylim(0.0, 1.0)
        _style_axes(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(handles),
            frameon=False,
        )
        figure.subplots_adjust(top=0.84, bottom=0.09, hspace=0.35, wspace=0.28)
    return _save_figure(
        figure,
        output_dir / f"{field}_by_budget_{method}",
        formats,
        status_note,
    )


def _plot_inspect_rates(
    plt,
    metric_rows: Sequence[dict[str, str]],
    method: str,
    output_dir: Path,
    formats: Sequence[str],
    status_note: str,
) -> list[Path]:
    rows = [
        row
        for row in metric_rows
        if row["method"] == method and row["budget"].lower() == "full"
    ]
    formulas = _ordered((row["formula"] for row in rows), FORMULA_ORDER)
    if not formulas:
        return []
    partitions = _ordered((row["partition"] for row in rows), PARTITION_ORDER)
    figure, axes = _formula_axes(
        plt,
        formulas,
        f"Full-budget Inspect@K rate — {METHOD_LABELS.get(method, method)}",
        "Cases localized (%)",
    )
    width = 0.18
    inspect_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    centers = list(range(len(partitions)))
    for axis, formula in zip(axes, formulas):
        formula_rows = [row for row in rows if row["formula"] == formula]
        common_cases = _common_cases(formula_rows, partitions)
        for offset_index, (field, label) in enumerate(INSPECT_FIELDS):
            offset = (offset_index - 1.5) * width
            rates = []
            for partition in partitions:
                members = [
                    row
                    for row in formula_rows
                    if row["partition"] == partition and row["case"] in common_cases
                ]
                rates.append(
                    100.0
                    * _mean(row[field].strip().lower() == "true" for row in members)
                )
            axis.bar(
                [center + offset for center in centers],
                rates,
                width=width,
                color=inspect_colors[offset_index],
                label=label,
            )
        axis.set_xticks(
            centers,
            [PARTITION_LABELS.get(partition, partition) for partition in partitions],
            rotation=12,
            ha="right",
        )
        axis.set_ylim(0, 100)
        axis.set_xlabel("Localization approach")
        _style_axes(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(top=0.84, bottom=0.13, hspace=0.4, wspace=0.28)
    return _save_figure(
        figure,
        output_dir / f"inspect_at_k_full_{method}",
        formats,
        status_note,
    )


def _plot_exam_distribution(
    plt,
    metric_rows: Sequence[dict[str, str]],
    method: str,
    output_dir: Path,
    formats: Sequence[str],
    status_note: str,
) -> list[Path]:
    rows = [
        row
        for row in metric_rows
        if row["method"] == method and row["budget"].lower() == "full"
    ]
    formulas = _ordered((row["formula"] for row in rows), FORMULA_ORDER)
    if not formulas:
        return []
    partitions = _ordered((row["partition"] for row in rows), PARTITION_ORDER)
    figure, axes = _formula_axes(
        plt,
        formulas,
        f"Full-budget EXAM Score distribution — {METHOD_LABELS.get(method, method)}",
        "EXAM Score (lower is better)",
    )
    for axis, formula in zip(axes, formulas):
        formula_rows = [row for row in rows if row["formula"] == formula]
        common_cases = _common_cases(formula_rows, partitions)
        samples = [
            [
                float(row["exam_score"])
                for row in formula_rows
                if row["partition"] == partition and row["case"] in common_cases
            ]
            for partition in partitions
        ]
        keep = [(partition, values) for partition, values in zip(partitions, samples) if values]
        if not keep:
            axis.set_visible(False)
            continue
        boxes = axis.boxplot(
            [values for _, values in keep],
            tick_labels=[PARTITION_LABELS.get(partition, partition) for partition, _ in keep],
            patch_artist=True,
            showmeans=True,
        )
        for patch, (partition, _) in zip(boxes["boxes"], keep):
            patch.set_facecolor(PARTITION_COLORS.get(partition, "#777777"))
            patch.set_alpha(0.7)
        axis.tick_params(axis="x", labelrotation=12)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Localization approach")
        _style_axes(axis)
    figure.subplots_adjust(top=0.9, bottom=0.13, hspace=0.4, wspace=0.28)
    return _save_figure(
        figure,
        output_dir / f"exam_score_distribution_full_{method}",
        formats,
        status_note,
    )


def _plot_exam_heatmap(
    plt,
    metric_rows: Sequence[dict[str, str]],
    method: str,
    output_dir: Path,
    formats: Sequence[str],
    status_note: str,
) -> list[Path]:
    import numpy as np

    rows = [
        row
        for row in metric_rows
        if row["method"] == method
        and row["budget"].lower() == "full"
        and row["formula"] == "ochiai"
    ]
    if not rows:
        return []
    cases = sorted({row["case"] for row in rows})
    partitions = _ordered((row["partition"] for row in rows), PARTITION_ORDER)
    values = {
        (row["case"], row["partition"]): float(row["exam_score"]) for row in rows
    }
    matrix = np.array(
        [
            [values.get((case, partition), math.nan) for partition in partitions]
            for case in cases
        ]
    )
    figure, axis = plt.subplots(figsize=(8.5, max(4.2, 0.42 * len(cases) + 2.0)))
    image = axis.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_title(f"Per-bug Ochiai EXAM Score — {METHOD_LABELS.get(method, method)}")
    axis.set_xticks(
        range(len(partitions)),
        [PARTITION_LABELS.get(partition, partition) for partition in partitions],
        rotation=12,
        ha="right",
    )
    axis.set_yticks(range(len(cases)), cases)
    axis.set_xlabel("Localization approach")
    axis.set_ylabel("BugsInPy case")
    for row_index in range(len(cases)):
        for column_index in range(len(partitions)):
            value = matrix[row_index, column_index]
            if not math.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.45 else "black",
                    fontsize=8,
                )
    colorbar = figure.colorbar(image, ax=axis, pad=0.03)
    colorbar.set_label("EXAM Score (lower is better)")
    figure.subplots_adjust(left=0.18, right=0.88, top=0.9, bottom=0.14)
    return _save_figure(
        figure,
        output_dir / f"exam_score_heatmap_full_ochiai_{method}",
        formats,
        status_note,
    )


def _plot_outcome_balance(
    plt,
    status_rows: Sequence[dict[str, str]],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = [row for row in status_rows if row.get("partition")]
    if not rows:
        return []
    partitions = _ordered((row["partition"] for row in rows), PARTITION_ORDER)
    counts = defaultdict(Counter)
    time_limits = Counter()
    run_counts = Counter()
    for row in rows:
        partition = row["partition"]
        run_counts[partition] += 1
        if row.get("status", "").upper() == "TIME_LIMIT":
            time_limits[partition] += 1
        for outcome in ("fail", "pass", "invalid"):
            try:
                counts[partition][outcome] += int(row.get(outcome, "0") or 0)
            except ValueError:
                continue

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bottoms = [0.0] * len(partitions)
    outcome_specs = (
        ("fail", "FAIL", "#D55E00"),
        ("pass", "PASS", "#0072B2"),
        ("invalid", "INVALID", "#999999"),
    )
    totals = [sum(counts[partition].values()) for partition in partitions]
    for outcome, label, color in outcome_specs:
        shares = [
            100.0 * counts[partition][outcome] / total if total else 0.0
            for partition, total in zip(partitions, totals)
        ]
        bars = axis.bar(partitions, shares, bottom=bottoms, color=color, label=label)
        for bar, share in zip(bars, shares):
            if share >= 7.0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        bottoms = [bottom + share for bottom, share in zip(bottoms, shares)]
    axis.set_xticks(
        range(len(partitions)),
        [
            f"{PARTITION_LABELS.get(partition, partition)}\n"
            f"{run_counts[partition]} runs, {time_limits[partition]} time-limited"
            for partition in partitions
        ],
    )
    for index, total in enumerate(totals):
        axis.text(index, 101.5, f"n={total}", ha="center", va="bottom", fontsize=9)
    axis.set_ylim(0, 108)
    axis.set_ylabel("Completed candidate executions (%)")
    axis.set_xlabel("Localization approach")
    figure.suptitle("PASS / FAIL / INVALID evidence balance", fontsize=15, y=0.98)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
    )
    _style_axes(axis)
    figure.subplots_adjust(top=0.82, bottom=0.18)
    return _save_figure(figure, output_dir / "candidate_outcome_balance", formats)


def generate_benchmark_graphs(
    results_dir: str | Path,
    *,
    formats: Sequence[str] = OUTPUT_FORMATS,
) -> tuple[Path, ...]:
    """Generate all supported plots from one completed benchmark report set."""

    unsupported = sorted(set(formats).difference(OUTPUT_FORMATS))
    if unsupported:
        raise ValueError(f"unsupported graph format(s): {', '.join(unsupported)}")
    if not formats:
        raise ValueError("at least one graph format is required")

    results = Path(results_dir)
    metric_rows = _read_csv(results / "all_bugs_metrics.csv")
    status_rows = _read_csv(results / "all_bugs_status.csv")
    output_dir = results / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    generated: list[Path] = []
    note = _status_note(status_rows)

    for method in _ordered((row["method"] for row in metric_rows), METHOD_ORDER):
        generated.extend(
            _plot_budget_metric(
                plt,
                metric_rows,
                method,
                "exam_score",
                _mean,
                "Mean EXAM Score (lower is better)",
                f"Mean EXAM Score by execution budget — {METHOD_LABELS.get(method, method)}",
                output_dir,
                formats,
                note,
            )
        )
        generated.extend(
            _plot_budget_metric(
                plt,
                metric_rows,
                method,
                "tie_average_rank",
                _median,
                "Median tie-aware faulty-line rank (lower is better)",
                f"Faulty-line rank by execution budget — {METHOD_LABELS.get(method, method)}",
                output_dir,
                formats,
                note,
            )
        )
        generated.extend(
            _plot_inspect_rates(
                plt, metric_rows, method, output_dir, formats, note
            )
        )
        generated.extend(
            _plot_exam_distribution(
                plt, metric_rows, method, output_dir, formats, note
            )
        )
        generated.extend(
            _plot_exam_heatmap(
                plt, metric_rows, method, output_dir, formats, note
            )
        )
    generated.extend(_plot_outcome_balance(plt, status_rows, output_dir, formats))
    plt.close("all")
    return tuple(generated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate HDD-LOC benchmark PNG/PDF graphs from CSV reports."
    )
    parser.add_argument("results_dir", nargs="?", default="benchmark_results")
    args = parser.parse_args(list(argv) if argv is not None else None)
    generated = generate_benchmark_graphs(args.results_dir)
    print(f"Generated {len(generated)} graph files in {Path(args.results_dir) / 'graphs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
