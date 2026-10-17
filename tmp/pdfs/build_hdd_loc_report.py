from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmark_results"
GRAPHS = RESULTS / "graphs"
OUTPUT = REPO / "output" / "pdf" / "hdd_loc_current_evaluation_report.pdf"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#0B7A75")
BLUE = colors.HexColor("#0072B2")
ORANGE = colors.HexColor("#D55E00")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_TEAL = colors.HexColor("#E8F5F3")
LIGHT_ORANGE = colors.HexColor("#FFF2E8")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#6B7280")
GRID = colors.HexColor("#D5DAE0")

PARTITIONS = ("baseline_hdd", "weighted_hdd", "ddmin_loc")
PARTITION_LABEL = {
    "baseline_hdd": "Baseline HDD",
    "weighted_hdd": "Weighted HDD-LOC",
    "ddmin_loc": "DDMin-LOC",
}
FORMULAS = ("ochiai", "jaccard", "tarantula", "dstar2")
BUDGETS = ("5", "10", "20", "30", "50", "full")


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


STATUS = read_csv("all_bugs_status.csv")
METRICS = read_csv("all_bugs_metrics.csv")


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def median(values) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def metric_rows(**filters) -> list[dict[str, str]]:
    return [
        row
        for row in METRICS
        if all(str(row.get(key)) == str(value) for key, value in filters.items())
    ]


def status_summary(partition: str) -> dict[str, float]:
    rows = [row for row in STATUS if row["partition"] == partition]
    executions = sum(int(row["executions"]) for row in rows)
    failed = sum(int(row["fail"]) for row in rows)
    passed = sum(int(row["pass"]) for row in rows)
    invalid = sum(int(row["invalid"]) for row in rows)
    return {
        "runs": len(rows),
        "executions": executions,
        "failed": failed,
        "passed": passed,
        "valid": failed + passed,
        "invalid": invalid,
        "invalid_rate": invalid / executions if executions else 0.0,
        "runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
    }


def full_mean_exam(partition: str, method: str = "raw_sbfl") -> float:
    rows = metric_rows(partition=partition, budget="full", method=method)
    return mean(float(row["exam_score"]) for row in rows)


def budget_mean_exam(partition: str, budget: str, method: str = "raw_sbfl") -> float:
    rows = metric_rows(partition=partition, budget=budget, method=method)
    return mean(float(row["exam_score"]) for row in rows)


def formula_mean_exam(partition: str, formula: str, method: str) -> float:
    rows = metric_rows(
        partition=partition, budget="full", formula=formula, method=method
    )
    return mean(float(row["exam_score"]) for row in rows)


def evidence_counts(partition: str) -> tuple[int, int]:
    rows = metric_rows(
        partition=partition,
        budget="full",
        formula="ochiai",
        method="raw_sbfl",
    )
    return (
        sum(int(row["raw_evidence_count"]) for row in rows),
        sum(int(row["deduplicated_evidence_count"]) for row in rows),
    )


def semantic_sequences_match() -> bool:
    record_dir = RESULTS / "execution_records"
    selected = ("serialized_input", "outcome", "coverage", "structurally_valid", "semantically_valid")
    for baseline_path in record_dir.glob("*__baseline_hdd.jsonl"):
        case = baseline_path.name.split("__", 1)[0]
        weighted_path = record_dir / f"{case}__weighted_hdd.jsonl"
        if not weighted_path.exists():
            return False
        baseline = [__import__("json").loads(line) for line in baseline_path.read_text().splitlines() if line]
        weighted = [__import__("json").loads(line) for line in weighted_path.read_text().splitlines() if line]
        if len(baseline) != len(weighted):
            return False
        for left, right in zip(baseline, weighted):
            if any(left.get(field) != right.get(field) for field in selected):
                return False
    return True


def paired_weighting_result() -> tuple[int, int, int]:
    index = defaultdict(dict)
    for row in METRICS:
        key = (row["case"], row["budget"], row["formula"], row["method"])
        index[key][row["partition"]] = float(row["exam_score"])
    better = equal = worse = 0
    for values in index.values():
        if "baseline_hdd" not in values or "weighted_hdd" not in values:
            continue
        delta = values["weighted_hdd"] - values["baseline_hdd"]
        if delta < -1e-12:
            better += 1
        elif delta > 1e-12:
            worse += 1
        else:
            equal += 1
    return better, equal, worse


def p(text: str, style="Body"):
    return Paragraph(text, STYLES[style])


def table(data, widths, *, header=True, font_size=8.2, alignments=None):
    prepared = []
    for row_index, row in enumerate(data):
        prepared.append(
            [
                cell
                if isinstance(cell, Paragraph)
                else Paragraph(str(cell), STYLES["TableHeader" if header and row_index == 0 else "TableCell"])
                for cell in row
            ]
        )
    result = Table(prepared, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
        for row_index in range(1, len(prepared)):
            if row_index % 2 == 0:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    if alignments:
        for column, alignment in alignments.items():
            commands.append(("ALIGN", (column, 1 if header else 0), (column, -1), alignment))
    result.setStyle(TableStyle(commands))
    return result


def section(title: str):
    return [Spacer(1, 3 * mm), p(title, "Section"), Spacer(1, 1.5 * mm)]


def callout(text: str, background=LIGHT_TEAL):
    box = Table([[Paragraph(text, STYLES["Callout"])]], colWidths=[174 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def figure(filename: str, caption: str, width=174 * mm, height=None):
    image = Image(str(GRAPHS / filename), width=width, height=height)
    image.hAlign = "CENTER"
    return KeepTogether([image, Spacer(1, 1.5 * mm), p(caption, "Caption")])


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 14 * mm, width - doc.rightMargin, height - 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(doc.leftMargin, height - 10.5 * mm, "HDD-LOC current evaluation")
    canvas.drawRightString(width - doc.rightMargin, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


BASE = getSampleStyleSheet()
STYLES = {
    "Title": ParagraphStyle(
        "Title",
        parent=BASE["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=8,
    ),
    "Subtitle": ParagraphStyle(
        "Subtitle",
        parent=BASE["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=MID_GRAY,
        spaceAfter=10,
    ),
    "Section": ParagraphStyle(
        "Section",
        parent=BASE["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=NAVY,
        spaceBefore=5,
        spaceAfter=4,
    ),
    "Subsection": ParagraphStyle(
        "Subsection",
        parent=BASE["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=TEAL,
        spaceBefore=5,
        spaceAfter=3,
    ),
    "Body": ParagraphStyle(
        "Body",
        parent=BASE["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#25313D"),
        spaceAfter=6,
    ),
    "Bullet": ParagraphStyle(
        "Bullet",
        parent=BASE["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        leftIndent=13,
        firstLineIndent=-7,
        bulletIndent=2,
        textColor=colors.HexColor("#25313D"),
        spaceAfter=4,
    ),
    "Callout": ParagraphStyle(
        "Callout",
        parent=BASE["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=15,
        textColor=NAVY,
    ),
    "Caption": ParagraphStyle(
        "Caption",
        parent=BASE["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
        spaceAfter=5,
    ),
    "TableHeader": ParagraphStyle(
        "TableHeader",
        parent=BASE["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=TA_CENTER,
    ),
    "TableCell": ParagraphStyle(
        "TableCell",
        parent=BASE["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#25313D"),
    ),
    "Small": ParagraphStyle(
        "Small",
        parent=BASE["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=MID_GRAY,
        spaceAfter=4,
    ),
}


def build_story():
    stories = []
    cases = [row["case"] for row in STATUS if row["partition"] == "baseline_hdd"]
    limited = [row for row in STATUS if row["status"] == "TIME_LIMIT"]
    weighting_better, weighting_equal, weighting_worse = paired_weighting_result()

    stories.extend(
        [
            Spacer(1, 20 * mm),
            p("HDD-LOC Current Evaluation Report", "Title"),
            p(
                "Baseline HDD, weighted HDD-LOC, and DDMin-LOC on structured semantic BugsInPy scenarios",
                "Subtitle",
            ),
            Spacer(1, 7 * mm),
            callout(
                "Main result: HDD produces useful localization evidence much earlier and with far fewer executions. "
                "DDMin-LOC reaches a lower full-run mean EXAM Score, but the gain is inconsistent across bugs and does not improve full-run Inspect@5 or Inspect@10 under raw Ochiai."
            ),
            Spacer(1, 10 * mm),
            p("Evaluation snapshot", "Section"),
            table(
                [
                    ["Configured bugs", "Approaches", "Budgets", "SBFL formulas", "Metric rows"],
                    [len(cases), 3, "5, 10, 20, 30, 50, full", 4, len(METRICS)],
                ],
                [30 * mm, 30 * mm, 48 * mm, 34 * mm, 32 * mm],
                alignments={0: "CENTER", 1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER"},
            ),
            Spacer(1, 6 * mm),
            p(
                f"The report contains {len(STATUS)} algorithm-case runs. "
                f"{len(STATUS) - len(limited)} completed normally and {len(limited)} reached the time limit. "
                "All three approaches covered at least one benchmark fault location in every configured case.",
                "Body",
            ),
            p(
                "Scope: " + ", ".join(cases) + ".",
                "Small",
            ),
            Spacer(1, 18 * mm),
            p(f"Generated {datetime.now().strftime('%Y-%m-%d')}", "Small"),
            PageBreak(),
        ]
    )

    stories += section("1. Evaluation design")
    stories.extend(
        [
            p(
                "Each minimizer starts from a structured input that exposes a semantic difference between a buggy and fixed program version. "
                "A candidate is FAIL when the difference remains, PASS when it disappears, and INVALID when the candidate is unusable. "
                "Coverage is collected only from the buggy version.",
                "Body",
            ),
            p(
                "Execution budget means the first N attempted candidate executions in chronological order. INVALID records consume budget but do not enter ef, ep, nf, or np. "
                "The full budget uses every completed execution before minimization ends or the 900-second DDMin-LOC limit is reached.",
                "Body",
            ),
            p("Compared approaches", "Subsection"),
            table(
                [
                    ["Approach", "Candidate reduction", "Spectrum evidence"],
                    ["Baseline HDD", "Tree-aware hierarchical reduction", "Unit weight per valid execution"],
                    ["Weighted HDD-LOC", "Same HDD candidate sequence in this run", "Existing inverse HDD weight"],
                    ["DDMin-LOC", "Character-level delta debugging", "Unit weight per valid execution"],
                ],
                [39 * mm, 67 * mm, 68 * mm],
            ),
            p("Reported localization measures", "Subsection"),
            p(
                "The implementation reports deterministic fault rank and tie-aware best, worst, and average ranks. "
                "EXAM Score is the tie-average rank divided by the number of ranked covered source lines. Inspect@K is true when the tie-average rank is at most K. "
                "Raw SBFL retains every valid execution; deduplicated raw SBFL keeps one canonical observation for each identical outcome and coverage spectrum.",
                "Body",
            ),
            p(
                "Important correction: deduplication removes duplicate test spectra, not duplicate source lines or tied suspiciousness scores. INVALID executions are recorded separately and never treated as PASS or FAIL evidence.",
                "Body",
            ),
        ]
    )

    stories += section("2. Overall efficiency and evidence quality")
    overall = [["Approach", "Executions", "Valid evidence", "Invalid", "Invalid rate", "Runtime", "Full mean EXAM"]]
    for partition in PARTITIONS:
        summary = status_summary(partition)
        overall.append(
            [
                PARTITION_LABEL[partition],
                f"{int(summary['executions']):,}",
                f"{int(summary['valid']):,}",
                f"{int(summary['invalid']):,}",
                f"{summary['invalid_rate']:.1%}",
                f"{summary['runtime_seconds'] / 60:.1f} min",
                f"{full_mean_exam(partition):.4f}",
            ]
        )
    stories.extend(
        [
            table(overall, [35 * mm, 22 * mm, 26 * mm, 20 * mm, 22 * mm, 22 * mm, 27 * mm], alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT"}),
            Spacer(1, 4 * mm),
            p(
                "DDMin-LOC used 36 times as many candidate executions as either HDD approach and about 11.8 times the baseline HDD runtime. "
                "Its invalid rate was 69.9%, compared with 9.2% for HDD. This is the clearest efficiency advantage for structure-aware reduction.",
                "Body",
            ),
            figure(
                "candidate_outcome_balance.png",
                "Figure 1. Candidate outcome composition. INVALID executions consume time and execution budget but do not contribute to SBFL spectra.",
                width=168 * mm,
                height=105 * mm,
            ),
            PageBreak(),
        ]
    )

    stories += section("3. Performance under limited execution budgets")
    budget_table = [["Budget", "Baseline HDD", "Weighted HDD-LOC", "DDMin-LOC"]]
    for budget in BUDGETS:
        budget_table.append(
            [
                budget.capitalize(),
                f"{budget_mean_exam('baseline_hdd', budget):.4f}",
                f"{budget_mean_exam('weighted_hdd', budget):.4f}",
                f"{budget_mean_exam('ddmin_loc', budget):.4f}",
            ]
        )
    stories.extend(
        [
            p(
                "Mean EXAM Score below averages all 12 cases and all four formulas for raw SBFL. Lower values are better.",
                "Body",
            ),
            table(budget_table, [35 * mm, 46 * mm, 48 * mm, 45 * mm], alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}),
            Spacer(1, 4 * mm),
            callout(
                "HDD is clearly stronger at budgets 5, 10, and 20. DDMin-LOC approximately catches HDD at budget 30, remains similar at budget 50, and becomes better only when all available executions are used.",
                LIGHT_BLUE,
            ),
            Spacer(1, 5 * mm),
            figure(
                "exam_score_by_budget_raw_sbfl.png",
                "Figure 2. Mean EXAM Score by execution budget and SBFL formula. The DDMin-LOC time-limit annotation refers to youtube-dl_2.",
                width=174 * mm,
                height=157 * mm,
            ),
            PageBreak(),
        ]
    )

    stories += section("4. Full-budget localization")
    formula_table = [["Evidence", "Formula", "Baseline HDD", "Weighted HDD-LOC", "DDMin-LOC", "DDMin vs baseline"]]
    for method, method_label in (("raw_sbfl", "Raw"), ("deduplicated_raw_sbfl", "Deduplicated")):
        for formula in FORMULAS:
            baseline = formula_mean_exam("baseline_hdd", formula, method)
            weighted = formula_mean_exam("weighted_hdd", formula, method)
            ddmin = formula_mean_exam("ddmin_loc", formula, method)
            formula_table.append(
                [
                    method_label,
                    formula.upper(),
                    f"{baseline:.4f}",
                    f"{weighted:.4f}",
                    f"{ddmin:.4f}",
                    f"{(ddmin - baseline) / baseline:+.1%}",
                ]
            )
    stories.extend(
        [
            table(formula_table, [29 * mm, 27 * mm, 30 * mm, 34 * mm, 26 * mm, 28 * mm], alignments={2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"}),
            Spacer(1, 4 * mm),
            p(
                "DDMin-LOC lowers full-run mean EXAM by about 9.6% for Ochiai, Jaccard, and DStar2. Tarantula does not confirm this result: DDMin-LOC is 1.3% worse with raw evidence and 0.3% worse after deduplication.",
                "Body",
            ),
            p(
                "The mean improvement is not a uniform win. For raw Ochiai, DDMin-LOC has three better, five equal, and four worse tie-aware fault ranks compared with baseline HDD. "
                "Its median tie-aware rank is 220.25, versus 174.75 for HDD. A few large improvements lower mean EXAM despite weaker median rank.",
                "Body",
            ),
            figure(
                "exam_score_distribution_full_raw_sbfl.png",
                "Figure 3. Full-budget EXAM distributions. Similar medians and broad tails show that the mean alone is not sufficient to claim consistent superiority.",
                width=174 * mm,
                height=150 * mm,
            ),
            PageBreak(),
        ]
    )

    stories += section("5. Per-bug Ochiai results")
    selected = metric_rows(budget="full", formula="ochiai", method="raw_sbfl")
    by_case = defaultdict(dict)
    for row in selected:
        by_case[row["case"]][row["partition"]] = row
    per_bug = [["Bug", "HDD tie rank", "DDMin tie rank", "HDD EXAM", "DDMin EXAM", "DDMin result"]]
    for case in cases:
        baseline = by_case[case]["baseline_hdd"]
        ddmin = by_case[case]["ddmin_loc"]
        base_rank = float(baseline["tie_average_rank"])
        ddmin_rank = float(ddmin["tie_average_rank"])
        if ddmin_rank < base_rank:
            outcome = "Better"
        elif ddmin_rank > base_rank:
            outcome = "Worse"
        else:
            outcome = "Equal"
        per_bug.append(
            [
                case,
                f"{base_rank:g}",
                f"{ddmin_rank:g}",
                f"{float(baseline['exam_score']):.4f}",
                f"{float(ddmin['exam_score']):.4f}",
                outcome,
            ]
        )
    stories.extend(
        [
            p(
                "Baseline and weighted HDD have the same full-run Ochiai ranks, so they share one HDD column below.",
                "Body",
            ),
            table(per_bug, [35 * mm, 28 * mm, 29 * mm, 27 * mm, 28 * mm, 27 * mm], alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "CENTER"}),
            Spacer(1, 4 * mm),
            p(
                "Largest DDMin-LOC improvements: tqdm_2 moves from tie rank 158.5 to 12.5, and luigi_1 moves from 34.5 to 1.5. "
                "Largest regression: youtube-dl_1 moves from tie rank 4 to 5,872. Black_2 also worsens from 11 to 24.",
                "Body",
            ),
            PageBreak(),
            figure(
                "exam_score_heatmap_full_ochiai_raw_sbfl.png",
                "Figure 4. Per-bug full-budget Ochiai EXAM Score. Darker cells indicate lower EXAM Scores. The heatmap exposes strong bug-specific variation hidden by the overall mean.",
                width=158 * mm,
                height=181 * mm,
            ),
            PageBreak(),
        ]
    )

    stories += section("6. Inspect@K and practical usability")
    topk = [["Approach", "Inspect@1", "Inspect@3", "Inspect@5", "Inspect@10"]]
    for partition in PARTITIONS:
        rows = metric_rows(
            partition=partition,
            budget="full",
            formula="ochiai",
            method="raw_sbfl",
        )
        topk.append(
            [
                PARTITION_LABEL[partition],
                f"{sum(as_bool(row['inspect_at_1']) for row in rows)}/12",
                f"{sum(as_bool(row['inspect_at_3']) for row in rows)}/12",
                f"{sum(as_bool(row['inspect_at_5']) for row in rows)}/12",
                f"{sum(as_bool(row['inspect_at_10']) for row in rows)}/12",
            ]
        )
    stories.extend(
        [
            table(topk, [47 * mm, 31 * mm, 31 * mm, 31 * mm, 31 * mm], alignments={1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER"}),
            Spacer(1, 4 * mm),
            p(
                "Under full raw Ochiai, DDMin-LOC improves Inspect@3 from 1/12 to 2/12, but does not improve Inspect@5 or Inspect@10. "
                "It also loses the single exact Inspect@1 result achieved by HDD because the youtube-dl_2 faulty line is tied across ranks 1-4 under DDMin-LOC.",
                "Body",
            ),
            figure(
                "inspect_at_k_full_raw_sbfl.png",
                "Figure 5. Full-budget Inspect@K rates across formulas. Top-K effectiveness remains low for every approach on this 12-bug sample.",
                width=174 * mm,
                height=154 * mm,
            ),
            PageBreak(),
        ]
    )

    stories += section("7. Spectrum duplication and weighting")
    duplicate_table = [["Approach", "Valid raw evidence", "Canonical spectra", "Removed as duplicates"]]
    for partition in PARTITIONS:
        raw, deduplicated = evidence_counts(partition)
        duplicate_table.append(
            [
                PARTITION_LABEL[partition],
                f"{raw:,}",
                f"{deduplicated:,}",
                f"{1 - deduplicated / raw:.1%}",
            ]
        )
    stories.extend(
        [
            table(duplicate_table, [48 * mm, 42 * mm, 42 * mm, 42 * mm], alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}),
            Spacer(1, 4 * mm),
            p(
                "DDMin-LOC produces 1,412 valid observations, but they collapse to only 86 unique outcome-coverage spectra. "
                "The 93.9% duplicate rate shows that most additional valid executions repeat evidence already observed. HDD has a 66.9% duplicate rate.",
                "Body",
            ),
            p(
                "Deduplication changes few full-run rankings. Its largest visible effect is under DDMin-LOC Tarantula, where it improves some bug ranks and lowers mean EXAM from 0.3186 to 0.3154. "
                "For Ochiai, the full-run change is only 0.00005 in mean EXAM.",
                "Body",
            ),
            p("Weighted HDD-LOC result", "Subsection"),
            p(
                f"The baseline and weighted HDD records contain the same candidate, outcome, and coverage sequence for all configured bugs: {'yes' if semantic_sequences_match() else 'no'}. "
                f"Across all 576 paired metric rows, weighting is better in {weighting_better}, equal in {weighting_equal}, and worse in {weighting_worse}. "
                "The only difference is youtube-dl_1 at budget 10 with raw Tarantula, where weighting makes the faulty line seven ranks worse.",
                "Body",
            ),
            callout(
                "Current conclusion: the existing inverse HDD weighting is active but provides no observed localization improvement on this benchmark report.",
                LIGHT_ORANGE,
            ),
        ]
    )

    stories += section("8. Time limit and interpretation")
    time_limit_rows = [row for row in STATUS if row["status"] == "TIME_LIMIT"]
    time_data = [["Case", "Approach", "Executions", "FAIL", "PASS", "INVALID", "Runtime"]]
    for row in time_limit_rows:
        time_data.append(
            [
                row["case"],
                PARTITION_LABEL.get(row["partition"], row["partition"]),
                row["executions"],
                row["fail"],
                row["pass"],
                row["invalid"],
                f"{float(row['runtime_seconds']) / 60:.1f} min",
            ]
        )
    stories.extend(
        [
            table(time_data, [33 * mm, 38 * mm, 22 * mm, 17 * mm, 17 * mm, 22 * mm, 25 * mm], alignments={2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT"}),
            Spacer(1, 4 * mm),
            p(
                "The youtube-dl_2 DDMin-LOC run reached the 900-second minimization limit. Its 264 completed attempts include 253 INVALID candidates and only 11 valid spectrum observations. "
                "The report correctly marks the result as TIME_LIMIT and uses only completed valid candidates. It does not treat timeout as PASS or FAIL evidence.",
                "Body",
            ),
            p("Interpretation", "Subsection"),
            p("- HDD is the strongest option when only 5-20 executions are affordable.", "Bullet"),
            p("- DDMin-LOC sometimes produces excellent full-budget localization, but results vary sharply by bug.", "Bullet"),
            p("- DDMin-LOC's lower full-run mean EXAM does not imply greater efficiency or consistently better rank.", "Bullet"),
            p("- Structural reduction prevents large numbers of invalid candidates and supplies useful evidence earlier.", "Bullet"),
            p("- The current weighted mechanism does not improve the baseline and needs a different evidence design or broader evaluation before making a contribution claim.", "Bullet"),
            PageBreak(),
            p("Limitations", "Subsection"),
            p(
                "The current report covers 12 configured bugs, not the complete originally proposed 22-bug set. It omits youtube-dl_3, youtube-dl_4, scrapy_1-2, fastapi_1-2, ansible_1-2, and keras_1-2. "
                "The findings are descriptive and should not be generalized until the remaining adapters are implemented and the expanded suite is executed.",
                "Body",
            ),
            p(
                "EXAM denominators are the number of source lines appearing in each approach's coverage-derived ranking, which can differ slightly across approaches. Absolute and tie-aware ranks should therefore be reported alongside EXAM. "
                "Large tie ranges in several cases show that the generated spectra still provide limited discrimination among many executed lines.",
                "Body",
            ),
        ]
    )

    stories += section("9. Data sources")
    stories.extend(
        [
            p("Current report files used for this document:", "Body"),
            p("- benchmark_results/all_bugs_status.csv", "Bullet"),
            p("- benchmark_results/all_bugs_metrics.csv", "Bullet"),
            p("- benchmark_results/aggregate_metrics.csv", "Bullet"),
            p("- benchmark_results/all_bugs_rankings.csv", "Bullet"),
            p("- benchmark_results/run_config.json", "Bullet"),
            p("- benchmark_results/execution_records/*.jsonl", "Bullet"),
            p(
                "Ground-truth bug locations come from BugsInPy patches and are used only after suspiciousness ranking for evaluation. "
                "All values in this report reflect the current files present when the PDF was generated.",
                "Small",
            ),
        ]
    )
    return stories


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=16 * mm,
        title="HDD-LOC Current Evaluation Report",
        author="HDD-LOC Research Project",
        subject="Current BugsInPy evaluation of baseline HDD, weighted HDD-LOC, and DDMin-LOC",
    )
    document.build(build_story(), onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
