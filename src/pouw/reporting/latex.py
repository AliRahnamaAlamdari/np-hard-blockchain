"""Emit LaTeX table code matching the paper's existing formatting.

Output uses ``booktabs`` rules and the same column specs and label names as
``sn-article.tex``, so a regenerated table is a drop-in replacement for the printed
one.  Nothing here writes to the ``.tex`` source; tables are written to
``results/latex/`` for the author to paste.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _escape(text: Any) -> str:
    """Escape the LaTeX specials that occur in these tables."""
    string = str(text)
    for character, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        string = string.replace(character, replacement)
    return string


def tabular(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    column_spec: str,
    caption: str,
    label: str,
    star: bool = False,
    placement: str = "t",
    font_size: str | None = "\\footnotesize",
    notes: str | None = None,
    escape_cells: bool = True,
) -> str:
    """Render a complete booktabs table environment."""
    environment = "table*" if star else "table"
    lines = [
        f"\\begin{{{environment}}}[{placement}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if font_size:
        lines.append(font_size)
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(f"\\textbf{{{header}}}" for header in headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        cells = [_escape(cell) if escape_cells else str(cell) for cell in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if notes:
        lines.append(f"\\par\\vspace{{2pt}}\\footnotesize {notes}")
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines)


def number(value: Any, precision: int = 4) -> str:
    """Format a numeric cell, tolerating the non-finite values that legitimately occur.

    ``inf`` appears in the max/min block ratio whenever a miner wins no blocks, and
    JSON serialisation turns it into the string ``"inf"``.  Rendering that as a LaTeX
    symbol is correct; crashing on it is not.
    """
    import math

    if isinstance(value, str):
        lowered = value.lower()
        if "inf" in lowered:
            return "$-\\infty$" if lowered.startswith("-") else "$\\infty$"
        if "nan" in lowered:
            return "---"
        try:
            value = float(value)
        except ValueError:
            return _escape(value)
    if value is None:
        return "---"
    if isinstance(value, float):
        if math.isnan(value):
            return "---"
        if math.isinf(value):
            return "$-\\infty$" if value < 0 else "$\\infty$"
    return f"{value:.{precision}f}"


def value_with_error(mean: Any, std: Any, precision: int = 4) -> str:
    """Format ``mean ± std`` for a table cell."""
    return f"{number(mean, precision)} $\\pm$ {number(std, precision)}"


def time_allocation_table(rows: Sequence[Mapping[str, Any]], precision: int = 4) -> str:
    """Regenerate the paper's Table 3 (``tab:timeallocation``) with error bars."""
    headers = ["Difficulty", "PoW (s)", "Useful (s)", "PoW (\\%)", "Useful (\\%)"]
    body = [
        [
            row["difficulty_label"],
            value_with_error(row["mean_pow_seconds"], row["std_pow_seconds"], precision),
            value_with_error(row["mean_solve_seconds"], row["std_solve_seconds"], precision),
            f"{row['pow_share_percent']:.2f}",
            value_with_error(row["useful_share_percent"], row.get("useful_share_std", 0.0), 2),
        ]
        for row in rows
    ]
    return tabular(
        headers=headers,
        rows=body,
        column_spec="lcccc",
        caption="Time share comparison based on PoW difficulty. Values are mean $\\pm$ standard deviation across independent seeds.",
        label="tab:timeallocation",
        escape_cells=False,
    )


def problem_size_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Regenerate the paper's Table 4 (``tab:problemsize``) with error bars.

    Uses more significant figures than the paper, whose 5-decimal column cannot
    reproduce its own percentages at small N (``PAPER_SPEC.md`` §11-I8).
    """
    headers = [
        "Instance ID",
        "Size ($N$)",
        "Avg PoW (s)",
        "Avg Useful Work (s)",
        "Useful Work Share",
    ]
    body = [
        [
            row["instance_id"],
            str(row["size"]),
            f"{row['mean_pow_seconds']:.6g}",
            f"{row['mean_solve_seconds']:.6g} $\\pm$ {row['std_solve_seconds']:.2g}",
            f"{row['useful_share_percent']:.2f}\\%",
        ]
        for row in rows
    ]
    return tabular(
        headers=headers,
        rows=body,
        column_spec="lcccc",
        caption="Impact of NP-hard problem size ($N$) on computation share. Useful-work times are mean $\\pm$ standard deviation over repeated runs.",
        label="tab:problemsize",
        escape_cells=False,
    )


def fairness_table(counts: Mapping[int, int], total: int) -> str:
    """Regenerate the paper's Table 5 (``tab:fairness``)."""
    headers = ["Miner ID", "Blocks Won", "Percentage"]
    body = [
        [f"Node {miner_id}", str(count), f"{count / total * 100:.2f}\\%"]
        for miner_id, count in sorted(counts.items())
    ]
    return tabular(
        headers=headers,
        rows=body,
        column_spec="lcc",
        caption=f"Fairness experiment block distribution ({len(counts)} nodes, {total} blocks).",
        label="tab:fairness",
        placement="h",
        font_size=None,
        escape_cells=False,
    )


def fairness_summary_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """A fairness table across seeds, with the significance test the paper omits."""
    headers = [
        "Statistic",
        "Mean",
        "Std",
        "Min",
        "Max",
    ]
    body = [
        [
            row["statistic"].replace("_", " "),
            number(row["mean"]),
            number(row["std"]),
            number(row["min"]),
            number(row["max"]),
        ]
        for row in rows
    ]
    return tabular(
        headers=headers,
        rows=body,
        column_spec="lcccc",
        caption="Fairness statistics across independent seeds, including a Pearson $\\chi^2$ goodness-of-fit test against a uniform block distribution.",
        label="tab:fairnesssummary",
        placement="h",
        escape_cells=False,
    )


def validation_cost_table(
    steps: Sequence[Mapping[str, Any]], total_ms: float, solve_ms: float, ratio: float
) -> str:
    """Regenerate the paper's Table 6 (``tab:validationcost``)."""
    headers = ["Validation Check Step", "Average Latency (ms)"]
    body = [
        [step["label"], f"{step['milliseconds']:.6f}"] for step in steps
    ]
    body.append(["\\textbf{Total Verification Time}", f"\\textbf{{{total_ms:.6f}}}"])
    notes = (
        f"Mean useful-work solve time: {solve_ms:.2f} ms. "
        f"Verification is {ratio:,.0f}$\\times$ faster than solving."
    )
    return tabular(
        headers=headers,
        rows=body,
        column_spec="lc",
        caption="Detailed validation steps performance report.",
        label="tab:validationcost",
        placement="h",
        font_size=None,
        notes=notes,
        escape_cells=False,
    )


def generic_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    caption: str,
    label: str,
    column_spec: str | None = None,
    star: bool = False,
) -> str:
    """Table builder for the ablation and sensitivity results, which the paper lacks."""
    spec = column_spec or "l" + "c" * (len(headers) - 1)
    return tabular(
        headers=headers,
        rows=rows,
        column_spec=spec,
        caption=caption,
        label=label,
        star=star,
        escape_cells=False,
    )


def write_table(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n")
    return path
