"""Markdown rendering for wm_bench results — honest by construction.

The report prints the same table cell whether LOCI wins or loses a metric,
always shows the brute-force oracle rows, and stamps every section with the
synthetic-data caveat.
"""

from __future__ import annotations


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _curve(values: list | None) -> str:
    if not values:
        return "n/a"
    return " → ".join(f"{v:.2f}" for v in values)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _section_future_analog(per_system: dict[str, dict]) -> list[str]:
    rows = [
        [name, _fmt(m.get("recall_at_k")), _fmt(m.get("recall_at_k_windowed"))]
        for name, m in per_system.items()
    ]
    any_m = next(iter(per_system.values()))
    return [
        f"### Future-state analog recall@{any_m.get('k')} (vs brute-force oracle)",
        "",
        *_table(["system", "recall@k", "recall@k (windowed)"], rows),
        "",
    ]


def _section_novelty(per_system: dict[str, dict]) -> list[str]:
    rows = [
        [
            name,
            _fmt(m.get("auc")),
            _fmt(m.get("auc_onset")),
            _fmt(m.get("median_familiar")),
            _fmt(m.get("median_ood")),
            "yes" if m.get("supported") else "no (null)",
        ]
        for name, m in per_system.items()
    ]
    return [
        "### Novelty ROC-AUC over OOD segments",
        "",
        *_table(
            ["system", "AUC (all OOD)", "AUC (onset)", "median familiar", "median OOD", "API"],
            rows,
        ),
        "",
        "Synthetic OOD segments are constructed near-orthogonal to the base manifold —",
        "an intentionally easy detection problem; read AUC as an upper bound.  'onset'",
        "scores only each excursion's first points (later in-segment points match their",
        "own predecessors and legitimately look familiar to an online detector).",
        "",
    ]


def _section_trajectory(per_system: dict[str, dict]) -> list[str]:
    rows = [
        [name, _fmt(m.get("coverage")), _fmt(m.get("order_fidelity")), str(m.get("method"))]
        for name, m in per_system.items()
    ]
    return [
        "### Trajectory reconstruction fidelity",
        "",
        *_table(["system", "coverage", "order fidelity", "method"], rows),
        "",
    ]


def _section_recall_vs_age(per_system: dict[str, dict]) -> list[str]:
    rows = [
        [
            name,
            _curve(m.get("recall_strict_by_age")),
            _curve(m.get("recall_covered_by_age")),
            _fmt(m.get("compression_ratio"), 2) + ("x" if m.get("compression_ratio") else ""),
            _fmt(m.get("resident_points")),
        ]
        for name, m in per_system.items()
    ]
    return [
        "### Recall vs age under consolidation (flight-recorder curve)",
        "",
        "Recall per age bucket, oldest → newest. 'strict' counts only surviving raw",
        "points; 'covered' also credits a consolidated summary whose source range",
        "covers the ground-truth point.",
        "",
        *_table(
            ["system", "strict recall by age", "covered recall by age", "compression", "resident"],
            rows,
        ),
        "",
    ]


def _section_latency(per_system: dict[str, dict]) -> list[str]:
    rows = [
        [
            name,
            _fmt(m.get("insert_p50_ms")),
            _fmt(m.get("insert_p95_ms")),
            _fmt(m.get("query_p50_ms")),
            _fmt(m.get("query_p95_ms")),
        ]
        for name, m in per_system.items()
    ]
    return [
        "### Sustained-load latency (interleaved insert/query, ms)",
        "",
        *_table(["system", "insert p50", "insert p95", "query p50", "query p95"], rows),
        "",
        "Latency is wall-clock and machine-dependent; compare systems within one run only.",
        "",
    ]


_SECTIONS = {
    "future_analog_recall": _section_future_analog,
    "novelty_auc": _section_novelty,
    "trajectory_fidelity": _section_trajectory,
    "recall_vs_age": _section_recall_vs_age,
    "sustained_load": _section_latency,
}


def _honest_headlines(results: dict[str, dict[str, dict]]) -> list[str]:
    """Auto-generated headlines, including where LOCI loses."""
    lines: list[str] = []
    latency = results.get("sustained_load", {})
    if "naive_qdrant" in latency:
        naive = latency["naive_qdrant"]
        for loci_name in ("loci_local", "loci_qdrant_memory"):
            loci = latency.get(loci_name)
            if not loci:
                continue
            for op in ("insert", "query"):
                l_ms, n_ms = loci[f"{op}_p50_ms"], naive[f"{op}_p50_ms"]
                if n_ms and l_ms > n_ms:
                    lines.append(
                        f"- LOCI loses on {op} latency: {loci_name} p50 {l_ms:.3f}ms vs "
                        f"naive_qdrant {n_ms:.3f}ms ({l_ms / n_ms:.1f}x slower)."
                    )
    age = results.get("recall_vs_age", {})
    cons = age.get("loci_local_consolidated")
    if cons and cons.get("compression_ratio"):
        strict = cons["recall_strict_by_age"]
        covered = cons["recall_covered_by_age"]
        lines.append(
            f"- Flight recorder: {cons['compression_ratio']}x storage compression; strict "
            f"recall on the oldest bucket drops to {strict[0]:.2f} "
            f"(covered-by-summary recall {covered[0]:.2f}) while the newest bucket holds "
            f"{strict[-1]:.2f}."
        )
    novelty = results.get("novelty_auc", {})
    for name, m in novelty.items():
        if m.get("supported") and name.startswith("loci"):
            lines.append(
                f"- Novelty AUC ({name}): {m['auc']:.3f} over all OOD points, "
                f"{m['auc_onset']:.3f} at excursion onset (constructed-easy OOD)."
            )
            break
    return lines


def format_markdown(result: dict) -> str:
    """Render a run document (from runner.run_suite) as markdown."""
    lines = [
        "",
        "## LOCI world-model memory benchmark (wm_bench)",
        "",
        "**DATA: SYNTHETIC.** " + result["synthetic_notice"],
        "",
        (
            f"Config: quick={result['quick']}, seed={result['seed'] or 'task defaults'}, "
            f"epoch_size_ms={result['config']['epoch_size_ms']}, "
            f"systems={', '.join(result['config']['systems'])}"
        ),
        "",
    ]
    for task_name, per_system in result["results"].items():
        section = _SECTIONS.get(task_name)
        if section and per_system:
            lines.extend(section(per_system))

    headlines = _honest_headlines(result["results"])
    if headlines:
        lines.extend(["### Headlines (wins and losses)", "", *headlines, ""])
    lines.append(
        "Ground truth for every recall metric is the brute_force oracle "
        "(exact numpy search over the full stream)."
    )
    lines.append("")
    return "\n".join(lines)
