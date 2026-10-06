#!/usr/bin/env python3
"""Run run-level paired or explicitly unpaired significance analyses."""

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


PAIR_KEYS = ("dataset", "seed", "split_id")


def read_rows(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = set(PAIR_KEYS + ("method",))
    if not rows:
        raise ValueError("Input CSV contains no rows.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError("Input CSV is missing columns: {}".format(sorted(missing)))
    return rows


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    n = values.size
    if n == 0:
        raise ValueError("At least one paired difference is required.")
    if n > 20:
        raise ValueError("Exact sign-flip test is limited to n <= 20.")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 1 << n
    for bits in range(total):
        signs = np.ones(n, dtype=np.float64)
        for index in range(n):
            if bits & (1 << index):
                signs[index] = -1.0
        statistic = abs(float((values * signs).mean()))
        if statistic >= observed - 1e-15:
            exceed += 1
    return float(exceed / float(total))


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    m = values.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(values)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return [float(x) for x in adjusted]


def _t_critical(df: float, confidence: float = 0.95) -> float:
    return float(stats.t.ppf(0.5 + confidence / 2.0, df))


def paired_summary(reference: np.ndarray, comparator: np.ndarray) -> Dict[str, Any]:
    if reference.shape != comparator.shape:
        raise ValueError("Paired arrays must have the same shape.")
    n = reference.size
    if n < 2:
        raise ValueError("At least two paired runs are required.")
    differences = reference - comparator
    mean_difference = float(differences.mean())
    sd_difference = float(differences.std(ddof=1))
    se = sd_difference / math.sqrt(n)
    critical = _t_critical(n - 1)
    t_result = stats.ttest_rel(reference, comparator)
    return {
        "design": "paired",
        "n_reference": int(n),
        "n_comparator": int(n),
        "n_pairs": int(n),
        "reference_mean": float(reference.mean()),
        "comparator_mean": float(comparator.mean()),
        "mean_difference_reference_minus_comparator": mean_difference,
        "ci95_low": float(mean_difference - critical * se),
        "ci95_high": float(mean_difference + critical * se),
        "paired_t_statistic": float(t_result.statistic),
        "p_value": float(t_result.pvalue),
        "exact_sign_flip_p_value": exact_sign_flip_pvalue(differences),
        "cohen_dz": (
            float(mean_difference / sd_difference)
            if sd_difference > 0
            else None
        ),
    }


def welch_summary(reference: np.ndarray, comparator: np.ndarray) -> Dict[str, Any]:
    if reference.size < 2 or comparator.size < 2:
        raise ValueError("Welch analysis requires at least two runs per method.")
    n1 = reference.size
    n2 = comparator.size
    mean_difference = float(reference.mean() - comparator.mean())
    var1 = float(reference.var(ddof=1))
    var2 = float(comparator.var(ddof=1))
    se2 = var1 / n1 + var2 / n2
    df_numerator = se2 ** 2
    df_denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_numerator / df_denominator if df_denominator > 0 else float("inf")
    critical = _t_critical(df)
    result = stats.ttest_ind(reference, comparator, equal_var=False)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = mean_difference / pooled_sd if pooled_sd > 0 else None
    return {
        "design": "unpaired_welch",
        "n_reference": int(n1),
        "n_comparator": int(n2),
        "n_pairs": None,
        "reference_mean": float(reference.mean()),
        "comparator_mean": float(comparator.mean()),
        "mean_difference_reference_minus_comparator": mean_difference,
        "ci95_low": float(mean_difference - critical * math.sqrt(se2)),
        "ci95_high": float(mean_difference + critical * math.sqrt(se2)),
        "welch_df": float(df),
        "welch_t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "exact_sign_flip_p_value": None,
        "cohen_d": float(cohen_d) if cohen_d is not None else None,
    }


def _method_rows(
    rows: Iterable[Dict[str, str]],
    dataset: str,
    method: str,
) -> List[Dict[str, str]]:
    return [
        row for row in rows
        if row["dataset"] == dataset and row["method"] == method
    ]


def _paired_values(
    rows: List[Dict[str, str]],
    dataset: str,
    reference: str,
    comparator: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    def index_for(method: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
        index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for row in _method_rows(rows, dataset, method):
            key = tuple(row[name] for name in PAIR_KEYS)
            if key in index:
                raise ValueError("Duplicate run key for {}: {}".format(method, key))
            index[key] = row
        return index

    reference_index = index_for(reference)
    comparator_index = index_for(comparator)
    keys = sorted(set(reference_index).intersection(comparator_index))
    if len(keys) < 2:
        raise ValueError(
            "Fewer than two matched runs for dataset={}, methods={} vs {}.".format(
                dataset, reference, comparator
            )
        )

    reference_values = []
    comparator_values = []
    for key in keys:
        reference_row = reference_index[key]
        comparator_row = comparator_index[key]
        ref_fingerprint = str(reference_row.get("data_fingerprint") or "").strip()
        cmp_fingerprint = str(comparator_row.get("data_fingerprint") or "").strip()
        if not ref_fingerprint or not cmp_fingerprint:
            raise ValueError("Both methods require a nonempty data fingerprint for paired key {}.".format(key))
        if ref_fingerprint != cmp_fingerprint:
            raise ValueError("Data fingerprint mismatch for paired key {}.".format(key))
        reference_values.append(float(reference_row[metric]))
        comparator_values.append(float(comparator_row[metric]))
    return (
        np.asarray(reference_values, dtype=np.float64),
        np.asarray(comparator_values, dtype=np.float64),
    )


def analyze(
    rows: List[Dict[str, str]],
    *,
    reference: str,
    comparators: Sequence[str],
    metrics: Sequence[str],
    unpaired: bool,
) -> List[Dict[str, Any]]:
    datasets = sorted({row["dataset"] for row in rows})
    results: List[Dict[str, Any]] = []

    for dataset in datasets:
        for comparator in comparators:
            for metric in metrics:
                if metric not in rows[0]:
                    raise ValueError("Input CSV is missing metric column {!r}.".format(metric))
                if unpaired:
                    reference_values = np.asarray(
                        [
                            float(row[metric])
                            for row in _method_rows(rows, dataset, reference)
                        ],
                        dtype=np.float64,
                    )
                    comparator_values = np.asarray(
                        [
                            float(row[metric])
                            for row in _method_rows(rows, dataset, comparator)
                        ],
                        dtype=np.float64,
                    )
                    summary = welch_summary(reference_values, comparator_values)
                else:
                    reference_values, comparator_values = _paired_values(
                        rows, dataset, reference, comparator, metric
                    )
                    summary = paired_summary(reference_values, comparator_values)
                result = {
                    "dataset": dataset,
                    "metric": metric,
                    "reference_method": reference,
                    "comparator_method": comparator,
                }
                result.update(summary)
                results.append(result)

    adjusted = holm_adjust([float(result["p_value"]) for result in results])
    for result, adjusted_value in zip(results, adjusted):
        result["holm_adjusted_p_value"] = adjusted_value
    return results


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def render_markdown(results: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Run-level significance analysis",
        "",
        "> Differences are defined as reference minus comparator. Statistical units are independent runs, not patches.",
        "",
        "| Dataset | Metric | Reference | Comparator | Design | n | Mean difference | 95% CI | p | Holm-adjusted p | Sign-flip p | Effect size |",
        "|:---|:---|:---|:---|:---|---:|---:|:---|---:|---:|---:|---:|",
    ]
    for result in results:
        effect = result.get("cohen_dz", result.get("cohen_d"))
        ci = "[{}, {}]".format(_format(result["ci95_low"]), _format(result["ci95_high"]))
        lines.append(
            "| {dataset} | {metric} | {reference_method} | {comparator_method} | "
            "{design} | {n} | {diff} | {ci} | {p} | {holm} | {perm} | {effect} |".format(
                dataset=result["dataset"],
                metric=result["metric"],
                reference_method=result["reference_method"],
                comparator_method=result["comparator_method"],
                design=result["design"],
                n=result.get("n_pairs") or "{} / {}".format(
                    result["n_reference"], result["n_comparator"]
                ),
                diff=_format(result["mean_difference_reference_minus_comparator"]),
                ci=ci,
                p=_format(result["p_value"]),
                holm=_format(result["holm_adjusted_p_value"]),
                perm=_format(result.get("exact_sign_flip_p_value")),
                effect=_format(effect),
            )
        )
    return "\n".join(lines) + "\n"


def write_results(results: Sequence[Dict[str, Any]], output_dir: str) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite output directory: {}".format(destination))
    destination.mkdir(parents=True)

    columns = sorted({key for result in results for key in result.keys()})
    with (destination / "run_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    (destination / "run_statistics.md").write_text(
        render_markdown(results), encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze run-level results with paired tests by default."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", default="padl")
    parser.add_argument(
        "--comparators",
        required=True,
        help="Comma-separated comparator method names.",
    )
    parser.add_argument(
        "--metrics",
        default="rmse",
        help="Comma-separated run-level metrics.",
    )
    parser.add_argument(
        "--unpaired",
        action="store_true",
        help="Explicitly use independent-sample Welch tests instead of pairing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_rows(args.input_csv)
    comparators = [item.strip() for item in args.comparators.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    results = analyze(
        rows,
        reference=args.reference,
        comparators=comparators,
        metrics=metrics,
        unpaired=args.unpaired,
    )
    destination = write_results(results, args.output_dir)
    print("Wrote run-level statistics to {}".format(destination))


if __name__ == "__main__":
    main()
