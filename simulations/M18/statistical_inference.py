"""Deterministic, standard-library statistical helpers for mission M18.

The functions favor transparent implementations over feature breadth so learners
can inspect every assumption used by the accompanying notebook.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_EXPERIMENT_COLUMNS = {"day", "group", "sessions", "conversions"}


def load_daily_experiment(path: str | Path) -> list[dict[str, int | str]]:
    """Load and validate grouped binomial observations from a CSV fixture."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_EXPERIMENT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        rows: list[dict[str, int | str]] = []
        seen: set[tuple[str, str]] = set()
        for line_number, raw in enumerate(reader, start=2):
            group = raw["group"].strip()
            day = raw["day"].strip()
            if group not in {"A", "B"}:
                raise ValueError(f"line {line_number}: group must be A or B")
            try:
                sessions = int(raw["sessions"])
                conversions = int(raw["conversions"])
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number}: sessions and conversions must be integers"
                ) from exc
            if sessions <= 0 or not 0 <= conversions <= sessions:
                raise ValueError(f"line {line_number}: invalid binomial counts")
            key = (day, group)
            if key in seen:
                raise ValueError(f"line {line_number}: duplicate day/group {key}")
            seen.add(key)
            rows.append(
                {
                    "day": day,
                    "group": group,
                    "sessions": sessions,
                    "conversions": conversions,
                }
            )

    if {str(row["group"]) for row in rows} != {"A", "B"}:
        raise ValueError("experiment must contain both A and B groups")
    return rows


def aggregate_counts(
    rows: Iterable[dict[str, int | str]],
) -> dict[str, dict[str, int | float]]:
    """Aggregate sessions, conversions, and conversion rates by group."""

    totals = {
        "A": {"sessions": 0, "conversions": 0},
        "B": {"sessions": 0, "conversions": 0},
    }
    for row in rows:
        group = str(row["group"])
        if group not in totals:
            raise ValueError(f"unknown group: {group}")
        totals[group]["sessions"] += int(row["sessions"])
        totals[group]["conversions"] += int(row["conversions"])

    result: dict[str, dict[str, int | float]] = {}
    for group, counts in totals.items():
        sessions = counts["sessions"]
        if sessions == 0:
            raise ValueError(f"group {group} contains no sessions")
        result[group] = {
            **counts,
            "rate": counts["conversions"] / sessions,
        }
    return result


def expand_binary_outcomes(
    rows: Iterable[dict[str, int | str]], group: str
) -> list[int]:
    """Expand grouped counts into a transparent 0/1 outcome vector."""

    if group not in {"A", "B"}:
        raise ValueError("group must be A or B")
    outcomes: list[int] = []
    for row in rows:
        if row["group"] != group:
            continue
        sessions = int(row["sessions"])
        conversions = int(row["conversions"])
        outcomes.extend([1] * conversions)
        outcomes.extend([0] * (sessions - conversions))
    if not outcomes:
        raise ValueError(f"group {group} contains no outcomes")
    return outcomes


def describe(values: Sequence[float | int]) -> dict[str, float | int]:
    """Return elementary distribution summaries using sample variance."""

    if not values:
        raise ValueError("describe requires at least one value")
    numeric = [float(value) for value in values]
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "variance": statistics.variance(numeric) if len(numeric) > 1 else 0.0,
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def difference_in_means(
    baseline: Sequence[float | int], treatment: Sequence[float | int]
) -> float:
    """Return treatment minus baseline on the original measurement scale."""

    if not baseline or not treatment:
        raise ValueError("both groups require observations")
    return statistics.fmean(treatment) - statistics.fmean(baseline)


def standard_error(values: Sequence[float | int]) -> float:
    """Estimate the standard error of a sample mean."""

    if len(values) < 2:
        raise ValueError("standard error requires at least two observations")
    return statistics.stdev(values) / math.sqrt(len(values))


def difference_standard_error(
    baseline: Sequence[float | int], treatment: Sequence[float | int]
) -> float:
    """Estimate the unpooled standard error for a difference in means."""

    if len(baseline) < 2 or len(treatment) < 2:
        raise ValueError("each group requires at least two observations")
    return math.sqrt(
        statistics.variance(baseline) / len(baseline)
        + statistics.variance(treatment) / len(treatment)
    )


def normal_confidence_interval(
    baseline: Sequence[float | int],
    treatment: Sequence[float | int],
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a normal-approximation interval for treatment minus baseline."""

    estimate = difference_in_means(baseline, treatment)
    margin = z * difference_standard_error(baseline, treatment)
    return estimate - margin, estimate + margin


def percentile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolate a percentile without external dependencies."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_differences(
    baseline: Sequence[float | int],
    treatment: Sequence[float | int],
    *,
    resamples: int = 4_000,
    seed: int = 18,
) -> list[float]:
    """Resample each observed group independently and return mean differences."""

    if not baseline or not treatment:
        raise ValueError("both groups require observations")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    baseline_values = list(baseline)
    treatment_values = list(treatment)
    differences: list[float] = []
    for _ in range(resamples):
        baseline_sample = rng.choices(baseline_values, k=len(baseline_values))
        treatment_sample = rng.choices(treatment_values, k=len(treatment_values))
        differences.append(difference_in_means(baseline_sample, treatment_sample))
    return differences


def bootstrap_interval(
    baseline: Sequence[float | int],
    treatment: Sequence[float | int],
    *,
    resamples: int = 4_000,
    seed: int = 18,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    differences = bootstrap_differences(
        baseline, treatment, resamples=resamples, seed=seed
    )
    tail = (1 - confidence) / 2
    return percentile(differences, tail), percentile(differences, 1 - tail)


def simulate_sample_means(
    population_probability: float,
    sample_size: int,
    *,
    repetitions: int = 2_000,
    seed: int = 18,
) -> list[float]:
    """Draw repeated Bernoulli samples to make sampling variation visible."""

    if not 0 <= population_probability <= 1:
        raise ValueError("population_probability must be between zero and one")
    if sample_size < 1 or repetitions < 1:
        raise ValueError("sample_size and repetitions must be positive")
    rng = random.Random(seed)
    return [
        sum(rng.random() < population_probability for _ in range(sample_size))
        / sample_size
        for _ in range(repetitions)
    ]


def normal_two_sided_p_value(z_score: float) -> float:
    """Two-sided tail probability for a standard normal statistic."""

    return math.erfc(abs(z_score) / math.sqrt(2))


def z_test_difference(
    baseline: Sequence[float | int], treatment: Sequence[float | int]
) -> dict[str, float]:
    """Test equality of two binary rates with a pooled null standard error."""

    if len(baseline) < 2 or len(treatment) < 2:
        raise ValueError("each group requires at least two observations")
    if any(value not in {0, 1} for value in [*baseline, *treatment]):
        raise ValueError("z_test_difference expects binary outcomes")
    pooled = (sum(baseline) + sum(treatment)) / (len(baseline) + len(treatment))
    null_se = math.sqrt(
        pooled * (1 - pooled) * (1 / len(baseline) + 1 / len(treatment))
    )
    estimate = difference_in_means(baseline, treatment)
    z_score = estimate / null_se if null_se else 0.0
    return {
        "estimate": estimate,
        "null_standard_error": null_se,
        "z_score": z_score,
        "p_value": normal_two_sided_p_value(z_score),
    }


def permutation_test(
    baseline: Sequence[float | int],
    treatment: Sequence[float | int],
    *,
    permutations: int = 4_000,
    seed: int = 18,
) -> dict[str, float | int]:
    """Approximate a two-sided randomization test under exchangeability."""

    if not baseline or not treatment:
        raise ValueError("both groups require observations")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed = abs(difference_in_means(baseline, treatment))
    pooled = list(baseline) + list(treatment)
    baseline_size = len(baseline)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = pooled.copy()
        rng.shuffle(shuffled)
        permuted = abs(
            difference_in_means(
                shuffled[:baseline_size], shuffled[baseline_size:]
            )
        )
        if permuted >= observed - 1e-12:
            extreme += 1
    return {
        "observed_difference": observed,
        "permutations": permutations,
        "extreme_permutations": extreme,
        "p_value": (extreme + 1) / (permutations + 1),
    }


def effect_sizes(
    baseline: Sequence[float | int], treatment: Sequence[float | int]
) -> dict[str, float]:
    """Report absolute and relative binary-outcome effects side by side."""

    baseline_rate = statistics.fmean(baseline)
    treatment_rate = statistics.fmean(treatment)
    if baseline_rate <= 0:
        raise ValueError("relative effect requires a positive baseline rate")
    difference = treatment_rate - baseline_rate
    relative_risk = treatment_rate / baseline_rate
    return {
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "risk_difference": difference,
        "relative_risk": relative_risk,
        "relative_lift": relative_risk - 1,
        "number_needed_to_treat": 1 / difference if difference > 0 else math.inf,
    }


def cherry_pick_null_comparisons(
    comparisons: int = 20, *, seed: int = 1800, alpha: float = 0.05
) -> dict[str, object]:
    """Generate independent null comparisons and select only small p-values."""

    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    rng = random.Random(seed)
    results: list[dict[str, float | str]] = []
    for index in range(1, comparisons + 1):
        z_score = rng.gauss(0, 1)
        results.append(
            {
                "metric": f"metric_{index:02d}",
                "estimated_effect": z_score * 0.01,
                "z_score": z_score,
                "p_value": normal_two_sided_p_value(z_score),
            }
        )
    selected = [result for result in results if float(result["p_value"]) < alpha]
    return {
        "alpha": alpha,
        "comparisons": results,
        "selected": selected,
        "minimum_p_value": min(float(result["p_value"]) for result in results),
        "bonferroni_threshold": alpha / comparisons,
    }


def simulate_familywise_false_positive_rate(
    comparisons: int = 20,
    *,
    families: int = 2_000,
    seed: int = 1818,
    alpha: float = 0.05,
) -> float:
    """Estimate how often a null family contains at least one small p-value."""

    if comparisons < 1 or families < 1:
        raise ValueError("comparisons and families must be positive")
    rng = random.Random(seed)
    families_with_false_positive = 0
    for _ in range(families):
        if any(
            normal_two_sided_p_value(rng.gauss(0, 1)) < alpha
            for _ in range(comparisons)
        ):
            families_with_false_positive += 1
    return families_with_false_positive / families


def load_confounding_fixture(path: str | Path) -> list[dict[str, float | str]]:
    """Load the small seasonal correlation fixture."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"month", "temperature_c", "ice_cream_sales", "drownings"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            rows.append(
                {
                    "month": raw["month"],
                    "temperature_c": float(raw["temperature_c"]),
                    "ice_cream_sales": float(raw["ice_cream_sales"]),
                    "drownings": float(raw["drownings"]),
                }
            )
    if len(rows) < 3:
        raise ValueError("correlation fixture requires at least three rows")
    return rows


def pearson_correlation(
    left: Sequence[float | int], right: Sequence[float | int]
) -> float:
    """Compute Pearson correlation while refusing degenerate inputs."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal-length samples of size >= 2")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        raise ValueError("correlation is undefined for a constant input")
    return (
        sum(a * b for a, b in zip(centered_left, centered_right, strict=True))
        / denominator
    )


def ascii_histogram(
    values: Sequence[float | int], bins: int = 10, width: int = 30
) -> str:
    """Render a compact deterministic distribution without plotting packages."""

    if not values or bins < 1 or width < 1:
        raise ValueError("values, bins, and width must be positive/non-empty")
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return f"{minimum:>8.3f} | {'#' * width} {len(values)}"
    counts = [0] * bins
    span = maximum - minimum
    for value in values:
        index = min(int((value - minimum) / span * bins), bins - 1)
        counts[index] += 1
    largest = max(counts)
    lines = []
    for index, count in enumerate(counts):
        lower = minimum + span * index / bins
        upper = minimum + span * (index + 1) / bins
        bar = "#" * round(width * count / largest) if largest else ""
        lines.append(f"{lower:>7.3f}–{upper:<7.3f} | {bar} {count}")
    return "\n".join(lines)
