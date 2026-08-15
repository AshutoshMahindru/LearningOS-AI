"""Regenerate the deterministic M12 classification fixture."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons


SEED = 12012
N_SAMPLES = 480
OUTPUT = Path(__file__).with_name("ensemble_fixture.csv")


def build_rows() -> list[dict[str, float | int]]:
    """Return fixture rows in a stable order with rounded numeric values."""
    features, target = make_moons(
        n_samples=N_SAMPLES,
        noise=0.31,
        random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []
    for (x1, x2), label in zip(features, target, strict=True):
        rows.append(
            {
                "x1": round(float(x1), 8),
                "x2": round(float(x2), 8),
                "linear_mix": round(float(0.65 * x1 + 0.35 * x2 + rng.normal(0, 0.18)), 8),
                "periodic": round(float(np.sin(2.4 * x1) + rng.normal(0, 0.15)), 8),
                "noise1": round(float(rng.normal()), 8),
                "noise2": round(float(rng.normal()), 8),
                "target": int(label),
            }
        )
    return rows


def main() -> None:
    rows = build_rows()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
