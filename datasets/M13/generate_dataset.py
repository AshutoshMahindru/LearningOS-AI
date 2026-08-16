"""Generate the deterministic, synthetic M13 KNN teaching dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


DEFAULT_OUTPUT = Path(__file__).with_name("knn_scale_cases.csv")


def build_dataset() -> pd.DataFrame:
    """Return balanced local geometry plus one independent high-scale feature."""
    coordinates, labels = make_moons(
        n_samples=96,
        noise=0.20,
        random_state=13,
    )
    weak_rng = np.random.default_rng(1301)

    return pd.DataFrame(
        {
            "case_id": [f"M13-{index:03d}" for index in range(1, 97)],
            "practice_hours": np.round(5.0 + 2.4 * coordinates[:, 0], 3),
            "assessment_score": np.round(60.0 + 18.0 * coordinates[:, 1], 3),
            "interface_event_count": weak_rng.integers(1000, 10000, size=96),
            "learning_route": np.where(labels == 1, "independent", "guided"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, lineterminator="\n")
    print(f"wrote {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
