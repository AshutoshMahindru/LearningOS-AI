from __future__ import annotations

import csv
import math
from pathlib import Path
import random


SEED = 902
ROW_COUNT = 180
FIELDNAMES = [
    "learner_id",
    "account_age_days",
    "weekly_sessions",
    "overdue_tasks",
    "assessment_score",
    "help_requests",
    "disengaged_next_30_days",
]


def generate_rows() -> list[dict[str, int | str]]:
    """Return the deterministic synthetic M09 dataset."""
    rng = random.Random(SEED)
    rows: list[dict[str, int | str]] = []

    for index in range(1, ROW_COUNT + 1):
        account_age_days = rng.randint(14, 720)
        weekly_sessions = rng.randint(0, 12)
        overdue_tasks = rng.randint(0, 8)
        assessment_score = rng.randint(40, 98)
        help_requests = rng.randint(0, 7)
        noise = rng.uniform(-1.35, 1.35)

        log_odds = (
            -4.2
            + 0.55 * overdue_tasks
            + 0.38 * help_requests
            - 0.20 * weekly_sessions
            - 0.045 * (assessment_score - 70)
            - 0.0015 * (account_age_days - 180)
            + noise
        )
        event_probability = 1.0 / (1.0 + math.exp(-log_odds))
        target = int(rng.random() < event_probability)

        rows.append(
            {
                "learner_id": f"L{index:03d}",
                "account_age_days": account_age_days,
                "weekly_sessions": weekly_sessions,
                "overdue_tasks": overdue_tasks,
                "assessment_score": assessment_score,
                "help_requests": help_requests,
                "disengaged_next_30_days": target,
            }
        )

    return rows


def write_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(generate_rows())


if __name__ == "__main__":
    output_path = Path(__file__).with_name("learner_disengagement.csv")
    write_dataset(output_path)
    print(f"wrote {ROW_COUNT} synthetic rows to {output_path}")
