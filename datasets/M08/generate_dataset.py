"""Generate the deterministic, synthetic M08 regression fixture."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import random


SEED = 8042
ROW_COUNT = 320
OUTPUT = Path(__file__).with_name("housing_regression.csv")
FIELDNAMES = [
    "transaction_id",
    "floor_area_m2",
    "bedrooms",
    "building_age_years",
    "distance_to_transit_km",
    "neighborhood_score",
    "renovation_quality",
    "energy_efficiency",
    "local_job_access_score",
    "post_sale_assessment_k",
    "sale_price_k",
]


def build_rows() -> list[dict[str, str | int]]:
    rng = random.Random(SEED)
    rows: list[dict[str, str | int]] = []

    for index in range(1, ROW_COUNT + 1):
        floor_area = rng.uniform(42.0, 225.0)
        bedrooms = max(1, min(6, round(floor_area / 42.0 + rng.gauss(0.0, 0.55))))
        age = rng.randint(0, 70)
        transit = rng.uniform(0.15, 9.0)
        neighborhood = rng.uniform(35.0, 96.0)
        renovation = rng.randint(1, 5)
        energy = min(99.0, max(38.0, 91.0 - 0.55 * age + rng.gauss(0.0, 9.0)))
        job_access = min(100.0, max(25.0, 103.0 - 6.8 * transit + rng.gauss(0.0, 8.0)))

        transit_bonus = 62.0 if transit < 1.0 else (25.0 if transit < 2.4 else 0.0)
        age_penalty = 1.0 * age + 0.018 * age**2
        area_neighborhood_interaction = 0.013 * floor_area * (neighborhood - 35.0)
        smooth_location_effect = 18.0 * math.sin(job_access / 16.0)
        noise = rng.gauss(0.0, 42.0)

        sale_price = (
            45.0
            + 1.65 * floor_area
            + 11.0 * bedrooms
            - age_penalty
            - 10.5 * transit
            + 2.15 * neighborhood
            + 13.5 * renovation
            + 0.85 * energy
            + 0.72 * job_access
            + area_neighborhood_interaction
            + transit_bonus
            + smooth_location_effect
            + noise
        )
        sale_price = max(95.0, sale_price)
        post_sale_assessment = sale_price * rng.uniform(0.992, 1.008) + rng.gauss(0.0, 2.0)

        rows.append(
            {
                "transaction_id": f"TX-{index:04d}",
                "floor_area_m2": f"{floor_area:.2f}",
                "bedrooms": bedrooms,
                "building_age_years": age,
                "distance_to_transit_km": f"{transit:.3f}",
                "neighborhood_score": f"{neighborhood:.2f}",
                "renovation_quality": renovation,
                "energy_efficiency": f"{energy:.2f}",
                "local_job_access_score": f"{job_access:.2f}",
                "post_sale_assessment_k": f"{post_sale_assessment:.2f}",
                "sale_price_k": f"{sale_price:.2f}",
            }
        )

    return rows


def main() -> None:
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"Wrote {ROW_COUNT} deterministic rows to {OUTPUT}")


if __name__ == "__main__":
    main()
