"""Deterministic optimization instrumentation for Mission M20.

The objective is deliberately small and transparent. Optimizers expose the gradient,
state-derived direction, applied update, and loss around every step so learners can
attribute observed dynamics without an ML framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


Vector = Sequence[float]
DEFAULT_CURVATURES = (1.0, 10.0)
DEFAULT_INITIALIZATION = (4.0, 4.0)
COMPONENT_GRADIENT_NOISE = (
    (-2.0, 1.0),
    (2.0, -1.0),
    (-1.0, -2.0),
    (1.0, 2.0),
)


@dataclass(frozen=True)
class OptimizationStep:
    """Inspectable evidence captured around one optimizer update."""

    step: int
    optimizer: str
    gradient_source: str
    parameters_before: tuple[float, ...]
    loss_before: float
    gradient: tuple[float, ...]
    optimizer_direction: tuple[float, ...]
    applied_update: tuple[float, ...]
    parameters_after: tuple[float, ...]
    loss_after: float


def _finite_vector(values: Vector, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _objective_inputs(
    parameters: Vector,
    curvatures: Vector,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    point = _finite_vector(parameters, "parameters")
    curvature = _finite_vector(curvatures, "curvatures")
    if len(point) != len(curvature):
        raise ValueError("parameters and curvatures must have equal length")
    if any(value <= 0 for value in curvature):
        raise ValueError("curvatures must be positive")
    return point, curvature


def quadratic_loss(
    parameters: Vector,
    curvatures: Vector = DEFAULT_CURVATURES,
) -> float:
    """Return 0.5 * sum(curvature_i * parameter_i**2)."""

    point, curvature = _objective_inputs(parameters, curvatures)
    return 0.5 * sum(scale * value * value for value, scale in zip(point, curvature))


def quadratic_gradient(
    parameters: Vector,
    curvatures: Vector = DEFAULT_CURVATURES,
) -> tuple[float, ...]:
    """Return the exact gradient of :func:`quadratic_loss`."""

    point, curvature = _objective_inputs(parameters, curvatures)
    return tuple(scale * value for value, scale in zip(point, curvature))


def component_noise_mean(
    noises: Sequence[Vector] = COMPONENT_GRADIENT_NOISE,
) -> tuple[float, ...]:
    """Return the coordinate mean of the committed SGD noise fixture."""

    rows = tuple(_finite_vector(row, "noise row") for row in noises)
    if not rows:
        raise ValueError("at least one noise row is required")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("noise rows must have equal length")
    return tuple(sum(row[index] for row in rows) / len(rows) for index in range(width))


def run_optimizer(
    optimizer: str,
    *,
    learning_rate: float,
    steps: int,
    initial_parameters: Vector = DEFAULT_INITIALIZATION,
    curvatures: Vector = DEFAULT_CURVATURES,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
    seed: int = 2020,
    stochastic_noise: Sequence[Vector] = COMPONENT_GRADIENT_NOISE,
) -> tuple[OptimizationStep, ...]:
    """Run GD, seeded SGD, classical momentum, or bias-corrected Adam.

    SGD samples without replacement within each shuffled fixture epoch. The noise
    rows average to zero, so their mean gradient equals the exact objective gradient.
    """

    if optimizer not in {"gd", "sgd", "momentum", "adam"}:
        raise ValueError("optimizer must be one of: gd, sgd, momentum, adam")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if steps < 1:
        raise ValueError("steps must be at least one")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    parameters, curvature = _objective_inputs(initial_parameters, curvatures)
    dimension = len(parameters)
    noise_rows = tuple(_finite_vector(row, "noise row") for row in stochastic_noise)
    if optimizer == "sgd":
        if not noise_rows or any(len(row) != dimension for row in noise_rows):
            raise ValueError("SGD noise must include dimension-matched rows")
        if any(abs(value) > 1.0e-12 for value in component_noise_mean(noise_rows)):
            raise ValueError("SGD component noise must have coordinate-wise zero mean")

    velocity = [0.0] * dimension
    first_moment = [0.0] * dimension
    second_moment = [0.0] * dimension
    randomizer = random.Random(seed)
    remaining_components: list[int] = []
    trace: list[OptimizationStep] = []

    for step_index in range(steps):
        loss_before = quadratic_loss(parameters, curvature)
        exact_gradient = quadratic_gradient(parameters, curvature)
        gradient_source = "exact"
        gradient = exact_gradient

        if optimizer == "sgd":
            if not remaining_components:
                remaining_components = list(range(len(noise_rows)))
                randomizer.shuffle(remaining_components)
            component_index = remaining_components.pop()
            gradient = tuple(
                value + noise
                for value, noise in zip(exact_gradient, noise_rows[component_index])
            )
            gradient_source = f"component:{component_index}"

        if optimizer in {"gd", "sgd"}:
            direction = gradient
        elif optimizer == "momentum":
            velocity = [
                momentum * previous + value
                for previous, value in zip(velocity, gradient)
            ]
            direction = tuple(velocity)
        else:
            first_moment = [
                beta1 * previous + (1.0 - beta1) * value
                for previous, value in zip(first_moment, gradient)
            ]
            second_moment = [
                beta2 * previous + (1.0 - beta2) * value * value
                for previous, value in zip(second_moment, gradient)
            ]
            time = step_index + 1
            corrected_first = [value / (1.0 - beta1**time) for value in first_moment]
            corrected_second = [value / (1.0 - beta2**time) for value in second_moment]
            direction = tuple(
                first / (math.sqrt(second) + epsilon)
                for first, second in zip(corrected_first, corrected_second)
            )

        applied_update = tuple(learning_rate * value for value in direction)
        parameters_after = tuple(
            value - update for value, update in zip(parameters, applied_update)
        )
        loss_after = quadratic_loss(parameters_after, curvature)
        if not math.isfinite(loss_after):
            raise OverflowError("optimizer trace became non-finite")

        trace.append(
            OptimizationStep(
                step=step_index,
                optimizer=optimizer,
                gradient_source=gradient_source,
                parameters_before=parameters,
                loss_before=loss_before,
                gradient=gradient,
                optimizer_direction=tuple(direction),
                applied_update=applied_update,
                parameters_after=parameters_after,
                loss_after=loss_after,
            )
        )
        parameters = parameters_after

    return tuple(trace)


def run_learning_rate_sweep(
    learning_rates: Sequence[float],
    *,
    steps: int,
    initial_parameters: Vector = DEFAULT_INITIALIZATION,
    curvatures: Vector = DEFAULT_CURVATURES,
) -> dict[float, tuple[OptimizationStep, ...]]:
    """Run controlled exact-gradient traces keyed by learning rate."""

    rates = tuple(float(value) for value in learning_rates)
    if not rates:
        raise ValueError("at least one learning rate is required")
    if len(set(rates)) != len(rates):
        raise ValueError("learning rates must be unique")
    return {
        rate: run_optimizer(
            "gd",
            learning_rate=rate,
            steps=steps,
            initial_parameters=initial_parameters,
            curvatures=curvatures,
        )
        for rate in rates
    }


def loss_history(trace: Sequence[OptimizationStep]) -> tuple[float, ...]:
    """Return initial loss followed by every post-update loss."""

    records = tuple(trace)
    if not records:
        raise ValueError("trace must not be empty")
    return (records[0].loss_before,) + tuple(record.loss_after for record in records)


def coordinate_sign_changes(
    trace: Sequence[OptimizationStep],
    coordinate: int,
) -> int:
    """Count non-zero sign changes in one parameter coordinate."""

    records = tuple(trace)
    if not records:
        raise ValueError("trace must not be empty")
    if coordinate < 0 or coordinate >= len(records[0].parameters_before):
        raise IndexError("coordinate is outside the parameter vector")
    values = (records[0].parameters_before[coordinate],) + tuple(
        record.parameters_after[coordinate] for record in records
    )
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    return sum(before != after for before, after in zip(signs, signs[1:]))


def diagnose_dynamics(
    trace: Sequence[OptimizationStep],
    *,
    coordinate: int = 1,
    stagnation_fraction: float = 0.01,
) -> str:
    """Classify a bounded trace from observable loss and sign evidence."""

    if not 0 < stagnation_fraction < 1:
        raise ValueError("stagnation_fraction must be between zero and one")
    losses = loss_history(trace)
    relative_improvement = (losses[0] - losses[-1]) / losses[0]
    sign_changes = coordinate_sign_changes(trace, coordinate)
    if losses[-1] > losses[0]:
        return "diverging"
    if relative_improvement < stagnation_fraction:
        return "stagnating"
    if sign_changes >= 2:
        return "oscillatory_convergence"
    return "converging"

