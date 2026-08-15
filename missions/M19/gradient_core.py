"""Small, deterministic gradient mechanics for Mission M19.

The implementation intentionally uses plain Python so the parameter-to-update
chain stays visible and the contract can be tested without an ML framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


Numbers = Sequence[float]


@dataclass(frozen=True)
class GradientStep:
    """Evidence captured around one update."""

    step: int
    parameters_before: tuple[float, ...]
    predictions: tuple[float, ...]
    loss_before: float
    gradient: tuple[float, ...]
    parameters_after: tuple[float, ...]
    loss_after: float


def _observations(xs: Numbers, ys: Numbers) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x_values = tuple(float(value) for value in xs)
    y_values = tuple(float(value) for value in ys)
    if not x_values:
        raise ValueError("at least one observation is required")
    if len(x_values) != len(y_values):
        raise ValueError("xs and ys must have equal length")
    if not all(math.isfinite(value) for value in x_values + y_values):
        raise ValueError("observations must be finite")
    return x_values, y_values


def predict_one_parameter(xs: Numbers, weight: float) -> tuple[float, ...]:
    """Predict y = weight * x."""

    x_values = tuple(float(value) for value in xs)
    return tuple(float(weight) * x for x in x_values)


def mean_squared_error(predictions: Numbers, targets: Numbers) -> float:
    predicted, actual = _observations(predictions, targets)
    return sum((prediction - target) ** 2 for prediction, target in zip(predicted, actual)) / len(actual)


def one_parameter_loss(xs: Numbers, ys: Numbers, weight: float) -> float:
    x_values, y_values = _observations(xs, ys)
    return mean_squared_error(predict_one_parameter(x_values, weight), y_values)


def finite_difference_gradient(
    xs: Numbers,
    ys: Numbers,
    weight: float,
    epsilon: float = 1.0e-5,
) -> float:
    """Estimate d(loss)/d(weight) with a central difference."""

    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    loss_plus = one_parameter_loss(xs, ys, weight + epsilon)
    loss_minus = one_parameter_loss(xs, ys, weight - epsilon)
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def analytic_weight_gradient(xs: Numbers, ys: Numbers, weight: float) -> float:
    """Return 2/n * sum(x * (weight*x - y))."""

    x_values, y_values = _observations(xs, ys)
    predictions = predict_one_parameter(x_values, weight)
    return 2.0 * sum(
        x * (prediction - target)
        for x, prediction, target in zip(x_values, predictions, y_values)
    ) / len(x_values)


def update_parameter(parameter: float, gradient: float, learning_rate: float) -> float:
    """Apply the descent convention: parameter - learning_rate * gradient."""

    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    return float(parameter) - learning_rate * float(gradient)


def one_parameter_step(
    xs: Numbers,
    ys: Numbers,
    weight: float,
    learning_rate: float,
    step: int = 0,
) -> GradientStep:
    """Trace parameter → prediction → loss → gradient → update once."""

    x_values, y_values = _observations(xs, ys)
    predictions = predict_one_parameter(x_values, weight)
    loss_before = mean_squared_error(predictions, y_values)
    gradient = analytic_weight_gradient(x_values, y_values, weight)
    weight_after = update_parameter(weight, gradient, learning_rate)
    loss_after = one_parameter_loss(x_values, y_values, weight_after)
    return GradientStep(
        step=step,
        parameters_before=(float(weight),),
        predictions=predictions,
        loss_before=loss_before,
        gradient=(gradient,),
        parameters_after=(weight_after,),
        loss_after=loss_after,
    )


def run_one_parameter_descent(
    xs: Numbers,
    ys: Numbers,
    initial_weight: float,
    learning_rate: float,
    steps: int,
) -> tuple[GradientStep, ...]:
    if steps < 1:
        raise ValueError("steps must be at least one")
    trace: list[GradientStep] = []
    weight = float(initial_weight)
    for step_index in range(steps):
        record = one_parameter_step(xs, ys, weight, learning_rate, step_index)
        trace.append(record)
        weight = record.parameters_after[0]
    return tuple(trace)


def predict_linear(xs: Numbers, weight: float, bias: float) -> tuple[float, ...]:
    """Predict y = weight * x + bias."""

    return tuple(float(weight) * float(x) + float(bias) for x in xs)


def linear_loss(xs: Numbers, ys: Numbers, weight: float, bias: float) -> float:
    x_values, y_values = _observations(xs, ys)
    return mean_squared_error(predict_linear(x_values, weight, bias), y_values)


def analytic_linear_gradients(
    xs: Numbers,
    ys: Numbers,
    weight: float,
    bias: float,
) -> tuple[float, float]:
    """Return partial derivatives with respect to weight and bias."""

    x_values, y_values = _observations(xs, ys)
    predictions = predict_linear(x_values, weight, bias)
    residuals = tuple(
        prediction - target for prediction, target in zip(predictions, y_values)
    )
    gradient_weight = 2.0 * sum(
        x * residual for x, residual in zip(x_values, residuals)
    ) / len(x_values)
    gradient_bias = 2.0 * sum(residuals) / len(x_values)
    return gradient_weight, gradient_bias


def linear_step(
    xs: Numbers,
    ys: Numbers,
    weight: float,
    bias: float,
    learning_rate: float,
    step: int = 0,
) -> GradientStep:
    """Apply one two-parameter gradient step and preserve its evidence."""

    x_values, y_values = _observations(xs, ys)
    predictions = predict_linear(x_values, weight, bias)
    loss_before = mean_squared_error(predictions, y_values)
    gradient_weight, gradient_bias = analytic_linear_gradients(
        x_values, y_values, weight, bias
    )
    weight_after = update_parameter(weight, gradient_weight, learning_rate)
    bias_after = update_parameter(bias, gradient_bias, learning_rate)
    loss_after = linear_loss(x_values, y_values, weight_after, bias_after)
    return GradientStep(
        step=step,
        parameters_before=(float(weight), float(bias)),
        predictions=predictions,
        loss_before=loss_before,
        gradient=(gradient_weight, gradient_bias),
        parameters_after=(weight_after, bias_after),
        loss_after=loss_after,
    )


def run_linear_descent(
    xs: Numbers,
    ys: Numbers,
    initial_weight: float,
    initial_bias: float,
    learning_rate: float,
    steps: int,
) -> tuple[GradientStep, ...]:
    if steps < 1:
        raise ValueError("steps must be at least one")
    trace: list[GradientStep] = []
    weight, bias = float(initial_weight), float(initial_bias)
    for step_index in range(steps):
        record = linear_step(
            xs, ys, weight, bias, learning_rate, step=step_index
        )
        trace.append(record)
        weight, bias = record.parameters_after
    return tuple(trace)
