"""Temperature scaling and multiclass calibration metrics."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

Rows = Sequence[Sequence[float]] | np.ndarray


def probability_rows(probs: Rows) -> list[np.ndarray]:
    rows = [np.asarray(row, dtype=float) for row in probs]
    if not rows:
        raise ValueError("At least one probability row is required")
    for row in rows:
        if (
            row.ndim != 1
            or not row.size
            or not np.isfinite(row).all()
            or (row < 0).any()
            or not np.isclose(row.sum(), 1)
        ):
            raise ValueError(
                "Each probability row must be finite, nonnegative, and sum to 1"
            )
    return rows


def checked_labels(rows: list[np.ndarray], labels: Sequence[int]) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (len(rows),) or values.dtype.kind not in "iu":
        raise ValueError("Supply one integer label index per row")
    if any(label < 0 or label >= len(row) for row, label in zip(rows, values)):
        raise ValueError("Label index is outside its probability row")
    return values


def apply_temperature(probs: Sequence[float], temperature: float) -> np.ndarray:
    """Scale probabilities with softmax(log(p) / T)."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    row = probability_rows([probs])[0]
    with np.errstate(divide="ignore"):
        logits = np.log(row) / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def fit_temperature(logits_or_probs: Rows, labels: Sequence[int]) -> float:
    """Fit one temperature, including rows with different option counts."""
    rows = [np.asarray(row, dtype=float) for row in logits_or_probs]
    if not rows or any(
        row.ndim != 1 or not row.size or not np.isfinite(row).all() for row in rows
    ):
        raise ValueError("Expected nonempty finite rows")
    targets = checked_labels(rows, labels)
    are_probs = all((row >= 0).all() and np.isclose(row.sum(), 1) for row in rows)
    logits = [np.log(np.clip(row, 1e-300, 1)) if are_probs else row for row in rows]
    # Padding permits one vectorized fit when questions have different option counts.
    matrix = np.full((len(rows), max(map(len, rows))), -np.inf)
    for index, row in enumerate(logits):
        matrix[index, : len(row)] = row - row.max()

    def loss(temperature: float) -> float:
        scaled = matrix / temperature
        return float(
            np.mean(
                np.log(np.exp(scaled).sum(axis=1))
                - scaled[np.arange(len(rows)), targets]
            )
        )

    grid = np.geomspace(0.2, 10, 100)
    for _ in range(3):
        best = int(np.argmin([loss(t) for t in grid]))
        temperature = float(grid[best])
        grid = np.geomspace(
            grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)], 100
        )
    return temperature


def ece(probs: Rows, labels: Sequence[int], bins: int = 10) -> float:
    """Return top-label expected calibration error with equal-width bins."""
    rows = probability_rows(probs)
    targets = checked_labels(rows, labels)
    if bins < 1:
        raise ValueError("bins must be positive")
    confidence = np.array([row.max() for row in rows])
    correct = np.array([row.argmax() == label for row, label in zip(rows, targets)])
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(bucket == b)
            * abs(correct[bucket == b].mean() - confidence[bucket == b].mean())
            for b in range(bins)
            if (bucket == b).any()
        )
    )


def brier(probs: Rows, labels: Sequence[int]) -> float:
    """Return the mean sum of squared multiclass errors."""
    rows = probability_rows(probs)
    targets = checked_labels(rows, labels)
    return float(
        np.mean(
            [
                np.square(row).sum() - 2 * row[label] + 1
                for row, label in zip(rows, targets)
            ]
        )
    )


def log_loss(probs: Rows, labels: Sequence[int]) -> float:
    rows = probability_rows(probs)
    targets = checked_labels(rows, labels)
    return float(
        np.mean([-np.log(max(row[label], 1e-300)) for row, label in zip(rows, targets)])
    )


def validate_temperatures(values: dict[str, float]) -> dict[str, float]:
    if not isinstance(values, dict) or any(
        key not in {"choice", "score", "noul"}
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for key, value in values.items()
    ):
        raise ValueError("Calibration must map question types to positive temperatures")
    return values


def load_calibration(path: Path = Path("calibration.json")) -> dict[str, float]:
    return validate_temperatures(json.loads(path.read_text())) if path.exists() else {}


def save_calibration(
    values: dict[str, float], path: Path = Path("calibration.json")
) -> None:
    path.write_text(json.dumps(validate_temperatures(values), indent=2) + "\n")
