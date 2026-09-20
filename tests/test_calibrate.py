import json

import numpy as np
import pytest

from system1 import SystemOne
from system1.backend import FakeBackend
from system1.calibrate import (
    apply_temperature,
    brier,
    ece,
    fit_temperature,
    load_calibration,
    log_loss,
    save_calibration,
)
from system1.schema import Request


@pytest.mark.parametrize("as_probs", [True, False])
def test_temperature_recovers_synthetic_scaling(as_probs):
    # Each block has exact empirical class frequencies, removing sampling noise.
    frequencies = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    logits = np.repeat(np.log(frequencies) * 2.5, 100, axis=0)
    labels = [
        label
        for row in frequencies
        for label, count in enumerate(np.rint(row * 100).astype(int))
        for _ in range(count)
    ]
    values = (
        np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        if as_probs
        else logits
    )
    assert fit_temperature(values, labels) == pytest.approx(2.5, abs=0.01)


def test_perfect_predictor_and_known_metric_values():
    assert ece([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == 0
    assert brier([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == 0
    assert log_loss([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == 0
    assert ece([[0.8, 0.2], [0.4, 0.6]], [0, 0]) == pytest.approx(0.4)
    assert brier([[0.8, 0.2], [0.4, 0.6]], [0, 0]) == pytest.approx(0.4)
    assert log_loss([[0.8, 0.2], [0.4, 0.6]], [0, 0]) == pytest.approx(
        -np.log(0.32) / 2
    )


def test_variable_option_counts_and_zero_mass():
    probs = [[0.8, 0.2], [0.1, 0.2, 0.7]]
    assert 0.2 <= fit_temperature(probs, [0, 2]) <= 10
    assert brier(probs, [0, 2]) == pytest.approx(0.11)
    assert apply_temperature([1.0, 0.0], 2).tolist() == [1.0, 0.0]


def test_save_load_and_decide_applies_temperature(tmp_path):
    path = tmp_path / "calibration.json"
    assert load_calibration(path) == {}
    save_calibration({"score": 2.0}, path)
    assert load_calibration(path) == {"score": 2.0}
    response = [
        {"token": "A", "logprob": np.log(0.8)},
        {"token": "B", "logprob": np.log(0.2)},
    ]
    engine = SystemOne(FakeBackend([response, response]), path)
    request = Request(
        state="",
        questions={
            "q": {"type": "score", "instructions": "", "levels": ["low", "high"]}
        },
        permutations=1,
    )
    calibrated = engine.decide(request)["answers"]["q"]["probabilities"]
    raw = engine.decide(request, calibrated=False)["answers"]["q"]["probabilities"]
    assert calibrated == pytest.approx({"low": 2 / 3, "high": 1 / 3})
    assert raw == pytest.approx({"low": 0.8, "high": 0.2})


@pytest.mark.parametrize(
    "values",
    [
        {"score": 0},
        {"noul": -1},
        {"choice": float("nan")},
        {"other": 1},
        {"score": True},
        [],
    ],
)
def test_invalid_calibration_is_not_ignored(values, tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_calibration(path)


@pytest.mark.parametrize(
    "probs,labels",
    [
        ([], []),
        ([[0.2, 0.2]], [0]),
        ([[0.7, 0.3]], [2]),
        ([[0.7, 0.3]], []),
        ([[0.7, 0.3]], [0.5]),
    ],
)
def test_metrics_reject_invalid_inputs(probs, labels):
    for metric in (ece, brier, log_loss):
        with pytest.raises(ValueError):
            metric(probs, labels)


def test_soft_metrics_known_values_and_rounding():
    from system1.calibrate import soft_brier, soft_log_loss

    assert soft_brier([[0.8, 0.2]], [[0.6, 0.4]]) == pytest.approx(0.04)
    assert soft_log_loss([[0.8, 0.2]], [[0.6, 0.4]]) == pytest.approx(
        -0.6 * np.log(0.8) - 0.4 * np.log(0.2)
    )
    assert soft_brier([[0.6, 0.4]], [[0.6, 0.4]]) == pytest.approx(0)
    assert soft_brier([[0.5, 0.5]], [[0.499999, 0.499999]]) == pytest.approx(0)
    assert np.isfinite(soft_log_loss([[1, 0]], [[0.5, 0.5]]))
    with pytest.raises(ValueError, match="Gold probabilities"):
        soft_brier([[0.5, 0.5]], [[0.2, 0.2]])
