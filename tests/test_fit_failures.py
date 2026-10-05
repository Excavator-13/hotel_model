"""未训练状态、拟合失败与最近成功权重保留测试（任务 2.3）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hotel_model import fitting as fitting_module
from hotel_model.fitting import FitResult, FittingConfig, LabeledPair, labeled_pairs, fit

from .conftest import make_samples, row
from .test_fitting import event, vectors


class _FakeResult:
    def __init__(self, *, success: bool, x: Any, message: str = "injected") -> None:
        self.success = success
        self.x = x
        self.message = message
        self.fun = 0.0


def _pairs() -> list[LabeledPair]:
    table = vectors(
        row("a", grade=5, nightly_price=200, room_area_sqm=40),
        row("b", grade=3, nightly_price=500, room_area_sqm=20),
    )
    return labeled_pairs([event("a", "b", 1.0)], table)


def test_successful_fit_reports_covered_label_count() -> None:
    pairs = _pairs()
    result = fit(pairs, FittingConfig())
    assert result.ok
    assert result.labels_covered == len(pairs) == 1


def test_optimizer_failure_returns_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_minimize(*args: Any, **kwargs: Any) -> _FakeResult:
        return _FakeResult(success=False, x=np.ones(8), message="迭代上限")

    monkeypatch.setattr(fitting_module, "minimize", fake_minimize)
    result = fit(_pairs(), FittingConfig())
    assert not result.ok
    assert result.weights is None
    assert result.training_cross_entropy is None
    assert "迭代上限" in result.message


@pytest.mark.parametrize(
    "bad",
    [
        np.full(8, np.nan),
        np.full(8, np.inf),
        np.full(8, -1.0),
        np.full(5, 1.0),
    ],
)
def test_non_finite_or_invalid_optimizer_output_rejected(
    monkeypatch: pytest.MonkeyPatch, bad: np.ndarray
) -> None:
    def fake_minimize(*args: Any, **kwargs: Any) -> _FakeResult:
        return _FakeResult(success=True, x=bad)

    monkeypatch.setattr(fitting_module, "minimize", fake_minimize)
    result = fit(_pairs(), FittingConfig())
    assert not result.ok
    assert result.weights is None


def test_optimizer_exception_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_minimize(*args: Any, **kwargs: Any) -> _FakeResult:
        raise RuntimeError("数值爆炸")

    monkeypatch.setattr(fitting_module, "minimize", fake_minimize)
    result = fit(_pairs(), FittingConfig())
    assert not result.ok
    assert "RuntimeError" in result.message


def test_failed_fit_does_not_overwrite_previous_weights() -> None:
    """失败结果不能覆盖最近成功的权重。"""

    from hotel_model.state import ModelSnapshot, model_after_fit

    good = fit(_pairs(), FittingConfig())
    assert good.ok and good.weights is not None
    snapshot = model_after_fit(ModelSnapshot.untrained(), good, label_count=1)
    assert snapshot.trained
    assert snapshot.labels_covered == 1
    assert snapshot.lagging_labels == 0
    saved = snapshot.weights.copy()

    failed = FitResult(False, None, None, 0, "注入的失败")
    after = model_after_fit(snapshot, failed, label_count=3)
    assert after.trained
    assert np.array_equal(after.weights, saved)
    assert after.labels_covered == 1
    assert after.lagging_labels == 2
    assert after.training_cross_entropy == snapshot.training_cross_entropy


def test_model_after_successful_refit_updates_coverage() -> None:
    from hotel_model.state import ModelSnapshot, model_after_fit

    snapshot = ModelSnapshot.untrained()
    first = fit(_pairs(), FittingConfig())
    snapshot = model_after_fit(snapshot, first, label_count=1)

    table = vectors(
        row("a", grade=5, nightly_price=200, room_area_sqm=40),
        row("b", grade=3, nightly_price=500, room_area_sqm=20),
        row("c", grade=4, nightly_price=300, room_area_sqm=30),
    )
    more = labeled_pairs(
        [event("a", "b", 1.0), event("a", "c", 0.5), event("c", "b", 1.0)],
        table,
    )
    second = fit(more, FittingConfig(), initial=snapshot.weights)
    assert second.ok
    snapshot = model_after_fit(snapshot, second, label_count=3)
    assert snapshot.labels_covered == 3
    assert snapshot.lagging_labels == 0


def test_untrained_snapshot_has_no_weights_or_loss() -> None:
    from hotel_model.state import ModelSnapshot

    snapshot = ModelSnapshot.untrained()
    assert not snapshot.trained
    assert snapshot.weights is None
    assert snapshot.training_cross_entropy is None
    assert snapshot.labels_covered == 0
    assert snapshot.lagging_labels == 0


def test_initial_weights_are_used_as_starting_point(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_minimize(objective: Any, start: Any, **kwargs: Any) -> _FakeResult:
        captured["start"] = np.array(start, dtype=float)
        return _FakeResult(success=True, x=np.array(start, dtype=float))

    monkeypatch.setattr(fitting_module, "minimize", fake_minimize)
    start = np.full(8, 0.25)
    result = fit(_pairs(), FittingConfig(), initial=start)
    assert result.ok
    assert np.allclose(captured["start"], start)
    assert np.allclose(result.weights, start)


def test_invalid_initial_weights_fall_back_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_minimize(objective: Any, start: Any, **kwargs: Any) -> _FakeResult:
        captured["start"] = np.array(start, dtype=float)
        return _FakeResult(success=True, x=np.zeros(8))

    monkeypatch.setattr(fitting_module, "minimize", fake_minimize)
    fit(_pairs(), FittingConfig(), initial=np.array([np.nan] * 8))
    assert np.allclose(captured["start"], np.zeros(8))
    fit(_pairs(), FittingConfig(), initial=np.ones(3))
    assert np.allclose(captured["start"], np.zeros(8))
