"""偏好标签语义与带约束拟合测试（任务 2.1–2.3）。"""

from __future__ import annotations

import numpy as np
import pytest

from hotel_model.fitting import (
    DEFAULT_L2,
    LEFT_WINS,
    RIGHT_WINS,
    TIE,
    FitResult,
    FittingConfig,
    LabeledPair,
    fit,
    labeled_pairs,
    loss_gradient,
    mean_cross_entropy,
    other_label,
    penalized_loss,
)

from .conftest import make_samples, row


def vectors(*rows: dict[str, object]) -> dict[str, np.ndarray]:
    samples = make_samples(list(rows))
    from hotel_model.features import to_features

    return {sample.hotel_id: to_features(sample) for sample in samples}


def event(left: str, right: str, label: float | None, kind: str = "label") -> dict[str, object]:
    return {"left_id": left, "right_id": right, "label": label, "kind": kind}


# --- 2.1 标签与差向量 -------------------------------------------------------


def test_label_constants() -> None:
    assert LEFT_WINS == 1.0
    assert RIGHT_WINS == 0.0
    assert TIE == 0.5


def test_other_label_inverts_strict_and_keeps_tie() -> None:
    assert other_label(LEFT_WINS) == RIGHT_WINS
    assert other_label(RIGHT_WINS) == LEFT_WINS
    assert other_label(TIE) == TIE


def test_difference_vector_is_left_minus_right() -> None:
    table = vectors(
        row("a", grade=5, nightly_price=200),
        row("b", grade=3, nightly_price=500),
    )
    pairs = labeled_pairs([event("a", "b", LEFT_WINS)], table)
    assert len(pairs) == 1
    expected = table["a"] - table["b"]
    assert np.allclose(pairs[0].difference, expected)


def test_swapping_display_and_flipping_label_keeps_target() -> None:
    """交换 A/B 并反转严格标签不改变拟合目标。"""

    table = vectors(
        row("a", grade=5, nightly_price=200, room_area_sqm=40),
        row("b", grade=3, nightly_price=500, room_area_sqm=20),
    )
    forward = labeled_pairs([event("a", "b", LEFT_WINS)], table)
    backward = labeled_pairs([event("b", "a", RIGHT_WINS)], table)

    # 标签按展示方向解释，因此交换后差向量取反、标签取反。
    assert np.allclose(forward[0].difference, -backward[0].difference)
    assert backward[0].target == other_label(forward[0].target)

    weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.03])
    loss_a = mean_cross_entropy(weights, np.vstack([p.difference for p in forward]), np.array([LEFT_WINS]))
    loss_b = mean_cross_entropy(weights, np.vstack([p.difference for p in backward]), np.array([RIGHT_WINS]))
    assert loss_a == pytest.approx(loss_b)


def test_flipped_pair_helper_is_equivalent() -> None:
    table = vectors(row("a", grade=5), row("b", grade=3))
    pair = labeled_pairs([event("a", "b", LEFT_WINS)], table)[0]
    assert pair.flipped().target == 0.0
    assert np.allclose(pair.flipped().difference, -pair.difference)


def test_tie_uses_half_label() -> None:
    table = vectors(row("a", grade=4), row("b", grade=4), )
    pairs = labeled_pairs([event("a", "b", TIE)], table)
    assert pairs[0].target == 0.5


def test_skip_does_not_change_training_sample_count() -> None:
    table = vectors(row("a", grade=5), row("b", grade=3), row("c", grade=4))
    events = [event("a", "b", LEFT_WINS), event("b", "c", None, kind="skip")]
    pairs = labeled_pairs(events, table)
    assert len(pairs) == 1
    pairs_all_labels = labeled_pairs([event("a", "b", LEFT_WINS)], table)
    assert len(pairs) == len(pairs_all_labels)


def test_skip_is_not_treated_as_tie() -> None:
    table = vectors(row("a", grade=5), row("b", grade=3))
    pairs = labeled_pairs([event("a", "b", None, kind="skip")], table)
    assert pairs == []


@pytest.mark.parametrize("label", [0.25, -1.0, 2.0])
def test_invalid_label_rejected(label: float) -> None:
    table = vectors(row("a", grade=5), row("b", grade=3))
    with pytest.raises(ValueError, match="标签非法"):
        labeled_pairs([event("a", "b", label)], table)


def test_unknown_sample_reference_rejected() -> None:
    table = vectors(row("a", grade=5), row("b", grade=3))
    with pytest.raises(ValueError, match="未知样本"):
        labeled_pairs([event("a", "zz", LEFT_WINS)], table)


# --- 2.2 目标函数、梯度与拟合 ----------------------------------------------


def dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    differences = rng.normal(scale=0.35, size=(6, 8))
    targets = np.array([1.0, 0.0, 0.5, 1.0, 0.0, 1.0])
    return differences, targets


def test_l2_must_be_finite_positive() -> None:
    assert FittingConfig().l2 == DEFAULT_L2
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="有限正数"):
            FittingConfig(l2=value)


def test_fitting_config_round_trip() -> None:
    config = FittingConfig(l2=0.25)
    assert FittingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        FittingConfig.from_dict({"version": "v9", "l2": 0.1})
    with pytest.raises(ValueError):
        FittingConfig.from_dict({"version": "v1", "l2": "strong"})


def test_analytic_gradient_matches_finite_difference() -> None:
    differences, targets = dataset()
    weights = np.array([0.3, 0.2, 0.1, 0.05, 0.0, 0.15, 0.25, 0.35])
    analytic = loss_gradient(weights, differences, targets, 0.1)

    numeric = np.zeros_like(weights)
    step = 1e-6
    for index in range(weights.size):
        up = weights.copy()
        down = weights.copy()
        up[index] += step
        down[index] -= step
        numeric[index] = (
            penalized_loss(up, differences, targets, 0.1)
            - penalized_loss(down, differences, targets, 0.1)
        ) / (2 * step)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_cross_entropy_is_stable_for_extreme_scores() -> None:
    weights = np.full(8, 600.0)
    differences = np.ones((1, 8))
    targets = np.array([1.0])
    assert np.isfinite(mean_cross_entropy(weights, differences, targets))
    assert mean_cross_entropy(weights, differences, targets) == pytest.approx(0.0, abs=1e-9)


def test_gradient_without_data_is_only_regularization() -> None:
    weights = np.full(8, 0.5)
    gradient = loss_gradient(weights, np.zeros((0, 8)), np.zeros((0,)), 0.2)
    assert np.allclose(gradient, 0.2 * weights)


def test_single_strict_label_fits_finite_non_negative_weights() -> None:
    table = vectors(
        row("a", grade=5, renovation_age_years=1, ctrip_rating=4.8,
            subway_walk_minutes=2, nightly_price=250, room_area_sqm=40),
        row("b", grade=3, renovation_age_years=12, ctrip_rating=4.5,
            subway_walk_minutes=15, nightly_price=500, room_area_sqm=20),
    )
    pairs = labeled_pairs([event("a", "b", LEFT_WINS)], table)
    result = fit(pairs, FittingConfig())
    assert result.ok
    assert result.weights is not None
    assert result.weights.shape == (8,)
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)
    assert result.weights.sum() > 0.0
    assert result.training_cross_entropy is not None
    assert 0.0 <= result.training_cross_entropy < 0.7


def test_all_ties_give_zero_weights() -> None:
    table = vectors(row("a", grade=4), row("b", grade=3))
    pairs = labeled_pairs([event("a", "b", TIE), event("a", "b", TIE)], table)
    result = fit(pairs, FittingConfig())
    assert result.ok
    assert result.weights is not None
    assert np.allclose(result.weights, 0.0, atol=1e-6)
    assert result.training_cross_entropy == pytest.approx(np.log(2.0))


def test_fully_separable_data_stays_finite() -> None:
    differences = np.zeros((4, 8))
    differences[:, 0] = [5.0, 5.0, 5.0, 5.0]
    targets = np.array([1.0, 1.0, 1.0, 1.0])
    pairs = [
        LabeledPair("a", "b", differences[i], float(targets[i])) for i in range(4)
    ]
    result = fit(pairs, FittingConfig(l2=0.1))
    assert result.ok
    assert result.weights is not None
    assert np.all(np.isfinite(result.weights))
    # 完全可分时 L2 让权重保持在有限范围。
    assert result.weights[0] < 100.0


def test_contradictory_labels_give_balanced_weights() -> None:
    table = vectors(
        row("a", grade=5, nightly_price=200, room_area_sqm=40),
        row("b", grade=3, nightly_price=500, room_area_sqm=20),
    )
    pairs = labeled_pairs(
        [event("a", "b", LEFT_WINS), event("b", "a", LEFT_WINS)],
        table,
    )
    result = fit(pairs, FittingConfig())
    assert result.ok
    assert result.weights is not None
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)
    # 相互矛盾的标签应把权重压向零附近。
    assert float(np.max(result.weights)) < 0.5


def test_monotonic_direction_for_favoured_features() -> None:
    """用户偏好更好特征时，对应权重应为正。"""

    table = vectors(
        row("good", grade=5, renovation_age_years=1, ctrip_rating=4.8,
            subway_walk_minutes=2, mall_transit_minutes=3,
            attraction_taxi_minutes=4, room_area_sqm=45, nightly_price=200),
        row("bad", grade=2, renovation_age_years=14, ctrip_rating=4.2,
            subway_walk_minutes=25, mall_transit_minutes=35,
            attraction_taxi_minutes=45, room_area_sqm=18, nightly_price=700),
    )
    events = [event("good", "bad", LEFT_WINS) for _ in range(5)]
    result = fit(labeled_pairs(events, table), FittingConfig())
    assert result.ok
    assert result.weights is not None
    assert np.all(result.weights > 0.0), "所有维度方向一致时权重应为正"


def test_multi_city_shares_one_weight_vector() -> None:
    table = vectors(
        row("hz-1", city="杭州", grade=5, nightly_price=200),
        row("hz-2", city="杭州", grade=3, nightly_price=500),
        row("cd-1", city="成都", grade=5, nightly_price=210),
        row("cd-2", city="成都", grade=3, nightly_price=490),
    )
    pairs = labeled_pairs(
        [event("hz-1", "hz-2", LEFT_WINS), event("cd-1", "cd-2", LEFT_WINS)],
        table,
    )
    result = fit(pairs, FittingConfig())
    assert result.ok
    assert result.weights is not None
    assert result.weights.shape == (8,)
    assert result.labels_covered == 2


def test_empty_dataset_returns_untrained_result() -> None:
    result = fit([], FittingConfig())
    assert not result.ok
    assert result.weights is None
    assert result.training_cross_entropy is None
    assert result.labels_covered == 0
    assert "未训练" in result.message


def test_fit_result_serialization() -> None:
    result = fit([], FittingConfig())
    payload = result.to_dict()
    assert payload["weights"] is None
    assert payload["ok"] is False
    assert isinstance(FitResult(**payload), FitResult)
