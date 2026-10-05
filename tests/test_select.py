"""主动选题测试（任务 3.1–3.3）。"""

from __future__ import annotations

import numpy as np
import pytest

from hotel_model.data import DataError
from hotel_model.features import to_features
from hotel_model.select import (
    COLD_START_LABELS,
    DEFAULT_EXPLORATION_RATE,
    DEFAULT_SEED,
    MIN_DIFFERENCE_NORM,
    Question,
    SelectionConfig,
    active_score,
    build_candidates,
    cold_score,
    diversity,
    is_tradeoff,
    magnitude,
    score_candidates,
    select_next,
)

from .conftest import make_samples, row


def table(*rows: dict[str, object]) -> tuple[list, dict[str, np.ndarray]]:
    samples = make_samples(list(rows))
    vectors = {sample.hotel_id: to_features(sample) for sample in samples}
    return samples, vectors


def ids(candidates) -> list[tuple[str, str]]:
    return [(c.left_id, c.right_id) for c in candidates]


# --- 3.1 候选构造 -----------------------------------------------------------


def test_defaults() -> None:
    assert DEFAULT_SEED == 42
    assert DEFAULT_EXPLORATION_RATE == 0.2
    assert COLD_START_LABELS == 8
    assert SelectionConfig().seed == 42


def test_candidates_are_same_city_and_sorted() -> None:
    samples, vectors = table(
        row("b", city="成都", grade=5, nightly_price=500),
        row("a", city="杭州", grade=3, nightly_price=300),
        row("c", city="杭州", grade=5, nightly_price=500),
    )
    candidates = build_candidates(samples, vectors, [])
    assert ids(candidates) == [("a", "c")]
    assert all(left < right for left, right in ids(candidates))


def test_mixed_city_produces_only_intra_city_pairs() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3),
        row("hz-2", city="杭州", grade=5),
        row("cd-1", city="成都", grade=3),
        row("cd-2", city="成都", grade=5),
    )
    candidates = build_candidates(samples, vectors, [])
    assert ids(candidates) == [("cd-1", "cd-2"), ("hz-1", "hz-2")]


def test_single_sample_city_yields_no_candidate() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3),
        row("hz-2", city="杭州", grade=5),
        row("cd-1", city="成都", grade=3),
    )
    assert ids(build_candidates(samples, vectors, [])) == [("hz-1", "hz-2")]


def test_identical_features_excluded() -> None:
    same = dict(
        grade=4,
        renovation_age_years=3,
        ctrip_rating=4.7,
        subway_walk_minutes=5,
        mall_transit_minutes=10,
        attraction_taxi_minutes=15,
        room_area_sqm=30,
        nightly_price=350,
    )
    samples, vectors = table(
        row("hz-1", city="杭州", **same),
        row("hz-2", city="杭州", **same),
    )
    assert build_candidates(samples, vectors, []) == []


def test_answered_pair_removed_in_both_directions() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
        row("hz-3", city="杭州", grade=4, nightly_price=400),
    )
    forward = build_candidates(samples, vectors, [{"left_id": "hz-1", "right_id": "hz-2"}])
    backward = build_candidates(samples, vectors, [{"left_id": "hz-2", "right_id": "hz-1"}])
    assert ("hz-1", "hz-2") not in ids(forward)
    assert ("hz-1", "hz-2") not in ids(backward)
    assert ids(forward) == ids(backward)


def test_pending_question_is_treated_as_processed() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
    )
    pending = Question(left_id="hz-2", right_id="hz-1")
    assert build_candidates(samples, vectors, [pending]) == []


def test_exhausted_candidates_return_none() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
    )
    rng = np.random.default_rng(1)
    assert (
        select_next(
            samples,
            processed=[{"left_id": "hz-1", "right_id": "hz-2"}],
            vectors=vectors,
            weights=None,
            label_count=0,
            rng=rng,
        )
        is None
    )


def test_missing_vector_reports_error() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3),
        row("hz-2", city="杭州", grade=5),
    )
    with pytest.raises(DataError, match="缺少特征向量"):
        build_candidates(samples, {"hz-1": vectors["hz-1"]}, [])


def test_pair_as_pairs_rejects_same_sample() -> None:
    from hotel_model.select import as_pairs

    with pytest.raises(ValueError):
        as_pairs([{"left_id": "a", "right_id": "a"}])


# --- 3.2 取舍优先与冷启动 ---------------------------------------------------


def test_is_tradeoff_detects_both_directions() -> None:
    assert is_tradeoff(np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert not is_tradeoff(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert not is_tradeoff(np.zeros(8))
    assert not is_tradeoff(np.full(8, -1.0))


def test_magnitude_normalized_and_clipped() -> None:
    assert magnitude(np.zeros(8)) == 0.0
    assert magnitude(np.full(8, 1.0)) == pytest.approx(1.0)
    assert magnitude(np.full(8, 10.0)) == 1.0


def test_diversity_is_one_without_history() -> None:
    difference = np.array([1.0] + [0.0] * 7)
    assert diversity(difference, []) == 1.0


def test_diversity_penalizes_repeated_direction() -> None:
    difference = np.array([1.0] + [0.0] * 7)
    unit = difference / np.linalg.norm(difference)
    assert diversity(difference, [unit]) == pytest.approx(0.0)
    assert diversity(difference, [np.array([0.0, 1.0] + [0.0] * 6)]) == pytest.approx(1.0)


def test_cold_start_prefers_larger_difference_when_direction_repeated() -> None:
    """冷启动不依赖未训练预测：相同覆盖程度下优先差异更大的候选。"""

    big = np.zeros(8)
    big[0] = 1.0
    small = np.zeros(8)
    small[0] = 0.01
    assert cold_score(big, []) > cold_score(small, [])


def test_active_score_prefers_larger_difference_at_equal_uncertainty() -> None:
    """相同预测不确定性、相同覆盖程度时，差异范数更大的候选得分更高。"""

    big = np.zeros(8)
    big[0] = 1.0
    tiny = np.zeros(8)
    tiny[0] = 0.01
    assert active_score(big, 0.5, []) > active_score(tiny, 0.5, [])


def test_active_score_favours_uncertainty() -> None:
    difference = np.zeros(8)
    difference[0] = 1.0
    uncertainty_peak = active_score(difference, 0.5, [])
    confident = active_score(difference, 0.99, [])
    assert uncertainty_peak > confident


def test_cold_start_deterministic_score_used_before_eight_labels() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=4, nightly_price=350),
        row("hz-3", city="杭州", grade=5, nightly_price=500),
    )
    candidates = build_candidates(samples, vectors, [])
    # 未训练时权重为 None，仍然能给出确定性分数。
    scores = score_candidates(candidates, None, 0, [], vectors, SelectionConfig())
    assert all(np.isfinite(scores))
    assert max(scores) > min(scores)


def test_phase_switch_changes_scoring_after_eight_labels() -> None:
    """有效标签少于 8 用冷启动分数，达到 8 后改用主动分数。"""

    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
    )
    candidates = build_candidates(samples, vectors, [])
    difference = candidates[0].difference
    # 权重取零使预测为 0.5，此时 U=1，主动分数与冷启动分数可解析对比。
    weights = np.zeros(8)
    expected_cold = cold_score(difference, [])
    expected_active = active_score(difference, 0.5, [])

    cold = score_candidates(candidates, weights, 7, [], vectors, SelectionConfig())
    active = score_candidates(candidates, weights, 8, [], vectors, SelectionConfig())
    assert cold[0] == pytest.approx(expected_cold)
    assert active[0] == pytest.approx(expected_active)
    assert cold[0] != pytest.approx(active[0])


def test_non_explore_branch_prefers_tradeoff_candidates() -> None:
    """存在取舍候选时优先从取舍候选中选择，即使其差异更小。"""

    samples, vectors = table(
        row("hz-1", city="杭州", grade=9, nightly_price=100),
        row("hz-2", city="杭州", grade=1, nightly_price=1000),
        row("hz-3", city="杭州", grade=4, nightly_price=300),
        row("hz-4", city="杭州", grade=5, nightly_price=900),
    )
    # 让探索分支失效，强制走确定性分支。
    config = SelectionConfig(exploration_rate=0.0)
    question = select_next(
        samples,
        processed=[],
        vectors=vectors,
        weights=None,
        label_count=0,
        rng=np.random.default_rng(0),
        config=config,
    )
    assert question is not None
    assert question.pair in {("hz-1", "hz-2"), ("hz-3", "hz-4")}


def test_fallback_when_no_tradeoff_candidates() -> None:
    """没有取舍候选时回退到全部候选，不返回 None。"""

    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, ctrip_rating=4.6, nightly_price=1000,
            subway_walk_minutes=30, mall_transit_minutes=40, attraction_taxi_minutes=50),
        row("hz-2", city="杭州", grade=5, ctrip_rating=4.8, nightly_price=100, room_area_sqm=50,
            subway_walk_minutes=1, mall_transit_minutes=2, attraction_taxi_minutes=3),
    )
    candidates = build_candidates(samples, vectors, [])
    assert not any(is_tradeoff(c.difference) for c in candidates)
    question = select_next(
        samples,
        processed=[],
        vectors=vectors,
        weights=None,
        label_count=0,
        rng=np.random.default_rng(0),
        config=SelectionConfig(exploration_rate=0.0),
    )
    assert question is not None
    assert question.pair in {("hz-1", "hz-2")}


# --- 3.3 随机探索与可复现 ---------------------------------------------------


def test_exploration_can_select_every_legal_candidate() -> None:
    """探索分支在多次抽取中覆盖全部合法候选。"""

    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
        row("hz-3", city="杭州", grade=4, nightly_price=400),
    )
    legal = {("hz-1", "hz-2"), ("hz-1", "hz-3"), ("hz-2", "hz-3")}
    rng = np.random.default_rng(12345)
    seen: set[tuple[str, str]] = set()
    for _ in range(200):
        question = select_next(
            samples,
            processed=[],
            vectors=vectors,
            weights=None,
            label_count=0,
            rng=rng,
        )
        assert question is not None
        assert question.pair in legal
        seen.add(question.pair)
    assert seen == legal


def test_random_exploration_only_picks_legal_candidates() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
        row("cd-1", city="成都", grade=3, nightly_price=300),
        row("cd-2", city="成都", grade=5, nightly_price=500),
    )
    rng = np.random.default_rng(9)
    for _ in range(100):
        question = select_next(
            samples, processed=[], vectors=vectors, weights=None, label_count=0, rng=rng
        )
        assert question is not None
        city = {s.hotel_id: s.city for s in samples}
        assert city[question.left_id] == city[question.right_id]


def test_exploration_never_returns_processed_pair() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
    )
    rng = np.random.default_rng(3)
    processed = [{"left_id": "hz-1", "right_id": "hz-2"}]
    assert (
        select_next(
            samples, processed=processed, vectors=vectors, weights=None, label_count=0, rng=rng
        )
        is None
    )


def test_left_right_order_is_randomized() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
    )
    rng = np.random.default_rng(0)
    orders = set()
    for _ in range(50):
        question = select_next(
            samples, processed=[], vectors=vectors, weights=None, label_count=0, rng=rng
        )
        assert question is not None
        orders.add((question.left_id, question.right_id))
    assert len(orders) == 2, "左右展示顺序应随机化"


def test_same_seed_and_inputs_reproduce_question_sequence() -> None:
    samples, vectors = table(
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
        row("hz-3", city="杭州", grade=4, nightly_price=400),
    )

    def sequence() -> list[tuple[str, str]]:
        rng = np.random.default_rng(42)
        processed: list[object] = []
        picked: list[tuple[str, str]] = []
        for _ in range(3):
            question = select_next(
                samples,
                processed=processed,
                vectors=vectors,
                weights=None,
                label_count=0,
                rng=rng,
                config=SelectionConfig(exploration_rate=0.0),
            )
            assert question is not None
            picked.append((question.left_id, question.right_id))
            processed.append(question)
        return picked

    assert sequence() == sequence()


def test_question_round_trip_and_version_check() -> None:
    question = Question(left_id="a", right_id="b")
    assert Question.from_dict(question.to_dict()) == question
    assert question.pair == ("a", "b")
    with pytest.raises(ValueError):
        Question.from_dict({"left_id": "a", "right_id": "a", "version": "v1"})
    with pytest.raises(ValueError):
        Question.from_dict({"left_id": "a", "right_id": "b", "version": "v9"})


def test_selection_config_validation_and_round_trip() -> None:
    config = SelectionConfig(seed=7, exploration_rate=0.5, cold_start_labels=3)
    assert SelectionConfig.from_dict(config.to_dict()) == config
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="探索比例"):
            SelectionConfig(exploration_rate=bad)
    with pytest.raises(ValueError, match="种子"):
        SelectionConfig(seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="冷启动"):
        SelectionConfig(cold_start_labels=0)
    with pytest.raises(ValueError):
        SelectionConfig.from_dict({"version": "v9"})


def test_min_difference_norm_is_tiny() -> None:
    assert MIN_DIFFERENCE_NORM == 1e-9
