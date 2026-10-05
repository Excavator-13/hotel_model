"""版本化状态、内容指纹、原子替换与恢复校验测试（任务 4.1–4.3）。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hotel_model.data import Sample
from hotel_model.features import FeatureConfig, to_features
from hotel_model.fitting import FittingConfig
from hotel_model.select import Question, SelectionConfig
from hotel_model.state import (
    SCHEMA_VERSION,
    ModelSnapshot,
    PendingQuestion,
    StateError,
    TrainingState,
    content_fingerprint,
    default_random_state,
    load_state,
    rng_from_state,
    save_state,
    state_from_dict,
    state_to_json,
)

from .conftest import csv_text, make_samples, row


def make_state(samples: list[Sample] | None = None, **overrides: object) -> TrainingState:
    pool = samples if samples is not None else make_samples(
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("hz-2", city="杭州", grade=5, nightly_price=500),
        ]
    )
    defaults: dict[str, object] = {
        "preprocessing": FeatureConfig(),
        "fitting": FittingConfig(),
        "selection": SelectionConfig(),
        "random_state": default_random_state(42),
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    defaults.update(overrides)
    return TrainingState.new(pool, **defaults)  # type: ignore[arg-type]


# --- 4.1 版本化 JSON 与指纹 -------------------------------------------------


def test_new_state_shape() -> None:
    state = make_state()
    assert state.schema_version == SCHEMA_VERSION == 1
    assert state.events == ()
    assert state.pending is None
    assert not state.model.trained
    assert state.fingerprint == content_fingerprint(state.samples)
    assert len(state.random_state) > 0


def test_empty_pool_rejected() -> None:
    with pytest.raises(StateError, match="至少需要一条样本"):
        make_state([])


def test_serialization_round_trip_preserves_content() -> None:
    state = make_state()
    restored = state_from_dict(json.loads(state_to_json(state)))
    assert restored == state
    assert restored.samples == state.samples
    assert restored.fingerprint == state.fingerprint
    assert restored.random_state == state.random_state
    assert restored.model == state.model


def test_round_trip_with_events_and_pending() -> None:
    samples = make_samples(
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("hz-2", city="杭州", grade=5, nightly_price=500),
            row("hz-3", city="杭州", grade=4, nightly_price=400),
        ]
    )
    state = make_state(samples)
    event = {
        "left_id": "hz-1",
        "right_id": "hz-2",
        "city": "杭州",
        "kind": "label",
        "label": 1.0,
        "answered_at": "2024-01-01T00:01:00+00:00",
        "asked_at": "2024-01-01T00:00:00+00:00",
        "answer": "a",
    }
    weights = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    state = state.with_(
        events=[event],
        pending=PendingQuestion(Question("hz-3", "hz-1"), "2024-01-01T00:02:00+00:00"),
        model=ModelSnapshot(True, weights, 1, 0.31, 0),
    )
    restored = state_from_dict(json.loads(state_to_json(state)))
    assert restored == state
    assert np.allclose(restored.model.weights, weights)
    assert restored.pending is not None
    assert restored.pending.question.left_id == "hz-3"


def test_file_round_trip(tmp_path: Path) -> None:
    state = make_state()
    path = tmp_path / "session.json"
    save_state(state, path)
    assert load_state(path) == state


def test_fingerprint_ignores_row_order_and_bom() -> None:
    rows = [
        row("hz-1", city="杭州", grade=3, nightly_price=300),
        row("hz-2", city="杭州", grade=5, nightly_price=500),
    ]
    forward = make_samples(rows)
    backward = make_samples(list(reversed(rows)))
    assert content_fingerprint(forward) == content_fingerprint(backward)


def test_fingerprint_ignores_extra_columns() -> None:
    rows = [row("hz-1", city="杭州", grade=3, nightly_price=300)]
    plain = make_samples(rows)
    with_extra = make_samples([dict(rows[0], **{"备注": "虚构"})])
    assert content_fingerprint(plain) == content_fingerprint(with_extra)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nightly_price", 301),
        ("name", "改名了"),
        ("city", "成都"),
        ("grade", 4),
        ("room_area_sqm", 26),
    ],
)
def test_fingerprint_changes_when_required_field_changes(field: str, value: object) -> None:
    base_row = row("hz-1", city="杭州", grade=3, nightly_price=300)
    changed_row = dict(base_row)
    changed_row[field] = value
    base = make_samples([base_row])
    changed = make_samples([changed_row])
    assert content_fingerprint(base) != content_fingerprint(changed)


def test_state_rejects_fingerprint_mismatch() -> None:
    state = make_state()
    payload = state.to_dict()
    payload["samples"][0]["nightly_price"] = 999.0  # type: ignore[index]
    with pytest.raises(StateError, match="指纹"):
        state_from_dict(payload)


@pytest.mark.parametrize("version", [0, 2, 99])
def test_unsupported_schema_version_rejected(version: int) -> None:
    payload = make_state().to_dict()
    payload["schema_version"] = version
    with pytest.raises(StateError, match="不支持的状态版本"):
        state_from_dict(payload)


def test_missing_schema_version_rejected() -> None:
    payload = make_state().to_dict()
    del payload["schema_version"]
    with pytest.raises(StateError, match="schema_version"):
        state_from_dict(payload)


def test_load_state_rejects_non_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StateError, match="不是合法 JSON"):
        load_state(path)


def test_load_state_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="找不到状态文件"):
        load_state(tmp_path / "nope.json")


def test_random_state_round_trip_reproduces_stream() -> None:
    state = make_state()
    first = rng_from_state(state.random_state)
    second = rng_from_state(state.random_state)
    assert first.random() == second.random()
    assert first.integers(0, 10) == second.integers(0, 10)
    # 序列化后仍可复现。
    reloaded = state_from_dict(json.loads(state_to_json(state)))
    third = rng_from_state(reloaded.random_state)
    fourth = rng_from_state(reloaded.random_state)
    assert third.random() == fourth.random()


# --- 4.2 原子替换与两步保存 ------------------------------------------------


def test_save_state_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    state = make_state()
    path = tmp_path / "session.json"
    save_state(state, path)
    save_state(state.with_(updated_at="2024-01-02T00:00:00+00:00"), path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "session.json"]
    assert leftovers == []
    assert load_state(path).updated_at == "2024-01-02T00:00:00+00:00"


def test_save_failure_keeps_previous_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = make_state()
    path = tmp_path / "session.json"
    save_state(state, path)
    original = path.read_text(encoding="utf-8")

    def broken_replace(source: object, target: object) -> None:
        raise OSError("磁盘写入失败")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        save_state(state.with_(updated_at="2024-09-09T00:00:00+00:00"), path)

    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == original
    assert load_state(path) == state
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "session.json"]
    assert leftovers == [], "失败后应清理临时文件"


def test_answer_persisted_before_fitting(tmp_path: Path) -> None:
    """事件在拟合之前写入磁盘，因此拟合失败不会丢失答案。"""

    from hotel_model.train import ScriptedConsole, TrainOptions, new_session

    options = TrainOptions(state_path=tmp_path / "session.json", data_path=None)
    samples = make_samples(
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("hz-2", city="杭州", grade=5, nightly_price=500),
        ]
    )
    session = new_session(samples, options, now="2024-01-01T00:00:00+00:00")
    session = session.persist(session.state)
    assert load_state(options.state_path).events == ()


def test_model_coverage_bookkeeping_matches_event_count() -> None:
    state = make_state()
    weights = np.ones(8) * 0.1
    state = state.with_(model=ModelSnapshot(True, weights, 0, 0.5, 2))
    payload = state.to_dict()
    with pytest.raises(StateError, match="覆盖数"):
        state_from_dict(payload)


# --- 4.3 恢复校验 -----------------------------------------------------------


def _state_with_two_events() -> TrainingState:
    samples = make_samples(
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("hz-2", city="杭州", grade=5, nightly_price=500),
            row("hz-3", city="杭州", grade=4, nightly_price=400),
        ]
    )
    state = make_state(samples)
    events = [
        {
            "left_id": "hz-1",
            "right_id": "hz-2",
            "city": "杭州",
            "kind": "label",
            "label": 1.0,
            "answered_at": "2024-01-01T00:01:00+00:00",
        },
        {
            "left_id": "hz-1",
            "right_id": "hz-3",
            "city": "杭州",
            "kind": "skip",
            "label": None,
            "answered_at": "2024-01-01T00:02:00+00:00",
        },
    ]
    weights = np.ones(8) * 0.1
    return state.with_(events=events, model=ModelSnapshot(True, weights, 1, 0.4, 0))


def test_valid_state_with_events_loads() -> None:
    state = _state_with_two_events()
    restored = state_from_dict(json.loads(state_to_json(state)))
    assert len(restored.events) == 2
    assert len(restored.label_events) == 1
    assert len(restored.skip_events) == 1
    assert restored.model.labels_covered == 1


def test_event_referencing_unknown_sample_rejected() -> None:
    payload = _state_with_two_events().to_dict()
    payload["events"][0]["left_id"] = "ghost"  # type: ignore[index]
    with pytest.raises(StateError, match="不存在"):
        state_from_dict(payload)


def test_cross_city_event_rejected() -> None:
    samples = make_samples(
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("cd-1", city="成都", grade=5, nightly_price=500),
        ]
    )
    state = make_state(samples)
    event = {
        "left_id": "hz-1",
        "right_id": "cd-1",
        "kind": "label",
        "label": 1.0,
        "answered_at": "2024-01-01T00:01:00+00:00",
    }
    payload = state.with_(events=[event], model=ModelSnapshot(True, np.ones(8) * 0.1, 1, 0.4, 0)).to_dict()
    with pytest.raises(StateError, match="跨城市"):
        state_from_dict(payload)


def test_duplicate_event_pair_rejected() -> None:
    payload = _state_with_two_events().to_dict()
    duplicate = dict(payload["events"][0])  # type: ignore[index]
    duplicate["left_id"], duplicate["right_id"] = "hz-2", "hz-1"
    payload["events"] = list(payload["events"]) + [duplicate]  # type: ignore[arg-type]
    payload["model"]["labels_covered"] = 2  # type: ignore[index]
    with pytest.raises(StateError, match="重复"):
        state_from_dict(payload)


def test_invalid_label_value_rejected() -> None:
    payload = _state_with_two_events().to_dict()
    payload["events"][0]["label"] = 0.25  # type: ignore[index]
    with pytest.raises(StateError, match="标签非法"):
        state_from_dict(payload)


def test_pending_pair_repeating_event_rejected() -> None:
    payload = _state_with_two_events().to_dict()
    payload["pending"] = {
        "left_id": "hz-2",
        "right_id": "hz-1",
        "version": "v1",
        "asked_at": "2024-01-01T00:03:00+00:00",
    }
    with pytest.raises(StateError, match="待答题与已处理"):
        state_from_dict(payload)


def test_cross_city_pending_rejected() -> None:
    samples = make_samples(
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("cd-1", city="成都", grade=5, nightly_price=500),
        ]
    )
    payload = make_state(samples).to_dict()
    payload["pending"] = {
        "left_id": "hz-1",
        "right_id": "cd-1",
        "version": "v1",
        "asked_at": "2024-01-01T00:03:00+00:00",
    }
    with pytest.raises(StateError, match="跨城市"):
        state_from_dict(payload)


def test_trained_model_requires_weights_and_loss() -> None:
    payload = make_state().to_dict()
    payload["model"] = {"trained": True, "weights": None, "labels_covered": 0,
                        "training_cross_entropy": None, "lagging_labels": 0}
    with pytest.raises(StateError, match="缺少权重"):
        state_from_dict(payload)


def test_untrained_model_rejects_weights() -> None:
    payload = make_state().to_dict()
    payload["model"] = {"trained": False, "weights": [0.1] * 8, "labels_covered": 0,
                        "training_cross_entropy": None, "lagging_labels": 0}
    with pytest.raises(StateError, match="未训练"):
        state_from_dict(payload)


def test_negative_and_non_finite_weights_rejected() -> None:
    for bad in ([-0.1] * 8, [float("nan")] * 8, [1.0] * 7):
        payload = make_state().to_dict()
        payload["model"] = {"trained": True, "weights": bad, "labels_covered": 0,
                            "training_cross_entropy": 0.5, "lagging_labels": 0}
        with pytest.raises(StateError):
            state_from_dict(payload)


def test_duplicate_sample_ids_in_snapshot_rejected() -> None:
    payload = make_state().to_dict()
    payload["samples"] = [payload["samples"][0], dict(payload["samples"][0])]  # type: ignore[index]
    with pytest.raises(StateError, match="重复"):
        state_from_dict(payload)


def test_unsupported_preprocessing_version_in_state_rejected() -> None:
    payload = make_state().to_dict()
    payload["preprocessing"]["version"] = "v2"  # type: ignore[index]
    with pytest.raises(StateError, match="预处理版本"):
        state_from_dict(payload)


def test_unsupported_selection_version_in_state_rejected() -> None:
    payload = make_state().to_dict()
    payload["selection"]["version"] = "v9"  # type: ignore[index]
    with pytest.raises(StateError, match="选题配置版本"):
        state_from_dict(payload)
