"""跨模块终端集成测试（任务 6.2）。

使用两个城市的数据依次执行选择、平局、跳过、退出与恢复，并模拟一次拟合失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from hotel_model import train as train_module
from hotel_model.fitting import FitResult, labeled_pairs
from hotel_model.state import load_state, rng_from_state
from hotel_model.train import ScriptedConsole, TrainOptions, run_training

from .conftest import write_csv

CLOCK = "2024-06-01T09:00:00+00:00"

# 两个城市各有 4 家，保证每个城市都能产生多道题。
ROWS = [
    {"hotel_id": "hz-1", "name": "杭州 1", "city": "杭州", "grade": 3, "renovation_age_years": 2,
     "ctrip_rating": 4.7, "subway_walk_minutes": 6, "mall_transit_minutes": 15,
     "attraction_taxi_minutes": 22, "room_area_sqm": 25, "nightly_price": 300},
    {"hotel_id": "hz-2", "name": "杭州 2", "city": "杭州", "grade": 4, "renovation_age_years": 6,
     "ctrip_rating": 4.6, "subway_walk_minutes": 12, "mall_transit_minutes": 8,
     "attraction_taxi_minutes": 18, "room_area_sqm": 30, "nightly_price": 360},
    {"hotel_id": "hz-3", "name": "杭州 3", "city": "杭州", "grade": 5, "renovation_age_years": 3,
     "ctrip_rating": 4.8, "subway_walk_minutes": 9, "mall_transit_minutes": 20,
     "attraction_taxi_minutes": 25, "room_area_sqm": 35, "nightly_price": 480},
    {"hotel_id": "hz-4", "name": "杭州 4", "city": "杭州", "grade": 4, "renovation_age_years": 12,
     "ctrip_rating": 4.5, "subway_walk_minutes": 3, "mall_transit_minutes": 25,
     "attraction_taxi_minutes": 30, "room_area_sqm": 28, "nightly_price": 260},
    {"hotel_id": "cd-1", "name": "成都 1", "city": "成都", "grade": 3, "renovation_age_years": 1,
     "ctrip_rating": 4.7, "subway_walk_minutes": 4, "mall_transit_minutes": 10,
     "attraction_taxi_minutes": 15, "room_area_sqm": 26, "nightly_price": 300},
    {"hotel_id": "cd-2", "name": "成都 2", "city": "成都", "grade": 4, "renovation_age_years": 5,
     "ctrip_rating": 4.6, "subway_walk_minutes": 10, "mall_transit_minutes": 14,
     "attraction_taxi_minutes": 20, "room_area_sqm": 31, "nightly_price": 355},
    {"hotel_id": "cd-3", "name": "成都 3", "city": "成都", "grade": 5, "renovation_age_years": 2,
     "ctrip_rating": 4.8, "subway_walk_minutes": 8, "mall_transit_minutes": 18,
     "attraction_taxi_minutes": 24, "room_area_sqm": 36, "nightly_price": 470},
    {"hotel_id": "cd-4", "name": "成都 4", "city": "成都", "grade": 4, "renovation_age_years": 11,
     "ctrip_rating": 4.5, "subway_walk_minutes": 2, "mall_transit_minutes": 22,
     "attraction_taxi_minutes": 28, "room_area_sqm": 29, "nightly_price": 255},
]


@pytest.fixture
def data_csv(tmp_path: Path) -> Path:
    return write_csv(tmp_path / "cities.csv", ROWS)


def ask_sequence(console: ScriptedConsole) -> list[tuple[str, str]]:
    """从控制台输出还原每道题的 (A id, B id) 顺序。

    恢复时若同一待答题在保存前后各展示一次，会被视为同一道题。
    """

    questions: list[tuple[str, str]] = []
    current: dict[str, str] = {}
    for line in console.lines:
        if line.startswith("A 候选"):
            current = {}
        match = re.search(r"hotel_id=(\S+)\s", line)
        if match and "A 候选" not in line:
            if "A" not in current:
                current["A"] = match.group(1)
            else:
                current["B"] = match.group(1)
                pair = (current["A"], current["B"])
                if not questions or questions[-1] != pair:
                    questions.append(pair)
    return questions


def test_two_city_session_asks_only_intra_city_pairs(tmp_path: Path, data_csv: Path) -> None:
    console = ScriptedConsole(["a", "t", "s", "b", "a", "b", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    cities = {sample.hotel_id: sample.city for sample in outcome.state.samples}
    asked = ask_sequence(console)
    assert asked, "应至少展示一道题"
    for left, right in asked:
        assert cities[left] == cities[right]
        assert left != right


def test_tie_skip_and_strict_are_recorded_distinctly(tmp_path: Path, data_csv: Path) -> None:
    console = ScriptedConsole(["a", "t", "s", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    state = outcome.state
    assert [event["label"] for event in state.events] == [1.0, 0.5, None]
    assert [event["kind"] for event in state.events] == ["label", "label", "skip"]
    assert len(state.label_events) == 2
    assert len(state.skip_events) == 1


def test_each_effective_label_triggers_refit(tmp_path: Path, data_csv: Path) -> None:
    console = ScriptedConsole(["a", "b", "t", "s", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    state = outcome.state
    effective = len(state.label_events)
    assert effective == 3
    assert state.model.trained
    assert state.model.labels_covered == effective
    assert state.model.lagging_labels == 0
    # 覆盖数与状态计数一致。
    assert state.model.labels_covered + state.model.lagging_labels == effective


def test_skips_do_not_enter_fitting(tmp_path: Path, data_csv: Path) -> None:
    console = ScriptedConsole(["s", "s", "a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    state = outcome.state
    assert len(state.skip_events) == 2
    assert len(state.label_events) == 1
    vectors = {
        sample.hotel_id: train_module.to_features(sample, state.preprocessing)
        for sample in state.samples
    }
    pairs = labeled_pairs(state.events, vectors)
    assert len(pairs) == 1
    assert state.model.labels_covered == 1


def test_state_counts_match_recomputed_weights(tmp_path: Path, data_csv: Path) -> None:
    console = ScriptedConsole(["a", "b", "a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    state = outcome.state
    vectors = {
        sample.hotel_id: train_module.to_features(sample, state.preprocessing)
        for sample in state.samples
    }
    pairs = labeled_pairs(state.events, vectors)
    from hotel_model.fitting import fit

    recomputed = fit(pairs, state.fitting)
    assert recomputed.ok
    assert np.allclose(state.model.weights, recomputed.weights)
    assert state.model.training_cross_entropy == pytest.approx(
        recomputed.training_cross_entropy
    )


def test_answered_pairs_do_not_repeat_and_skipped_pairs_are_gone(
    tmp_path: Path, data_csv: Path
) -> None:
    console = ScriptedConsole(["a", "t", "s", "b", "a", "b", "a", "t", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    asked = [tuple(sorted(pair)) for pair in ask_sequence(console)]
    assert len(asked) == len(set(asked)), "同一无序样本对不应重复提问"

    recorded = [tuple(sorted((e["left_id"], e["right_id"]))) for e in outcome.state.events]
    assert len(recorded) == len(set(recorded))
    # 每道被展示的题按顺序消耗一个回答，因此已记录的对是展示序列的前缀。
    assert asked[: len(recorded)] == recorded
    # 唯一被展示但未记录的对，必须正是用 q 留下的待答题。
    assert asked[len(recorded):] == [tuple(sorted(outcome.state.pending.question.pair))]
    # 跳过的对同样永久移除，不会再出现在后续展示中。
    skipped = {
        tuple(sorted((e["left_id"], e["right_id"])))
        for e in outcome.state.skip_events
    }
    assert skipped == {asked[2]}
    assert asked.count(asked[2]) == 1


class _FailingFit:
    """第一次拟合失败，之后恢复真实实现。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> FitResult:
        self.calls += 1
        return FitResult(False, None, None, 0, "注入的优化失败")


def test_fit_failure_preserves_answer_and_stops(
    tmp_path: Path, data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "s.json"
    failing = _FailingFit()
    monkeypatch.setattr(train_module, "fit", failing)

    console = ScriptedConsole(["a", "b", "q"])
    outcome = run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    assert outcome.stop_reason == "fit_failed"
    assert outcome.fit_failures == 1
    assert "拟合失败" in console.output

    # 答案已先持久化，状态文件可读。
    persisted = load_state(state_path)
    assert len(persisted.events) == 1
    assert persisted.model.lagging_labels == 1
    assert not persisted.model.trained


def test_resume_after_fit_failure_refits_and_continues(
    tmp_path: Path, data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "s.json"
    monkeypatch.setattr(train_module, "fit", _FailingFit())
    run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=ScriptedConsole(["a", "q"]),
        now=lambda: CLOCK,
    )
    broken = load_state(state_path)
    assert broken.model.lagging_labels == 1

    monkeypatch.undo()
    console = ScriptedConsole(["b", "q"])
    outcome = run_training(
        TrainOptions(state_path=state_path, resume=True, data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    assert outcome.stop_reason == "quit"
    assert outcome.state.model.trained
    assert outcome.state.model.labels_covered == 2
    assert outcome.state.model.lagging_labels == 0
    assert "先重新拟合" in console.output or outcome.fit_failures == 0
    assert len(outcome.state.events) == 2


class _InterruptingFit:
    """在第 ``at_call`` 次拟合抛 KeyboardInterrupt，模拟拟合过程被中断。"""

    def __init__(self, at_call: int, real) -> None:  # type: ignore[no-untyped-def]
        self.at_call = at_call
        self.real = real
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == self.at_call:
            raise KeyboardInterrupt
        return self.real(*args, **kwargs)


def test_interrupted_first_fit_leaves_loadable_checkpoint_and_resume_refits(
    tmp_path: Path, data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """答案已保存但第一次拟合被中断时，检查点必须可读且恢复后先重拟合。"""

    state_path = tmp_path / "s.json"
    real_fit = train_module.fit
    monkeypatch.setattr(train_module, "fit", _InterruptingFit(1, real_fit))

    outcome = run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=ScriptedConsole(["a"]),
        now=lambda: CLOCK,
    )
    assert outcome.stop_reason == "interrupted"
    monkeypatch.undo()

    # 该检查点在拟合之前写入，必须能被独立读取。
    checkpoint = load_state(state_path)
    assert len(checkpoint.label_events) == 1
    assert not checkpoint.model.trained
    assert checkpoint.model.labels_covered == 0
    assert checkpoint.model.lagging_labels == 1
    first_pair = tuple(sorted((checkpoint.events[0]["left_id"], checkpoint.events[0]["right_id"])))

    # 恢复时先重拟合落后标签，再继续提问；已处理的对不重复。
    console = ScriptedConsole(["b", "q"])
    resumed = run_training(
        TrainOptions(state_path=state_path, resume=True),
        console=console,
        now=lambda: CLOCK,
    )
    assert resumed.state.model.trained
    assert resumed.state.model.labels_covered == 2
    assert resumed.state.model.lagging_labels == 0
    assert len(resumed.state.events) == 2
    assert "先重新拟合" in console.output
    resumed_pairs = [tuple(sorted(pair)) for pair in ask_sequence(console)]
    assert resumed_pairs
    assert first_pair not in resumed_pairs


def test_answer_checkpoint_after_later_answer_keeps_previous_weights(
    tmp_path: Path, data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第二个答案写盘后、拟合被中断：保留上一份权重，落后数精确为 1。"""

    state_path = tmp_path / "s.json"
    real_fit = train_module.fit
    monkeypatch.setattr(train_module, "fit", _InterruptingFit(2, real_fit))

    outcome = run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=ScriptedConsole(["a", "b"]),
        now=lambda: CLOCK,
    )
    assert outcome.stop_reason == "interrupted"
    monkeypatch.undo()

    checkpoint = load_state(state_path)
    assert len(checkpoint.label_events) == 2
    assert checkpoint.model.trained
    assert checkpoint.model.labels_covered == 1
    assert checkpoint.model.lagging_labels == 1
    # 与第一次成功拟合的结果一致。
    assert np.allclose(checkpoint.model.weights, outcome.state.model.weights)


def test_failed_answer_save_keeps_previous_checkpoint(
    tmp_path: Path, data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """答案保存失败时，之前的完整状态必须保持不变且可读。"""

    state_path = tmp_path / "s.json"
    real_save = train_module.save_state

    def flaky_save(state, path):  # type: ignore[no-untyped-def]
        if state.events:
            raise OSError("注入的答案保存失败")
        return real_save(state, path)

    monkeypatch.setattr(train_module, "save_state", flaky_save)
    with pytest.raises(OSError):
        run_training(
            TrainOptions(state_path=state_path, data_path=data_csv),
            console=ScriptedConsole(["a"]),
            now=lambda: CLOCK,
        )
    monkeypatch.undo()

    checkpoint = load_state(state_path)
    assert checkpoint.events == ()
    assert checkpoint.pending is not None
    assert not checkpoint.model.trained
    assert checkpoint.model.lagging_labels == 0


def test_failed_model_save_keeps_answer_checkpoint_and_resume_recovers(
    tmp_path: Path, data_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型保存失败时事件已持久化；恢复必须从答案检查点重拟合。"""

    state_path = tmp_path / "s.json"
    real_save = train_module.save_state

    def flaky_save(state, path):  # type: ignore[no-untyped-def]
        if state.model.trained and state.model.lagging_labels == 0:
            raise OSError("注入的模型保存失败")
        return real_save(state, path)

    monkeypatch.setattr(train_module, "save_state", flaky_save)
    with pytest.raises(OSError):
        run_training(
            TrainOptions(state_path=state_path, data_path=data_csv),
            console=ScriptedConsole(["a"]),
            now=lambda: CLOCK,
        )
    monkeypatch.undo()

    checkpoint = load_state(state_path)
    assert len(checkpoint.label_events) == 1
    assert not checkpoint.model.trained
    assert checkpoint.model.lagging_labels == 1
    first_pair = tuple(sorted((checkpoint.events[0]["left_id"], checkpoint.events[0]["right_id"])))

    console = ScriptedConsole(["b", "q"])
    resumed = run_training(
        TrainOptions(state_path=state_path, resume=True),
        console=console,
        now=lambda: CLOCK,
    )
    assert resumed.state.model.trained
    assert resumed.state.model.labels_covered == 2
    assert resumed.state.model.lagging_labels == 0
    assert "先重新拟合" in console.output
    resumed_pairs = [tuple(sorted(pair)) for pair in ask_sequence(console)]
    assert first_pair not in resumed_pairs


def test_resume_keeps_pending_question_and_direction(tmp_path: Path, data_csv: Path) -> None:
    state_path = tmp_path / "s.json"
    first = ScriptedConsole(["q"])
    outcome = run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=first,
        now=lambda: CLOCK,
    )
    pending = outcome.state.pending
    assert pending is not None
    assert pending.question.left_id != pending.question.right_id

    second = ScriptedConsole(["a", "q"])
    resumed = run_training(
        TrainOptions(state_path=state_path, resume=True),
        console=second,
        now=lambda: CLOCK,
    )
    # 恢复后第一道展示的题就是原待答题，且左右顺序不变。
    assert (pending.question.left_id, pending.question.right_id) == ask_sequence(second)[0]
    assert resumed.state.events[0]["left_id"] == pending.question.left_id
    assert resumed.state.events[0]["right_id"] == pending.question.right_id
    assert resumed.state.events[0]["label"] == 1.0


def test_resume_does_not_repeat_answered_questions(tmp_path: Path, data_csv: Path) -> None:
    state_path = tmp_path / "s.json"
    first = ScriptedConsole(["a", "t", "q"])
    before = run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=first,
        now=lambda: CLOCK,
    )
    answered = {tuple(sorted((e["left_id"], e["right_id"]))) for e in before.state.events}

    second = ScriptedConsole(["b", "a", "b", "q"])
    after = run_training(
        TrainOptions(state_path=state_path, resume=True),
        console=second,
        now=lambda: CLOCK,
    )
    resumed_pairs = [tuple(sorted(pair)) for pair in ask_sequence(second)]
    assert resumed_pairs, "恢复后应继续提问"
    assert not (set(resumed_pairs) & answered)
    all_pairs = [
        tuple(sorted((e["left_id"], e["right_id"]))) for e in after.state.events
    ]
    assert len(all_pairs) == len(set(all_pairs))


def test_reproducible_questions_and_labels_with_same_seed(tmp_path: Path, data_csv: Path) -> None:
    """相同数据、配置、种子与回答序列应产生相同的选题与展示方向。"""

    def run(tag: str, answers: list[str]) -> tuple[list[tuple[str, str]], list[float | None]]:
        console = ScriptedConsole(answers)
        outcome = run_training(
            TrainOptions(
                state_path=tmp_path / f"{tag}.json",
                data_path=data_csv,
                seed=42,
            ),
            console=console,
            now=lambda: CLOCK,
        )
        return ask_sequence(console), [event["label"] for event in outcome.state.events]

    answers = ["a", "b", "t", "s", "a"]
    first_questions, first_labels = run("a", answers)
    second_questions, second_labels = run("b", answers)
    assert first_questions == second_questions
    assert first_labels == second_labels == [1.0, 0.0, 0.5, None, 1.0]


def test_label_matches_displayed_direction(tmp_path: Path, data_csv: Path) -> None:
    """输入 a 表示展示中的 A 更好，差向量按展示方向取。"""

    console = ScriptedConsole(["a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=data_csv),
        console=console,
        now=lambda: CLOCK,
    )
    asked = ask_sequence(console)[0]
    event = outcome.state.events[0]
    assert (event["left_id"], event["right_id"]) == asked
    assert event["label"] == 1.0

    # 拟合目标与「左样本更好」一致。
    state = outcome.state
    vectors = {
        sample.hotel_id: train_module.to_features(sample, state.preprocessing)
        for sample in state.samples
    }
    difference = vectors[event["left_id"]] - vectors[event["right_id"]]
    margin = float(np.dot(state.model.weights, difference))
    assert margin > 0.0, "记录为 A 更好时，展示方向的得分差应为正"


def test_random_state_advances_and_survives_resume(tmp_path: Path, data_csv: Path) -> None:
    state_path = tmp_path / "s.json"
    first = run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=ScriptedConsole(["q"]),
        now=lambda: CLOCK,
    )
    loaded = load_state(state_path)
    assert loaded.random_state == first.state.random_state
    # 保存的状态能重建同一随机流。
    a = rng_from_state(loaded.random_state)
    b = rng_from_state(first.state.random_state)
    assert a.random() == b.random()


def test_resume_ignores_seed_and_l2_flags(tmp_path: Path, data_csv: Path) -> None:
    """恢复时训练与选题配置固定，命令行 --seed/--l2 不生效。"""

    state_path = tmp_path / "s.json"
    run_training(
        TrainOptions(state_path=state_path, data_path=data_csv, seed=42, l2=0.1),
        console=ScriptedConsole(["q"]),
        now=lambda: CLOCK,
    )
    before = load_state(state_path)

    run_training(
        TrainOptions(state_path=state_path, resume=True, seed=999, l2=5.0),
        console=ScriptedConsole(["q"]),
        now=lambda: CLOCK,
    )
    after = load_state(state_path)
    assert after.selection == before.selection
    assert after.fitting == before.fitting
    assert after.selection.seed == 42
    assert after.fitting.l2 == 0.1


def test_data_change_is_rejected_on_resume(tmp_path: Path, data_csv: Path) -> None:
    state_path = tmp_path / "s.json"
    run_training(
        TrainOptions(state_path=state_path, data_path=data_csv),
        console=ScriptedConsole(["q"]),
        now=lambda: CLOCK,
    )
    before = state_path.read_text(encoding="utf-8")

    changed = [dict(item) for item in ROWS]
    changed[0]["nightly_price"] = 999
    other = write_csv(tmp_path / "changed.csv", changed)
    from hotel_model.state import StateError

    with pytest.raises(StateError, match="不一致"):
        run_training(
            TrainOptions(state_path=state_path, resume=True, data_path=other),
            console=ScriptedConsole(["a", "q"]),
            now=lambda: CLOCK,
        )
    assert state_path.read_text(encoding="utf-8") == before, "拒绝恢复时不得改写原文件"
