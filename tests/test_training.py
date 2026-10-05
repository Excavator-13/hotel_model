"""终端训练循环、命令参数与诊断测试（任务 5.1–5.2）。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from hotel_model import train as train_module
from hotel_model.cli import main, options_from_args, build_parser
from hotel_model.data import DataError
from hotel_model.diagnostics import RAW_FIELD_LABELS, format_diagnostics, format_sample, summarize
from hotel_model.fitting import FitResult
from hotel_model.state import StateError, load_state
from hotel_model.train import (
    ANSWER_CHOICES,
    PROMPT,
    REPROMPT,
    ScriptedConsole,
    TrainOptions,
    run_training,
)

from .conftest import make_samples, row, write_csv


def fixed_clock() -> str:
    return "2024-05-01T12:00:00+00:00"


def options(tmp_path: Path, *, data: Path | None = None, **overrides: object) -> TrainOptions:
    values: dict[str, object] = {
        "state_path": tmp_path / "session.json",
        "data_path": data,
    }
    values.update(overrides)
    return TrainOptions(**values)  # type: ignore[arg-type]


@pytest.fixture
def hz_city_csv(tmp_path: Path) -> Path:
    rows = [
        row("hz-1", city="杭州", grade=3, renovation_age_years=2, ctrip_rating=4.7,
            subway_walk_minutes=6, nightly_price=300, room_area_sqm=25),
        row("hz-2", city="杭州", grade=4, renovation_age_years=6, ctrip_rating=4.6,
            subway_walk_minutes=12, nightly_price=360, room_area_sqm=30),
        row("hz-3", city="杭州", grade=5, renovation_age_years=3, ctrip_rating=4.8,
            subway_walk_minutes=9, nightly_price=480, room_area_sqm=35),
        row("hz-4", city="杭州", grade=4, renovation_age_years=12, ctrip_rating=4.5,
            subway_walk_minutes=3, nightly_price=260, room_area_sqm=28),
    ]
    return write_csv(tmp_path / "hotels.csv", rows)


# --- 5.1 参数校验与训练循环 -------------------------------------------------


def test_answer_choice_table() -> None:
    assert set(ANSWER_CHOICES) == {"a", "b", "t", "s", "q"}
    assert "A更好" in ANSWER_CHOICES["a"].replace(" ", "") or "A 更好" == ANSWER_CHOICES["a"]
    assert "退出" in ANSWER_CHOICES["q"]


def test_new_session_requires_data_and_state(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(StateError, match="--state"):
        options_from_args(parser.parse_args(["train", "--data", "x.csv"]))
    with pytest.raises(StateError, match="--data"):
        options_from_args(parser.parse_args(["train", "--state", "s.json"]))


def test_resume_requires_state(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(StateError, match="--state|状态文件"):
        options_from_args(parser.parse_args(["train", "--resume"]))
    parsed = options_from_args(parser.parse_args(["train", "--resume", "--state", "s.json"]))
    assert parsed.resume
    assert parsed.data_path is None


def test_resume_with_data_is_optional(tmp_path: Path) -> None:
    parser = build_parser()
    parsed = options_from_args(
        parser.parse_args(["train", "--resume", "--state", "s.json", "--data", "hotels.csv"])
    )
    assert parsed.resume
    assert parsed.data_path == Path("hotels.csv")


def test_new_session_refuses_to_overwrite(tmp_path: Path, hz_city_csv: Path) -> None:
    state_path = tmp_path / "session.json"
    state_path.write_text("{}", encoding="utf-8")
    console = ScriptedConsole(["q"])
    with pytest.raises(StateError, match="已存在"):
        run_training(
            TrainOptions(state_path=state_path, data_path=hz_city_csv),
            console=console,
        )
    assert state_path.read_text(encoding="utf-8") == "{}"


def test_display_shows_all_raw_fields_and_units(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["q"])
    run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    output = console.output
    assert "A 候选" in output and "B 候选" in output
    for _, label in RAW_FIELD_LABELS:
        assert label in output, f"缺少字段展示：{label}"
    assert "（" in output and "）" in output, "字段应带单位"
    assert "half高档" not in output


def test_only_same_city_pairs_are_asked(tmp_path: Path, two_city_csv: Path) -> None:
    console = ScriptedConsole(["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=two_city_csv),
        console=console,
        now=fixed_clock,
    )
    cities = {sample.hotel_id: sample.city for sample in outcome.state.samples}
    for event in outcome.state.events:
        assert cities[event["left_id"]] == cities[event["right_id"]]


@pytest.mark.parametrize("answer", ["a", "b", "t", "s"])
def test_each_answer_recorded_with_expected_label(tmp_path: Path, hz_city_csv: Path, answer: str) -> None:
    console = ScriptedConsole([answer, "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert len(outcome.state.events) == 1
    event = outcome.state.events[0]
    expected = {"a": 1.0, "b": 0.0, "t": 0.5, "s": None}[answer]
    assert event["answer"] == answer
    assert event["label"] == expected
    assert event["kind"] == ("skip" if answer == "s" else "label")


@pytest.mark.parametrize("answer", [" A ", "A", "a "])
def test_answer_input_is_normalized(tmp_path: Path, hz_city_csv: Path, answer: str) -> None:
    console = ScriptedConsole([answer, "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert len(outcome.state.events) == 1
    assert outcome.state.events[0]["label"] == 1.0


def test_invalid_input_keeps_question_and_records_nothing(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["x", "maybe", "", "b", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert console.output.count(REPROMPT) == 3
    assert len(outcome.state.events) == 1
    assert outcome.state.events[0]["label"] == 0.0


def test_invalid_input_does_not_change_pending_question(tmp_path: Path, hz_city_csv: Path) -> None:
    first = ScriptedConsole(["zzz", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=first,
        now=fixed_clock,
    )
    pending = outcome.state.pending
    assert pending is not None
    assert outcome.state.events == ()


def test_q_does_not_create_label_and_keeps_pending(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert outcome.stop_reason == "quit"
    assert outcome.state.events == ()
    assert outcome.state.pending is not None
    assert not outcome.state.model.trained


def test_eof_stops_without_answer(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole([])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert outcome.stop_reason == "eof"
    assert outcome.state.events == ()
    assert outcome.state.pending is not None


class _InterruptingConsole:
    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self.lines.append(text)

    def read_line(self, prompt: str) -> str | None:
        if not self._answers:
            raise KeyboardInterrupt
        return self._answers.pop(0)

    @property
    def output(self) -> str:
        return "\n".join(self.lines)


def test_ctrl_c_does_not_create_answer(tmp_path: Path, hz_city_csv: Path) -> None:
    console = _InterruptingConsole(["a", "b"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert outcome.stop_reason == "interrupted"
    assert len(outcome.state.events) == 2
    assert outcome.state.pending is not None


def test_candidate_exhaustion_reports_and_stops(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "two.csv",
        [
            row("hz-1", city="杭州", grade=3, nightly_price=300),
            row("hz-2", city="杭州", grade=5, nightly_price=500),
        ],
    )
    console = ScriptedConsole(["a", "a", "a"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=csv_path),
        console=console,
        now=fixed_clock,
    )
    assert outcome.stop_reason == "exhausted"
    assert "没有可比较的候选" in console.output
    assert len(outcome.state.events) == 1
    assert outcome.state.pending is None


def test_single_sample_city_reports_no_candidate(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "single.csv",
        [row("hz-1", city="杭州", grade=3, nightly_price=300)],
    )
    console = ScriptedConsole([])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=csv_path),
        console=console,
        now=fixed_clock,
    )
    assert outcome.stop_reason == "exhausted"
    assert "没有可比较的候选" in console.output


def test_identical_feature_pairs_exhaust_candidates(tmp_path: Path) -> None:
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
    csv_path = write_csv(
        tmp_path / "same.csv",
        [row("hz-1", city="杭州", **same), row("hz-2", city="杭州", **same)],
    )
    console = ScriptedConsole([])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=csv_path),
        console=console,
        now=fixed_clock,
    )
    assert outcome.stop_reason == "exhausted"
    assert "没有可比较的候选" in console.output


def test_invalid_data_rejected_before_state_created(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "bad.csv",
        [row("hz-1", city="杭州", nightly_price=-5)],
    )
    state_path = tmp_path / "s.json"
    console = ScriptedConsole([])
    with pytest.raises(DataError):
        run_training(TrainOptions(state_path=state_path, data_path=csv_path), console=console)
    assert not state_path.exists()


def test_prompt_text_lists_all_choices(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["q"])
    run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    assert PROMPT in console.output


def test_new_session_reports_configuration(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["q"])
    run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv, seed=7, l2=0.25),
        console=console,
        now=fixed_clock,
    )
    assert "种子 7" in console.output
    assert "L2=0.25" in console.output


def test_cli_returns_error_code_for_bad_arguments(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main(["train", "--state", str(tmp_path / "s.json")], stdout=out, stderr=err)
    assert code == 2
    assert "参数错误" in err.getvalue()


def test_cli_prints_help_without_command() -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main([], stdout=out, stderr=err)
    assert code == 2
    assert "train" in out.getvalue()


def test_cli_new_then_resume(tmp_path: Path, hz_city_csv: Path) -> None:
    state_path = tmp_path / "s.json"
    out1, err1 = io.StringIO(), io.StringIO()
    code = main(
        ["train", "--data", str(hz_city_csv), "--state", str(state_path), "--quiet"],
        stdin=io.StringIO("q\n"),
        stdout=out1,
        stderr=err1,
    )
    assert code == 0, err1.getvalue()
    assert state_path.is_file()

    out2, err2 = io.StringIO(), io.StringIO()
    code = main(
        ["train", "--resume", "--state", str(state_path), "--quiet"],
        stdin=io.StringIO("a\nq\n"),
        stdout=out2,
        stderr=err2,
    )
    assert code == 0, err2.getvalue()
    assert len(load_state(state_path).events) == 1


def test_cli_reports_missing_data(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["train", "--data", str(tmp_path / "nope.csv"), "--state", str(tmp_path / "s.json")],
        stdin=io.StringIO(""),
        stdout=out,
        stderr=err,
    )
    assert code == 2
    assert "训练中止" in err.getvalue()


# --- 5.2 诊断 ---------------------------------------------------------------


def test_diagnostics_with_only_skips(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["s", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    output = console.output
    assert "未训练" in output
    assert "跳过=1" in output
    assert "训练交叉熵" not in output
    assert "权重" not in output, "未训练时不应展示权重"
    assert "有效比较=0" in output


def test_diagnostics_counts_after_tie_and_strict(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["a", "t", "s", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    output = console.output
    assert outcome.state.model.trained
    assert "有效比较=2" in output
    assert "严格偏好=1" in output
    assert "平局=1" in output
    assert "跳过=1" in output
    assert "训练交叉熵=" in output
    assert "不代表校准或泛化精度" in output


def test_diagnostics_lists_weights_in_feature_order(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv),
        console=console,
        now=fixed_clock,
    )
    output = console.output
    assert "权重（顺序：grade, renovation, rating, subway, mall, attraction, area, price）" in output
    for name in ("grade", "renovation", "rating", "subway", "mall", "attraction", "area", "price"):
        assert f"{name}=" in output


def test_diagnostics_reports_city_coverage(tmp_path: Path, two_city_csv: Path) -> None:
    console = ScriptedConsole(["a", "a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=two_city_csv),
        console=console,
        now=fixed_clock,
    )
    output = console.output
    assert "按城市有效标签" in output
    assert "杭州" in output or "成都" in output


def test_diagnostics_flags_feature_without_comparison_information(tmp_path: Path) -> None:
    """所有比较中档次相同，应提示档次缺少比较信息。"""

    csv_path = write_csv(
        tmp_path / "same_grade.csv",
        [
            row("hz-1", city="杭州", grade=4, ctrip_rating=4.6, nightly_price=300),
            row("hz-2", city="杭州", grade=4, ctrip_rating=4.8, nightly_price=300),
        ],
    )
    console = ScriptedConsole(["a", "q"])
    outcome = run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=csv_path),
        console=console,
        now=fixed_clock,
    )
    output = console.output
    assert "缺少比较信息的特征" in output
    assert "grade" in output
    assert "不代表用户不在意" in output


def test_diagnostics_quiet_suppresses_output(tmp_path: Path, hz_city_csv: Path) -> None:
    console = ScriptedConsole(["a", "q"])
    run_training(
        TrainOptions(state_path=tmp_path / "s.json", data_path=hz_city_csv, show_diagnostics=False),
        console=console,
        now=fixed_clock,
    )
    assert "训练诊断" not in console.output


def test_format_sample_includes_grade_name_and_units() -> None:
    sample = make_samples([row("hz-1", grade=4, renovation_age_years=2.5, ctrip_rating=4.7)])[0]
    text = format_sample(sample)
    assert "半高档" in text
    assert "装修年数（年）=2.5" in text
    assert "评分（0-5）=4.7" in text
    assert "每晚价格（元）" in text
    assert "地铁步行（分钟）" in text
    assert "面积（平方米）" in text


def test_summarize_counts_and_uninformative_features() -> None:
    from hotel_model.features import FeatureConfig, to_features
    from hotel_model.fitting import FittingConfig
    from hotel_model.select import SelectionConfig
    from hotel_model.state import ModelSnapshot, TrainingState

    samples = make_samples(
        [
            row("hz-1", city="杭州", grade=4, ctrip_rating=4.6, nightly_price=300),
            row("hz-2", city="杭州", grade=4, ctrip_rating=4.8, nightly_price=300),
        ]
    )
    state = TrainingState.new(
        samples,
        preprocessing=FeatureConfig(),
        fitting=FittingConfig(),
        selection=SelectionConfig(),
        random_state={
            "bit_generator": "PCG64",
            "state": {"state": 1, "inc": 2},
            "has_uint32": 0,
            "uinteger": 0,
        },
        created_at="2024-01-01T00:00:00+00:00",
    )
    event = {
        "left_id": "hz-1",
        "right_id": "hz-2",
        "city": "杭州",
        "kind": "label",
        "label": 1.0,
        "answered_at": "2024-01-01T00:01:00+00:00",
    }
    state = state.with_(events=[event], model=ModelSnapshot(True, np.ones(8) * 0.2, 1, 0.3, 0))
    vectors = {sample.hotel_id: to_features(sample) for sample in samples}
    counts = summarize(state, vectors)
    assert counts.valid_labels == 1
    assert counts.by_city == {"杭州": 1}
    assert "grade" in counts.uninformative_features
    lines = format_diagnostics(state, counts)
    assert any("训练交叉熵" in line for line in lines)
