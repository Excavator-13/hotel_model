"""README、CSV 模板与虚构示例数据的核对测试（任务 6.1）。"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from hotel_model.data import REQUIRED_COLUMNS, load_csv
from hotel_model.features import FEATURE_NAMES
from hotel_model.select import build_candidates
from hotel_model.features import to_features

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TEMPLATE = ROOT / "examples" / "hotels_template.csv"
EXAMPLE = ROOT / "examples" / "hotels_example.csv"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists(readme_text: str) -> None:
    assert readme_text.startswith("# hotel-model")


def test_template_has_header_and_no_rows() -> None:
    lines = [line for line in TEMPLATE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].split(",") == list(REQUIRED_COLUMNS)


def test_template_is_readable_as_empty_shape() -> None:
    # 空模板只有表头，读取时应报告没有数据行，而不是解析错误。
    from hotel_model.data import DataError, load_csv_text

    with pytest.raises(DataError, match="没有数据行"):
        load_csv_text(TEMPLATE.read_text(encoding="utf-8"))


def test_example_data_is_readable() -> None:
    samples = load_csv(EXAMPLE)
    assert len(samples) >= 30
    cities = Counter(sample.city for sample in samples)
    assert len(cities) >= 2
    assert all(count >= 2 for count in cities.values())


def test_example_data_covers_core_grades() -> None:
    samples = load_csv(EXAMPLE)
    grades = {sample.grade for sample in samples}
    # 采样建议以舒适、半高档、高档为核心，示例应覆盖这三档。
    assert {3, 4, 5} <= grades
    assert all(1 <= grade <= 9 for grade in grades)


def test_example_data_produces_candidates_with_tradeoffs() -> None:
    samples = load_csv(EXAMPLE)
    vectors = {sample.hotel_id: to_features(sample) for sample in samples}
    candidates = build_candidates(samples, vectors, [])
    assert len(candidates) > 50, "示例应产生足够的同城候选"
    from hotel_model.select import is_tradeoff

    assert any(is_tradeoff(candidate.difference) for candidate in candidates)


def test_example_data_is_labelled_fictional() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "虚构" in text


def test_readme_documents_units_and_formulas(readme_text: str) -> None:
    for column in REQUIRED_COLUMNS:
        assert column in readme_text, f"README 未说明字段 {column}"
    assert "平方米" in readme_text
    assert "人民币" in readme_text or "元" in readme_text
    assert "分钟" in readme_text
    # v1 公式的每个特征都应出现。
    for name in FEATURE_NAMES:
        assert name in readme_text, f"README 未说明特征 {name}"
    assert "-log(nightly_price / 300)" in readme_text
    assert "log(room_area_sqm / 25)" in readme_text
    assert "(ctrip_rating - 4.7) / 0.2" in readme_text
    assert "min(renovation_age_years, 15) / 5" in readme_text


def test_readme_documents_commands_and_answers(readme_text: str) -> None:
    assert "--data" in readme_text and "--state" in readme_text
    assert "--resume" in readme_text
    assert "--seed" in readme_text and "42" in readme_text
    assert "--l2" in readme_text
    for token in ("`a`", "`b`", "`t`", "`s`", "`q`"):
        assert token in readme_text, f"README 未说明回答 {token}"


def test_documented_commands_actually_parse(readme_text: str) -> None:
    """README 中出现的每条 hotel-model 命令都必须能被 CLI 解析。"""

    from hotel_model.cli import build_parser

    parser = build_parser()
    commands = re.findall(r"hotel-model\s+train[^\n`]*", readme_text)
    assert commands, "README 应包含可执行命令"
    checked = 0
    for command in commands:
        stripped = command.strip()
        if stripped.endswith("--help"):
            continue
        tokens = stripped.split()
        # 不应触发 SystemExit（参数非法会退出码 2）。
        parsed = parser.parse_args(tokens[1:])
        assert parsed.command == "train"
        checked += 1
    assert checked >= 3, f"应核对多条命令，实际 {checked}"


def test_readme_documents_comparison_conditions(readme_text: str) -> None:
    assert "同一房型" in readme_text
    assert "退订" in readme_text
    assert "早餐" in readme_text
    assert "目标景点" in readme_text


def test_readme_sampling_advice_is_advisory_and_three_tier_centred(readme_text: str) -> None:
    assert "16" in readme_text
    assert "不是程序限制" in readme_text
    assert "不强制档次配额" in readme_text or "不是程序限制" in readme_text
    for tier in ("舒适", "半高档", "高档"):
        assert tier in readme_text


def test_readme_documents_single_state_single_process(readme_text: str) -> None:
    assert "一个状态文件只运行一个训练进程" in readme_text


def test_readme_states_nine_grade_and_cross_city_limits(readme_text: str) -> None:
    assert "九档等距假设未经检验" in readme_text
    assert "跨城市迁移尚未验证" in readme_text
    assert "不是校准度、验证误差或泛化精度" in readme_text
    assert "不代表" in readme_text and "不在意" in readme_text


def test_readme_has_no_scoring_or_web_entry(readme_text: str) -> None:
    assert "没有" in readme_text
    assert "评分" in readme_text
    # 不应存在评分/推荐命令示例。
    for forbidden in ("hotel-model score", "hotel-model rank", "hotel-model serve", "hotel-model web"):
        assert forbidden not in readme_text


def test_repo_has_no_scoring_or_web_command() -> None:
    import io

    from hotel_model.cli import build_parser

    out = io.StringIO()
    build_parser().print_help(out)
    help_text = out.getvalue()
    # 顶层 help 只提到 train 这一个入口。
    assert "train" in help_text
    for forbidden in ("score", "rank", "recommend", "serve", "web"):
        assert forbidden not in help_text.lower()
    source = (ROOT / "src" / "hotel_model" / "cli.py").read_text(encoding="utf-8")
    for forbidden in ("score", "recommend", "serve", "http"):
        assert forbidden not in source.lower()


def test_parser_exposes_only_train_subcommand() -> None:
    import argparse

    from hotel_model.cli import build_parser

    parser = build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers) == 1
    assert set(subparsers[0].choices) == {"train"}


def test_project_declares_single_console_script() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" in pyproject
    scripts_section = pyproject.split("[project.scripts]", 1)[1].split("[", 1)[0]
    # 只暴露训练入口这一个控制台脚本。
    entry_points = [line for line in scripts_section.splitlines() if "=" in line]
    assert len(entry_points) == 1
    assert entry_points[0].strip().startswith("hotel-model")
    assert "hotel_model.cli:main" in entry_points[0]


def test_modules_under_src_are_training_only() -> None:
    modules = sorted(path.name for path in (ROOT / "src" / "hotel_model").glob("*.py"))
    assert modules == [
        "__init__.py",
        "cli.py",
        "data.py",
        "diagnostics.py",
        "features.py",
        "fitting.py",
        "select.py",
        "state.py",
        "train.py",
    ]
