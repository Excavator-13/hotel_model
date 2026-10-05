"""原始 CSV 数据契约测试（任务 1.2）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hotel_model.data import (
    GRADE_NAMES,
    DataError,
    grade_name,
    load_csv,
    load_csv_text,
)

from .conftest import BASE_ROW, HEADER, csv_text, row


def test_reads_valid_mixed_types() -> None:
    samples = load_csv_text(csv_text([BASE_ROW]))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.hotel_id == "hz-01"
    assert sample.grade == 3
    assert sample.grade_name == "舒适"
    assert sample.renovation_age_years == 2.5
    assert sample.ctrip_rating == 4.7
    assert sample.nightly_price == 320


def test_half_grade_mapping() -> None:
    samples = load_csv_text(
        csv_text(
            [
                row("a", grade=4, renovation_age_years=2.5, ctrip_rating=4.7),
            ]
        )
    )
    assert samples[0].grade_name == "半高档"
    assert samples[0].renovation_age_years == 2.5
    assert samples[0].ctrip_rating == 4.7


def test_all_nine_grade_names() -> None:
    assert len(GRADE_NAMES) == 9
    assert GRADE_NAMES[0] == "经济"
    assert GRADE_NAMES[-1] == "奢华"
    assert grade_name(1) == "经济"
    assert grade_name(9) == "奢华"
    for value in range(1, 10):
        assert grade_name(value) == GRADE_NAMES[value - 1]


def test_utf8_bom_and_whitespace_tolerated() -> None:
    text = csv_text([row(" b ", name=" 示例 ", city=" 杭州 ")], bom=True)
    samples = load_csv_text(text)
    assert samples[0].hotel_id == "b"
    assert samples[0].name == "示例"
    assert samples[0].city == "杭州"


def test_file_loader_reads_bom_file(tmp_path: Path) -> None:
    path = tmp_path / "hotels.csv"
    path.write_text(csv_text([BASE_ROW], bom=True), encoding="utf-8")
    assert load_csv(path)[0].hotel_id == "hz-01"


def test_file_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="找不到数据文件"):
        load_csv(tmp_path / "nope.csv")


def test_empty_file_rejected() -> None:
    with pytest.raises(DataError, match="缺少必需列|为空文件"):
        load_csv_text("")


def test_missing_column_rejected() -> None:
    columns = [column for column in HEADER if column != "nightly_price"]
    with pytest.raises(DataError, match="nightly_price"):
        load_csv_text(csv_text([BASE_ROW], columns=columns))


def test_duplicate_column_rejected() -> None:
    columns = list(HEADER) + ["city"]
    with pytest.raises(DataError, match="重复列 `city`"):
        load_csv_text(csv_text([BASE_ROW], columns=columns))


def test_duplicate_id_reports_first_line() -> None:
    rows = [row("hz-01"), row("hz-01")]
    with pytest.raises(DataError) as error:
        load_csv_text(csv_text(rows))
    message = str(error.value)
    assert "第 3 行" in message
    assert "重复" in message
    assert "第 2 行" in message


@pytest.mark.parametrize("field", ["hotel_id", "name", "city"])
def test_empty_identity_rejected(field: str) -> None:
    record = row("hz-01")
    if field == "hotel_id":
        record["hotel_id"] = "  "
    else:
        record[field] = "  "
    with pytest.raises(DataError, match=field):
        load_csv_text(csv_text([record]))


@pytest.mark.parametrize("field", ["grade", "nightly_price", "ctrip_rating"])
def test_empty_number_rejected(field: str) -> None:
    with pytest.raises(DataError) as error:
        load_csv_text(csv_text([row("hz-01", **{field: ""})]))
    assert field in str(error.value)
    assert "第 2 行" in str(error.value)


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "infinity"])
def test_non_finite_rejected(value: str) -> None:
    with pytest.raises(DataError) as error:
        load_csv_text(csv_text([row("hz-01", nightly_price=value)]))
    assert "nightly_price" in str(error.value)
    assert "第 2 行" in str(error.value)


def test_non_numeric_rejected() -> None:
    with pytest.raises(DataError, match="不是数字"):
        load_csv_text(csv_text([row("hz-01", grade="四星")]))


@pytest.mark.parametrize("value", ["3.5", "0", "10", "-1"])
def test_non_integer_or_out_of_range_grade(value: str) -> None:
    with pytest.raises(DataError, match="grade"):
        load_csv_text(csv_text([row("hz-01", grade=value)]))


@pytest.mark.parametrize("value", ["-0.1", "5.1"])
def test_rating_out_of_range(value: str) -> None:
    with pytest.raises(DataError, match="ctrip_rating"):
        load_csv_text(csv_text([row("hz-01", ctrip_rating=value)]))


@pytest.mark.parametrize(
    "field",
    ["renovation_age_years", "subway_walk_minutes", "mall_transit_minutes", "attraction_taxi_minutes"],
)
def test_negative_minutes_rejected(field: str) -> None:
    with pytest.raises(DataError, match=field):
        load_csv_text(csv_text([row("hz-01", **{field: "-1"})]))


@pytest.mark.parametrize("field", ["room_area_sqm", "nightly_price"])
def test_non_positive_size_or_price_rejected(field: str) -> None:
    with pytest.raises(DataError, match=field):
        load_csv_text(csv_text([row("hz-01", **{field: "0"})]))


def test_zero_minutes_and_zero_rating_accepted() -> None:
    samples = load_csv_text(
        csv_text(
            [
                row(
                    "hz-01",
                    subway_walk_minutes=0,
                    mall_transit_minutes=0,
                    attraction_taxi_minutes=0,
                    ctrip_rating=0,
                    renovation_age_years=0,
                )
            ]
        )
    )
    assert samples[0].subway_walk_minutes == 0.0
    assert samples[0].ctrip_rating == 0.0


def test_extra_columns_ignored() -> None:
    columns = list(HEADER) + ["备注", "review_count"]
    extra = dict(BASE_ROW, **{"备注": "虚构", "review_count": "999"})
    samples = load_csv_text(csv_text([extra], columns=columns))
    assert len(samples) == 1
    assert not hasattr(samples[0], "review_count")


def test_extra_columns_in_any_position() -> None:
    columns = ["备注"] + list(HEADER)
    extra = dict(BASE_ROW, **{"备注": "虚构"})
    samples = load_csv_text(csv_text([extra], columns=columns))
    assert samples[0].nightly_price == 320


def test_ragged_row_rejected() -> None:
    text = "".join(
        [
            ",".join(HEADER) + "\n",
            "hz-01,只写了两列\n",
        ]
    )
    with pytest.raises(DataError, match="不一致"):
        load_csv_text(text)


def test_blank_lines_skipped() -> None:
    text = ",".join(HEADER) + "\n" + ",".join(str(BASE_ROW[c]) for c in HEADER) + "\n\n"
    assert len(load_csv_text(text)) == 1


def test_header_only_rejected() -> None:
    with pytest.raises(DataError, match="没有数据行"):
        load_csv_text(csv_text([]))


def test_unclosed_quote_reports_line() -> None:
    text = ",".join(HEADER) + '\nhz-01,"未闭合\n'
    with pytest.raises(DataError):
        load_csv_text(text)
