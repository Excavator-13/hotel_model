"""共享测试夹具：原始 CSV 构造与标准样本池。"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest

from hotel_model.data import REQUIRED_COLUMNS, Sample

HEADER = list(REQUIRED_COLUMNS)

BASE_ROW: dict[str, object] = {
    "hotel_id": "hz-01",
    "name": "示例舒适酒店",
    "city": "杭州",
    "grade": 3,
    "renovation_age_years": 2.5,
    "ctrip_rating": 4.7,
    "subway_walk_minutes": 6,
    "mall_transit_minutes": 15,
    "attraction_taxi_minutes": 22,
    "room_area_sqm": 25,
    "nightly_price": 320,
}


def csv_text(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str] = HEADER,
    bom: bool = False,
) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(column, "") for column in columns])
    text = buffer.getvalue()
    return ("\ufeff" + text) if bom else text


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], **kwargs: object) -> Path:
    path.write_text(csv_text(rows, **kwargs), encoding="utf-8")
    return path


def row(hotel_id: str, **overrides: object) -> dict[str, object]:
    record = dict(BASE_ROW)
    record["hotel_id"] = hotel_id
    record["name"] = f"示例酒店 {hotel_id}"
    record.update(overrides)
    return record


@pytest.fixture
def hangzhou_rows() -> list[dict[str, object]]:
    """杭州 4 家：档次、新旧、价格彼此有取舍。"""

    return [
        row("hz-01", city="杭州", grade=3, renovation_age_years=2, ctrip_rating=4.7,
            subway_walk_minutes=6, mall_transit_minutes=15, attraction_taxi_minutes=22,
            room_area_sqm=25, nightly_price=320),
        row("hz-02", city="杭州", grade=4, renovation_age_years=5, ctrip_rating=4.6,
            subway_walk_minutes=12, mall_transit_minutes=8, attraction_taxi_minutes=18,
            room_area_sqm=30, nightly_price=380),
        row("hz-03", city="杭州", grade=5, renovation_age_years=3, ctrip_rating=4.8,
            subway_walk_minutes=9, mall_transit_minutes=20, attraction_taxi_minutes=25,
            room_area_sqm=35, nightly_price=480),
        row("hz-04", city="杭州", grade=4, renovation_age_years=12, ctrip_rating=4.5,
            subway_walk_minutes=3, mall_transit_minutes=25, attraction_taxi_minutes=30,
            room_area_sqm=28, nightly_price=260),
    ]


@pytest.fixture
def chengdu_rows() -> list[dict[str, object]]:
    return [
        row("cd-01", city="成都", grade=3, renovation_age_years=1, ctrip_rating=4.7,
            subway_walk_minutes=4, mall_transit_minutes=10, attraction_taxi_minutes=15,
            room_area_sqm=26, nightly_price=300),
        row("cd-02", city="成都", grade=5, renovation_age_years=6, ctrip_rating=4.8,
            subway_walk_minutes=14, mall_transit_minutes=22, attraction_taxi_minutes=28,
            room_area_sqm=38, nightly_price=520),
    ]


@pytest.fixture
def two_city_rows(
    hangzhou_rows: list[dict[str, object]],
    chengdu_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return list(hangzhou_rows) + list(chengdu_rows)


@pytest.fixture
def two_city_csv(tmp_path: Path, two_city_rows: list[dict[str, object]]) -> Path:
    return write_csv(tmp_path / "hotels.csv", two_city_rows)


@pytest.fixture
def samples(hangzhou_rows: list[dict[str, object]]) -> list[Sample]:
    from hotel_model.data import load_csv_text

    return load_csv_text(csv_text(hangzhou_rows))


def make_samples(rows: Iterable[Mapping[str, object]]) -> list[Sample]:
    from hotel_model.data import load_csv_text

    return load_csv_text(csv_text(list(rows)))
