"""原始酒店 CSV 的读取与校验。

数据契约见 ``openspec/changes/train-hotel-preference/specs``：只接受原始字段，
不做缺失填补，并在创建训练状态前拒绝整份不合法数据。
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# 顺序即规格中列出的顺序，用于错误信息与状态快照。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "hotel_id",
    "name",
    "city",
    "grade",
    "renovation_age_years",
    "ctrip_rating",
    "subway_walk_minutes",
    "mall_transit_minutes",
    "attraction_taxi_minutes",
    "room_area_sqm",
    "nightly_price",
)

# 九档名称映射，1 至 9。
GRADE_NAMES: tuple[str, ...] = (
    "经济",
    "半舒适",
    "舒适",
    "半高档",
    "高档",
    "半豪华",
    "豪华",
    "半奢华",
    "奢华",
)

# 八项数值字段，即需要范围校验和有限性校验的字段。
NUMERIC_COLUMNS: tuple[str, ...] = (
    "grade",
    "renovation_age_years",
    "ctrip_rating",
    "subway_walk_minutes",
    "mall_transit_minutes",
    "attraction_taxi_minutes",
    "room_area_sqm",
    "nightly_price",
)

BOM = "\ufeff"


class DataError(ValueError):
    """原始数据不满足契约；消息包含可定位的行与字段。"""


@dataclass(frozen=True)
class Sample:
    """一条经过校验的原始样本，对应一个具体酒店房型。"""

    hotel_id: str
    name: str
    city: str
    grade: int
    renovation_age_years: float
    ctrip_rating: float
    subway_walk_minutes: float
    mall_transit_minutes: float
    attraction_taxi_minutes: float
    room_area_sqm: float
    nightly_price: float

    @property
    def grade_name(self) -> str:
        return grade_name(self.grade)


def grade_name(grade: int) -> str:
    """把 1 至 9 的整数档次映射为名称。"""

    if not isinstance(grade, int) or isinstance(grade, bool):
        raise DataError(f"档次必须是整数，收到 {grade!r}")
    if not 1 <= grade <= 9:
        raise DataError(f"档次必须在 1 至 9 之间，收到 {grade}")
    return GRADE_NAMES[grade - 1]


def _clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace(BOM, "").strip()


def _number(raw: str, *, field: str, line: int, required: bool) -> float | None:
    text = raw.strip()
    if text == "":
        if required:
            raise DataError(f"第 {line} 行字段 `{field}` 不能为空")
        return None
    try:
        value = float(text)
    except ValueError:
        raise DataError(f"第 {line} 行字段 `{field}` 不是数字：{raw!r}") from None
    if not math.isfinite(value):
        raise DataError(f"第 {line} 行字段 `{field}` 必须是有限数值，收到 {raw!r}")
    return value


def _check_ranges(values: Mapping[str, float], line: int) -> None:
    grade_value = values["grade"]
    if grade_value != int(grade_value):
        raise DataError(
            f"第 {line} 行字段 `grade` 必须是整数档次，收到 {grade_value!r}"
        )
    grade = int(grade_value)
    if not 1 <= grade <= 9:
        raise DataError(f"第 {line} 行字段 `grade` 必须在 1 至 9 之间，收到 {grade}")

    for field in ("renovation_age_years", "subway_walk_minutes",
                  "mall_transit_minutes", "attraction_taxi_minutes"):
        if values[field] < 0:
            raise DataError(f"第 {line} 行字段 `{field}` 不能为负：{values[field]!r}")

    rating = values["ctrip_rating"]
    if not 0.0 <= rating <= 5.0:
        raise DataError(f"第 {line} 行字段 `ctrip_rating` 必须在 0 至 5 之间，收到 {rating!r}")

    if values["room_area_sqm"] <= 0:
        raise DataError(
            f"第 {line} 行字段 `room_area_sqm` 必须大于零，收到 {values['room_area_sqm']!r}"
        )
    if values["nightly_price"] <= 0:
        raise DataError(
            f"第 {line} 行字段 `nightly_price` 必须大于零，收到 {values['nightly_price']!r}"
        )


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DataError(f"`{path}` 不是 UTF-8 编码的 CSV：{error}") from None


def load_csv_text(text: str, *, source: str = "<memory>") -> list[Sample]:
    """解析并校验 CSV 文本；任何不合法之处都抛出 :class:`DataError`。"""

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise DataError(f"`{source}` 为空文件，缺少必需列：{', '.join(REQUIRED_COLUMNS)}") from None

    columns = [_clean_cell(cell) for cell in header]
    seen: dict[str, int] = {}
    for index, column in enumerate(columns):
        if column == "":
            raise DataError(f"`{source}` 表头第 {index + 1} 列名为空")
        if column in seen:
            raise DataError(f"`{source}` 表头存在重复列 `{column}`")
        seen[column] = index

    missing = [column for column in REQUIRED_COLUMNS if column not in seen]
    if missing:
        raise DataError(
            f"`{source}` 缺少必需列：{', '.join(missing)}"
        )

    samples: list[Sample] = []
    by_id: dict[str, int] = {}
    for offset, row in enumerate(reader):
        line = offset + 2  # 表头占第 1 行
        if not row or all(_clean_cell(cell) == "" for cell in row):
            continue
        if len(row) != len(columns):
            raise DataError(
                f"第 {line} 行有 {len(row)} 个字段，与表头 {len(columns)} 列不一致"
            )

        cells = {column: _clean_cell(row[seen[column]]) for column in columns}
        for field in ("hotel_id", "name", "city"):
            if cells[field] == "":
                raise DataError(f"第 {line} 行字段 `{field}` 不能为空")

        values: dict[str, float] = {}
        for field in NUMERIC_COLUMNS:
            number = _number(cells[field], field=field, line=line, required=True)
            assert number is not None  # required=True 时不会返回 None
            values[field] = number
        _check_ranges(values, line)

        hotel_id = cells["hotel_id"]
        if hotel_id in by_id:
            raise DataError(
                f"第 {line} 行字段 `hotel_id` 重复：{hotel_id!r} 首次出现在第 {by_id[hotel_id]} 行"
            )
        by_id[hotel_id] = line

        samples.append(
            Sample(
                hotel_id=hotel_id,
                name=cells["name"],
                city=cells["city"],
                grade=int(values["grade"]),
                renovation_age_years=values["renovation_age_years"],
                ctrip_rating=values["ctrip_rating"],
                subway_walk_minutes=values["subway_walk_minutes"],
                mall_transit_minutes=values["mall_transit_minutes"],
                attraction_taxi_minutes=values["attraction_taxi_minutes"],
                room_area_sqm=values["room_area_sqm"],
                nightly_price=values["nightly_price"],
            )
        )

    if not samples:
        raise DataError(f"`{source}` 没有数据行")
    return samples


def load_csv(path: str | Path) -> list[Sample]:
    """读取并校验一个 UTF-8（可带 BOM）CSV 文件。"""

    resolved = Path(path)
    if not resolved.is_file():
        raise DataError(f"找不到数据文件 `{resolved}`")
    return load_csv_text(_decode(resolved), source=str(resolved))


def samples_from_ids(pool: Sequence[Sample], hotel_ids: Iterable[str]) -> list[Sample]:
    """按给定 ID 顺序取出样本，缺失即报错。"""

    index = {sample.hotel_id: sample for sample in pool}
    picked: list[Sample] = []
    for hotel_id in hotel_ids:
        if hotel_id not in index:
            raise DataError(f"样本 ID `{hotel_id}` 不在当前数据中")
        picked.append(index[hotel_id])
    return picked


def sample_to_dict(sample: Sample) -> dict[str, object]:
    return {
        "hotel_id": sample.hotel_id,
        "name": sample.name,
        "city": sample.city,
        "grade": sample.grade,
        "renovation_age_years": sample.renovation_age_years,
        "ctrip_rating": sample.ctrip_rating,
        "subway_walk_minutes": sample.subway_walk_minutes,
        "mall_transit_minutes": sample.mall_transit_minutes,
        "attraction_taxi_minutes": sample.attraction_taxi_minutes,
        "room_area_sqm": sample.room_area_sqm,
        "nightly_price": sample.nightly_price,
    }


def sample_from_dict(payload: Mapping[str, object]) -> Sample:
    """从状态快照恢复样本，并复用同一套范围校验。"""

    def text(field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or value.strip() == "":
            raise DataError(f"状态快照字段 `{field}` 必须是非空字符串")
        return value

    def number(field: str) -> float:
        value = payload.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise DataError(f"状态快照字段 `{field}` 必须是数值")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise DataError(f"状态快照字段 `{field}` 必须是有限数值")
        return as_float

    values = {field: number(field) for field in NUMERIC_COLUMNS}
    try:
        _check_ranges(values, 0)
        grade = int(values["grade"])
    except DataError as error:
        raise DataError(f"状态快照样本不合法：{error}") from None

    return Sample(
        hotel_id=text("hotel_id"),
        name=text("name"),
        city=text("city"),
        grade=grade,
        renovation_age_years=values["renovation_age_years"],
        ctrip_rating=values["ctrip_rating"],
        subway_walk_minutes=values["subway_walk_minutes"],
        mall_transit_minutes=values["mall_transit_minutes"],
        attraction_taxi_minutes=values["attraction_taxi_minutes"],
        room_area_sqm=values["room_area_sqm"],
        nightly_price=values["nightly_price"],
    )
