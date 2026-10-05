"""固定八维 v1 特征变换。

规格要求不做逐城市或逐样本池归一化，因此这里的变换只依赖单条样本的原始数值。
版本、顺序和常数随训练状态一起保存。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import Sample

FEATURE_NAMES: tuple[str, ...] = (
    "grade",
    "renovation",
    "rating",
    "subway",
    "mall",
    "attraction",
    "area",
    "price",
)

FEATURE_COUNT = len(FEATURE_NAMES)
PREPROCESSING_VERSION = "v1"

# v1 常数：价格与面积中心、评分中心与尺度、时间的 5 分钟尺度、装修封顶 15 年。
FEATURE_CONSTANTS: dict[str, float] = {
    "grade_center": 4.0,
    "renovation_cap_years": 15.0,
    "renovation_scale_years": 5.0,
    "rating_center": 4.7,
    "rating_scale": 0.2,
    "minutes_scale": 5.0,
    "area_center_sqm": 25.0,
    "price_center": 300.0,
}


@dataclass(frozen=True)
class FeatureConfig:
    """预处理配置；目前只有 v1 一种取值。"""

    version: str = PREPROCESSING_VERSION
    names: tuple[str, ...] = FEATURE_NAMES
    constants: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.version != PREPROCESSING_VERSION:
            raise ValueError(f"不支持的预处理版本：{self.version!r}")
        if tuple(self.names) != FEATURE_NAMES:
            raise ValueError("特征顺序与 v1 不一致")
        constants = dict(FEATURE_CONSTANTS if self.constants is None else self.constants)
        if constants != FEATURE_CONSTANTS:
            raise ValueError("v1 特征常数不可修改；变更曲线必须提升预处理版本")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "constants", constants)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "names": list(self.names),
            "constants": {key: self.constants[key] for key in sorted(self.constants)},  # type: ignore[index]
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureConfig":
        version = payload.get("version")
        if version != PREPROCESSING_VERSION:
            raise ValueError(f"不支持的预处理版本：{version!r}")
        names = payload.get("names")
        if not isinstance(names, list) or tuple(names) != FEATURE_NAMES:
            raise ValueError("状态中的特征顺序与 v1 不一致")
        constants = payload.get("constants")
        if not isinstance(constants, Mapping):
            raise ValueError("状态中缺少预处理常数")
        restored = {str(key): float(value) for key, value in constants.items()}  # type: ignore[arg-type]
        if restored != FEATURE_CONSTANTS:
            raise ValueError("状态中的预处理常数与 v1 不一致")
        return cls(version=version, names=tuple(names), constants=restored)


def to_features(sample: Sample, config: FeatureConfig | None = None) -> np.ndarray:
    """把一条原始样本变换为八维特征向量。"""

    settings = config if config is not None else FeatureConfig()
    constants = settings.constants
    assert constants is not None

    grade = float(sample.grade) - constants["grade_center"]
    renovation = (
        -min(sample.renovation_age_years, constants["renovation_cap_years"])
        / constants["renovation_scale_years"]
    )
    rating = (sample.ctrip_rating - constants["rating_center"]) / constants["rating_scale"]
    scale = constants["minutes_scale"]
    subway = -math.log1p(sample.subway_walk_minutes / scale)
    mall = -math.log1p(sample.mall_transit_minutes / scale)
    attraction = -math.log1p(sample.attraction_taxi_minutes / scale)
    # `log(x / center)` 在 x 为极小的正数时，除法先下溢为零而丢失定义域；
    # 用等价的 `log(x) - log(center)` 保证契约允许的全部正数都有有限特征。
    area = math.log(sample.room_area_sqm) - math.log(constants["area_center_sqm"])
    price = math.log(constants["price_center"]) - math.log(sample.nightly_price)

    vector = np.array(
        [grade, renovation, rating, subway, mall, attraction, area, price],
        dtype=float,
    )
    if vector.shape != (FEATURE_COUNT,):
        raise AssertionError("特征向量维度错误")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"样本 `{sample.hotel_id}` 产生非有限特征")
    return vector


def feature_matrix(samples: Sequence[Sample], config: FeatureConfig | None = None) -> np.ndarray:
    settings = config if config is not None else FeatureConfig()
    return np.vstack([to_features(sample, settings) for sample in samples])


def feature_vector_to_list(vector: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(vector, dtype=float).reshape(-1)]


def config_fingerprint(config: FeatureConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False)
