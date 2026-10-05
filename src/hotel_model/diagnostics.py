"""训练诊断：计数、权重、覆盖状态与训练交叉熵。

诊断只汇报可观察事实。训练交叉熵表示拟合程度，不是校准度或泛化精度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .data import Sample
from .features import FEATURE_NAMES
from .fitting import LABEL_DIFF_TOLERANCE, KIND_LABEL, KIND_SKIP
from .state import TrainingState

# 展示顺序即规格中的原始字段顺序，括号内为单位。
RAW_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("grade", "档次"),
    ("renovation_age_years", "装修年数（年）"),
    ("ctrip_rating", "评分（0-5）"),
    ("subway_walk_minutes", "地铁步行（分钟）"),
    ("mall_transit_minutes", "商场公交（分钟）"),
    ("attraction_taxi_minutes", "景点打车（分钟）"),
    ("room_area_sqm", "面积（平方米）"),
    ("nightly_price", "每晚价格（元）"),
)


@dataclass
class TrainingCounts:
    """按规格要求的比较计数。"""

    valid_labels: int = 0
    strict_labels: int = 0
    ties: int = 0
    skips: int = 0
    by_city: dict[str, int] = field(default_factory=dict)
    uninformative_features: list[str] = field(default_factory=list)
    model_trained: bool = False
    labels_covered: int = 0
    lagging_labels: int = 0
    training_cross_entropy: float | None = None


def summarize(
    state: TrainingState,
    vectors: Mapping[str, np.ndarray],
) -> TrainingCounts:
    """从事件序列与模型快照计算诊断计数。"""

    counts = TrainingCounts()
    counts.model_trained = state.model.trained
    counts.labels_covered = state.model.labels_covered
    counts.lagging_labels = state.model.lagging_labels
    counts.training_cross_entropy = (
        state.model.training_cross_entropy if state.model.trained else None
    )

    sample_map = state.sample_map()
    differences: list[np.ndarray] = []
    for event in state.events:
        if event.get("kind") == KIND_SKIP:
            counts.skips += 1
            continue
        if event.get("kind") != KIND_LABEL:
            continue
        label = float(event.get("label", 0.5))
        if label == 0.5:
            counts.ties += 1
        else:
            counts.strict_labels += 1
        counts.valid_labels += 1
        pair = (str(event.get("left_id")), str(event.get("right_id")))
        city = sample_map[pair[0]].city if pair[0] in sample_map else str(event.get("city", ""))
        counts.by_city[city] = counts.by_city.get(city, 0) + 1
        if pair[0] in vectors and pair[1] in vectors:
            differences.append(
                np.asarray(vectors[pair[0]], dtype=float) - np.asarray(vectors[pair[1]], dtype=float)
            )

    if counts.valid_labels == 0:
        counts.uninformative_features = list(FEATURE_NAMES)
    elif differences:
        matrix = np.vstack(differences)
        counts.uninformative_features = [
            name
            for index, name in enumerate(FEATURE_NAMES)
            if float(np.max(np.abs(matrix[:, index]))) <= LABEL_DIFF_TOLERANCE
        ]
    return counts


def format_sample(sample: Sample) -> str:
    """渲染一家酒店的全部原始字段，附档次名称与单位。"""

    lines = [
        f"  hotel_id={sample.hotel_id}  name={sample.name}  city={sample.city}",
        f"  档次={sample.grade}（{sample.grade_name}）",
    ]
    values = {
        "renovation_age_years": sample.renovation_age_years,
        "ctrip_rating": sample.ctrip_rating,
        "subway_walk_minutes": sample.subway_walk_minutes,
        "mall_transit_minutes": sample.mall_transit_minutes,
        "attraction_taxi_minutes": sample.attraction_taxi_minutes,
        "room_area_sqm": sample.room_area_sqm,
        "nightly_price": sample.nightly_price,
    }
    for field_name, label in RAW_FIELD_LABELS[1:]:
        lines.append(f"  {label}={values[field_name]:g}")
    return "\n".join(lines)


def format_diagnostics(state: TrainingState, counts: TrainingCounts) -> list[str]:
    """按规格生成诊断文本行。"""

    lines = ["", "训练诊断"]
    lines.append(
        f"  有效比较={counts.valid_labels} 严格偏好={counts.strict_labels} "
        f"平局={counts.ties} 跳过={counts.skips}"
    )
    if counts.by_city:
        cities = "，".join(f"{city}={count}" for city, count in sorted(counts.by_city.items()))
        lines.append(f"  按城市有效标签：{cities}")
    else:
        lines.append("  按城市有效标签：暂无")

    if counts.valid_labels == 0:
        lines.append("  模型：未训练（没有有效标签，不展示拟合指标）")
        return lines

    lines.append(f"  权重（顺序：{', '.join(FEATURE_NAMES)}）")
    if counts.model_trained and state.model.weights is not None:
        for name, value in zip(FEATURE_NAMES, state.model.weights):
            lines.append(f"    {name}={float(value):.6g}")
        assert counts.training_cross_entropy is not None
        lines.append(
            f"  训练交叉熵={counts.training_cross_entropy:.6g}（仅表示训练拟合程度，"
            "不含正则项，不代表校准或泛化精度）"
        )
    else:
        lines.append("    模型未训练，暂无权重")

    if counts.lagging_labels > 0:
        lines.append(
            f"  模型覆盖状态：落后 {counts.lagging_labels} 个有效标签"
            f"（已覆盖 {counts.labels_covered}）"
        )
    else:
        lines.append(f"  模型覆盖状态：已覆盖全部 {counts.labels_covered} 个有效标签")

    if counts.uninformative_features:
        lines.append(
            "  缺少比较信息的特征（有效比较中差值近似为零）："
            + "，".join(counts.uninformative_features)
        )
        lines.append("  提示：这些维度无法从当前比较中估计，权重接近零不代表用户不在意。")
    return lines
