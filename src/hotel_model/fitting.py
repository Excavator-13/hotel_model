"""偏好标签语义与带约束的成对逻辑回归。

约定 ``d = z_left - z_right``，严格标签为 1 或 0，平局为软标签 0.5。
跳过记录不产生 :class:`LabeledPair`，也不参与损失。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .features import FEATURE_COUNT

LEFT_WINS = 1.0
RIGHT_WINS = 0.0
TIE = 0.5
VALID_LABELS: tuple[float, ...] = (LEFT_WINS, TIE, RIGHT_WINS)

# 事件种类：有效标签与跳过。
KIND_LABEL = "label"
KIND_SKIP = "skip"
VALID_KINDS: tuple[str, ...] = (KIND_LABEL, KIND_SKIP)

DEFAULT_L2 = 0.1
MIN_L2 = 1e-9
# 判断某特征的比较差值是否近似为零。
LABEL_DIFF_TOLERANCE = 1e-12
# 权重上界仅用于兜底非有限结果，远大于任何合理取值。
MAX_WEIGHT = 1e6


def other_label(label: float) -> float:
    """交换 A、B 后的等价标签：严格标签反转，平局保持不变。"""

    return 1.0 - label


@dataclass(frozen=True)
class LabeledPair:
    """一条面向展示顺序的有效标签。"""

    left_id: str
    right_id: str
    difference: np.ndarray
    target: float

    def flipped(self) -> "LabeledPair":
        """交换展示方向后的等价标签，目标值不变。"""

        return LabeledPair(
            left_id=self.right_id,
            right_id=self.left_id,
            difference=-self.difference,
            target=other_label(self.target),
        )


@dataclass(frozen=True)
class FittingConfig:
    """拟合配置；``lambda`` 必须是有限正数，默认 0.1。"""

    l2: float = DEFAULT_L2
    version: str = "v1"

    def __post_init__(self) -> None:
        value = float(self.l2)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"L2 正则强度必须是有限正数，收到 {self.l2!r}")
        object.__setattr__(self, "l2", value)

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "l2": self.l2}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FittingConfig":
        if payload.get("version") != "v1":
            raise ValueError(f"不支持的拟合配置版本：{payload.get('version')!r}")
        l2 = payload.get("l2")
        if not isinstance(l2, (int, float)) or isinstance(l2, bool):
            raise ValueError("状态中的 L2 正则强度必须是数值")
        return cls(l2=float(l2))


def labeled_pairs(
    events: Iterable[Mapping[str, object]],
    vectors: Mapping[str, np.ndarray],
) -> list[LabeledPair]:
    """把事件序列转换为拟合用样本，跳过记录被忽略。"""

    pairs: list[LabeledPair] = []
    for index, event in enumerate(events):
        kind = event.get("kind")
        if kind == KIND_SKIP:
            continue
        if kind != KIND_LABEL:
            raise ValueError(f"第 {index + 1} 条事件种类非法：{kind!r}")
        left_id = event.get("left_id")
        right_id = event.get("right_id")
        label = event.get("label")
        if not isinstance(left_id, str) or not isinstance(right_id, str):
            raise ValueError(f"第 {index + 1} 条事件缺少样本 ID")
        if not isinstance(label, (int, float)) or isinstance(label, bool):
            raise ValueError(f"第 {index + 1} 条事件标签必须是数值")
        target = float(label)
        if target not in VALID_LABELS:
            raise ValueError(f"第 {index + 1} 条事件标签非法：{target!r}")
        if left_id not in vectors or right_id not in vectors:
            raise ValueError(f"第 {index + 1} 条事件引用了未知样本")
        difference = np.asarray(vectors[left_id], dtype=float) - np.asarray(
            vectors[right_id], dtype=float
        )
        pairs.append(
            LabeledPair(
                left_id=left_id,
                right_id=right_id,
                difference=difference,
                target=target,
            )
        )
    return pairs


def _stack(pairs: Sequence[LabeledPair]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.zeros((0, FEATURE_COUNT), dtype=float), np.zeros((0,), dtype=float)
    differences = np.vstack([np.asarray(pair.difference, dtype=float) for pair in pairs])
    targets = np.array([pair.target for pair in pairs], dtype=float)
    if differences.shape[1] != FEATURE_COUNT:
        raise ValueError("差向量维度与特征数不一致")
    if not np.all(np.isfinite(differences)) or not np.all(np.isfinite(targets)):
        raise ValueError("拟合样本包含非有限值")
    return differences, targets


def mean_cross_entropy(
    weights: np.ndarray,
    differences: np.ndarray,
    targets: np.ndarray,
) -> float:
    """不含正则项的平均二元交叉熵，使用稳定的 logaddexp 形式。"""

    if differences.shape[0] == 0:
        return 0.0
    scores = differences @ np.asarray(weights, dtype=float)
    terms = np.logaddexp(0.0, scores) - targets * scores
    value = float(np.mean(terms))
    if not math.isfinite(value):
        raise ValueError("交叉熵不是有限值")
    return value


def penalized_loss(
    weights: np.ndarray,
    differences: np.ndarray,
    targets: np.ndarray,
    l2: float,
) -> float:
    value = mean_cross_entropy(weights, differences, targets) + 0.5 * l2 * float(
        np.dot(weights, weights)
    )
    if not math.isfinite(value):
        raise ValueError("目标函数不是有限值")
    return value


def loss_gradient(
    weights: np.ndarray,
    differences: np.ndarray,
    targets: np.ndarray,
    l2: float,
) -> np.ndarray:
    """解析梯度：``mean((p - y) * d) + lambda * w``。"""

    vector = np.asarray(weights, dtype=float)
    if differences.shape[0] == 0:
        gradient = l2 * vector
    else:
        scores = differences @ vector
        # 稳定 sigmoid：score >= 0 时用 1 / (1 + exp(-score))，否则用 exp(score) / (1 + exp(score))。
        probabilities = np.where(
            scores >= 0.0,
            1.0 / (1.0 + np.exp(-np.abs(scores))),
            np.exp(-np.abs(scores)) / (1.0 + np.exp(-np.abs(scores))),
        )
        gradient = (differences.T @ (probabilities - targets)) / differences.shape[0] + l2 * vector
    if not np.all(np.isfinite(gradient)):
        raise ValueError("梯度不是有限值")
    return gradient


@dataclass(frozen=True)
class FitResult:
    """一次拟合的结果；失败时 ``weights`` 为 ``None``。"""

    ok: bool
    weights: np.ndarray | None
    training_cross_entropy: float | None
    labels_covered: int
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "weights": None if self.weights is None else [float(value) for value in self.weights],
            "training_cross_entropy": self.training_cross_entropy,
            "labels_covered": self.labels_covered,
            "message": self.message,
        }


def fit(
    pairs: Sequence[LabeledPair],
    config: FittingConfig,
    initial: np.ndarray | None = None,
) -> FitResult:
    """用 SciPy L-BFGS-B 拟合非负权重；失败时返回 ``ok=False`` 而不抛出。"""

    if not pairs:
        return FitResult(
            ok=False,
            weights=None,
            training_cross_entropy=None,
            labels_covered=0,
            message="没有有效标签，模型保持未训练",
        )

    try:
        differences, targets = _stack(pairs)
    except ValueError as error:
        return FitResult(False, None, None, 0, str(error))

    if initial is None:
        start = np.zeros(FEATURE_COUNT, dtype=float)
    else:
        start = np.asarray(initial, dtype=float).reshape(-1).copy()
        if start.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(start)):
            start = np.zeros(FEATURE_COUNT, dtype=float)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            value = penalized_loss(weights, differences, targets, config.l2)
            gradient = loss_gradient(weights, differences, targets, config.l2)
        except ValueError as error:
            return math.inf, np.zeros_like(weights)
        return value, gradient

    try:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * FEATURE_COUNT,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
    except Exception as error:  # 优化器自身异常也要转成可报告的失败
        return FitResult(
            False,
            None,
            None,
            0,
            f"优化器异常：{type(error).__name__}: {error}",
        )

    weights = np.asarray(getattr(result, "x", np.full(FEATURE_COUNT, np.nan)), dtype=float)
    accepted = (
        bool(getattr(result, "success", False))
        and weights.shape == (FEATURE_COUNT,)
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and float(np.max(np.abs(weights))) <= MAX_WEIGHT
    )
    if not accepted:
        message = str(getattr(result, "message", "优化未收敛"))
        return FitResult(False, None, None, 0, f"拟合未成功：{message}")

    try:
        cross_entropy = mean_cross_entropy(weights, differences, targets)
    except ValueError as error:
        return FitResult(False, None, None, 0, f"拟合结果不可用：{error}")

    return FitResult(
        ok=True,
        weights=weights,
        training_cross_entropy=cross_entropy,
        labels_covered=len(pairs),
        message=str(getattr(result, "message", "")),
    )
