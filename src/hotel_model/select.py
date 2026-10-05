"""轻量主动选题。

候选来自同城无序样本对，按城市与 ID 排序保证稳定顺序。非探索分支先看是否存在
双方各有优势的取舍候选，再用确定性分数排序；每次选题保留随机探索比例。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import Sample, DataError
from .features import FEATURE_COUNT

DEFAULT_EXPLORATION_RATE = 0.2
DEFAULT_SEED = 42
COLD_START_LABELS = 8
# 差向量小于该范数视为没有实际差异。
MIN_DIFFERENCE_NORM = 1e-9
DEFAULT_SELECTION_VERSION = "v1"
RANDOM_STATE_VERSION = "v1"


@dataclass(frozen=True)
class SelectionConfig:
    """选题配置；恢复时不可修改。"""

    seed: int = DEFAULT_SEED
    exploration_rate: float = DEFAULT_EXPLORATION_RATE
    cold_start_labels: int = COLD_START_LABELS
    version: str = DEFAULT_SELECTION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("随机种子必须是整数")
        rate = float(self.exploration_rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError(f"探索比例必须在 0 至 1 之间，收到 {self.exploration_rate!r}")
        if not isinstance(self.cold_start_labels, int) or self.cold_start_labels < 1:
            raise ValueError("冷启动标签数必须是正整数")
        object.__setattr__(self, "exploration_rate", rate)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "exploration_rate": self.exploration_rate,
            "cold_start_labels": self.cold_start_labels,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelectionConfig":
        if payload.get("version") != DEFAULT_SELECTION_VERSION:
            raise ValueError(f"不支持的选题配置版本：{payload.get('version')!r}")
        seed = payload.get("seed")
        rate = payload.get("exploration_rate")
        cold = payload.get("cold_start_labels")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("状态中的随机种子必须是整数")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError("状态中的探索比例必须是数值")
        if not isinstance(cold, int) or isinstance(cold, bool):
            raise ValueError("状态中的冷启动标签数必须是整数")
        return cls(
            seed=int(seed),
            exploration_rate=float(rate),
            cold_start_labels=int(cold),
        )


@dataclass(frozen=True)
class Question:
    """一道待答问题；左右顺序即展示顺序。"""

    left_id: str
    right_id: str
    version: str = DEFAULT_SELECTION_VERSION

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.left_id, self.right_id)))

    @property
    def unordered(self) -> tuple[str, str]:
        return self.pair

    def to_dict(self) -> dict[str, object]:
        return {"left_id": self.left_id, "right_id": self.right_id, "version": self.version}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Question":
        left = payload.get("left_id")
        right = payload.get("right_id")
        version = payload.get("version", DEFAULT_SELECTION_VERSION)
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise ValueError("状态中的待答题不是两个不同样本")
        if version != DEFAULT_SELECTION_VERSION:
            raise ValueError(f"不支持的选题版本：{version!r}")
        return cls(left_id=left, right_id=right, version=str(version))


@dataclass(frozen=True)
class Candidate:
    """一个候选无序样本对。"""

    left_id: str
    right_id: str
    difference: np.ndarray

    @property
    def key(self) -> tuple[str, str]:
        return (self.left_id, self.right_id)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.difference))


def as_pairs(processed: Iterable[object]) -> set[tuple[str, str]]:
    """把事件或待答题集合规整为已处理的无序 ID 对。"""

    pairs: set[tuple[str, str]] = set()
    for item in processed:
        if isinstance(item, Question):
            pairs.add(item.pair)
            continue
        if isinstance(item, Mapping):
            left = item.get("left_id")
            right = item.get("right_id")
        else:
            left = getattr(item, "left_id", None)
            right = getattr(item, "right_id", None)
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise ValueError("已处理记录必须引用两个不同样本")
        pairs.add(tuple(sorted((left, right))))
    return pairs


def build_candidates(
    pool: Sequence[Sample],
    vectors: Mapping[str, np.ndarray],
    processed: Iterable[object],
) -> list[Candidate]:
    """构造稳定排序的同城无序候选，排除已处理与近似零差异的对。"""

    done = as_pairs(processed)
    cities: dict[str, list[str]] = {}
    for sample in pool:
        cities.setdefault(sample.city, []).append(sample.hotel_id)

    candidates: list[Candidate] = []
    for city in sorted(cities):
        ids = sorted(cities[city])
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                if (left, right) in done:
                    continue
                if left not in vectors or right not in vectors:
                    raise DataError(f"样本 `{left}` 或 `{right}` 缺少特征向量")
                difference = np.asarray(vectors[left], dtype=float) - np.asarray(
                    vectors[right], dtype=float
                )
                if float(np.linalg.norm(difference)) <= MIN_DIFFERENCE_NORM:
                    continue
                candidates.append(Candidate(left_id=left, right_id=right, difference=difference))
    return candidates


def is_tradeoff(difference: np.ndarray, *, tolerance: float = MIN_DIFFERENCE_NORM) -> bool:
    """差向量同时含有正分量和负分量时，两个样本各有优势。"""

    values = np.asarray(difference, dtype=float)
    return bool(np.any(values > tolerance) and np.any(values < -tolerance))


def magnitude(difference: np.ndarray) -> float:
    """差异幅度，按 sqrt(特征数) 归一化后截到 1。"""

    value = float(np.linalg.norm(np.asarray(difference, dtype=float))) / math.sqrt(FEATURE_COUNT)
    return min(max(value, 0.0), 1.0)


def diversity(difference: np.ndarray, historical_units: Sequence[np.ndarray]) -> float:
    """方向覆盖：与历史单位差向量方向越不重复，取值越接近 1。"""

    norm = float(np.linalg.norm(np.asarray(difference, dtype=float)))
    if norm <= MIN_DIFFERENCE_NORM:
        return 0.0
    unit = np.asarray(difference, dtype=float) / norm
    if not historical_units:
        return 1.0
    overlap = max(abs(float(np.dot(unit, other))) for other in historical_units)
    return float(min(max(1.0 - overlap, 0.0), 1.0))


def cold_score(difference: np.ndarray, historical_units: Sequence[np.ndarray]) -> float:
    return 0.6 * magnitude(difference) + 0.4 * diversity(difference, historical_units)


def active_score(
    difference: np.ndarray,
    probability: float,
    historical_units: Sequence[np.ndarray],
) -> float:
    uncertainty = 4.0 * probability * (1.0 - probability)
    return uncertainty * (0.5 + 0.5 * magnitude(difference)) * (
        0.5 + 0.5 * diversity(difference, historical_units)
    )


def _historical_units(
    processed: Iterable[object],
    vectors: Mapping[str, np.ndarray],
) -> list[np.ndarray]:
    units: list[np.ndarray] = []
    for item in processed:
        if isinstance(item, Question):
            left, right = item.left_id, item.right_id
        elif isinstance(item, Mapping):
            left, right = item.get("left_id"), item.get("right_id")
        else:
            left, right = getattr(item, "left_id", None), getattr(item, "right_id", None)
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        if left not in vectors or right not in vectors:
            continue
        difference = np.asarray(vectors[left], dtype=float) - np.asarray(vectors[right], dtype=float)
        norm = float(np.linalg.norm(difference))
        if norm > MIN_DIFFERENCE_NORM:
            units.append(difference / norm)
    return units


def score_candidates(
    candidates: Sequence[Candidate],
    weights: np.ndarray | None,
    label_count: int,
    processed: Iterable[object],
    vectors: Mapping[str, np.ndarray],
    config: SelectionConfig,
) -> list[float]:
    """非探索分支的确定性分数，索引与 ``candidates`` 对齐。"""

    historical = _historical_units(processed, vectors)
    weight_vector = None if weights is None else np.asarray(weights, dtype=float)
    scores: list[float] = []
    for candidate in candidates:
        if label_count < config.cold_start_labels or weight_vector is None:
            scores.append(cold_score(candidate.difference, historical))
            continue
        logit = float(np.dot(weight_vector, candidate.difference))
        if logit >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            probability = math.exp(logit) / (1.0 + math.exp(logit))
        scores.append(active_score(candidate.difference, probability, historical))
    return scores


def select_next(
    pool: Sequence[Sample],
    processed: Iterable[object],
    vectors: Mapping[str, np.ndarray],
    weights: np.ndarray | None,
    label_count: int,
    rng: np.random.Generator,
    config: SelectionConfig | None = None,
) -> Question | None:
    """选出一道题目；没有合法候选时返回 ``None``。

    ``processed`` 必须包含全部已回答、已跳过以及当前待答的样本对。
    """

    settings = config if config is not None else SelectionConfig()
    processed_list = list(processed)
    candidates = build_candidates(pool, vectors, processed_list)
    if not candidates:
        return None

    if rng.random() < settings.exploration_rate:
        chosen = candidates[int(rng.integers(0, len(candidates)))]
    else:
        tradeoffs = [c for c in candidates if is_tradeoff(c.difference)]
        branch = tradeoffs if tradeoffs else candidates
        scores = score_candidates(
            branch, weights, label_count, processed_list, vectors, settings
        )
        best = 0
        for index in range(1, len(branch)):
            if scores[index] > scores[best]:
                best = index
        chosen = branch[best]

    # 独立选取左右展示顺序，标签随后按展示方向解释。
    if rng.integers(0, 2, endpoint=True) == 1:
        question = Question(left_id=chosen.left_id, right_id=chosen.right_id)
    else:
        question = Question(left_id=chosen.right_id, right_id=chosen.left_id)
    assert chosen.key == question.pair
    return question
