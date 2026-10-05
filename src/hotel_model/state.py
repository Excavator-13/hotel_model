"""可恢复的本地训练状态：版本化 JSON、内容指纹与原子替换。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import Sample, sample_from_dict, sample_to_dict
from .features import FEATURE_COUNT, FeatureConfig
from .fitting import DEFAULT_L2, FitResult, FittingConfig
from .select import (
    DEFAULT_SEED,
    Question,
    SelectionConfig,
    as_pairs,
)

SCHEMA_VERSION = 1
FINGERPRINT_FIELDS: tuple[str, ...] = (
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


class StateError(ValueError):
    """状态文件缺失、版本不支持或内容自相矛盾。"""


# 用于区分“不修改该字段”与“显式设置为 None”的哨兵。
_UNSET = object()


@dataclass(frozen=True)
class ModelSnapshot:
    """最近一次成功拟合的模型；未训练或拟合失败时 ``trained`` 为假。"""

    trained: bool
    weights: np.ndarray | None
    labels_covered: int
    training_cross_entropy: float | None
    lagging_labels: int = 0

    @classmethod
    def untrained(cls) -> "ModelSnapshot":
        return cls(trained=False, weights=None, labels_covered=0, training_cross_entropy=None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelSnapshot):
            return NotImplemented
        if (
            self.trained != other.trained
            or self.labels_covered != other.labels_covered
            or self.training_cross_entropy != other.training_cross_entropy
            or self.lagging_labels != other.lagging_labels
        ):
            return False
        if self.weights is None or other.weights is None:
            return self.weights is None and other.weights is None
        return bool(np.array_equal(self.weights, other.weights))

    __hash__ = None  # type: ignore[assignment]  # 含数组字段，不可哈希

    def to_dict(self) -> dict[str, object]:
        return {
            "trained": self.trained,
            "weights": None if self.weights is None else [float(v) for v in self.weights],
            "labels_covered": int(self.labels_covered),
            "training_cross_entropy": self.training_cross_entropy,
            "lagging_labels": int(self.lagging_labels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelSnapshot":
        trained = payload.get("trained")
        if not isinstance(trained, bool):
            raise StateError("模型字段 `trained` 必须是布尔值")
        covered = payload.get("labels_covered")
        if not isinstance(covered, int) or isinstance(covered, bool) or covered < 0:
            raise StateError("模型字段 `labels_covered` 必须是非负整数")
        lagging = payload.get("lagging_labels", 0)
        if not isinstance(lagging, int) or isinstance(lagging, bool) or lagging < 0:
            raise StateError("模型字段 `lagging_labels` 必须是非负整数")
        loss = payload.get("training_cross_entropy")
        if loss is not None:
            if not isinstance(loss, (int, float)) or isinstance(loss, bool):
                raise StateError("模型字段 `training_cross_entropy` 必须是数值或空")
            loss = float(loss)
            if not np.isfinite(loss):
                raise StateError("模型字段 `training_cross_entropy` 必须是有限值")
        raw_weights = payload.get("weights")
        weights: np.ndarray | None = None
        if raw_weights is not None:
            if not isinstance(raw_weights, list) or len(raw_weights) != FEATURE_COUNT:
                raise StateError(f"模型权重必须是 {FEATURE_COUNT} 维数组")
            values = np.array(
                [float(v) for v in raw_weights if isinstance(v, (int, float)) and not isinstance(v, bool)],
                dtype=float,
            )
            if values.shape != (FEATURE_COUNT,):
                raise StateError(f"模型权重必须是 {FEATURE_COUNT} 维数组")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise StateError("模型权重必须是有限非负数值")
            weights = values
        if trained and weights is None:
            raise StateError("模型标记为已训练但缺少权重")
        if trained and loss is None:
            raise StateError("模型标记为已训练但缺少训练交叉熵")
        if not trained and weights is not None:
            raise StateError("模型未训练却带有权重")
        return cls(
            trained=trained,
            weights=weights,
            labels_covered=int(covered),
            training_cross_entropy=loss,
            lagging_labels=int(lagging),
        )


@dataclass(frozen=True)
class PendingQuestion:
    """当前待答问题；左右顺序即展示顺序。"""

    question: Question
    asked_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "left_id": self.question.left_id,
            "right_id": self.question.right_id,
            "version": self.question.version,
            "asked_at": self.asked_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PendingQuestion":
        asked_at = payload.get("asked_at")
        if not isinstance(asked_at, str) or asked_at == "":
            raise StateError("待答题缺少 `asked_at` 时间戳")
        try:
            question = Question.from_dict(payload)
        except ValueError as error:
            raise StateError(f"待答题不合法：{error}") from None
        return cls(question=question, asked_at=asked_at)


@dataclass(frozen=True)
class TrainingState:
    """一个训练会话的完整事实来源。"""

    schema_version: int
    created_at: str
    updated_at: str
    preprocessing: FeatureConfig
    fitting: FittingConfig
    selection: SelectionConfig
    samples: tuple[Sample, ...]
    fingerprint: str
    events: tuple[dict[str, object], ...]
    pending: PendingQuestion | None
    random_state: dict[str, Any]
    model: ModelSnapshot

    @classmethod
    def new(
        cls,
        samples: Sequence[Sample],
        *,
        preprocessing: FeatureConfig,
        fitting: FittingConfig,
        selection: SelectionConfig,
        random_state: Mapping[str, Any],
        created_at: str,
    ) -> "TrainingState":
        pool = tuple(samples)
        if not pool:
            raise StateError("训练会话至少需要一条样本")
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=created_at,
            updated_at=created_at,
            preprocessing=preprocessing,
            fitting=fitting,
            selection=selection,
            samples=pool,
            fingerprint=content_fingerprint(pool),
            events=(),
            pending=None,
            random_state=dict(random_state),
            model=ModelSnapshot.untrained(),
        )

    @property
    def label_events(self) -> tuple[dict[str, object], ...]:
        return tuple(event for event in self.events if event.get("kind") == "label")

    @property
    def skip_events(self) -> tuple[dict[str, object], ...]:
        return tuple(event for event in self.events if event.get("kind") == "skip")

    def sample_map(self) -> dict[str, Sample]:
        return {sample.hotel_id: sample for sample in self.samples}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "preprocessing": self.preprocessing.to_dict(),
            "fitting": self.fitting.to_dict(),
            "selection": self.selection.to_dict(),
            "samples": [sample_to_dict(sample) for sample in self.samples],
            "fingerprint": self.fingerprint,
            "events": [dict(event) for event in self.events],
            "pending": None if self.pending is None else self.pending.to_dict(),
            "random_state": self.random_state,
            "model": self.model.to_dict(),
        }

    def with_(
        self,
        *,
        updated_at: str | None = None,
        events: Sequence[Mapping[str, object]] | None = None,
        pending: "PendingQuestion | None | object" = _UNSET,
        random_state: Mapping[str, Any] | None = None,
        model: ModelSnapshot | None = None,
    ) -> "TrainingState":
        return replace(
            self,
            updated_at=self.updated_at if updated_at is None else updated_at,
            events=self.events if events is None else tuple(dict(e) for e in events),
            pending=self.pending if pending is _UNSET else pending,  # type: ignore[arg-type]
            random_state=self.random_state if random_state is None else dict(random_state),
            model=self.model if model is None else model,
        )


def content_fingerprint(samples: Sequence[Sample]) -> str:
    """按 ID 排序、只哈希必需字段的内容指纹。

    行顺序、BOM 和额外列不影响结果；必需字段内容变化会改变指纹。
    """

    payload = []
    for sample in sorted(samples, key=lambda item: item.hotel_id):
        record = sample_to_dict(sample)
        payload.append({field: record[field] for field in FINGERPRINT_FIELDS})
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def random_state_to_json(state: Mapping[str, Any]) -> dict[str, Any]:
    """把 ``numpy.random.Generator`` 状态转成纯 JSON 类型。"""

    return json.loads(json.dumps(dict(state), default=_json_default))


def random_state_from_json(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(payload), default=_json_default))


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"随机状态包含无法序列化的类型：{type(value).__name__}")


def _require_text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or value == "":
        raise StateError(f"状态字段 `{field}` 必须是非空字符串")
    return value


def _validated_events(
    payload: Iterable[object],
    samples: Mapping[str, Sample],
    pending: PendingQuestion | None,
) -> tuple[dict[str, object], ...]:
    events: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise StateError(f"第 {index + 1} 条事件必须是对象")
        left = raw.get("left_id")
        right = raw.get("right_id")
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise StateError(f"第 {index + 1} 条事件引用了非法样本对")
        if left not in samples or right not in samples:
            raise StateError(f"第 {index + 1} 条事件引用了快照中不存在的样本")
        if samples[left].city != samples[right].city:
            raise StateError(f"第 {index + 1} 条事件跨城市：{left!r} 与 {right!r}")
        pair = tuple(sorted((left, right)))
        if pair in seen:
            raise StateError(f"第 {index + 1} 条事件重复了样本对 {pair}")
        seen.add(pair)
        kind = raw.get("kind")
        if kind not in ("label", "skip"):
            raise StateError(f"第 {index + 1} 条事件种类非法：{kind!r}")
        record = dict(raw)
        record["city"] = samples[left].city
        if kind == "label":
            label = raw.get("label")
            if not isinstance(label, (int, float)) or isinstance(label, bool):
                raise StateError(f"第 {index + 1} 条事件标签必须是数值")
            if float(label) not in (0.0, 0.5, 1.0):
                raise StateError(f"第 {index + 1} 条事件标签非法：{label!r}")
            record["label"] = float(label)
        else:
            record["label"] = None
        if not isinstance(raw.get("answered_at"), str):
            raise StateError(f"第 {index + 1} 条事件缺少 `answered_at`")
        events.append(record)

    if pending is not None and pending.question.pair in seen:
        raise StateError("待答题与已处理样本对重复")
    return tuple(events)


def state_from_dict(payload: Mapping[str, object]) -> TrainingState:
    """从 JSON 对象恢复状态，并校验全部不变量。"""

    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise StateError("状态缺少整数 `schema_version`")
    if version != SCHEMA_VERSION:
        raise StateError(f"不支持的状态版本：{version}（当前支持 {SCHEMA_VERSION}）")

    created_at = _require_text(payload, "created_at")
    updated_at = _require_text(payload, "updated_at")

    raw_preprocessing = payload.get("preprocessing")
    raw_fitting = payload.get("fitting")
    raw_selection = payload.get("selection")
    if not isinstance(raw_preprocessing, Mapping):
        raise StateError("状态缺少 `preprocessing`")
    if not isinstance(raw_fitting, Mapping):
        raise StateError("状态缺少 `fitting`")
    if not isinstance(raw_selection, Mapping):
        raise StateError("状态缺少 `selection`")
    try:
        preprocessing = FeatureConfig.from_dict(raw_preprocessing)
        fitting = FittingConfig.from_dict(raw_fitting)
        selection = SelectionConfig.from_dict(raw_selection)
    except (ValueError, TypeError) as error:
        raise StateError(f"状态配置不合法：{error}") from None

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise StateError("状态缺少样本快照")
    samples: list[Sample] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise StateError(f"第 {index + 1} 条样本快照必须是对象")
        try:
            samples.append(sample_from_dict(raw))
        except Exception as error:
            raise StateError(f"第 {index + 1} 条样本快照不合法：{error}") from None
    ids = [sample.hotel_id for sample in samples]
    if len(set(ids)) != len(ids):
        raise StateError("样本快照存在重复 `hotel_id`")
    sample_index = {sample.hotel_id: sample for sample in samples}

    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint == "":
        raise StateError("状态缺少内容指纹")
    if fingerprint != content_fingerprint(samples):
        raise StateError("状态内容指纹与样本快照不一致")

    raw_pending = payload.get("pending")
    pending: PendingQuestion | None
    if raw_pending is None:
        pending = None
    elif isinstance(raw_pending, Mapping):
        pending = PendingQuestion.from_dict(raw_pending)
        for hotel_id in (pending.question.left_id, pending.question.right_id):
            if hotel_id not in sample_index:
                raise StateError(f"待答题引用了快照中不存在的样本 `{hotel_id}`")
        if sample_index[pending.question.left_id].city != sample_index[pending.question.right_id].city:
            raise StateError("待答题跨城市")
    else:
        raise StateError("状态字段 `pending` 必须是对象或空")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise StateError("状态缺少事件序列")
    events = _validated_events(raw_events, sample_index, pending)

    raw_random = payload.get("random_state")
    if not isinstance(raw_random, Mapping):
        raise StateError("状态缺少随机状态")
    random_state = random_state_from_json(raw_random)

    raw_model = payload.get("model")
    if not isinstance(raw_model, Mapping):
        raise StateError("状态缺少模型字段")
    model = ModelSnapshot.from_dict(raw_model)

    label_count = sum(1 for event in events if event.get("kind") == "label")
    if model.labels_covered + model.lagging_labels != label_count:
        raise StateError(
            "模型覆盖数与有效标签数不一致："
            f"{model.labels_covered} + {model.lagging_labels} != {label_count}"
        )

    return TrainingState(
        schema_version=version,
        created_at=created_at,
        updated_at=updated_at,
        preprocessing=preprocessing,
        fitting=fitting,
        selection=selection,
        samples=tuple(samples),
        fingerprint=fingerprint,
        events=events,
        pending=pending,
        random_state=random_state,
        model=model,
    )


def state_to_json(state: TrainingState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def load_state(path: str | Path) -> TrainingState:
    resolved = Path(path)
    if not resolved.is_file():
        raise StateError(f"找不到状态文件 `{resolved}`")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StateError(f"状态文件 `{resolved}` 不是合法 JSON：{error}") from None
    if not isinstance(payload, Mapping):
        raise StateError(f"状态文件 `{resolved}` 的顶层必须是对象")
    return state_from_dict(payload)


def save_state(state: TrainingState, path: str | Path) -> Path:
    """先写同目录临时文件再原子替换，失败时保留原有文件。"""

    resolved = Path(path)
    directory = resolved.parent if str(resolved.parent) != "" else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    payload = state_to_json(state)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(directory),
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - 清理失败不应掩盖原错误
            pass
        raise
    return resolved


def model_after_fit(
    previous: ModelSnapshot,
    result: FitResult,
    label_count: int,
) -> ModelSnapshot:
    """把一次拟合结果合并进模型；失败时保留最近成功的权重并记录落后标签数。"""

    if result.ok and result.weights is not None:
        return ModelSnapshot(
            trained=True,
            weights=np.asarray(result.weights, dtype=float),
            labels_covered=int(result.labels_covered),
            training_cross_entropy=result.training_cross_entropy,
            lagging_labels=max(label_count - int(result.labels_covered), 0),
        )
    return replace(previous, lagging_labels=max(label_count - previous.labels_covered, 0))


def lagging_count(state: TrainingState) -> int:
    """尚未被最近成功模型覆盖的有效标签数。"""

    return max(len(state.label_events) - state.model.labels_covered, 0)


def default_random_state(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed if isinstance(seed, int) else DEFAULT_SEED)
    return random_state_to_json(rng.bit_generator.state)


def rng_from_state(random_state: Mapping[str, Any]) -> np.random.Generator:
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = random_state_from_json(random_state)
    return generator


def processed_keys(state: TrainingState) -> set[tuple[str, str]]:
    """已处理的无序样本对，包含当前待答题。"""

    items: list[object] = list(state.events)
    if state.pending is not None:
        items.append(state.pending.question)
    return as_pairs(items)


def default_fitting_config(l2: float = DEFAULT_L2) -> FittingConfig:
    return FittingConfig(l2=l2)
