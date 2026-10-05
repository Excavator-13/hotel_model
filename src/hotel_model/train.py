"""终端训练循环：新建/恢复会话、展示同城问题、记录答案并重新拟合。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, TextIO

import numpy as np

from .data import Sample, load_csv, samples_from_ids
from .diagnostics import format_diagnostics, format_sample, summarize
from .features import FeatureConfig, feature_vector_to_list, to_features
from .fitting import (
    KIND_LABEL,
    KIND_SKIP,
    LEFT_WINS,
    RIGHT_WINS,
    TIE,
    FitResult,
    FittingConfig,
    fit,
    labeled_pairs,
)
from .select import (
    DEFAULT_SEED,
    EXHAUSTED,
    Question,
    SelectionConfig,
    reason_for_empty_candidates,
    select_next,
)
from .state import (
    PendingQuestion,
    StateError,
    TrainingState,
    content_fingerprint,
    default_random_state,
    load_state,
    model_after_fit,
    model_after_label,
    random_state_to_json,
    rng_from_state,
    save_state,
    state_to_json,
)

ANSWER_CHOICES: Mapping[str, str] = {
    "a": "A 更好",
    "b": "B 更好",
    "t": "差不多",
    "s": "无法判断",
    "q": "退出",
}

PROMPT = "请选择 [a] A更好 / [b] B更好 / [t] 差不多 / [s] 无法判断 / [q] 退出："
REPROMPT = "无法识别的输入，请重新输入 a / b / t / s / q。"


class Console(Protocol):
    """终端输入输出接口，便于在测试中注入受控输入。"""

    def write(self, text: str) -> None: ...

    def read_line(self, prompt: str) -> str | None: ...


class StreamConsole:
    """基于文本流的终端实现；EOF 返回 ``None``，Ctrl-C 向上传播。"""

    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text + "\n")
        self._stdout.flush()

    def read_line(self, prompt: str) -> str | None:
        self._stdout.write(prompt)
        self._stdout.flush()
        line = self._stdin.readline()
        if line == "":
            return None
        return line.rstrip("\n")


class ScriptedConsole:
    """脚本化输入，供测试和示例复现使用。"""

    def __init__(self, answers: Sequence[str], *, echo: bool = True) -> None:
        self._answers = list(answers)
        self._index = 0
        self.lines: list[str] = []
        self._echo = echo

    def write(self, text: str) -> None:
        self.lines.append(text)

    def read_line(self, prompt: str) -> str | None:
        if self._echo:
            self.lines.append(prompt)
        if self._index >= len(self._answers):
            return None
        answer = self._answers[self._index]
        self._index += 1
        if self._echo:
            self.lines.append(answer)
        return answer

    @property
    def output(self) -> str:
        return "\n".join(self.lines)


@dataclass
class TrainOptions:
    """一次 ``hotel-model train`` 调用的参数。"""

    state_path: Path
    data_path: Path | None = None
    resume: bool = False
    seed: int = DEFAULT_SEED
    l2: float = 0.1
    show_diagnostics: bool = True


@dataclass
class Session:
    """训练状态及其持久化路径；所有写入都经过它。"""

    state: TrainingState
    path: Path

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self.state.samples

    def vectors(self) -> dict[str, np.ndarray]:
        return {
            sample.hotel_id: to_features(sample, self.state.preprocessing)
            for sample in self.state.samples
        }

    def label_count(self) -> int:
        return len(self.state.label_events)

    def labels(self) -> list:
        return labeled_pairs(self.state.events, self.vectors())

    def processed(self) -> list[object]:
        items: list[object] = list(self.state.events)
        if self.state.pending is not None:
            items.append(self.state.pending.question)
        return items

    def refit(self) -> FitResult:
        return fit(self.labels(), self.state.fitting, initial=self.state.model.weights)

    def persist(self, updated: TrainingState) -> "Session":
        save_state(updated, self.path)
        return Session(state=updated, path=self.path)

    def with_pending(self, question: Question, asked_at: str, rng: np.random.Generator) -> "Session":
        pending = PendingQuestion(question=question, asked_at=asked_at)
        updated = self.state.with_(
            updated_at=asked_at,
            pending=pending,
            random_state=random_state_to_json(rng.bit_generator.state),
        )
        return self.persist(updated)


@dataclass
class TrainOutcome:
    """训练结束后的结果摘要。"""

    session: Session
    stop_reason: str
    questions_asked: int = 0
    fit_failures: int = 0

    @property
    def state(self) -> TrainingState:
        return self.session.state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session(
    samples: Sequence[Sample],
    options: TrainOptions,
    *,
    now: str | None = None,
) -> Session:
    """创建新会话；调用方必须已确认状态文件不存在。"""

    timestamp = now if now is not None else _now()
    state = TrainingState.new(
        samples,
        preprocessing=FeatureConfig(),
        fitting=FittingConfig(l2=options.l2),
        selection=SelectionConfig(seed=options.seed),
        random_state=default_random_state(options.seed),
        created_at=timestamp,
    )
    return Session(state=state, path=Path(options.state_path))


def resume_session(options: TrainOptions) -> Session:
    """从状态文件恢复；可选 CSV 只用于一致性校验。"""

    state = load_state(options.state_path)
    if options.data_path is not None:
        fresh = load_csv(options.data_path)
        if content_fingerprint(fresh) != state.fingerprint:
            raise StateError(
                f"`{options.data_path}` 的内容与状态快照不一致，拒绝把旧标签套用到新数据"
            )
    return Session(state=state, path=Path(options.state_path))


def run_training(
    options: TrainOptions,
    *,
    console: Console,
    now: Callable[[], str] | None = None,
) -> TrainOutcome:
    """执行完整训练循环，直到退出、候选耗尽或输入结束。"""

    clock = now if now is not None else _now
    state_path = Path(options.state_path)

    if options.resume:
        session = resume_session(options)
    else:
        if state_path.exists():
            raise StateError(f"状态文件 `{state_path}` 已存在，新建会话不会覆盖它")
        if options.data_path is None:
            raise StateError("新建会话必须提供 `--data` 数据文件")
        samples = load_csv(options.data_path)
        session = new_session(samples, options, now=clock())
        session = session.persist(session.state)
        console.write(
            f"已创建训练会话：{state_path}（样本 {len(samples)} 条，"
            f"城市 {len({s.city for s in samples})} 个，种子 {session.state.selection.seed}，"
            f"L2={session.state.fitting.l2:g}）"
        )

    console.write("输入 a/b/t/s/q 回答；q、EOF 或 Ctrl-C 都会保留已保存的进度。")
    questions_asked = 0
    fit_failures = 0
    stop_reason = "exhausted"

    # 恢复时若最近成功模型落后于已有标签，先重新拟合再继续提问。
    if session.state.model.lagging_labels > 0 and session.label_count() > 0:
        console.write(
            f"检测到 {session.state.model.lagging_labels} 个有效标签尚未拟合，先重新拟合。"
        )
        session, result = _apply_fit(session, session.refit(), now=clock(), console=console)
        if not result.ok:
            _report(session, console, options)
            return TrainOutcome(session, "fit_failed", questions_asked, fit_failures + 1)

    show_pending = False
    try:
        while True:
            if session.state.pending is None:
                selected = _select(session, console, now=clock())
                if selected is None:
                    stop_reason = "exhausted"
                    break
                session, question = selected
                questions_asked += 1
                _display(session.state, question, console)
                show_pending = True
                continue

            if not show_pending:
                _display(session.state, session.state.pending.question, console)
                show_pending = True

            line = console.read_line(PROMPT)
            if line is None:
                stop_reason = "eof"
                break
            answer = line.strip().lower()
            if answer == "q":
                stop_reason = "quit"
                break
            if answer not in ANSWER_CHOICES:
                console.write(REPROMPT)
                continue

            session = _record_answer(session, answer, now=clock())
            console.write(f"已记录：{ANSWER_CHOICES[answer]}")

            if session.label_count() == 0:
                # 还没有有效标签：模型保持未训练，这不是拟合失败。
                show_pending = False
                continue

            session, result = _apply_fit(session, session.refit(), now=clock(), console=console)
            if not result.ok:
                fit_failures += 1
                _report(session, console, options)
                return TrainOutcome(session, "fit_failed", questions_asked, fit_failures)
            show_pending = False
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        console.write("")
        console.write("收到中断，已保存的进度保留。")

    _report(session, console, options)
    return TrainOutcome(session, stop_reason, questions_asked, fit_failures)


def _select(
    session: Session,
    console: Console,
    *,
    now: str,
) -> tuple[Session, Question] | None:
    """选题并先保存待答题与抽取后的随机状态。"""

    rng = rng_from_state(session.state.random_state)
    samples = session.state.samples
    vectors = session.vectors()
    question = select_next(
        samples,
        processed=session.processed(),
        vectors=vectors,
        weights=session.state.model.weights,
        label_count=session.label_count(),
        rng=rng,
        config=session.state.selection,
    )
    if question is None:
        if reason_for_empty_candidates(samples, vectors) == EXHAUSTED:
            console.write("所有符合条件的同城样本对都已回答或跳过，问题已耗尽。")
            console.write("程序正常结束，不重复提问，也不构造跨城市问题。")
        else:
            console.write(
                "没有可比较的候选：每个城市不足两个样本，或所有未处理的同城样本对特征相同。"
            )
            console.write("程序正常结束，不构造跨城市问题。")
        return None
    return (session.with_pending(question, now, rng), question)


def _display(state: TrainingState, question: Question, console: Console) -> None:
    samples = state.sample_map()
    left = samples[question.left_id]
    right = samples[question.right_id]
    console.write("")
    console.write(f"A 候选（{left.city}）")
    console.write(format_sample(left))
    console.write(f"B 候选（{right.city}）")
    console.write(format_sample(right))


def _record_answer(session: Session, answer: str, *, now: str) -> Session:
    """先追加事件并清除待答题，原子保存后再由调用方拟合。

    答案检查点必须在拟合之前就能被 ``load_state`` 读取，因此这里同时把模型的
    落后标签数更新到最新有效标签数：权重仍是最近一次成功拟合的结果，覆盖数与
    有效标签数保持自洽。
    """

    pending = session.state.pending
    assert pending is not None
    question = pending.question
    if answer == "a":
        kind, label = KIND_LABEL, LEFT_WINS
    elif answer == "b":
        kind, label = KIND_LABEL, RIGHT_WINS
    elif answer == "t":
        kind, label = KIND_LABEL, TIE
    else:
        kind, label = KIND_SKIP, None

    samples = session.state.sample_map()
    event: dict[str, object] = {
        "left_id": question.left_id,
        "right_id": question.right_id,
        "city": samples[question.left_id].city,
        "kind": kind,
        "label": label,
        "answered_at": now,
        "asked_at": pending.asked_at,
        "answer": answer,
    }
    events = list(session.state.events) + [event]
    label_count = sum(1 for item in events if item.get("kind") == KIND_LABEL)
    updated = session.state.with_(
        updated_at=now,
        events=events,
        pending=None,
        model=model_after_label(session.state.model, label_count),
    )
    return session.persist(updated)


def _apply_fit(
    session: Session,
    result: FitResult,
    *,
    now: str,
    console: Console,
) -> tuple[Session, FitResult]:
    if result.ok:
        updated = session.state.with_(
            updated_at=now,
            model=model_after_fit(session.state.model, result, session.label_count()),
        )
        return session.persist(updated), result

    console.write(f"拟合失败：{result.message}")
    console.write("已保留最近成功的权重，并停止继续选题；下次恢复会先重新拟合。")
    updated = session.state.with_(
        updated_at=now,
        model=model_after_fit(session.state.model, result, session.label_count()),
    )
    return session.persist(updated), result


def _report(session: Session, console: Console, options: TrainOptions) -> None:
    if not options.show_diagnostics:
        return
    counts = summarize(session.state, session.vectors())
    for line in format_diagnostics(session.state, counts):
        console.write(line)
    console.write("")
    console.write(f"状态文件：{session.path}")


def render_state_summary(state: TrainingState) -> str:
    """供测试与调试使用的完整状态序列化。"""

    return state_to_json(state)


def features_for(
    samples: Sequence[Sample],
    config: FeatureConfig | None = None,
) -> dict[str, list[float]]:
    settings = config if config is not None else FeatureConfig()
    return {
        sample.hotel_id: feature_vector_to_list(to_features(sample, settings))
        for sample in samples
    }


def selected_samples(state: TrainingState, hotel_ids: Sequence[str]) -> list[Sample]:
    return samples_from_ids(state.samples, hotel_ids)
