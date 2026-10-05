"""``hotel-model`` 命令行入口。

第一版只暴露训练入口；没有评分、推荐或网页命令。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

from .data import DataError
from .fitting import DEFAULT_L2
from .select import DEFAULT_SEED
from .state import StateError
from .train import StreamConsole, TrainOptions, run_training

PROGRAM = "hotel-model"
DESCRIPTION = "本地酒店偏好训练器：用同城两两比较训练共享线性权重（仅训练，不提供评分或推荐）。"

EXIT_OK = 0
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM, description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", metavar="train")

    train = subparsers.add_parser(
        "train",
        help="新建或恢复一个训练会话",
        description=(
            "新建会话：hotel-model train --data hotels.csv --state session.json\n"
            "恢复会话：hotel-model train --resume session.json [--data hotels.csv]"
        ),
    )
    train.add_argument("--data", type=Path, default=None, help="原始酒店 CSV 路径（新建时必需）")
    train.add_argument("--state", type=Path, default=None, help="训练状态 JSON 路径")
    train.add_argument("--resume", action="store_true", help="从已有状态文件恢复训练")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子（默认 {DEFAULT_SEED}）")
    train.add_argument("--l2", type=float, default=DEFAULT_L2, help=f"L2 正则强度（默认 {DEFAULT_L2:g}）")
    train.add_argument("--quiet", action="store_true", help="结束后不打印诊断")
    return parser


def options_from_args(args: argparse.Namespace) -> TrainOptions:
    if args.resume:
        if args.state is None:
            if args.data is None:
                raise StateError("恢复训练需要 `--resume` 指定状态文件")
            raise StateError("恢复训练需要 `--state` 指定状态文件")
        return TrainOptions(
            state_path=Path(args.state),
            data_path=Path(args.data) if args.data is not None else None,
            resume=True,
            seed=args.seed,
            l2=args.l2,
            show_diagnostics=not args.quiet,
        )
    if args.state is None:
        raise StateError("新建训练需要 `--state` 指定状态文件路径")
    if args.data is None:
        raise StateError("新建训练需要 `--data` 指定数据文件")
    return TrainOptions(
        state_path=Path(args.state),
        data_path=Path(args.data),
        resume=False,
        seed=args.seed,
        l2=args.l2,
        show_diagnostics=not args.quiet,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    if args.command != "train":
        parser.print_help(out)
        return EXIT_ERROR

    try:
        options = options_from_args(args)
    except StateError as error:
        err.write(f"参数错误：{error}\n")
        return EXIT_ERROR

    console = StreamConsole(stdin=stdin, stdout=out)
    try:
        outcome = run_training(options, console=console)
    except (StateError, DataError, ValueError) as error:
        err.write(f"训练中止：{error}\n")
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        err.write("\n已中断，已保存的进度保留。\n")
        return EXIT_ERROR

    out.write(f"训练结束（{outcome.stop_reason}）。\n")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
