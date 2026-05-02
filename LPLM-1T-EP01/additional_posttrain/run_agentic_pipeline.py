"""Manage additional agentic SWE post-training scripts.

The runner is intentionally dry-run first: by default it prints the exact
commands for the selected stages. Pass --execute when you want it to run them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_yaml, resolve_path


SCRIPT_ROOT = Path(__file__).resolve().parent
VALID_STEPS = ("validate", "distill_hints", "shape_rewards", "train_rollouts")


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def default_shaped_output(config: dict, *, config_path: Path) -> Path:
    output_dir = resolve_path(
        config.get("model", {}).get(
            "output_dir",
            "outputs/additional_posttrain/agentic_swe_hrl_v1",
        ),
        base_dir=config_path.parent,
    )
    return output_dir.parent / "rollouts" / "shaped_rollouts.jsonl"


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    config = load_yaml(args.config)
    commands: list[list[str]] = []

    steps = args.steps or ["validate"]
    shaped_output = args.shaped_output or default_shaped_output(config, config_path=args.config)

    for step in steps:
        if step == "validate":
            commands.append(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "validate_data.py"),
                    "--config",
                    str(args.config),
                ]
            )
        elif step == "distill_hints":
            if args.tasks is None or args.hint_output is None:
                raise SystemExit(
                    "distill_hints requires --tasks and --hint-output."
                )
            commands.append(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "distill_hints_from_diffs.py"),
                    "--tasks",
                    str(args.tasks),
                    "--output",
                    str(args.hint_output),
                ]
            )
        elif step == "shape_rewards":
            if args.rollouts is None:
                raise SystemExit("shape_rewards requires --rollouts.")
            commands.append(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "shape_rollout_rewards.py"),
                    "--config",
                    str(args.config),
                    "--rollouts",
                    str(args.rollouts),
                    "--output",
                    str(shaped_output),
                ]
            )
        elif step == "train_rollouts":
            if "shape_rewards" in steps:
                rollouts = shaped_output
            elif args.rollouts is not None:
                rollouts = args.rollouts
            else:
                raise SystemExit(
                    "train_rollouts requires --rollouts unless shape_rewards is also selected."
                )
            command = [
                sys.executable,
                str(SCRIPT_ROOT / "train_from_rollouts.py"),
                "--config",
                str(args.config),
                "--rollouts",
                str(rollouts),
                "--max_steps",
                str(args.max_steps or config.get("training", {}).get("max_steps", 5000)),
                "--max_length",
                str(args.max_length),
                "--min_reward",
                str(args.min_reward),
            ]
            if args.output_dir is not None:
                command.extend(["--output_dir", str(args.output_dir)])
            commands.append(command)
        else:  # pragma: no cover
            raise SystemExit(f"Unknown step: {step}")
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or run the additional agentic SWE post-training pipeline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=VALID_STEPS,
        default=None,
        help="Pipeline steps to print or execute. Defaults to validate.",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument("--tasks", type=Path, default=None, help="Task JSONL for hint distillation.")
    parser.add_argument("--hint-output", type=Path, default=None, help="Output JSONL for distilled hints.")
    parser.add_argument("--rollouts", type=Path, default=None, help="Raw or shaped rollout JSONL.")
    parser.add_argument("--shaped-output", type=Path, default=None, help="Output path for shaped rollouts.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Training output directory override.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--min_reward", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)

    for command in commands:
        print(display_command(command))

    if not args.execute:
        print("dry run only; pass --execute to run these commands")
        return

    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
