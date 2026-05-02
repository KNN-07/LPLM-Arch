"""Validate post-training YAML templates and print launch commands.

The YAML files in `post_training/templates` are configuration templates. This
helper makes them operationally visible: validate them, inspect their resolved
stage type, and print the trainer command that should consume each config.
Actual trainer scripts are intentionally separate from the templates.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to read post-training configs. Install with: "
        "pip install pyyaml"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
POST_TRAINING_ROOT = Path(__file__).resolve().parent

TRAINER_BY_STAGE_TYPE = {
    "continued_pretraining": POST_TRAINING_ROOT / "train_cpt.py",
    "supervised_fine_tuning": POST_TRAINING_ROOT / "train_sft.py",
    "preference_optimization": POST_TRAINING_ROOT / "train_preference.py",
    "reasoning_rl_verifiable_rewards": POST_TRAINING_ROOT
    / "train_reasoning_rlvr.py",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "continued_pretraining": ("stage", "model", "data", "training", "promotion_gate"),
    "supervised_fine_tuning": ("stage", "model", "data", "training", "promotion_gate"),
    "preference_optimization": (
        "stage",
        "model",
        "method",
        "data",
        "training",
        "promotion_gate",
    ),
    "reasoning_rl_verifiable_rewards": (
        "stage",
        "model",
        "algorithm",
        "reward",
        "training",
        "promotion_gate",
    ),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return REPO_ROOT / path


def validate_stage_config(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    stage = config.get("stage")
    if not isinstance(stage, dict):
        raise ValueError(f"{path} is missing a `stage` mapping.")

    stage_type = stage.get("type")
    if stage_type not in TRAINER_BY_STAGE_TYPE:
        raise ValueError(
            f"{path} has unsupported stage.type={stage_type!r}. Supported: "
            f"{', '.join(TRAINER_BY_STAGE_TYPE)}"
        )

    missing = [
        key for key in REQUIRED_TOP_LEVEL_KEYS[stage_type] if key not in config
    ]
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")

    model = config.get("model", {})
    training = config.get("training", {})
    if not isinstance(model, dict) or "input_checkpoint" not in model:
        raise ValueError(f"{path} model.input_checkpoint is required.")
    if not isinstance(model, dict) or "output_dir" not in model:
        raise ValueError(f"{path} model.output_dir is required.")
    if not isinstance(training, dict) or "optimizer" not in training:
        raise ValueError(f"{path} training.optimizer is required.")

    return {
        "id": stage.get("id", path.stem),
        "type": stage_type,
        "input_checkpoint": model["input_checkpoint"],
        "output_dir": model["output_dir"],
        "optimizer": training["optimizer"],
        "trainer": TRAINER_BY_STAGE_TYPE[stage_type],
    }


def command_for_stage(
    config_path: Path,
    trainer_path: Path,
    *,
    accelerate_config: str | None,
) -> list[str]:
    command = ["accelerate", "launch"]
    if accelerate_config:
        command.extend(["--config_file", accelerate_config])
    command.extend([str(trainer_path), "--config", str(config_path)])
    return command


def command_to_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def materialize_manifest_stage_configs(manifest_path: Path) -> list[Path]:
    manifest = load_yaml(manifest_path)
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{manifest_path} must contain a `stages` list.")

    output_root = REPO_ROOT / "outputs" / "post_training" / "resolved_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    config_paths: list[Path] = []
    for stage in stages:
        if not isinstance(stage, dict) or "template" not in stage:
            raise ValueError("Each manifest stage must include `template`.")
        template_path = resolve_path(stage["template"], base_dir=manifest_path.parent)
        config = copy.deepcopy(load_yaml(template_path))

        config.setdefault("stage", {})
        if "id" in stage:
            config["stage"]["id"] = stage["id"]
        config.setdefault("model", {})
        if "input_checkpoint" in stage:
            config["model"]["input_checkpoint"] = stage["input_checkpoint"]
        if "output_checkpoint" in stage:
            config["model"]["output_dir"] = stage["output_checkpoint"]

        resolved_path = output_root / f"{config['stage'].get('id', template_path.stem)}.yaml"
        with resolved_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        config_paths.append(resolved_path)
    return config_paths


def print_stage_plan(
    config_path: Path,
    *,
    accelerate_config: str | None,
    execute: bool,
) -> None:
    config = load_yaml(config_path)
    summary = validate_stage_config(config_path, config)
    command = command_for_stage(
        config_path,
        summary["trainer"],
        accelerate_config=accelerate_config,
    )

    print(f"stage: {summary['id']}")
    print(f"type: {summary['type']}")
    print(f"input_checkpoint: {summary['input_checkpoint']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"optimizer: {summary['optimizer']}")
    print(f"trainer: {summary['trainer']}")
    print(f"command: {command_to_text(command)}")

    if not summary["trainer"].exists():
        print(
            "status: trainer_missing. Implement this trainer or point the "
            "command at an existing trainer before using --execute."
        )
        if execute:
            raise FileNotFoundError(f"Trainer script not found: {summary['trainer']}")
        return

    if execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    else:
        print("status: dry_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate post-training config templates and print launch commands."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Single stage config YAML, such as templates/sft_config.yaml.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Stage manifest YAML. Prints a plan for every listed stage.",
    )
    parser.add_argument(
        "--accelerate_config",
        default=None,
        help="Optional accelerate config file passed to accelerate launch.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the trainer command. Dry-run is the default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.config) == bool(args.manifest):
        raise SystemExit("Pass exactly one of --config or --manifest.")

    if args.config:
        config_paths = [resolve_path(args.config, base_dir=Path.cwd())]
    else:
        manifest_path = resolve_path(args.manifest, base_dir=Path.cwd())
        config_paths = materialize_manifest_stage_configs(manifest_path)

    for index, config_path in enumerate(config_paths):
        if index:
            print("")
        print_stage_plan(
            config_path,
            accelerate_config=args.accelerate_config,
            execute=args.execute,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
