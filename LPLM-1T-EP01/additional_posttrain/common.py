"""Utilities for additional agentic SWE post-training scripts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install PyYAML to run these scripts: pip install pyyaml") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
ADDITIONAL_ROOT = Path(__file__).resolve().parent
PATCH_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
HUNK_RE = re.compile(r"^@@ .* @@")


def parse_config_arg(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, allow_special_tokens=True)
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def expand_patterns(patterns: list[str], *, base_dir: Path | None = None) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        search = str(pattern_path if pattern_path.is_absolute() else REPO_ROOT / pattern)
        matches = [Path(match).resolve() for match in glob.glob(search, recursive=True)]
        if not matches and base_dir is not None:
            matches = [
                Path(match).resolve()
                for match in glob.glob(str(base_dir / pattern), recursive=True)
            ]
        files.extend(matches)
    return sorted(set(files))


def extract_patch_files(diff_text: str) -> list[str]:
    files: list[str] = []
    for line in diff_text.splitlines():
        match = PATCH_FILE_RE.match(line)
        if match:
            files.append(match.group(1))
    return files


def count_patch_hunks(diff_text: str) -> int:
    return sum(1 for line in diff_text.splitlines() if HUNK_RE.match(line))


def load_task_records(config: dict[str, Any], *, config_path: Path) -> dict[str, dict[str, Any]]:
    data = config.get("data", {})
    paths = data.get("task_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    files = expand_patterns([str(path) for path in paths], base_dir=config_path.parent)
    records: dict[str, dict[str, Any]] = {}
    for file in files:
        for record in load_jsonl(file):
            task_id = record.get("id")
            if not isinstance(task_id, str):
                raise ValueError(f"Task without string id in {file}")
            records[task_id] = record
    return records


def load_hint_records(config: dict[str, Any], *, config_path: Path) -> dict[str, dict[str, Any]]:
    data = config.get("data", {})
    paths = data.get("hint_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    files = expand_patterns([str(path) for path in paths], base_dir=config_path.parent)
    records: dict[str, dict[str, Any]] = {}
    for file in files:
        for record in load_jsonl(file):
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError(f"Hint without string task_id in {file}")
            records[task_id] = record
    return records


def provenance_cap(provenance: str, reward_config: dict[str, Any]) -> float:
    caps = reward_config.get("reward", {}).get("provenance_caps", {})
    return float(caps.get(provenance, 0.0))


def hint_discount(hint_level: str, reward_config: dict[str, Any]) -> float:
    discounts = reward_config.get("reward", {}).get("hint_discounts", {})
    return float(discounts.get(hint_level, 0.0))


def plan_score(plan_reward: dict[str, Any], reward_config: dict[str, Any]) -> float:
    if not plan_reward:
        return 0.0
    if "score" in plan_reward:
        return float(plan_reward["score"])
    components = reward_config.get("reward", {}).get("plan_reward", {}).get("components", {})
    checks = plan_reward.get("checks", {})
    score = 0.0
    for name, weight in components.items():
        score += float(weight) if bool(checks.get(name, False)) else 0.0
    return max(0.0, min(score, 1.0))


def penalty_total(attempt: dict[str, Any], reward_config: dict[str, Any]) -> float:
    penalties = reward_config.get("reward", {}).get("penalties", {})
    flags = attempt.get("penalties", {})
    if not isinstance(flags, dict):
        return 0.0
    total = 0.0
    for key, enabled in flags.items():
        if enabled:
            total += float(penalties.get(key, 0.0))
    return total


def success_score(attempt: dict[str, Any], reward_config: dict[str, Any]) -> float:
    tests = attempt.get("tests", {})
    if bool(tests.get("private_passed", False)):
        return 1.0
    if "private_pass_rate" in tests:
        private_rate = float(tests.get("private_pass_rate", 0.0))
        public_rate = float(tests.get("public_pass_rate", 0.0))
        success_cfg = reward_config.get("reward", {}).get("execution_success", {})
        return (
            float(success_cfg.get("private_test_weight", 1.0)) * private_rate
            + float(success_cfg.get("public_test_weight", 0.0)) * public_rate
        )
    return 1.0 if bool(tests.get("public_passed", False)) and bool(tests.get("private_passed", False)) else 0.0


def final_reward(
    *,
    task: dict[str, Any],
    rollout: dict[str, Any],
    reward_config: dict[str, Any],
) -> dict[str, float | str]:
    provenance = str(task.get("provenance", "unknown"))
    cap = provenance_cap(provenance, reward_config)
    plan = plan_score(rollout.get("plan_reward", {}), reward_config)
    plan_weight = float(
        reward_config.get("reward", {}).get("plan_reward", {}).get("weight", 0.1)
    )

    best_reward = float(reward_config.get("hinting", {}).get("failure", {}).get("reward", -0.1))
    best_hint = "failure"
    best_success = 0.0
    best_penalty = 0.0
    for attempt in rollout.get("attempts", []):
        hint_level = str(attempt.get("hint_level", "failure"))
        success = success_score(attempt, reward_config)
        penalties = penalty_total(attempt, reward_config)
        execution_reward = cap * hint_discount(hint_level, reward_config) * success
        shaped = execution_reward + plan_weight * plan - penalties
        shaped = max(-1.0, min(shaped, cap))
        if shaped > best_reward:
            best_reward = shaped
            best_hint = hint_level
            best_success = success
            best_penalty = penalties

    if math.isnan(best_reward):
        best_reward = -1.0
    return {
        "provenance": provenance,
        "provenance_cap": cap,
        "plan_score": plan,
        "hint_level": best_hint,
        "success_score": best_success,
        "penalties": best_penalty,
        "final_reward": best_reward,
    }
