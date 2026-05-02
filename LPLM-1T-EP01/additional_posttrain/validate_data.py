"""Validate agentic SWE task and hint JSONL data."""

from __future__ import annotations

from common import load_hint_records, load_task_records, load_yaml, parse_config_arg


VALID_PROVENANCE = {"human_verified", "synthetic"}


def validate_task(task_id: str, task: dict) -> list[str]:
    errors: list[str] = []
    if task.get("provenance") not in VALID_PROVENANCE:
        errors.append(f"{task_id}: invalid provenance {task.get('provenance')!r}")
    repo = task.get("repo", {})
    issue = task.get("issue", {})
    reward = task.get("reward", {})
    if not repo.get("url"):
        errors.append(f"{task_id}: repo.url is required")
    if not repo.get("base_commit"):
        errors.append(f"{task_id}: repo.base_commit is required")
    if not issue.get("title") or not issue.get("body"):
        errors.append(f"{task_id}: issue.title and issue.body are required")
    if "provenance_cap" not in reward:
        errors.append(f"{task_id}: reward.provenance_cap is required")
    return errors


def validate_hint(task_id: str, hint: dict) -> list[str]:
    errors: list[str] = []
    hints = hint.get("hints", {})
    for key in ("conceptual", "localized", "procedural"):
        value = hints.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{task_id}: hints.{key} is required")
    if not hint.get("grounding", {}).get("diff_path"):
        errors.append(f"{task_id}: grounding.diff_path is required")
    return errors


def main() -> None:
    args = parse_config_arg("Validate agentic SWE task and hint data.")
    config = load_yaml(args.config)
    tasks = load_task_records(config, config_path=args.config)
    hints = load_hint_records(config, config_path=args.config)

    errors: list[str] = []
    if not tasks:
        errors.append("no task records found from data.task_paths")
    if not hints:
        errors.append("no hint records found from data.hint_paths")
    for task_id, task in tasks.items():
        errors.extend(validate_task(task_id, task))
        if task_id not in hints:
            errors.append(f"{task_id}: missing hint record")
    for task_id, hint in hints.items():
        errors.extend(validate_hint(task_id, hint))
        if task_id not in tasks:
            errors.append(f"{task_id}: hint has no matching task")

    if errors:
        print("validation failed")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"validation ok: {len(tasks)} tasks, {len(hints)} hints")


if __name__ == "__main__":
    main()
