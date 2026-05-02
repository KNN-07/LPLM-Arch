"""Create heuristic hint records from task diffs.

This is an offline bootstrapper. It does not call an LLM; it extracts touched
files and hunk counts from verified diffs and writes static hints that can be
manually reviewed or later replaced by higher-quality orchestrator output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import count_patch_hunks, extract_patch_files, load_jsonl, resolve_path, write_jsonl


def make_hint(task: dict, *, task_file: Path) -> dict:
    task_id = task["id"]
    diff_path_value = task.get("gold", {}).get("diff_path")
    if not diff_path_value:
        raise ValueError(f"{task_id}: gold.diff_path is required")
    diff_path = resolve_path(diff_path_value, base_dir=task_file.parent)
    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    files = extract_patch_files(diff_text)
    hunks = count_patch_hunks(diff_text)
    file_hint = ", ".join(files[:3]) if files else "the files changed by the reference fix"

    return {
        "task_id": task_id,
        "hint_source": "heuristic_diff_distilled",
        "hints": {
            "conceptual": "Focus on reproducing the reported failure, then make the smallest code change that addresses the root cause.",
            "localized": f"The verified fix touches {file_hint}. Inspect that area before editing unrelated code.",
            "procedural": f"Reproduce the issue, inspect {file_hint}, apply a minimal fix, then rerun the task tests. The reference patch has {hunks} hunks.",
        },
        "grounding": {
            "diff_path": str(diff_path_value),
            "files_touched": files,
            "hunk_count": hunks,
        },
        "quality": {
            "verified_against_gold_diff": True,
            "manual_reviewed": False,
            "heuristic": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill heuristic hints from task diffs.")
    parser.add_argument("--tasks", type=Path, required=True, help="Task JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output hint JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.tasks)
    hints = [make_hint(task, task_file=args.tasks) for task in tasks]
    write_jsonl(args.output, hints)
    print(f"wrote {len(hints)} hints to {args.output}")


if __name__ == "__main__":
    main()
