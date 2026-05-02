"""Compute provenance-discounted shaped rewards for agent rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import final_reward, load_jsonl, load_task_records, load_yaml, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shape rewards for agentic SWE rollouts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True, help="Execution rollout JSONL.")
    parser.add_argument("--reward_config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    reward_config_path = args.reward_config
    if reward_config_path is None:
        reward_config_path = resolve_path(
            config.get("reward", {}).get("shaping_file", "templates/provenance_reward.yaml"),
            base_dir=args.config.parent,
        )
    reward_config = load_yaml(reward_config_path)
    hinting_file = config.get("hinting", {}).get("curriculum_file")
    if hinting_file:
        hinting_config = load_yaml(resolve_path(hinting_file, base_dir=args.config.parent))
        reward_config["hinting"] = hinting_config.get("hinting", {})
        reward_config["hinting"]["failure"] = hinting_config.get("failure", {})

    tasks = load_task_records(config, config_path=args.config)
    shaped = []
    for rollout in load_jsonl(args.rollouts):
        task_id = rollout.get("task_id")
        if task_id not in tasks:
            raise ValueError(f"Rollout references unknown task_id: {task_id}")
        reward = final_reward(task=tasks[task_id], rollout=rollout, reward_config=reward_config)
        enriched = dict(rollout)
        enriched["shaped_reward"] = reward
        shaped.append(enriched)

    write_jsonl(args.output, shaped)
    print(f"wrote {len(shaped)} shaped rollouts to {args.output}")


if __name__ == "__main__":
    main()
