# Additional Post-Training: Agentic SWE RL

This folder documents an additional post-training approach for agentic software
engineering: provenance-discounted hierarchical reinforcement learning with a
human-grounded hinting curriculum.

The approach is intended to run after the base post-training stack in
`LPLM-1T-EP01/post_training` has produced a stable instruction/tool-capable
checkpoint.

## Contents

| Path | Purpose |
| --- | --- |
| `paper/provenance_discounted_hrl.tex` | LaTeX write-up of the proposed method. |
| `APPROACH.md` | Engineering interpretation of the method and training flow. |
| `DATA_SCHEMAS.md` | JSONL schemas for SWE tasks, hints, plans, rollouts, and rewards. |
| `REWARD_SHAPING.md` | Reward formulas for planning, hint discounting, and provenance caps. |
| `RUNBOOK.md` | Operational steps to prepare data and run the curriculum. |
| `run_agentic_pipeline.py` | Dry-run-first manager for validation, hint distillation, reward shaping, and rollout training. |
| `validate_data.py` | Checks task and hint JSONL records referenced by the config. |
| `distill_hints_from_diffs.py` | Creates heuristic static hints from verified patch diffs. |
| `shape_rollout_rewards.py` | Adds provenance-discounted shaped rewards to rollout JSONL. |
| `train_from_rollouts.py` | Offline weighted causal-LM trainer over shaped agent transcripts using MuonClip. |
| `templates/agentic_swe_rl_config.yaml` | End-to-end config template for the method. |
| `templates/hint_curriculum.yaml` | Hint schedule and intervention template. |
| `templates/provenance_reward.yaml` | Reward cap and provenance weighting template. |

## High-Level Sequence

1. Start from a post-SFT or preference-tuned checkpoint.
2. Build a dataset of software engineering tasks from human PRs and synthetic
   tasks.
3. Distill static hint curricula from verified human diffs.
4. Train a planner objective on `<PLAN>` blocks.
5. Run grouped agent rollouts with fallback hints.
6. Shape rewards by autonomy level and data provenance.
7. Optimize with GRPO or GiGPO while keeping task provenance hidden from the
   model prompt.

This is a design and implementation contract. It is separate from the current
single-turn post-training trainers because it requires a stateful code
environment, repository checkout manager, test runner, and rollout database.

## Script Quick Start

Print the validation command for the default config:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\run_agentic_pipeline.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml
```

Run validation directly:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\validate_data.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml
```

Create reviewable heuristic hints from a task file with verified diffs:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\distill_hints_from_diffs.py --tasks .\data\agentic_swe\human_verified\tasks.jsonl --output .\data\agentic_swe\hints\heuristic_hints.jsonl
```

Shape raw rollout rewards:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\shape_rollout_rewards.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml --rollouts .\rollouts\agentic_swe\raw_rollouts.jsonl --output .\rollouts\agentic_swe\shaped_rollouts.jsonl
```

Train from shaped rollouts:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\train_from_rollouts.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml --rollouts .\rollouts\agentic_swe\shaped_rollouts.jsonl --max_steps 1000
```

Run a selected pipeline sequence after reviewing the printed commands:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\run_agentic_pipeline.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml --steps validate shape_rewards train_rollouts --rollouts .\rollouts\agentic_swe\raw_rollouts.jsonl --shaped-output .\rollouts\agentic_swe\shaped_rollouts.jsonl --execute
```
