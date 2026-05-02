# Runbook

This runbook describes how to operationalize the additional agentic SWE RL
approach. The included scripts cover config/data validation, offline hint
bootstrapping, reward shaping, and offline training from captured rollouts. A
stateful repository sandbox is still required to produce the raw agent rollouts.

## 1. Prepare Checkpoint

Start from a checkpoint that has already completed:

1. SFT.
2. Preference tuning.
3. Tool-use validation.
4. Basic code RLVR.

Set it in `templates/agentic_swe_rl_config.yaml`:

```yaml
model:
  input_checkpoint: outputs/post_training/pref_v1
```

## 2. Build Human Anchor Tasks

For each human-verified issue:

1. Save repo URL and base commit.
2. Save issue title/body.
3. Save merged PR diff.
4. Save public and private test commands.
5. Assign `provenance: human_verified`.
6. Set `provenance_cap: 1.0`.

Store records using `DATA_SCHEMAS.md#swe-task-record`.

## 3. Build Synthetic Volume Tasks

For synthetic tasks:

1. Generate task prompt, solution patch, and tests.
2. Verify the solution patch passes tests from a clean checkout.
3. Reject tasks that require hidden external services.
4. Assign `provenance: synthetic`.
5. Set `provenance_cap: 0.6`.

Synthetic tasks should teach tool mechanics and common edit patterns, not define
the model's final code style.

## 4. Distill Hints Offline

For every task, create static hints:

```text
conceptual -> broad strategy
localized  -> subsystem or file-level pointer
procedural -> explicit step sequence
```

Hints for human tasks must be grounded in the human PR diff. Do not generate
runtime hints during RL rollout unless you are explicitly running an ablation.

To bootstrap hints from `gold.diff_path` fields:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\distill_hints_from_diffs.py --tasks .\data\agentic_swe\human_verified\tasks.jsonl --output .\data\agentic_swe\hints\heuristic_hints.jsonl
```

These heuristic hints are intentionally conservative and should be manually
reviewed before large runs.

## 5. Validate Blind Prompting

Before training, confirm prompts do not reveal:

- `provenance`
- reward cap
- whether the task is synthetic
- whether hints came from a human diff

Only the reward engine sees this metadata.

Validate the task and hint files referenced by the config:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\validate_data.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml
```

## 6. Run Planner Warmup

Train or score `<PLAN>` blocks before full agent RL:

1. Sample task.
2. Ask model for `<PLAN>`.
3. Score with PRM or rule-based rubric.
4. Use this as a dense auxiliary reward.

This can be implemented as SFT plus reward-model scoring before full GRPO.

## 7. Run Agentic GRPO

For each task:

1. Sample `G` independent rollouts.
2. Run no-hint attempt.
3. If tests fail, reset repo and inject conceptual hint.
4. If tests fail again, reset repo and inject localized/procedural hint.
5. Compute final shaped reward.
6. Normalize advantages within the group.
7. Apply GRPO update with KL control.

The online sandbox rollout collector is the remaining environment integration
point. After it writes raw rollout JSONL records, apply the reward model:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\shape_rollout_rewards.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml --rollouts .\rollouts\agentic_swe\raw_rollouts.jsonl --output .\rollouts\agentic_swe\shaped_rollouts.jsonl
```

Then run the offline weighted rollout trainer:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\train_from_rollouts.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml --rollouts .\rollouts\agentic_swe\shaped_rollouts.jsonl --max_steps 1000
```

To manage the same steps from one entrypoint, start with a dry run:

```powershell
python .\LPLM-1T-EP01\additional_posttrain\run_agentic_pipeline.py --config .\LPLM-1T-EP01\additional_posttrain\templates\agentic_swe_rl_config.yaml --steps validate shape_rewards train_rollouts --rollouts .\rollouts\agentic_swe\raw_rollouts.jsonl --shaped-output .\rollouts\agentic_swe\shaped_rollouts.jsonl
```

Add `--execute` only after the printed commands are correct.

## 8. Evaluate Promotion

Required held-out gates:

- Human-verified pass@1 improves.
- Hint reliance decreases.
- Synthetic success does not mask human-task regression.
- Patch size and touched-file count stay bounded.
- No test deletion or weakening.
- Tool transcripts are reproducible.

Do not promote a checkpoint solely because synthetic pass rate improves.
