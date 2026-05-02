# LPLM-1T-EP01 Post-Training

This directory contains the post-training plan and templates for turning the
FineWeb-pretrained EP01 base model into usable instruction, reasoning, tool,
and safety variants.

Recommended sequence:

1. Continue pretraining on curated domain mixtures.
2. Supervised fine-tuning with the LPLM chat template.
3. Preference optimization with DPO, SimPO, or KTO.
4. Reasoning RL with verifiable rewards, preferably GRPO/RLVR style.
5. Safety, tool-use, and refusal polishing.
6. Evaluation gates before every promotion.

## Files

| Path | Purpose |
| --- | --- |
| `POST_TRAINING_STRATEGY.md` | Detailed post-training recommendations and stage criteria. |
| `DATA_SCHEMAS.md` | JSONL schemas for SFT, preference, KTO, RLVR, and eval data. |
| `RUNNING_TEMPLATES.md` | How to copy, validate, dry-run, and execute stage configs. |
| `run_stage.py` | Helper that validates stage YAML and prints the expected launch command. |
| `common.py` | Shared config, tokenizer, model, dataset, W&B, and Trainer utilities. |
| `train_cpt.py` | Continued pretraining trainer for JSONL text data. |
| `train_sft.py` | SFT trainer for LPLM-rendered conversation JSONL. |
| `train_preference.py` | SimPO, DPO, and KTO-style preference trainer. |
| `train_reasoning_rlvr.py` | GRPO/RLVR-style trainer with verifiable rewards. |
| `templates/stage_manifest.yaml` | End-to-end stage manifest template. |
| `templates/data_mixture.yaml` | Dataset mixture template for CPT and SFT. |
| `templates/cpt_config.yaml` | Continued pretraining stage template. |
| `templates/sft_config.yaml` | Supervised fine-tuning stage template. |
| `templates/preference_config.yaml` | DPO, SimPO, and KTO stage template. |
| `templates/reasoning_rlvr_config.yaml` | GRPO/RLVR-style reasoning template. |
| `templates/eval_gate.yaml` | Promotion gate and evaluation template. |
| `templates/chat_template.lplm.jinja` | LPLM chat rendering template for post-training data. |

## Operational Notes

- Keep the base checkpoint immutable. Each stage should write to a new output
  directory and include a W&B run id in metadata.
- Keep general replay data in CPT and SFT to reduce catastrophic forgetting.
- Track MoE health during every stage: expert load balance, routing entropy,
  dead experts, gate saturation, per-layer active expert distribution, and
  token throughput.
- Keep MuonClip/QK-Clip enabled unless a controlled ablation shows a better
  stability profile.
- Do not promote a checkpoint unless it passes the evaluation gate for the
  target stage.

## Quick Validation

Validate one stage template:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --config .\LPLM-1T-EP01\post_training\templates\sft_config.yaml
```

Validate the full manifest:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --manifest .\LPLM-1T-EP01\post_training\templates\stage_manifest.yaml
```

See `RUNNING_TEMPLATES.md` for execution details and the trainer script
contract.

Run a stage after copying and editing a config:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --config .\runs\post_training\sft_instruct_v1.yaml `
  --execute
```
