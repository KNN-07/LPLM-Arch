# Running The Post-Training Templates

The files in `post_training/templates` are stage configs. They are meant to be
copied, edited, validated, then passed to a trainer script.

This repo includes config templates, stage-specific trainers, and a
validation/launch helper at `post_training/run_stage.py`. The helper prints the
exact `accelerate launch` shape for each config and can execute it with
`--execute`.

## 1. Install Runtime Dependencies

```powershell
pip install pyyaml accelerate transformers datasets torch wandb
```

For large EP01 runs, also install the distributed stack you plan to use:

```powershell
pip install deepspeed
```

## 2. Copy A Template Before Editing

Keep the checked-in templates stable. Copy one into a run-specific file:

```powershell
New-Item -ItemType Directory -Force .\runs\post_training
Copy-Item .\LPLM-1T-EP01\post_training\templates\sft_config.yaml `
  .\runs\post_training\sft_instruct_v1.yaml
```

Edit the copied file:

- `model.input_checkpoint`
- `model.output_dir`
- dataset paths under `data`
- `training.max_steps` or `training.num_train_epochs`
- W&B project and run metadata

## 3. Validate One Stage

Dry-run a single config:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --config .\runs\post_training\sft_instruct_v1.yaml
```

Expected output includes:

```text
stage: sft_instruct_v1
type: supervised_fine_tuning
input_checkpoint: outputs/post_training/cpt_domain_v1
output_dir: outputs/post_training/sft_instruct_v1
optimizer: muonclip
trainer: D:\Repos\LPLM-Arch\LPLM-1T-EP01\post_training\train_sft.py
command: accelerate launch ... train_sft.py --config ...sft_instruct_v1.yaml
status: dry_run
```

## 4. Validate The Full Manifest

Run the manifest dry-run:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --manifest .\LPLM-1T-EP01\post_training\templates\stage_manifest.yaml
```

This prints the planned command for each stage in order:

1. CPT
2. SFT
3. Preference optimization
4. Reasoning RLVR
5. Safety/tool polish

Manifest stages can override `input_checkpoint` and `output_checkpoint`. The
helper materializes effective per-stage configs under:

```text
outputs/post_training/resolved_configs/
```

The printed commands point at those resolved configs.

## 5. Execute A Stage

Run one stage:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --config .\runs\post_training\sft_instruct_v1.yaml `
  --execute
```

With a custom Accelerate config:

```powershell
python .\LPLM-1T-EP01\post_training\run_stage.py `
  --config .\runs\post_training\sft_instruct_v1.yaml `
  --accelerate_config .\configs\accelerate_zero3.yaml `
  --execute
```

The helper expands that to:

```powershell
accelerate launch --config_file .\configs\accelerate_zero3.yaml `
  .\LPLM-1T-EP01\post_training\train_sft.py `
  --config .\runs\post_training\sft_instruct_v1.yaml
```

## 6. Trainer Script Contract

Each future trainer script should accept exactly this minimum interface:

```powershell
python train_sft.py --config path\to\stage.yaml
```

Required trainer scripts by stage:

| Stage type | Expected trainer |
| --- | --- |
| `continued_pretraining` | `post_training/train_cpt.py` |
| `supervised_fine_tuning` | `post_training/train_sft.py` |
| `preference_optimization` | `post_training/train_preference.py` |
| `reasoning_rl_verifiable_rewards` | `post_training/train_reasoning_rlvr.py` |

Implemented trainers:

- `train_cpt.py` loads JSONL text records, packs tokens, and runs causal-LM
  continued pretraining.
- `train_sft.py` loads conversation JSONL and masks non-assistant tokens by
  default.
- `train_preference.py` supports SimPO, DPO, and KTO-style binary feedback.
- `train_reasoning_rlvr.py` supports grouped generation with exact-match,
  JSON schema, unit-test, and substring-grounding rewards.

## 7. Current Status

The stage-specific trainers are implemented and config-driven. They require the
input checkpoint and JSONL data paths referenced by the selected config to
exist before `--execute` can run successfully.
