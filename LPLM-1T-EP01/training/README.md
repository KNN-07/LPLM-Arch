# LPLM-1T-EP01 FineWeb Pretraining

`pretrain_fineweb.py` starts LPLM-1T-EP01 from random weights using the
local EP01 config and Kimi tiktoken tokenizer. It streams Hugging Face FineWeb,
packs text into fixed-length causal-LM blocks, trains with MuonClip by default,
and reports training to W&B.

The default dataset is `HuggingFaceFW/fineweb` with
`--dataset_config latest`, which resolves the newest `CC-MAIN-*` config from
the Hugging Face dataset metadata at runtime.

## Install

```powershell
pip install torch transformers datasets accelerate wandb tiktoken tokenizers
pip install deepspeed
```

Set W&B before launching:

```powershell
$env:WANDB_API_KEY = "<your-key>"
```

The checked-in `tiktoken.model` may be a Git LFS pointer in a fresh checkout.
Fetch the real tokenizer asset with `git lfs pull`, or pass
`--tokenizer_name_or_path` to a complete Kimi tokenizer directory or Hub id.

## Smoke Test

Use the smaller EP01 variant first:

```powershell
python .\LPLM-1T-EP01\training\pretrain_fineweb.py `
  --variant 300M `
  --dataset_config sample-10BT `
  --max_steps 20 `
  --block_size 1024 `
  --per_device_train_batch_size 1 `
  --optimizer muonclip `
  --wandb_project lplm-pretraining
```

## 1T Launch Shape

The 1T variant requires distributed training infrastructure. A ZeRO-3 bf16
DeepSpeed config is provided as a starting point:

```powershell
accelerate launch .\LPLM-1T-EP01\training\pretrain_fineweb.py `
  --variant 1T `
  --dataset_config latest `
  --block_size 4096 `
  --max_steps 100000 `
  --gradient_accumulation_steps 16 `
  --deepspeed_config .\LPLM-1T-EP01\training\deepspeed_zero3_bf16.json `
  --optimizer muonclip `
  --qk_clip_threshold 100 `
  --wandb_project lplm-pretraining
```

FineWeb is text-only, so the script defaults to `--train_scope text`, which
initializes and trains the EP01 language stack. Use `--train_scope full` only if
you need the multimodal wrapper initialized too.

MuonClip uses Muon updates for eligible 2D hidden-layer matrices and AdamW for
embeddings, heads, norms, router weights, biases, and other non-2D tensors.
QK-Clip is enabled by default with `--qk_clip_threshold 100`.
