"""Offline policy training from shaped agentic SWE rollouts.

This script expects precomputed rollout records with shaped rewards. It trains
on assistant/tool transcripts using weighted causal-LM loss, where higher
reward rollouts contribute larger positive weight and failed rollouts can be
downweighted or skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ADDITIONAL_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ADDITIONAL_ROOT.parents[0]
TRAINING_ROOT = MODEL_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from muonclip import MuonClipTrainer, QKClipCallback  # noqa: E402
from common import load_jsonl, load_yaml, resolve_path, tokenize_text  # noqa: E402


class RolloutDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class WeightedCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad(self, tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(tensor.numel() for tensor in tensors)
        out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
        for index, tensor in enumerate(tensors):
            out[index, : tensor.numel()] = tensor
        return out

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = self._pad([feature["input_ids"] for feature in features], self.pad_token_id)
        labels = self._pad([feature["labels"] for feature in features], -100)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
            "labels": labels,
            "weights": torch.tensor([feature["weight"] for feature in features], dtype=torch.float32),
        }


class WeightedCausalLMTrainer(MuonClipTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["labels"][:, 1:]
        mask = labels.ne(-100)
        safe_labels = labels.masked_fill(~mask, 0)
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)
        seq_loss = (losses * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        loss = (seq_loss * weights).mean()
        return (loss, outputs) if return_outputs else loss


def transcript_text(rollout: dict[str, Any]) -> str | None:
    if isinstance(rollout.get("transcript"), str):
        return rollout["transcript"]
    if isinstance(rollout.get("messages"), list):
        return "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in rollout["messages"]
        )
    transcript_path = rollout.get("transcript_path")
    if transcript_path:
        path = resolve_path(transcript_path, base_dir=ADDITIONAL_ROOT)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def build_dataset(rollouts: list[dict[str, Any]], tokenizer, *, max_length: int, min_reward: float) -> RolloutDataset:
    examples: list[dict[str, Any]] = []
    for rollout in rollouts:
        reward = rollout.get("shaped_reward", {}).get("final_reward", rollout.get("final_reward"))
        if reward is None:
            continue
        reward = float(reward)
        if reward < min_reward:
            continue
        text = transcript_text(rollout)
        if not text:
            continue
        input_ids = tokenize_text(tokenizer, text)[:max_length]
        if not input_ids:
            continue
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        examples.append(
            {
                "input_ids": input_tensor,
                "labels": input_tensor.clone(),
                "weight": max(0.0, reward),
            }
        )
    if not examples:
        raise ValueError("No trainable rollout examples found.")
    return RolloutDataset(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline from shaped agentic SWE rollouts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--min_reward", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_cfg = config["model"]
    training_cfg = config.get("training", {})
    monitoring_cfg = config.get("monitoring", {})
    if monitoring_cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", str(monitoring_cfg["wandb_project"]))
    output_dir = args.output_dir or resolve_path(model_cfg["output_dir"], base_dir=args.config.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_path(model_cfg["tokenizer_name_or_path"], base_dir=args.config.parent),
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolve_path(model_cfg["input_checkpoint"], base_dir=args.config.parent)
    )
    model.config.use_cache = False

    dataset = build_dataset(
        load_jsonl(args.rollouts),
        tokenizer,
        max_length=args.max_length,
        min_reward=args.min_reward,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-7)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        bf16=bool(training_cfg.get("bf16", True)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        save_steps=int(training_cfg.get("save_steps", 250)),
        remove_unused_columns=False,
        report_to=["wandb"] if monitoring_cfg.get("wandb_project") else [],
    )
    training_args.lplm_optimizer = training_cfg.get("optimizer", "muonclip")
    training_args.adam_beta1 = float(training_cfg.get("adam_beta1", 0.9))
    training_args.adam_beta2 = float(training_cfg.get("adam_beta2", 0.95))
    training_args.adam_epsilon = float(training_cfg.get("adam_epsilon", 1e-8))
    training_args.muon_momentum = float(training_cfg.get("muon_momentum", 0.95))
    training_args.muon_nesterov = bool(training_cfg.get("muon_nesterov", True))
    training_args.muon_ns_steps = int(training_cfg.get("muon_ns_steps", 5))
    training_args.muon_ns_coeff_a = float(training_cfg.get("muon_ns_coeff_a", 3.4445))
    training_args.muon_ns_coeff_b = float(training_cfg.get("muon_ns_coeff_b", -4.775))
    training_args.muon_ns_coeff_c = float(training_cfg.get("muon_ns_coeff_c", 2.0315))
    training_args.muon_eps = float(training_cfg.get("muon_eps", 1e-7))
    training_args.muon_adjust_lr_fn = str(training_cfg.get("muon_adjust_lr_fn", "match_rms_adamw"))
    training_args.muon_adamw_lr_ratio = float(training_cfg.get("adamw_fallback_lr_ratio", 0.1))

    trainer_cls: type[Trainer] = WeightedCausalLMTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=WeightedCollator(tokenizer.pad_token_id or 0),
        processing_class=tokenizer,
        callbacks=[
            QKClipCallback(
                threshold=float(training_cfg.get("qk_clip_threshold", 100.0)),
                enabled=bool(training_cfg.get("qk_clip", True)),
            )
        ],
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
