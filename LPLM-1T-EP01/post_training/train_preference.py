"""Preference optimization trainer for LPLM post-training configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import (
    JsonlDataset,
    MuonClipTrainer,
    load_model,
    load_tokenizer,
    make_training_args,
    read_stage_records,
    render_chat,
    resolve_path,
    setup_stage,
    setup_wandb,
    tokenize_text,
    trainer_callbacks,
)


def render_completion_example(tokenizer, prompt: list[dict[str, Any]], completion: list[dict[str, Any]], *, tools=None) -> tuple[list[int], list[int]]:
    prefix_text = tokenizer.apply_chat_template(
        prompt,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        thinking=False,
    )
    full_text = render_chat(tokenizer, prompt + completion, tools=tools, thinking=False)
    input_ids = tokenize_text(tokenizer, full_text)
    prefix_len = len(tokenize_text(tokenizer, prefix_text))
    labels = [-100] * len(input_ids)
    for pos in range(min(prefix_len, len(labels)), len(labels)):
        labels[pos] = input_ids[pos]
    return input_ids, labels


def build_preference_dataset(records: list[dict[str, Any]], *, tokenizer, method: str, max_length: int) -> JsonlDataset:
    examples: list[dict[str, Any]] = []
    for record in records:
        prompt = record.get("prompt")
        tools = record.get("tools")
        if not isinstance(prompt, list):
            raise ValueError("Preference records must include a prompt message list.")

        if method in ("dpo", "simpo"):
            chosen = record.get("chosen")
            rejected = record.get("rejected")
            if not isinstance(chosen, list) or not isinstance(rejected, list):
                raise ValueError("DPO/SimPO records need chosen and rejected lists.")
            chosen_ids, chosen_labels = render_completion_example(tokenizer, prompt, chosen, tools=tools)
            rejected_ids, rejected_labels = render_completion_example(tokenizer, prompt, rejected, tools=tools)
            examples.append(
                {
                    "chosen_input_ids": torch.tensor(chosen_ids[:max_length], dtype=torch.long),
                    "chosen_labels": torch.tensor(chosen_labels[:max_length], dtype=torch.long),
                    "rejected_input_ids": torch.tensor(rejected_ids[:max_length], dtype=torch.long),
                    "rejected_labels": torch.tensor(rejected_labels[:max_length], dtype=torch.long),
                }
            )
        elif method == "kto":
            completion = record.get("completion")
            label = record.get("label")
            if not isinstance(completion, list) or label not in ("desirable", "undesirable"):
                raise ValueError("KTO records need completion plus desirable/undesirable label.")
            input_ids, labels = render_completion_example(tokenizer, prompt, completion, tools=tools)
            examples.append(
                {
                    "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long),
                    "labels": torch.tensor(labels[:max_length], dtype=torch.long),
                    "preference_label": 1 if label == "desirable" else 0,
                }
            )
        else:
            raise ValueError(f"Unsupported preference method: {method}")

    if not examples:
        raise ValueError("Preference preprocessing produced no examples.")
    return JsonlDataset(examples)


class PreferenceCollator:
    def __init__(self, pad_token_id: int, method: str) -> None:
        self.pad_token_id = pad_token_id
        self.method = method

    def _pad(self, tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(tensor.numel() for tensor in tensors)
        out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
        for index, tensor in enumerate(tensors):
            out[index, : tensor.numel()] = tensor
        return out

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.method in ("dpo", "simpo"):
            batch = {
                "chosen_input_ids": self._pad([f["chosen_input_ids"] for f in features], self.pad_token_id),
                "chosen_labels": self._pad([f["chosen_labels"] for f in features], -100),
                "rejected_input_ids": self._pad([f["rejected_input_ids"] for f in features], self.pad_token_id),
                "rejected_labels": self._pad([f["rejected_labels"] for f in features], -100),
            }
            batch["chosen_attention_mask"] = (batch["chosen_input_ids"] != self.pad_token_id).long()
            batch["rejected_attention_mask"] = (batch["rejected_input_ids"] != self.pad_token_id).long()
            return batch

        batch = {
            "input_ids": self._pad([f["input_ids"] for f in features], self.pad_token_id),
            "labels": self._pad([f["labels"] for f in features], -100),
            "preference_label": torch.tensor([f["preference_label"] for f in features], dtype=torch.float32),
        }
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        return batch


def sequence_logps(model, input_ids, attention_mask, labels, *, average: bool) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    sums = token_logps.sum(dim=-1)
    if average:
        return sums / mask.sum(dim=-1).clamp_min(1)
    return sums


class PreferenceTrainer(MuonClipTrainer):
    def __init__(self, *args, method_config: dict[str, Any], reference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = method_config["name"]
        self.method_config = method_config
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.method_name in ("dpo", "simpo"):
            loss = self._pairwise_loss(model, inputs)
        elif self.method_name == "kto":
            loss = self._kto_loss(model, inputs)
        else:
            raise ValueError(f"Unsupported method: {self.method_name}")
        return (loss, {}) if return_outputs else loss

    def _pairwise_loss(self, model, inputs) -> torch.Tensor:
        average = self.method_name == "simpo"
        chosen_logps = sequence_logps(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_labels"],
            average=average,
        )
        rejected_logps = sequence_logps(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_labels"],
            average=average,
        )
        diff = chosen_logps - rejected_logps

        if self.method_name == "dpo":
            beta = float(self.method_config.get("dpo", {}).get("beta", 0.1))
            if self.reference_model is not None:
                self.reference_model.to(chosen_logps.device)
                with torch.no_grad():
                    ref_chosen = sequence_logps(
                        self.reference_model,
                        inputs["chosen_input_ids"],
                        inputs["chosen_attention_mask"],
                        inputs["chosen_labels"],
                        average=False,
                    )
                    ref_rejected = sequence_logps(
                        self.reference_model,
                        inputs["rejected_input_ids"],
                        inputs["rejected_attention_mask"],
                        inputs["rejected_labels"],
                        average=False,
                    )
                diff = diff - (ref_chosen - ref_rejected)
            return -F.logsigmoid(beta * diff).mean()

        simpo = self.method_config.get("simpo", {})
        beta = float(simpo.get("beta", 2.0))
        gamma_beta_ratio = float(simpo.get("gamma_beta_ratio", 0.5))
        return -F.logsigmoid(beta * (diff - gamma_beta_ratio)).mean()

    def _kto_loss(self, model, inputs) -> torch.Tensor:
        kto = self.method_config.get("kto", {})
        beta = float(kto.get("beta", 0.1))
        desirable_weight = float(kto.get("desirable_weight", 1.0))
        undesirable_weight = float(kto.get("undesirable_weight", 1.0))
        logps = sequence_logps(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
            average=True,
        )
        labels = inputs["preference_label"].to(logps.device)
        desirable_loss = -F.logsigmoid(beta * logps) * labels * desirable_weight
        undesirable_loss = -F.logsigmoid(-beta * logps) * (1 - labels) * undesirable_weight
        return (desirable_loss + undesirable_loss).mean()


def load_reference_model_if_needed(config: dict[str, Any], output_dir: Path):
    method = config.get("method", {})
    name = method.get("name", "simpo")
    if name != "dpo" or bool(method.get("dpo", {}).get("reference_free", False)):
        return None
    reference_checkpoint = config.get("model", {}).get("reference_checkpoint")
    if not reference_checkpoint:
        return None
    reference_config = copy.deepcopy(config)
    reference_config["model"]["input_checkpoint"] = reference_checkpoint
    return load_model(reference_config, output_dir=output_dir)


def main() -> None:
    args, config = setup_stage(
        Path(__file__),
        "Run preference optimization from a post-training YAML config.",
    )
    stage = config["stage"]
    model_config = config["model"]
    method_config = config["method"]
    method_name = method_config.get("name", "simpo")
    output_dir = resolve_path(model_config["output_dir"], base_dir=args.config.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config, output_dir=output_dir)
    records = read_stage_records(config, args.config)
    dataset = build_preference_dataset(
        records,
        tokenizer=tokenizer,
        method=method_name,
        max_length=int(config.get("data", {}).get("max_seq_length", 8192)),
    )

    model = load_model(config, output_dir=output_dir)
    reference_model = load_reference_model_if_needed(config, output_dir)
    report_to = setup_wandb(config)
    training_args = make_training_args(
        config,
        output_dir=output_dir,
        report_to=report_to,
        stage_name=stage.get("id", "preference"),
    )
    metadata = {
        "stage": stage,
        "model": model_config,
        "method": method_config,
        "num_examples": len(dataset),
    }

    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PreferenceCollator(tokenizer.pad_token_id or 0, method_name),
        processing_class=tokenizer,
        callbacks=trainer_callbacks(config, metadata),
        method_config=method_config,
        reference_model=reference_model,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
