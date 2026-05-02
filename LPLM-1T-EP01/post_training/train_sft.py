"""Supervised fine-tuning trainer for LPLM post-training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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


def assistant_label_spans(tokenizer, record: dict[str, Any], *, thinking: bool) -> tuple[list[int], list[int]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT records must include a `messages` list.")

    tools = record.get("tools")
    full_text = render_chat(tokenizer, messages, tools=tools, thinking=thinking)
    input_ids = tokenize_text(tokenizer, full_text)
    labels = [-100] * len(input_ids)

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        prefix_messages = messages[:index]
        target_messages = messages[: index + 1]
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            thinking=thinking,
        )
        target_text = render_chat(tokenizer, target_messages, tools=tools, thinking=thinking)
        prefix_len = len(tokenize_text(tokenizer, prefix_text))
        target_len = len(tokenize_text(tokenizer, target_text))
        end = min(target_len, len(labels))
        for pos in range(min(prefix_len, end), end):
            labels[pos] = input_ids[pos]

    return input_ids, labels


def build_sft_dataset(records: list[dict[str, Any]], *, tokenizer, max_seq_length: int, mask_user_tokens: bool) -> JsonlDataset:
    examples: list[dict[str, torch.Tensor]] = []
    for record in records:
        thinking = bool(record.get("metadata", {}).get("preserve_thinking", False))
        if mask_user_tokens:
            input_ids, labels = assistant_label_spans(tokenizer, record, thinking=thinking)
        else:
            text = render_chat(tokenizer, record["messages"], tools=record.get("tools"), thinking=thinking)
            input_ids = tokenize_text(tokenizer, text)
            labels = list(input_ids)

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        if not input_ids or all(label == -100 for label in labels):
            continue

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        examples.append(
            {
                "input_ids": input_tensor,
                "attention_mask": torch.ones_like(input_tensor),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )

    if not examples:
        raise ValueError("SFT preprocessing produced no trainable examples.")
    return JsonlDataset(examples)


def main() -> None:
    args, config = setup_stage(
        Path(__file__),
        "Run supervised fine-tuning from a post-training YAML config.",
    )
    stage = config["stage"]
    model_config = config["model"]
    data_config = config["data"]
    output_dir = resolve_path(model_config["output_dir"], base_dir=args.config.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config, output_dir=output_dir)
    records = read_stage_records(config, args.config)
    dataset = build_sft_dataset(
        records,
        tokenizer=tokenizer,
        max_seq_length=int(data_config.get("max_seq_length", 8192)),
        mask_user_tokens=bool(data_config.get("mask_user_tokens", True)),
    )

    model = load_model(config, output_dir=output_dir)
    report_to = setup_wandb(config)
    training_args = make_training_args(
        config,
        output_dir=output_dir,
        report_to=report_to,
        stage_name=stage.get("id", "sft"),
    )
    metadata = {
        "stage": stage,
        "model": model_config,
        "data": data_config,
        "num_examples": len(dataset),
    }

    trainer = MuonClipTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=trainer_callbacks(config, metadata, block_size=data_config.get("max_seq_length")),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
