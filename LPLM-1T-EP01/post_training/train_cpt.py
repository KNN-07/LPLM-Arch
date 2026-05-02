"""Continued pretraining trainer for LPLM post-training configs."""

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
    resolve_path,
    setup_stage,
    setup_wandb,
    tokenize_text,
    trainer_callbacks,
)


def pack_cpt_records(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    block_size: int,
    text_column: str,
    add_bos: bool,
    add_eos: bool,
) -> JsonlDataset:
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    buffer: list[int] = []
    examples: list[dict[str, torch.Tensor]] = []

    for record in records:
        text = record.get(text_column, record.get("text"))
        if not isinstance(text, str) or not text:
            continue
        if add_bos and bos_token_id is not None:
            buffer.append(int(bos_token_id))
        buffer.extend(tokenize_text(tokenizer, text))
        if add_eos and eos_token_id is not None:
            buffer.append(int(eos_token_id))

        while len(buffer) >= block_size:
            block = buffer[:block_size]
            del buffer[:block_size]
            input_ids = torch.tensor(block, dtype=torch.long)
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }
            )

    if not examples:
        raise ValueError("CPT packing produced no examples.")
    return JsonlDataset(examples)


def main() -> None:
    args, config = setup_stage(
        Path(__file__),
        "Run continued pretraining from a post-training YAML config.",
    )
    stage = config["stage"]
    model_config = config["model"]
    data_config = config["data"]
    output_dir = resolve_path(model_config["output_dir"], base_dir=args.config.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config, output_dir=output_dir)
    records = read_stage_records(config, args.config)
    dataset = pack_cpt_records(
        records,
        tokenizer=tokenizer,
        block_size=int(data_config.get("block_size", 4096)),
        text_column=str(data_config.get("text_column", "text")),
        add_bos=bool(data_config.get("add_bos", True)),
        add_eos=bool(data_config.get("add_eos", True)),
    )

    model = load_model(config, output_dir=output_dir)
    report_to = setup_wandb(config)
    training_args = make_training_args(
        config,
        output_dir=output_dir,
        report_to=report_to,
        stage_name=stage.get("id", "cpt"),
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
        callbacks=trainer_callbacks(
            config,
            metadata,
            block_size=int(data_config.get("block_size", 4096)),
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
