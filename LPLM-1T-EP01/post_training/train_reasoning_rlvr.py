"""GRPO/RLVR-style reasoning trainer for LPLM post-training configs."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tempfile
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
    read_jsonl_files,
    render_chat,
    resolve_existing_patterns,
    resolve_path,
    setup_stage,
    setup_wandb,
    tokenize_text,
    trainer_callbacks,
)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def exact_match_reward(completion: str, record: dict[str, Any]) -> float:
    expected = record.get("answer")
    if expected is None:
        return 0.0
    return 1.0 if normalize_answer(completion) == normalize_answer(str(expected)) else 0.0


def schema_match_reward(completion: str, record: dict[str, Any]) -> float:
    try:
        parsed = json.loads(completion)
    except json.JSONDecodeError:
        return 0.0
    schema = record.get("reward", {}).get("schema")
    if not isinstance(schema, dict):
        return 1.0
    required = schema.get("required", [])
    if isinstance(required, list) and all(key in parsed for key in required):
        return 1.0
    return 0.0


def unit_test_reward(completion: str, record: dict[str, Any]) -> float:
    reward = record.get("reward", {})
    tests = reward.get("tests", [])
    if not isinstance(tests, list) or not tests:
        return 0.0
    code = extract_code(completion)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(code + "\n\n" + "\n".join(str(test) for test in tests), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(reward.get("timeout_seconds", 5.0)),
            check=False,
        )
    return 1.0 if result.returncode == 0 else 0.0


def score_completion(completion: str, record: dict[str, Any]) -> float:
    reward = record.get("reward", {})
    reward_type = reward.get("type")
    if reward_type == "exact_match":
        return exact_match_reward(completion, record)
    if reward_type == "schema_match":
        return schema_match_reward(completion, record)
    if reward_type == "unit_tests":
        return unit_test_reward(completion, record)
    if reward_type == "tool_result_grounding":
        expected = reward.get("expected_substrings", [])
        if isinstance(expected, list) and expected:
            lowered = completion.lower()
            return 1.0 if all(str(item).lower() in lowered for item in expected) else 0.0
    return exact_match_reward(completion, record)


class RLVRDataCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {"records": features}


def pad_sequences(sequences: list[list[int]], *, pad_value: int) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        out[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return out


def sequence_logps(model, input_ids, attention_mask, labels) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


class GRPORLVRTrainer(MuonClipTrainer):
    def __init__(self, *args, tokenizer, algorithm: dict[str, Any], reference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.algorithm = algorithm
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        records = inputs["records"]
        device = next(model.parameters()).device
        group_size = int(self.algorithm.get("group_size", 8))
        max_completion_length = int(self.algorithm.get("max_completion_length", 1024))
        temperature = float(self.algorithm.get("temperature", 0.8))
        top_p = float(self.algorithm.get("top_p", 0.95))
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        all_rewards: list[float] = []

        for record in records:
            prompt = record.get("prompt")
            if not isinstance(prompt, list):
                raise ValueError("RLVR records must include a prompt message list.")
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                thinking=True,
            )
            prompt_ids = tokenize_text(self.tokenizer, prompt_text)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            prompt_tensor = prompt_tensor.repeat(group_size, 1)
            attention_mask = torch.ones_like(prompt_tensor)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=prompt_tensor,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_completion_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            group_rewards: list[float] = []
            group_inputs: list[list[int]] = []
            group_labels: list[list[int]] = []
            for row in generated.tolist():
                completion_ids = row[len(prompt_ids) :]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                reward = score_completion(completion_text, record)
                labels = [-100] * len(prompt_ids) + completion_ids
                group_inputs.append(row)
                group_labels.append(labels[: len(row)])
                group_rewards.append(reward)

            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            advantages = rewards_tensor - rewards_tensor.mean()
            std = rewards_tensor.std(unbiased=False)
            if float(std.item()) > 1e-6:
                advantages = advantages / (std + 1e-6)

            all_input_ids.extend(group_inputs)
            all_labels.extend(group_labels)
            all_rewards.extend(float(item) for item in advantages.tolist())

        input_ids = pad_sequences(all_input_ids, pad_value=pad_token_id).to(device)
        labels = pad_sequences(all_labels, pad_value=-100).to(device)
        attention_mask = input_ids.ne(pad_token_id).long()
        advantages = torch.tensor(all_rewards, dtype=torch.float32, device=device)

        logps = sequence_logps(model, input_ids, attention_mask, labels)
        loss = -(advantages * logps).mean()

        kl_config = self.algorithm.get("kl_penalty", {})
        if self.reference_model is not None and bool(kl_config.get("enabled", True)):
            beta = float(kl_config.get("beta", 0.02))
            self.reference_model.to(device)
            with torch.no_grad():
                ref_logps = sequence_logps(self.reference_model, input_ids, attention_mask, labels)
            loss = loss + beta * (logps - ref_logps).mean()

        return (loss, {"mean_advantage": advantages.mean()}) if return_outputs else loss


def read_reward_records(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    reward = config.get("reward", {})
    sources = reward.get("reward_sources", [])
    if not isinstance(sources, list):
        raise ValueError("reward.reward_sources must be a list.")
    paths: list[str] = []
    for source in sources:
        if isinstance(source, dict) and "path" in source:
            paths.append(str(source["path"]))
    files = resolve_existing_patterns(paths, base_dir=config_path.parent)
    if not files:
        raise FileNotFoundError(f"No RLVR reward files matched: {paths}")
    return read_jsonl_files(files)


def load_reference_model_if_needed(config: dict[str, Any], output_dir: Path):
    kl_config = config.get("algorithm", {}).get("kl_penalty", {})
    if not bool(kl_config.get("enabled", True)):
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
        "Run reasoning RLVR from a post-training YAML config.",
    )
    stage = config["stage"]
    model_config = config["model"]
    output_dir = resolve_path(model_config["output_dir"], base_dir=args.config.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config, output_dir=output_dir)
    records = read_reward_records(config, args.config)
    dataset = JsonlDataset(records)

    model = load_model(config, output_dir=output_dir)
    reference_model = load_reference_model_if_needed(config, output_dir)
    report_to = setup_wandb(config)
    training_args = make_training_args(
        config,
        output_dir=output_dir,
        report_to=report_to,
        stage_name=stage.get("id", "reasoning_rlvr"),
    )
    metadata = {
        "stage": stage,
        "model": model_config,
        "algorithm": config.get("algorithm", {}),
        "reward": config.get("reward", {}),
        "num_prompts": len(dataset),
    }

    trainer = GRPORLVRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=RLVRDataCollator(),
        processing_class=tokenizer,
        callbacks=trainer_callbacks(config, metadata),
        tokenizer=tokenizer,
        algorithm=config.get("algorithm", {}),
        reference_model=reference_model,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
