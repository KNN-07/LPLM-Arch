"""Pretrain LPLM-1T-EP01 from scratch on Hugging Face FineWeb.

This script intentionally initializes the model from the local EP01 config
instead of loading pretrained weights. The default dataset config resolves to
the latest `CC-MAIN-*` FineWeb dump available from Hugging Face at runtime.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import get_dataset_config_names, load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from muonclip import MuonClipTrainer, QKClipCallback

try:
    from transformers.integrations import HfDeepSpeedConfig
except ImportError:  # pragma: no cover - older transformers
    HfDeepSpeedConfig = None


LOGGER = logging.getLogger("lplm.pretrain_fineweb")
FINEWEB_DUMP_RE = re.compile(r"^CC-MAIN-(\d{4})-(\d{2})$")


class PackedFineWebDataset(IterableDataset):
    """Stream FineWeb records, tokenize them, and pack causal-LM blocks."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: str | None,
        dataset_revision: str | None,
        split: str,
        text_column: str,
        tokenizer: Any,
        block_size: int,
        seed: int,
        shuffle_buffer_size: int,
        add_bos: bool,
        add_eos: bool,
        bos_token_id: int | None,
        eos_token_id: int | None,
        min_text_chars: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_revision = dataset_revision
        self.split = split
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.min_text_chars = min_text_chars

    def __iter__(self):
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
            revision=self.dataset_revision,
        )

        worker_info = get_worker_info()
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        num_shards = max(1, world_size * num_workers)
        shard_index = rank * num_workers + worker_id

        dataset = dataset.shard(num_shards=num_shards, index=shard_index)
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                seed=self.seed + shard_index,
                buffer_size=self.shuffle_buffer_size,
            )

        buffer: list[int] = []
        for row in dataset:
            text = row.get(self.text_column)
            if not isinstance(text, str) or len(text) < self.min_text_chars:
                continue

            if self.add_bos and self.bos_token_id is not None:
                buffer.append(self.bos_token_id)
            buffer.extend(self.tokenizer.encode(text, allow_special_tokens=False))
            if self.add_eos and self.eos_token_id is not None:
                buffer.append(self.eos_token_id)

            while len(buffer) >= self.block_size:
                block = buffer[:self.block_size]
                del buffer[:self.block_size]
                input_ids = torch.tensor(block, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }


class WandbTokenCallback(TrainerCallback):
    """Log estimated token counts and throughput to W&B."""

    def __init__(self, *, block_size: int, metadata: dict[str, Any]) -> None:
        self.block_size = block_size
        self.metadata = metadata
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()
        if not state.is_world_process_zero:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.config.update(self.metadata, allow_val_change=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.start_time is None:
            return
        tokens_seen = (
            state.global_step
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * max(1, args.world_size)
            * self.block_size
        )
        elapsed = max(1e-6, time.perf_counter() - self.start_time)
        payload = {
            "tokens_seen_estimate": tokens_seen,
            "tokens_per_second_estimate": tokens_seen / elapsed,
        }
        if logs is not None:
            logs.update(payload)
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(payload, step=state.global_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain LPLM-1T-EP01 from scratch on FineWeb."
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the LPLM-1T-EP01 directory.",
    )
    parser.add_argument(
        "--variant",
        choices=("1T", "300M"),
        default="1T",
        help="EP01 config variant to initialize from scratch.",
    )
    parser.add_argument(
        "--train_scope",
        choices=("text", "full"),
        default="text",
        help="Use only EP01 text stack for FineWeb, or the full multimodal wrapper.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "flash_attention_2"),
        default="eager",
        help="Attention backend to set on the text config.",
    )
    parser.add_argument(
        "--dataset_name",
        default="HuggingFaceFW/fineweb",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--dataset_config",
        default="latest",
        help="Dataset config/subset. Use 'latest' for newest CC-MAIN dump.",
    )
    parser.add_argument(
        "--dataset_revision",
        default=None,
        help="Optional dataset revision, branch, or tag.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help=(
            "Optional tokenizer path or Hub id. Defaults to the local "
            "tokenizer under model_root/building/tokenizer."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no_bos", action="store_true")
    parser.add_argument("--no_eos", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/lplm-ep01-fineweb"),
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=("muonclip", "adamw_torch"),
        default="muonclip",
        help="Optimizer family. MuonClip is the default for pretraining.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Transformers optimizer name used when --optimizer adamw_torch.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument(
        "--muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_ns_coeff_a", type=float, default=3.4445)
    parser.add_argument("--muon_ns_coeff_b", type=float, default=-4.775)
    parser.add_argument("--muon_ns_coeff_c", type=float, default=2.0315)
    parser.add_argument("--muon_eps", type=float, default=1e-7)
    parser.add_argument(
        "--muon_adjust_lr_fn",
        choices=("match_rms_adamw", "original", "none"),
        default="match_rms_adamw",
    )
    parser.add_argument(
        "--muon_adamw_lr_ratio",
        type=float,
        default=0.1,
        help="LR multiplier for AdamW fallback params under MuonClip.",
    )
    parser.add_argument(
        "--qk_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply MuonClip QK projection rescaling after optimizer steps.",
    )
    parser.add_argument("--qk_clip_threshold", type=float, default=100.0)
    parser.add_argument("--qk_clip_log_only", action="store_true")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bf16 training. Use --no-bf16 to disable.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing. Use --no-gradient_checkpointing to disable.",
    )
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--deepspeed_config", default=None)
    parser.add_argument(
        "--fsdp",
        default=None,
        help="Optional FSDP mode string passed to TrainingArguments.",
    )
    parser.add_argument("--wandb_project", default="lplm-pretraining")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--wandb_log_model",
        choices=("false", "checkpoint", "end"),
        default="false",
    )
    return parser.parse_args()


def resolve_latest_fineweb_config(
    dataset_name: str,
    dataset_config: str | None,
    dataset_revision: str | None,
) -> str | None:
    if dataset_config != "latest":
        return dataset_config

    config_names = get_dataset_config_names(dataset_name, revision=dataset_revision)
    dumps: list[tuple[int, int, str]] = []
    for config_name in config_names:
        match = FINEWEB_DUMP_RE.match(config_name)
        if match:
            dumps.append((int(match.group(1)), int(match.group(2)), config_name))

    if not dumps:
        raise ValueError(
            f"Could not resolve latest CC-MAIN config for {dataset_name}. "
            "Pass --dataset_config explicitly."
        )
    return max(dumps)[2]


def configure_hf_dynamic_module_cache(output_dir: Path) -> None:
    hf_modules_cache = output_dir / "runtime" / "hf_modules"
    hf_modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_modules_cache))
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils

        dynamic_module_utils.HF_MODULES_CACHE = str(hf_modules_cache)
    except ImportError:
        pass


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return False
    return first_line == b"version https://git-lfs.github.com/spec/v1"


def prepare_tokenizer_dir(model_root: Path, output_dir: Path) -> Path:
    configure_hf_dynamic_module_cache(output_dir)
    tokenizer_dir = model_root / "building" / "tokenizer"
    required_files = (
        "tokenizer_config.json",
        "tokenization_lplm.py",
        "tool_declaration_ts.py",
        "tiktoken.model",
    )
    missing = [filename for filename in required_files if not (tokenizer_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Local tokenizer directory is incomplete: {tokenizer_dir}. "
            f"Missing: {', '.join(missing)}."
        )

    vocab_file = tokenizer_dir / "tiktoken.model"
    if is_git_lfs_pointer(vocab_file):
        raise RuntimeError(
            f"{vocab_file} is still a Git LFS pointer. Fetch the real "
            "tokenizer asset with `git lfs pull`, or pass "
            "--tokenizer_name_or_path with a complete tokenizer path/Hub id."
        )
    return tokenizer_dir


def add_model_source_to_path(model_root: Path) -> None:
    building_dir = model_root / "building"
    if str(building_dir) not in sys.path:
        sys.path.insert(0, str(building_dir))


def setup_wandb_env(args: argparse.Namespace) -> list[str]:
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    return [] if args.wandb_mode == "disabled" else ["wandb"]


def set_text_special_ids(text_config, ep01_config, tokenizer) -> None:
    bos_token_id = ep01_config.bos_token_id
    if bos_token_id is None:
        bos_token_id = text_config.bos_token_id
    eos_token_id = ep01_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = text_config.eos_token_id
    pad_token_id = ep01_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = text_config.pad_token_id

    ep01_config.bos_token_id = bos_token_id
    ep01_config.eos_token_id = eos_token_id
    ep01_config.pad_token_id = pad_token_id
    text_config.vocab_size = tokenizer.vocab_size
    text_config.bos_token_id = bos_token_id
    text_config.eos_token_id = eos_token_id
    text_config.pad_token_id = pad_token_id
    text_config.use_cache = False


def build_model(args: argparse.Namespace, tokenizer):
    add_model_source_to_path(args.model_root)
    from configuration_lplm_ep01 import LPLMEP01Config
    from modeling_deepseek import DeepseekV3ForCausalLM
    from modeling_lplm_ep01 import LPLMEP01ForConditionalGeneration

    config = LPLMEP01Config(variant=args.variant)
    config.text_config._attn_implementation = args.attn_implementation
    set_text_special_ids(config.text_config, config, tokenizer)

    if args.block_size > config.text_config.max_position_embeddings:
        raise ValueError(
            f"--block_size {args.block_size} exceeds model max_position_embeddings "
            f"{config.text_config.max_position_embeddings}."
        )

    if args.train_scope == "text":
        model = DeepseekV3ForCausalLM(config.text_config)
    else:
        model = LPLMEP01ForConditionalGeneration(config)
        model.config.text_config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, config


def count_parameters(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = getattr(parameter, "ds_numel", parameter.numel())
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return total, trainable


def make_training_args(
    args: argparse.Namespace,
    *,
    report_to: list[str],
    run_name: str,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": False,
        "do_train": True,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "optim": args.optim,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "torch_compile": args.torch_compile,
        "remove_unused_columns": False,
        "report_to": report_to,
        "run_name": run_name,
        "save_safetensors": True,
    }
    if args.deepspeed_config:
        kwargs["deepspeed"] = args.deepspeed_config
    if args.fsdp:
        kwargs["fsdp"] = args.fsdp
    training_args = TrainingArguments(**kwargs)
    training_args.lplm_optimizer = args.optimizer
    training_args.adam_beta1 = args.adam_beta1
    training_args.adam_beta2 = args.adam_beta2
    training_args.adam_epsilon = args.adam_epsilon
    training_args.muon_momentum = args.muon_momentum
    training_args.muon_nesterov = args.muon_nesterov
    training_args.muon_ns_steps = args.muon_ns_steps
    training_args.muon_ns_coeff_a = args.muon_ns_coeff_a
    training_args.muon_ns_coeff_b = args.muon_ns_coeff_b
    training_args.muon_ns_coeff_c = args.muon_ns_coeff_c
    training_args.muon_eps = args.muon_eps
    training_args.muon_adjust_lr_fn = args.muon_adjust_lr_fn
    training_args.muon_adamw_lr_ratio = args.muon_adamw_lr_ratio
    return training_args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.bf16 and args.fp16:
        raise ValueError("Choose either bf16 or fp16, not both.")
    set_seed(args.seed)

    report_to = setup_wandb_env(args)
    resolved_dataset_config = resolve_latest_fineweb_config(
        args.dataset_name,
        args.dataset_config,
        args.dataset_revision,
    )
    run_name = args.wandb_run_name or (
        f"{args.model_root.name}-{args.variant}-{args.train_scope}-"
        f"{resolved_dataset_config}"
    )

    ds_config_ref = None
    if args.deepspeed_config and HfDeepSpeedConfig is not None:
        ds_config_ref = HfDeepSpeedConfig(args.deepspeed_config)

    configure_hf_dynamic_module_cache(args.output_dir)
    tokenizer_dir = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path is not None
        else prepare_tokenizer_dir(args.model_root, args.output_dir)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    model, ep01_config = build_model(args, tokenizer)
    total_params, trainable_params = count_parameters(model)

    LOGGER.info("Using FineWeb dataset config: %s", resolved_dataset_config)
    LOGGER.info("Initialized %s model from scratch.", args.train_scope)
    LOGGER.info("Total params: %s; trainable params: %s", total_params, trainable_params)
    if tokenizer.eos_token_id != ep01_config.eos_token_id:
        LOGGER.warning(
            "Tokenizer eos_token_id=%s differs from EP01 eos_token_id=%s; "
            "packed training will use the EP01 id.",
            tokenizer.eos_token_id,
            ep01_config.eos_token_id,
        )

    train_dataset = PackedFineWebDataset(
        dataset_name=args.dataset_name,
        dataset_config=resolved_dataset_config,
        dataset_revision=args.dataset_revision,
        split=args.split,
        text_column=args.text_column,
        tokenizer=tokenizer,
        block_size=args.block_size,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
        bos_token_id=ep01_config.bos_token_id,
        eos_token_id=ep01_config.eos_token_id,
        min_text_chars=args.min_text_chars,
    )

    training_args = make_training_args(args, report_to=report_to, run_name=run_name)
    metadata = {
        "model_root": str(args.model_root),
        "variant": args.variant,
        "train_scope": args.train_scope,
        "dataset_name": args.dataset_name,
        "dataset_config": resolved_dataset_config,
        "dataset_revision": args.dataset_revision or "main",
        "tokenizer_dir": str(tokenizer_dir),
        "block_size": args.block_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "optimizer": args.optimizer,
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn,
        "qk_clip": args.qk_clip,
        "qk_clip_threshold": args.qk_clip_threshold,
    }
    trainer = MuonClipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[
            WandbTokenCallback(block_size=args.block_size, metadata=metadata),
            QKClipCallback(
                threshold=args.qk_clip_threshold,
                enabled=args.qk_clip,
                log_only=args.qk_clip_log_only,
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    ep01_config.save_pretrained(args.output_dir / "ep01_config")

    if ds_config_ref is not None:
        # Keep the HfDeepSpeedConfig object alive until after model construction.
        LOGGER.debug("DeepSpeed config reference retained: %s", ds_config_ref)


if __name__ == "__main__":
    main()
