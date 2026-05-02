"""Shared utilities for LPLM post-training stage trainers."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments, set_seed

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install PyYAML to run post-training: pip install pyyaml") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]
POST_TRAINING_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = MODEL_ROOT / "training"
BUILDING_ROOT = MODEL_ROOT / "building"
LOGGER = logging.getLogger("lplm.post_training")


def ensure_import_paths() -> None:
    for path in (str(TRAINING_ROOT), str(BUILDING_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_import_paths()

from muonclip import MuonClipTrainer, QKClipCallback  # noqa: E402


class WandbMetadataCallback(TrainerCallback):
    """Attach config metadata and token throughput estimates to W&B."""

    def __init__(self, *, metadata: dict[str, Any], block_size: int | None = None) -> None:
        self.metadata = metadata
        self.block_size = block_size
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
        payload: dict[str, float] = {}
        if self.block_size is not None:
            tokens_seen = (
                state.global_step
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * max(1, args.world_size)
                * self.block_size
            )
            elapsed = max(1e-6, time.perf_counter() - self.start_time)
            payload["tokens_seen_estimate"] = float(tokens_seen)
            payload["tokens_per_second_estimate"] = float(tokens_seen / elapsed)
        if logs is not None:
            logs.update(payload)
        if not payload:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(payload, step=state.global_step)


class JsonlDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def parse_config_arg(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_existing_patterns(patterns: list[str], *, base_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        search_pattern = str(pattern_path if pattern_path.is_absolute() else REPO_ROOT / pattern)
        matches = [Path(match).resolve() for match in glob.glob(search_pattern, recursive=True)]
        if not matches and base_dir is not REPO_ROOT:
            search_pattern = str(base_dir / pattern)
            matches = [Path(match).resolve() for match in glob.glob(search_pattern, recursive=True)]
        files.extend(matches)
    return sorted(set(files))


def read_jsonl_files(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object in {path}:{line_number}")
                records.append(record)
    if not records:
        raise ValueError(f"No records loaded from {len(paths)} files.")
    return records


def read_stage_records(
    config: dict[str, Any],
    config_path: Path,
    *,
    key: str = "train_paths",
) -> list[dict[str, Any]]:
    data = config.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config `data` must be a mapping.")
    paths = data.get(key)
    if paths is None and "mixture_file" in data:
        mixture_path = resolve_path(data["mixture_file"], base_dir=config_path.parent)
        mixture = load_yaml(mixture_path)
        sources = mixture.get(data.get("sources_key", "sources"), [])
        if not isinstance(sources, list):
            raise ValueError(f"{mixture_path} sources must be a list.")
        paths = [source["path"] for source in sources if isinstance(source, dict) and "path" in source]
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        raise ValueError(f"Config data.{key} must be a string or list.")
    files = resolve_existing_patterns([str(path) for path in paths], base_dir=config_path.parent)
    if not files:
        raise FileNotFoundError(f"No JSONL files matched data.{key}: {paths}")
    LOGGER.info("Loading %s records from %d files.", key, len(files))
    return read_jsonl_files(files)


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


def prepare_local_kimi_tokenizer(output_dir: Path) -> Path:
    tokenizer_dir = MODEL_ROOT / "building" / "source" / "tokenizer"
    vocab_file = tokenizer_dir / "tiktoken.model"
    if is_git_lfs_pointer(vocab_file):
        raise RuntimeError(
            f"{vocab_file} is a Git LFS pointer. Run `git lfs pull` or set "
            "model.tokenizer_name_or_path to a complete Kimi tokenizer."
        )
    if (tokenizer_dir / "tool_declaration_ts.py").exists():
        return tokenizer_dir

    runtime_dir = output_dir / "runtime" / "kimi_tokenizer"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("tokenizer_config.json", "tokenization_kimi.py", "tiktoken.model"):
        shutil.copy2(tokenizer_dir / filename, runtime_dir / filename)
    shutil.copy2(
        MODEL_ROOT / "building" / "source" / "misc" / "tool_declaration_ts.py",
        runtime_dir / "tool_declaration_ts.py",
    )
    return runtime_dir


def load_tokenizer(config: dict[str, Any], *, output_dir: Path) -> Any:
    configure_hf_dynamic_module_cache(output_dir)
    model_config = config.get("model", {})
    tokenizer_path = model_config.get("tokenizer_name_or_path")
    if tokenizer_path is None:
        tokenizer_path = prepare_local_kimi_tokenizer(output_dir)
    else:
        tokenizer_path = resolve_path(tokenizer_path, base_dir=POST_TRAINING_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
    )
    chat_template = model_config.get("chat_template")
    if chat_template:
        chat_template_path = resolve_path(chat_template, base_dir=POST_TRAINING_ROOT)
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def load_model(config: dict[str, Any], *, output_dir: Path) -> torch.nn.Module:
    ensure_import_paths()
    from modeling_deepseek import DeepseekV3ForCausalLM
    from modeling_lplm_ep01 import LPLMEP01ForConditionalGeneration

    model_config = config.get("model", {})
    checkpoint = resolve_path(model_config["input_checkpoint"], base_dir=POST_TRAINING_ROOT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Input checkpoint does not exist: {checkpoint}")

    train_scope = model_config.get("train_scope", "text")
    cls = DeepseekV3ForCausalLM if train_scope == "text" else LPLMEP01ForConditionalGeneration
    LOGGER.info("Loading %s checkpoint from %s.", train_scope, checkpoint)
    model = cls.from_pretrained(checkpoint)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model


def setup_wandb(config: dict[str, Any]) -> list[str]:
    tracking = config.get("tracking", {})
    monitoring = config.get("monitoring", {})
    project = monitoring.get("wandb_project") or tracking.get("wandb_project")
    entity = monitoring.get("wandb_entity") or tracking.get("wandb_entity")
    log_model = tracking.get("log_model", "false")
    if project:
        os.environ["WANDB_PROJECT"] = str(project)
    if entity:
        os.environ["WANDB_ENTITY"] = str(entity)
    os.environ.setdefault("WANDB_LOG_MODEL", str(log_model))
    return ["wandb"] if project else []


def make_training_args(
    config: dict[str, Any],
    *,
    output_dir: Path,
    report_to: list[str],
    stage_name: str,
) -> TrainingArguments:
    training = config.get("training", {})
    optimizer = training.get("optimizer", "muonclip")
    bf16 = bool(training.get("bf16", True))
    fp16 = bool(training.get("fp16", False))
    if bf16 and fp16:
        raise ValueError("Choose either bf16 or fp16, not both.")

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "do_train": True,
        "overwrite_output_dir": False,
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training.get("learning_rate", 1e-5)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "logging_first_step": True,
        "save_steps": int(training.get("save_steps", 500)),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "remove_unused_columns": False,
        "report_to": report_to,
        "run_name": stage_name,
        "save_safetensors": True,
        "optim": str(training.get("optim", "adamw_torch")),
        "lr_scheduler_type": str(training.get("lr_scheduler_type", "cosine")),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
    }
    if training.get("max_steps") is not None:
        kwargs["max_steps"] = int(training["max_steps"])
    if training.get("num_train_epochs") is not None:
        kwargs["num_train_epochs"] = float(training["num_train_epochs"])
    if training.get("warmup_steps") is not None:
        kwargs["warmup_steps"] = int(training["warmup_steps"])
    if training.get("warmup_ratio") is not None:
        kwargs["warmup_ratio"] = float(training["warmup_ratio"])
    if training.get("deepspeed_config"):
        kwargs["deepspeed"] = str(resolve_path(training["deepspeed_config"], base_dir=POST_TRAINING_ROOT))
    if training.get("fsdp"):
        kwargs["fsdp"] = training["fsdp"]

    args = TrainingArguments(**kwargs)
    args.lplm_optimizer = optimizer
    args.adam_beta1 = float(training.get("adam_beta1", 0.9))
    args.adam_beta2 = float(training.get("adam_beta2", 0.95))
    args.adam_epsilon = float(training.get("adam_epsilon", 1e-8))
    args.muon_momentum = float(training.get("muon_momentum", 0.95))
    args.muon_nesterov = bool(training.get("muon_nesterov", True))
    args.muon_ns_steps = int(training.get("muon_ns_steps", 5))
    args.muon_ns_coeff_a = float(training.get("muon_ns_coeff_a", 3.4445))
    args.muon_ns_coeff_b = float(training.get("muon_ns_coeff_b", -4.775))
    args.muon_ns_coeff_c = float(training.get("muon_ns_coeff_c", 2.0315))
    args.muon_eps = float(training.get("muon_eps", 1e-7))
    args.muon_adjust_lr_fn = str(training.get("muon_adjust_lr_fn", "match_rms_adamw"))
    args.muon_adamw_lr_ratio = float(training.get("adamw_fallback_lr_ratio", 0.1))
    return args


def trainer_callbacks(config: dict[str, Any], metadata: dict[str, Any], *, block_size: int | None = None) -> list:
    training = config.get("training", {})
    return [
        WandbMetadataCallback(metadata=metadata, block_size=block_size),
        QKClipCallback(
            threshold=float(training.get("qk_clip_threshold", 100.0)),
            enabled=bool(training.get("qk_clip", True)),
            log_only=bool(training.get("qk_clip_log_only", False)),
        ),
    ]


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], *, tools=None, thinking: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            thinking=thinking,
        )
    return "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages)


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, allow_special_tokens=True)
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def setup_stage(config_path: Path, description: str) -> tuple[argparse.Namespace, dict[str, Any]]:
    args = parse_config_arg(description)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_yaml(args.config)
    seed = int(config.get("global_training", {}).get("seed", config.get("training", {}).get("seed", 1337)))
    set_seed(seed)
    return args, config
