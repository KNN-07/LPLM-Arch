"""MuonClip optimizer utilities for LPLM pretraining."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from transformers import Trainer, TrainerCallback


def _zeropower_via_newtonschulz(
    matrix: torch.Tensor,
    *,
    ns_steps: int,
    ns_coefficients: tuple[float, float, float],
    eps: float,
) -> torch.Tensor:
    """Approximate matrix sign by Newton-Schulz iterations."""

    if matrix.ndim != 2:
        raise ValueError("Muon update expects a 2D matrix.")

    a, b, c = ns_coefficients
    update = matrix.float()
    transposed = update.size(0) > update.size(1)
    if transposed:
        update = update.T

    update = update / (update.norm() + eps)
    for _ in range(ns_steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update

    if transposed:
        update = update.T
    return update


class MuonClipOptimizer(Optimizer):
    """Single Optimizer with Muon groups plus AdamW fallback groups.

    Muon is applied only to 2D hidden-layer matrices. Biases, embeddings,
    normalization parameters, output heads, and router weights use AdamW.
    QK clipping is applied by `QKClipCallback` after optimizer steps.
    """

    def __init__(
        self,
        param_groups: Iterable[dict[str, Any]],
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_lr_ratio: float = 0.1,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_ns_coefficients: tuple[float, float, float] = (
            3.4445,
            -4.775,
            2.0315,
        ),
        muon_eps: float = 1e-7,
        muon_adjust_lr_fn: str = "match_rms_adamw",
    ) -> None:
        if muon_adjust_lr_fn not in ("match_rms_adamw", "original", "none"):
            raise ValueError(
                "--muon_adjust_lr_fn must be one of match_rms_adamw, original, none."
            )
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "adamw_lr_ratio": adamw_lr_ratio,
            "muon_momentum": muon_momentum,
            "muon_nesterov": muon_nesterov,
            "muon_ns_steps": muon_ns_steps,
            "muon_ns_coefficients": muon_ns_coefficients,
            "muon_eps": muon_eps,
            "muon_adjust_lr_fn": muon_adjust_lr_fn,
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            algorithm = group.get("algorithm", "adamw")
            if algorithm == "muon":
                self._muon_step(group)
            elif algorithm == "adamw":
                self._adamw_step(group)
            else:
                raise ValueError(f"Unknown MuonClip algorithm group: {algorithm}")
        return loss

    @torch.no_grad()
    def _muon_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["muon_momentum"]
        nesterov = group["muon_nesterov"]
        ns_steps = group["muon_ns_steps"]
        ns_coefficients = group["muon_ns_coefficients"]
        eps = group["muon_eps"]
        adjust_lr_fn = group["muon_adjust_lr_fn"]

        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if grad.is_sparse:
                raise RuntimeError("MuonClip does not support sparse gradients.")
            if parameter.ndim != 2:
                raise RuntimeError("Muon groups must contain only 2D parameters.")

            state = self.state[parameter]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, device=parameter.device)
                state["momentum_buffer"] = torch.zeros_like(
                    parameter,
                    dtype=torch.float32,
                    memory_format=torch.preserve_format,
                )
            state["step"] += 1
            momentum_buffer = state["momentum_buffer"]
            momentum_buffer.mul_(momentum).add_(grad.float())
            update = grad.float().add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
            update = _zeropower_via_newtonschulz(
                update,
                ns_steps=ns_steps,
                ns_coefficients=ns_coefficients,
                eps=eps,
            )

            rows, cols = parameter.shape
            if adjust_lr_fn == "match_rms_adamw":
                update.mul_(0.2 * math.sqrt(max(rows, cols)))
            elif adjust_lr_fn == "original":
                update.mul_(math.sqrt(max(1.0, rows / cols)))

            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)
            parameter.add_(update.to(parameter.dtype), alpha=-lr)

    @torch.no_grad()
    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"] * group["adamw_lr_ratio"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]

        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if grad.is_sparse:
                raise RuntimeError("MuonClip AdamW fallback does not support sparse gradients.")

            state = self.state[parameter]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, device=parameter.device)
                state["exp_avg"] = torch.zeros_like(
                    parameter,
                    dtype=torch.float32,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    dtype=torch.float32,
                    memory_format=torch.preserve_format,
                )
            state["step"] += 1
            step = int(state["step"].item())
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)
            grad = grad.float()
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            update = exp_avg / denom
            parameter.add_(update.to(parameter.dtype), alpha=-step_size)


def _is_router_weight(name: str) -> bool:
    return name.endswith(".gate.weight")


def _uses_muon(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim != 2:
        return False
    excluded = (
        "embed_tokens",
        "lm_head",
        "score",
        "norm",
        "layernorm",
        "rotary",
    )
    lowered = name.lower()
    if any(part in lowered for part in excluded):
        return False
    if _is_router_weight(name):
        return False
    return True


def build_muonclip_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    muon_params: list[nn.Parameter] = []
    adamw_decay_params: list[nn.Parameter] = []
    adamw_no_decay_params: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _uses_muon(name, parameter):
            muon_params.append(parameter)
        elif parameter.ndim >= 2 and "norm" not in name.lower():
            adamw_decay_params.append(parameter)
        else:
            adamw_no_decay_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "weight_decay": weight_decay,
                "algorithm": "muon",
                "name": "muon_hidden_matrices",
            }
        )
    if adamw_decay_params:
        groups.append(
            {
                "params": adamw_decay_params,
                "weight_decay": weight_decay,
                "algorithm": "adamw",
                "name": "adamw_decay_fallback",
            }
        )
    if adamw_no_decay_params:
        groups.append(
            {
                "params": adamw_no_decay_params,
                "weight_decay": 0.0,
                "algorithm": "adamw",
                "name": "adamw_no_decay_fallback",
            }
        )
    return groups


class MuonClipTrainer(Trainer):
    """Trainer that creates MuonClipOptimizer when requested."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if getattr(self.args, "lplm_optimizer", "muonclip") != "muonclip":
            return super().create_optimizer()

        optimizer_grouped_parameters = build_muonclip_param_groups(
            self.model_wrapped if self.model_wrapped is not None else self.model,
            self.args.weight_decay,
        )
        self.optimizer = MuonClipOptimizer(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            adamw_betas=(self.args.adam_beta1, self.args.adam_beta2),
            adamw_eps=self.args.adam_epsilon,
            adamw_lr_ratio=self.args.muon_adamw_lr_ratio,
            muon_momentum=self.args.muon_momentum,
            muon_nesterov=self.args.muon_nesterov,
            muon_ns_steps=self.args.muon_ns_steps,
            muon_ns_coefficients=(
                self.args.muon_ns_coeff_a,
                self.args.muon_ns_coeff_b,
                self.args.muon_ns_coeff_c,
            ),
            muon_eps=self.args.muon_eps,
            muon_adjust_lr_fn=self.args.muon_adjust_lr_fn,
        )
        return self.optimizer


class QKClipCallback(TrainerCallback):
    """Apply per-head QK-Clip using logits recorded by DeepseekV3Attention."""

    def __init__(self, *, threshold: float, enabled: bool = True, log_only: bool = False):
        self.threshold = threshold
        self.enabled = enabled
        self.log_only = log_only

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or not self.enabled:
            return
        max_qk_logit = qk_clip_model(model, threshold=self.threshold, log_only=self.log_only)
        if not state.is_world_process_zero:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None and max_qk_logit is not None:
            wandb.log({"qk_clip/max_logit": max_qk_logit}, step=state.global_step)


@torch.no_grad()
def qk_clip_model(model: nn.Module, *, threshold: float, log_only: bool = False) -> float | None:
    max_seen: float | None = None
    for module in model.modules():
        per_head_max = getattr(module, "_last_qk_max_per_head", None)
        if per_head_max is None:
            continue
        per_head_max = per_head_max.to("cpu", dtype=torch.float32)
        layer_max = float(per_head_max.max().item())
        max_seen = layer_max if max_seen is None else max(max_seen, layer_max)
        if log_only:
            continue

        clip_heads = torch.nonzero(per_head_max > threshold, as_tuple=False).flatten()
        if clip_heads.numel() == 0:
            continue
        for head_idx_tensor in clip_heads:
            head_idx = int(head_idx_tensor.item())
            score = float(per_head_max[head_idx].item())
            sqrt_scale = math.sqrt(threshold / score)
            rope_scale = threshold / score
            _clip_attention_head(module, head_idx, sqrt_scale, rope_scale)

    return max_seen


@torch.no_grad()
def _clip_attention_head(module: nn.Module, head_idx: int, sqrt_scale: float, rope_scale: float) -> None:
    qk_nope_head_dim = getattr(module, "qk_nope_head_dim", None)
    qk_rope_head_dim = getattr(module, "qk_rope_head_dim", None)
    q_head_dim = getattr(module, "q_head_dim", None)
    v_head_dim = getattr(module, "v_head_dim", None)
    if None in (qk_nope_head_dim, qk_rope_head_dim, q_head_dim, v_head_dim):
        return

    q_start = head_idx * q_head_dim
    q_nope_end = q_start + qk_nope_head_dim
    q_rope_end = q_nope_end + qk_rope_head_dim

    if hasattr(module, "q_proj"):
        module.q_proj.weight[q_start:q_nope_end].mul_(sqrt_scale)
        module.q_proj.weight[q_nope_end:q_rope_end].mul_(rope_scale)
    elif hasattr(module, "q_b_proj"):
        module.q_b_proj.weight[q_start:q_nope_end].mul_(sqrt_scale)
        module.q_b_proj.weight[q_nope_end:q_rope_end].mul_(rope_scale)

    if hasattr(module, "kv_b_proj"):
        kv_stride = qk_nope_head_dim + v_head_dim
        k_start = head_idx * kv_stride
        k_end = k_start + qk_nope_head_dim
        module.kv_b_proj.weight[k_start:k_end].mul_(sqrt_scale)
