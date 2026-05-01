
import torch
import sys
import os
from transformers.utils import is_flash_attn_2_available

# Add current directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configuration_lplm_ep01 import LPLMEP01Config
from modeling_lplm_ep01 import LPLMEP01ForConditionalGeneration

def count_parameters(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())

def format_params(n):
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return str(n)

def iter_moe_layers(language_model):
    decoder = getattr(language_model, "model", None)
    layers = getattr(decoder, "layers", [])
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if getattr(mlp, "experts", None) is not None:
            yield mlp

def active_moe_breakdown(model, config):
    num_experts_per_tok = config.text_config.num_experts_per_tok
    moe_layers = list(iter_moe_layers(model.language_model))
    routed_experts_params = 0
    shared_experts_params = 0
    inactive_routed_params = 0
    one_routed_expert_params = None

    for moe_layer in moe_layers:
        expert_counts = [
            count_parameters(expert)
            for expert in getattr(moe_layer, "experts", [])
            if expert is not None
        ]
        if expert_counts:
            routed_experts_params += sum(expert_counts)
            if one_routed_expert_params is None:
                one_routed_expert_params = expert_counts[0]

            active_expert_count = getattr(
                moe_layer, "num_experts_per_tok", num_experts_per_tok
            )
            if active_expert_count is None:
                active_expert_count = len(expert_counts)
            active_expert_count = min(active_expert_count, len(expert_counts))

            if len(set(expert_counts)) == 1:
                active_routed_params = active_expert_count * expert_counts[0]
            else:
                active_routed_params = round(
                    sum(expert_counts) * active_expert_count / len(expert_counts)
                )
            inactive_routed_params += sum(expert_counts) - active_routed_params

        shared_experts_params += count_parameters(
            getattr(moe_layer, "shared_experts", None)
        )

    return {
        "num_moe_layers": len(moe_layers),
        "one_routed_expert_params": one_routed_expert_params,
        "total_expert_params": routed_experts_params + shared_experts_params,
        "inactive_routed_params": inactive_routed_params,
    }

def make_count_config(variant):
    config = LPLMEP01Config(variant=variant)
    vision_attn = getattr(config.vision_config, "_attn_implementation", None)
    if vision_attn == "flash_attention_2" and not is_flash_attn_2_available():
        config.vision_config._attn_implementation = "eager"
        print(
            "Parameter-count override: vision attention flash_attention_2 -> eager "
            "(flash_attn is not installed)."
        )
    return config

def run_calc(variant="1T"):
    print(f"--- Calculating parameters for variant: {variant} ---")
    config = make_count_config(variant)
    
    # Use meta device to avoid memory allocation
    with torch.device("meta"):
        model = LPLMEP01ForConditionalGeneration(config)
    
    total_params = count_parameters(model)
    vision_params = count_parameters(model.vision_tower)
    projector_params = count_parameters(model.mm_projector)
    language_params = count_parameters(model.language_model)
    
    print(f"Total Parameters: {format_params(total_params)} ({total_params:,})")
    print(f"Vision Tower: {format_params(vision_params)} ({vision_params:,})")
    print(f"MM Projector: {format_params(projector_params)} ({projector_params:,})")
    print(f"Language Model: {format_params(language_params)} ({language_params:,})")
    
    print("\nBreakdown of Language Model:")
    num_layers = config.text_config.num_hidden_layers
    moe_breakdown = active_moe_breakdown(model, config)
    active_params = total_params - moe_breakdown["inactive_routed_params"]
    
    print(f"Active Parameters per token (approx): {format_params(active_params)} ({active_params:,})")
    print(f"MoE Layers: {moe_breakdown['num_moe_layers']} / {num_layers}")
    if moe_breakdown["one_routed_expert_params"] is None:
        print("One Routed Expert: N/A")
    else:
        print(f"One Routed Expert: {format_params(moe_breakdown['one_routed_expert_params'])}")
    print(f"Total Experts (all MoE layers): {format_params(moe_breakdown['total_expert_params'])}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_calc(sys.argv[1])
    else:
        run_calc("1T")
        print("\n")
        run_calc("300M")
