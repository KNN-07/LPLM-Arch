
import torch
import sys
import os

# Add current directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configuration_lplm_ep01 import LPLMEP01Config
from modeling_lplm_ep01 import LPLMEP01ForConditionalGeneration

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def format_params(n):
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return str(n)

def run_calc(variant="1T"):
    print(f"--- Calculating parameters for variant: {variant} ---")
    config = LPLMEP01Config(variant=variant)
    
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
    
    # Estimate active parameters per token
    # Active = Vision Tower (all) + Projector (all) + Language Model (Embeddings + Attn + Active Experts + Norms)
    # This is a bit complex to calculate automatically without traversing the module tree specifically.
    
    print("\nBreakdown of Language Model:")
    # We can calculate active parameters per layer for MoE
    # For DeepSeek-V3 MoE:
    # Active per MoE layer = Shared Experts + num_experts_per_tok * Routed Expert
    
    num_layers = config.text_config.num_hidden_layers
    num_experts_per_tok = config.text_config.num_experts_per_tok
    
    # Calculate size of one routed expert
    # One expert has gate_proj, up_proj, down_proj
    # gate_proj: hidden_size -> moe_intermediate_size
    # up_proj: hidden_size -> moe_intermediate_size
    # down_proj: moe_intermediate_size -> hidden_size
    h = config.text_config.hidden_size
    mi = config.text_config.moe_intermediate_size
    one_expert_params = 3 * h * mi # gate, up, down
    
    # Shared experts
    n_shared = config.text_config.n_shared_experts
    shared_params = n_shared * one_expert_params if n_shared else 0
    
    # Routed experts total
    n_routed = config.text_config.n_routed_experts
    routed_total_params = n_routed * one_expert_params if n_routed else 0
    
    # Active experts per layer
    active_experts_params = (num_experts_per_tok * one_expert_params) + shared_params
    
    # Non-expert params per layer (Attn, Norms, etc.)
    # Rough estimate: Layer params - total expert params
    # We'll just do it more simply:
    
    # Calculate non-MoE params (Embeddings + Attn + Norms + Gate + etc.)
    # Total = Embeddings + Layers * (Attn + Norms + Experts) + Head
    # Active = Embeddings + Layers * (Attn + Norms + Active Experts) + Head
    
    inactive_params_per_layer = ((n_routed - num_experts_per_tok)
                                 * one_expert_params) if n_routed else 0
    total_inactive = num_layers * inactive_params_per_layer
    
    active_params = total_params - total_inactive
    
    print(f"Active Parameters per token (approx): {format_params(active_params)} ({active_params:,})")
    print(f"One Routed Expert: {format_params(one_expert_params)}")
    print(f"Total Experts (all layers): {format_params(num_layers * (routed_total_params + shared_params))}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_calc(sys.argv[1])
    else:
        run_calc("1T")
        print("\n")
        run_calc("300M")
