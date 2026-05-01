from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Union, Dict

try:
    from .configuration_deepseek import DeepseekV3Config
except ImportError:
    from configuration_deepseek import DeepseekV3Config

logger = logging.get_logger(__name__)

class LPLMEP01VisionConfig(PretrainedConfig):
    """Configuration for LPLMEP01 Vision Tower."""
    model_type = "lplm_ep01_vision"

    def __init__(
            self,
            patch_size: int = 14,
            init_pos_emb_height: int = 64,
            init_pos_emb_width: int = 64,
            init_pos_emb_time: int = 4,
            pos_emb_type: str = 'divided_fixed',
            vt_num_attention_heads: int = 16,
            vt_num_hidden_layers: int = 27,
            vt_hidden_size: int = 1152,
            vt_intermediate_size: int = 4304,
            merge_kernel_size: tuple = (2, 2),
            video_attn_type: str = 'spatial_temporal',
            merge_type: str = 'sd2_tpool',
            _attn_implementation: str = 'flash_attention_2',
            # MM Projector parameters
            mm_projector_type: str = 'patchmerger',
            mm_hidden_size: int | None = None,
            projector_hidden_act: str = "gelu",
            projector_ln_eps: float = 1e-5,
            # Other parameters
            ignore_index: int = -100,
            media_placeholder_token_id: int = 163605,
            pad_token_id: int = 163839,
            use_unified_vision_chunk: bool = True,
            video_placeholder="<|lplm_ep01_video_placeholder|>",
            text_hidden_size=7168,
            **vision_config_kwargs):

        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size
        super().__init__(pad_token_id=pad_token_id, **vision_config_kwargs)
        self._attn_implementation = _attn_implementation


class LPLMEP01Config(PretrainedConfig):
    """LPLMEP01 model configuration.
    
    Supports two variants:
    - 1T: Large MoE model (~1 trillion parameters)
    - 300M: Small variant for testing/evaluation
    """

    model_type = "lplm_ep01"

    def __init__(
        self,
        text_config: Union[Dict, DeepseekV3Config] = None,
        vision_config: Union[Dict, LPLMEP01VisionConfig] = None,
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 163839,
        use_unified_vision_chunk: bool = True,
        video_placeholder="<|lplm_ep01_video_placeholder|>",
        variant: str = "1T", # "1T" or "300M"
        **kwargs,
    ):
        if variant == "1T":
            text_defaults = {
                "model_type": "kimi_k2",
                "vocab_size": 163840,
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "intermediate_size": 18432,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "n_shared_experts": 1,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "num_nextn_predict_layers": 0,
                "ep_size": 1,
                "routed_scaling_factor": 2.827,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "topk_method": "noaux_tc",
                "n_group": 1,
                "topk_group": 1,
                "norm_topk_prob": True,
                "scoring_func": "sigmoid",
                "aux_loss_alpha": 0.001,
                "seq_aux": True,
                "hidden_act": "silu",
                "max_position_embeddings": 262144,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-5,
                "use_cache": True,
                "pad_token_id": 163839,
                "bos_token_id": 163584,
                "eos_token_id": 163586,
                "pretraining_tp": 1,
                "tie_word_embeddings": False,
                "rope_theta": 50000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 64.0,
                    "original_max_position_embeddings": 4096,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                },
                "attention_bias": False,
                "attention_dropout": 0.0,
            }
            vision_defaults = {
                "patch_size": 14,
                "init_pos_emb_height": 64,
                "init_pos_emb_width": 64,
                "init_pos_emb_time": 4,
                "pos_emb_type": "divided_fixed",
                "vt_hidden_size": 1152,
                "vt_num_hidden_layers": 27,
                "vt_num_attention_heads": 16,
                "vt_intermediate_size": 4304,
                "merge_kernel_size": [2, 2],
                "video_attn_type": "spatial_temporal",
                "merge_type": "sd2_tpool",
                "_attn_implementation": "flash_attention_2",
                "mm_projector_type": "patchmerger",
                "mm_hidden_size": 1152,
                "projector_hidden_act": "gelu",
                "projector_ln_eps": 1e-5,
                "text_hidden_size": 7168,
            }
        elif variant == "300M":
            text_defaults = {
                "model_type": "kimi_k2",
                "vocab_size": 163840,
                "hidden_size": 576,
                "num_hidden_layers": 14,
                "intermediate_size": 1536,
                "num_attention_heads": 9,
                "num_key_value_heads": 9,
                "n_shared_experts": 1,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 384,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 2,
                "num_nextn_predict_layers": 0,
                "routed_scaling_factor": 1.0,
                "kv_lora_rank": 128,
                "q_lora_rank": None,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 32,
                "v_head_dim": 64,
                "topk_method": "noaux_tc",
                "n_group": 1,
                "topk_group": 1,
                "norm_topk_prob": True,
                "scoring_func": "sigmoid",
                "aux_loss_alpha": 0.001,
                "seq_aux": True,
                "hidden_act": "silu",
                "max_position_embeddings": 32768,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-5,
                "use_cache": True,
                "pad_token_id": 163839,
                "bos_token_id": 163584,
                "eos_token_id": 163586,
                "pretraining_tp": 1,
                "tie_word_embeddings": False,
                "rope_theta": 50000.0,
                "rope_scaling": {
                    "type": "linear",
                    "factor": 1.0,
                },
                "attention_bias": False,
                "attention_dropout": 0.0,
            }
            vision_defaults = {
                "vt_hidden_size": 384,
                "vt_num_hidden_layers": 8,
                "vt_num_attention_heads": 8,
                "vt_intermediate_size": 1536,
                "mm_hidden_size": 384,
                "text_hidden_size": 576,
            }
        else:
            text_defaults = {}
            vision_defaults = {}

        if text_config is None:
            text_config = text_defaults
        elif isinstance(text_config, dict):
            # Update provided dict with defaults for missing keys
            for k, v in text_defaults.items():
                if k not in text_config:
                    text_config[k] = v
        
        if vision_config is None:
            vision_config = vision_defaults
        elif isinstance(vision_config, dict):
            for k, v in vision_defaults.items():
                if k not in vision_config:
                    vision_config[k] = v

        if isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        if isinstance(vision_config, dict):
            vision_config = LPLMEP01VisionConfig(**vision_config)
        
        self.text_config = text_config
        self.vision_config = vision_config
        self.variant = variant
        
        # Other config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder
        
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)
