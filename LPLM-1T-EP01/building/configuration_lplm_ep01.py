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
            pad_token_id: int = 0,
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
        self._attn_implementation = _attn_implementation

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size
        super().__init__(**vision_config_kwargs)


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
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = True,
        video_placeholder="<|lplm_ep01_video_placeholder|>",
        variant: str = "1T", # "1T" or "300M"
        **kwargs,
    ):
        if variant == "1T":
            text_defaults = {
                "hidden_size": 8192,
                "num_hidden_layers": 80,
                "intermediate_size": 24576,
                "num_attention_heads": 128,
                "n_shared_experts": 1,
                "n_routed_experts": 512,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
            }
            vision_defaults = {
                "vt_hidden_size": 1536,
                "vt_num_hidden_layers": 40,
                "text_hidden_size": 8192,
            }
        elif variant == "300M":
            text_defaults = {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "n_shared_experts": None,
                "n_routed_experts": None,
            }
            vision_defaults = {
                "vt_hidden_size": 768,
                "vt_num_hidden_layers": 12,
                "text_hidden_size": 1024,
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
