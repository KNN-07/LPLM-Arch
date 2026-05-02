---
license: apache-2.0
---

# LPLM-1T-EP01

LPLM-1T-EP01 is an experimental multimodal MoE model architecture with roughly 1 trillion total parameters. The language stack uses the `lplm_ep01_text` configuration in this repository and DeepSeek V3-style modeling code. The vision tower uses an LPLM 3D patch encoder.

This model is experimental and not intended for production use. Training data is planned from Hugging Face open source pretraining datasets filtered from CommonCrawl.

## Basic Configuration

| Component | Setting |
| --- | --- |
| Total parameters | 1,026,879,376,368 |
| Active parameters per token | ~33.33B |
| Language model parameters | 1,026,408,232,448 |
| Vision tower parameters | 416,866,032 |
| Multimodal projector parameters | 54,277,888 |
| Text model type | `lplm_ep01_text` |
| Attention implementation | `flash_attention_2` |
| Data type | `bfloat16` |

## Text Model

| Field | Value |
| --- | --- |
| Vocabulary size | 163,840 |
| Hidden size | 7,168 |
| Layers | 61 |
| Intermediate size | 18,432 |
| Attention heads | 64 |
| Key/value heads | 64 |
| Max position embeddings | 262,144 |
| RoPE theta | 50,000 |
| RoPE scaling | YaRN, factor 64, original context 4,096 |

## MoE

| Field | Value |
| --- | --- |
| MoE layers | 60 / 61 |
| First dense layers | 1 |
| Routed experts per MoE layer | 384 |
| Shared experts per MoE layer | 1 |
| Active routed experts per token | 8 |
| MoE intermediate size | 2,048 |
| One routed expert | ~44.04M parameters |
| Router scoring | `sigmoid` |
| Top-k method | `noaux_tc` |
| Routed scaling factor | 2.827 |

## Vision And Projector

| Field | Value |
| --- | --- |
| Vision hidden size | 1,152 |
| Vision layers | 27 |
| Vision attention heads | 16 |
| Vision intermediate size | 4,304 |
| Patch size | 14 |
| Position embedding | `divided_fixed` |
| Merge type | `sd2_tpool` |
| Merge kernel | `[2, 2]` |
| Projector type | `patchmerger` |
| Projector input hidden size | 1,152 |
| Projector output hidden size | 7,168 |

## Tokens

| Token | ID |
| --- | --- |
| BOS | 163584 |
| EOS | 163586 |
| PAD | 163839 |
| Media placeholder | 163605 |

## Test Variant

`LPLM-0.3B-EP01` is the small evaluation variant. It keeps the MoE routing pattern with smaller experts so local tests exercise the same dense-plus-MoE layer structure as the 1T configuration.

| Component | Setting |
| --- | --- |
| Total parameters | 299,998,816 |
| Active parameters per token | ~252.22M |
| Language model parameters | 280,756,384 |
| Vision tower parameters | 15,995,520 |
| Multimodal projector parameters | 3,246,912 |
| Text hidden size | 576 |
| Text layers | 14 |
| Attention heads | 9 |
| First dense layers | 2 |
| MoE layers | 12 / 14 |
| Routed experts per MoE layer | 8 |
| Shared experts per MoE layer | 1 |
| Active routed experts per token | 2 |
| MoE intermediate size | 384 |
| One routed expert | ~663.55K parameters |
| Router scoring | `sigmoid` |
| Top-k method | `noaux_tc` |
