# LPLM-Arch

Experimental architecture work for the LPLM language model family.

This repository tracks model architecture experiments, configuration files, and packaging code for LPLM variants. The code is research-oriented and should be treated as experimental until a model card or release note says otherwise.

## Current Experiments

| Model | Status | Notes |
| --- | --- | --- |
| `LPLM-1T-EP01` | Active | 1T-parameter experimental architecture adapted from DeepSeek V3 and multimodal modeling patterns. |
| `LPLM-0.3B-EP01` | Planned/test variant | Smaller evaluation variant described in the EP01 model card. |
| `LPLM-EP02` | Placeholder | Reserved for the next experiment series. |

## Repository Layout

```text
.
|-- LPLM-1T-EP01/
|   |-- README.md
|   |-- additional_posttrain/
|   |-- building/
|   |   |-- tokenizer/
|   |   `-- *.py
|   |-- post_training/
|   `-- training/
|-- LPLM-EP02/
|-- LICENSE
`-- README.md
```

## Notes

- `LPLM-1T-EP01/README.md` contains the model-card metadata for the EP01 experiment.
- `LPLM-1T-EP01/building/` contains the active EP01 model/configuration code and local tokenizer assets.
- Generated caches, local editor state, run outputs, and model checkpoints are intentionally ignored by git.

## License

The repository is licensed under the Apache License 2.0. Some files include additional upstream attribution and license notices in their headers; keep those notices intact when modifying derived code.
