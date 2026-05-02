# Post-Training Strategy

This plan assumes `LPLM-1T-EP01` has completed base pretraining and has a
stable Kimi-tokenizer checkpoint. The goal is to move from a base model to
instruction-following, tool-capable, safety-aware, and reasoning-specialized
variants without losing base capability.

## Stage 0: Base Checkpoint Readiness

Do not start post-training until the base checkpoint has:

- Stable validation loss on a held-out web mix.
- No obvious tokenizer/template mismatch.
- No sustained NaNs, exploding QK logits, or unstable router behavior.
- A reproducible W&B lineage: code commit, dataset revisions, tokenizer id,
  optimizer settings, hardware topology, and checkpoint path.

Recommended readiness metrics:

| Metric | Gate |
| --- | --- |
| Held-out perplexity | No late-run regression trend. |
| QK max logits | Stable under the configured QK-Clip threshold. |
| Expert load | No large group of dead experts. |
| Router entropy | Stable per MoE layer after warmup. |
| Throughput | Stable enough for full-stage cost estimates. |

## Stage 1: Continued Pretraining

Use CPT to specialize the base model before instruction tuning. This is still
next-token training, not chat training.

Recommended mixture:

- 40 percent high-quality general replay.
- 20 percent code.
- 15 percent math and symbolic reasoning.
- 10 percent tool/API/docs data.
- 10 percent multilingual high-quality text.
- 5 percent target-domain corpora.

Why this stage matters:

- It gives the base model domain vocabulary and task priors before instruction
  formatting narrows behavior.
- It is cheaper and less brittle than trying to teach all domain knowledge via
  SFT.
- It reduces the amount of synthetic instruction data needed later.

Training guidance:

- Keep MuonClip enabled.
- Use long-context samples gradually. Start with the normal block size, then
  add a long-context phase only after stability is proven.
- Keep a general replay mix large enough to prevent domain collapse.
- Log separate loss by source if the data loader supports it.

Promotion gate:

- Domain validation improves.
- General validation does not regress beyond the accepted threshold.
- No routing collapse or persistent expert starvation.

## Stage 2: Supervised Fine-Tuning

SFT should teach the assistant interface, not overload the model with every
possible skill. Use clean data and the Kimi chat template.

Core SFT buckets:

- General instruction following.
- Multi-turn chat.
- Code editing and explanation.
- Math and reasoning demonstrations.
- Tool-call formatting.
- Refusal and safe-completion examples.
- Grounded QA and citation behavior if retrieval is planned.

Data quality rules:

- Prefer fewer high-quality examples over large noisy instruction dumps.
- Normalize all conversations through the Kimi chat template.
- Keep assistant answers direct and task-solving oriented.
- Separate reasoning traces from final answers when the target behavior needs
  hidden or private reasoning controls.
- Do not mix tool results with assistant text unless the schema makes tool
  boundaries explicit.

Training guidance:

- Start with conservative LR and short runs.
- Use a held-out human-written validation set, not only synthetic examples.
- Keep a small base replay slice if instruction tuning causes capability loss.
- Watch length bias, refusal over-triggering, and tool-call hallucination.

Promotion gate:

- Instruction following improves on held-out tasks.
- Refusal behavior is accurate rather than broad.
- Tool-call JSON or TypeScript-style declarations are valid.
- General knowledge and code benchmarks do not materially regress.

## Stage 3: Preference Optimization

Run preference optimization after SFT. Do not apply it directly to a raw base
checkpoint.

Method choice:

| Method | Use When | Notes |
| --- | --- | --- |
| DPO | You have paired chosen/rejected responses and enough memory for a reference model. | Stable default with strong tooling support. |
| SimPO | You need lower memory overhead or want reference-free training. | Good fit for very large models when reference-model cost is high. |
| KTO | You have desirable/undesirable labels rather than preference pairs. | Useful for safety and style data where pair construction is expensive. |

Preference data buckets:

- Helpfulness and completeness.
- Honesty and calibrated uncertainty.
- Tool-call correctness.
- Code patch correctness.
- Safety and refusal boundary quality.
- Concision versus under-answering.
- Multi-turn consistency.

Training guidance:

- Keep preference data close to target deployment prompts.
- Include hard negatives from the current SFT model.
- Track reward margins or preference accuracy by bucket.
- Avoid tuning only for verbosity or generic friendliness.

Promotion gate:

- Pairwise win rate improves against the SFT checkpoint.
- Safety false positives do not spike.
- Tool-call validity does not regress.
- Long-answer factuality does not degrade.

## Stage 4: Reasoning RL

Use verifiable rewards for math, code, and structured reasoning. Prefer GRPO or
RLVR-style training where each prompt gets multiple sampled completions and a
relative score.

Good reward sources:

- Exact-answer math checks.
- Unit tests for code tasks.
- Static analysis and formatting checks.
- Symbolic solvers for constrained domains.
- Retrieval-grounded answer checks where labels are available.

Avoid rewards that are only style preferences. Reasoning RL should reward
correctness and verifiable progress.

Training guidance:

- Start from the SFT or preference-tuned checkpoint, not the base checkpoint.
- Use short context and small sample counts until rewards are reliable.
- Keep KL or reference constraints to prevent language quality drift.
- Evaluate pass@k and single-sample accuracy separately.

Promotion gate:

- Verifiable task accuracy improves.
- Single-sample behavior improves, not only large self-consistency.
- Natural-language answer quality remains acceptable.
- The model does not learn reward hacks.

## Stage 5: Safety And Tool Polish

This stage is for boundary behavior and production readiness.

Focus areas:

- Refusal precision.
- Non-evasive safe alternatives.
- Sensitive-domain uncertainty.
- Tool-call schema validity.
- No fabricated tool outputs.
- Prompt-injection resistance.
- Long-context instruction hierarchy.

Use a mix of SFT, KTO, DPO, and targeted rejection sampling. Keep this stage
small and evaluate heavily.

## Stage 6: Release Evaluation

Run a fixed eval suite for every candidate:

- General capability: MMLU-style, BBH-style, and internal held-out prompts.
- Math: GSM8K, MATH-style, and internal verified tasks.
- Code: HumanEval/MBPP-style, repo editing tasks, and unit-test tasks.
- Chat: MT-Bench-style, multi-turn instruction following, and style evals.
- Safety: jailbreaks, refusal boundaries, sensitive advice, and policy tests.
- Tool use: schema validity, argument correctness, and tool-result grounding.
- Long context: retrieval, needle tests, and multi-document synthesis.
- MoE health: expert usage and router stats.

Release only if the target-stage metrics improve without unacceptable
regressions in safety, tool use, routing, or core capability.

## References

- InstructGPT: supervised instruction tuning plus human preference training.
- DPO: direct preference optimization with paired preferences.
- SimPO: reference-free preference optimization.
- KTO: desirable/undesirable feedback optimization.
- DeepSeekMath: math-focused CPT plus GRPO for reasoning.
- Constitutional AI: critique, revision, and AI-feedback safety training.
