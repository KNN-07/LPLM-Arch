# Reward Shaping

This document defines the reward contract for provenance-discounted
hierarchical RL.

## Variables

| Symbol | Meaning |
| --- | --- |
| `P` | Planning reward in `[0, 1]`. |
| `S` | Binary or graded task success score in `[0, 1]`. |
| `D_hint` | Hint discount by autonomy level. |
| `C_prov` | Maximum reward cap by provenance. |
| `M` | Malus penalties for unsafe or invalid behavior. |
| `R_exec` | Execution reward. |
| `R_final` | Final shaped reward. |

## Provenance Cap

```text
C_prov(human_verified) = 1.0
C_prov(synthetic) = 0.6
```

The model prompt must not reveal which cap applies.

## Hint Discount

```text
D_hint(none)       = 1.0
D_hint(conceptual) = 0.6
D_hint(localized)  = 0.4
D_hint(procedural) = 0.2
D_hint(failure)    = 0.0
```

The paper version uses three attempts: no hint, high-level hint, and low-level
hint. The implementation template supports an optional localized middle level.

## Execution Reward

```text
R_exec = C_prov * D_hint * S
```

For binary private-test rewards:

```text
S = 1 if private tests pass else 0
```

For graded rewards:

```text
S = 0.5 * public_test_pass_rate + 0.5 * private_test_pass_rate
```

## Planning Reward

Plan reward should be small enough not to dominate task success:

```text
P = mean([
  reproduces_failure,
  localizes_relevant_area,
  proposes_minimal_edit,
  includes_verification,
  avoids_broad_rewrite
])
```

Default weight:

```text
W_plan = 0.1
```

## Final Reward

```text
R_final = clamp(R_exec + W_plan * P - M, min=-1.0, max=C_prov)
```

Suggested penalties:

| Penalty | Value |
| --- | --- |
| Deletes or weakens tests | `1.0` |
| Edits unrelated large surfaces | `0.3` |
| Syntax failure | `0.2` |
| Timeout | `0.2` |
| Unsafe command | `1.0` |
| Fabricated tool output | `1.0` |

## GRPO Advantage

For each task group with rollouts `i = 1..G`:

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

Use GiGPO when nesting by provenance:

```text
A_i = normalize_within_task(normalize_within_provenance(R_i))
```

This reduces synthetic-task dominance while preserving useful dense gradients
from synthetic volume.

## Reward Hacking Checks

Reject or penalize rollouts that:

- Delete, skip, or weaken tests.
- Modify lockfiles or generated artifacts without cause.
- Hard-code expected outputs.
- Claim tests passed without tool evidence.
- Touch unrelated subsystems.
- Depend on network access unless the task explicitly permits it.
