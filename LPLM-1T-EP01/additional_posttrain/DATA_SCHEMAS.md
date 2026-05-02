# Data Schemas

All records should be JSONL. Keep provenance out of prompts and inside reward
metadata only.

## SWE Task Record

```json
{
  "id": "swe-human-000001",
  "source": "swe_bench_verified",
  "provenance": "human_verified",
  "repo": {
    "url": "https://github.com/org/repo",
    "base_commit": "abc123",
    "language": "python"
  },
  "issue": {
    "title": "Fix parser crash on empty input",
    "body": "The parser raises IndexError when input is empty.",
    "public_tests": ["pytest tests/test_parser.py::test_empty_input"]
  },
  "gold": {
    "merged_pr": "https://github.com/org/repo/pull/123",
    "diff_path": "data/diffs/swe-human-000001.patch",
    "private_tests": ["pytest tests/test_parser.py"]
  },
  "reward": {
    "provenance_cap": 1.0,
    "success_metric": "private_tests_pass"
  },
  "metadata": {
    "license": "repo-license-id",
    "difficulty": "medium"
  }
}
```

## Hint Curriculum Record

```json
{
  "task_id": "swe-human-000001",
  "hint_source": "human_diff_distilled",
  "hints": {
    "conceptual": "Inspect how the parser handles sequence length before indexing.",
    "localized": "The empty-input path reaches parser.py before validation.",
    "procedural": "Add an early return for empty token lists in parser.py, then run the parser empty-input test."
  },
  "grounding": {
    "diff_path": "data/diffs/swe-human-000001.patch",
    "files_touched": ["src/parser.py", "tests/test_parser.py"]
  },
  "quality": {
    "verified_against_gold_diff": true,
    "manual_reviewed": false
  }
}
```

## Plan Rollout Record

```json
{
  "task_id": "swe-human-000001",
  "rollout_id": "rollout-000001-plan",
  "model_checkpoint": "outputs/post_training/pref_v1",
  "plan_text": "<PLAN>\n1. Reproduce the failure.\n2. Inspect parser.py.\n3. Add empty-input guard.\n4. Run parser tests.\n</PLAN>",
  "plan_reward": {
    "score": 0.85,
    "checks": {
      "reproduces_failure": true,
      "localizes_area": true,
      "minimal_edit": true,
      "verification_step": true
    }
  }
}
```

## Execution Rollout Record

```json
{
  "task_id": "swe-human-000001",
  "rollout_id": "rollout-000001-exec",
  "group_id": "grpo-group-000001",
  "attempts": [
    {
      "attempt": 1,
      "hint_level": "none",
      "transcript_path": "rollouts/rollout-000001/attempt1.jsonl",
      "patch_path": "rollouts/rollout-000001/attempt1.patch",
      "tests": {
        "public_passed": false,
        "private_passed": false
      },
      "terminal_status": "failed_tests"
    },
    {
      "attempt": 2,
      "hint_level": "conceptual",
      "transcript_path": "rollouts/rollout-000001/attempt2.jsonl",
      "patch_path": "rollouts/rollout-000001/attempt2.patch",
      "tests": {
        "public_passed": true,
        "private_passed": true
      },
      "terminal_status": "success"
    }
  ],
  "reward": {
    "plan_reward": 0.85,
    "execution_reward": 0.6,
    "provenance_cap": 1.0,
    "final_reward": 0.685
  }
}
```

## Tool Transcript Event

```json
{
  "rollout_id": "rollout-000001-exec",
  "attempt": 2,
  "step": 4,
  "role": "assistant",
  "tool_call": {
    "name": "shell",
    "arguments": {
      "command": "pytest tests/test_parser.py::test_empty_input"
    }
  },
  "observation": {
    "exit_code": 0,
    "stdout_path": "rollouts/rollout-000001/stdout-004.txt",
    "stderr_path": "rollouts/rollout-000001/stderr-004.txt"
  }
}
```
