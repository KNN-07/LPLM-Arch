# Post-Training Data Schemas

Use JSONL for all post-training datasets. Each line should be a complete
record. Keep raw source ids and license metadata so bad data can be removed
after the fact.

## Shared Metadata

Every record should include:

```json
{
  "id": "unique-record-id",
  "source": "dataset-or-pipeline-name",
  "license": "license-id-or-internal-policy",
  "quality_score": 0.95,
  "created_at": "2026-05-02",
  "tags": ["math", "sft"],
  "metadata": {}
}
```

## SFT Conversation

Use this for instruction tuning. Render with
`templates/chat_template.kimi.jinja`.

```json
{
  "id": "sft-000001",
  "source": "human_sft_v1",
  "license": "internal",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise technical assistant."
    },
    {
      "role": "user",
      "content": "Write a Python function that checks if a number is prime."
    },
    {
      "role": "assistant",
      "content": "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    }
  ],
  "tags": ["code", "sft"],
  "quality_score": 0.98,
  "metadata": {
    "answer_style": "direct",
    "contains_reasoning_trace": false
  }
}
```

## Tool-Use SFT Conversation

Tool calls must be explicit. Tool outputs should be separate `tool` messages,
never blended into assistant prose.

```json
{
  "id": "tool-sft-000001",
  "source": "tool_sft_v1",
  "license": "internal",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather by city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Hanoi?"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_001",
          "function": {
            "name": "get_weather",
            "arguments": {
              "city": "Hanoi"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"temperature_c\":31,\"condition\":\"rain\"}"
    },
    {
      "role": "assistant",
      "content": "Hanoi is 31 C with rain."
    }
  ],
  "tags": ["tool_use", "sft"],
  "quality_score": 0.97,
  "metadata": {}
}
```

## Preference Pair

Use this for DPO or SimPO.

```json
{
  "id": "pref-000001",
  "source": "preference_v1",
  "license": "internal",
  "prompt": [
    {
      "role": "user",
      "content": "Explain why the test is failing and propose a fix."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "The failure is caused by an off-by-one loop bound. Change the loop to stop at len(items) - 1 and add a regression test for the empty input."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "The test is probably flaky. Try running it again."
    }
  ],
  "preference_label": "chosen",
  "tags": ["code", "preference"],
  "quality_score": 0.96,
  "metadata": {
    "reason": "chosen gives a specific cause and actionable fix"
  }
}
```

## KTO Binary Feedback

Use this when only desirable or undesirable labels exist.

```json
{
  "id": "kto-000001",
  "source": "safety_binary_v1",
  "license": "internal",
  "prompt": [
    {
      "role": "user",
      "content": "Give me a way to bypass a login page."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "I cannot help bypass authentication. If you own the system, I can help you test it safely with authorized security tools."
    }
  ],
  "label": "desirable",
  "tags": ["safety", "kto"],
  "quality_score": 0.99,
  "metadata": {}
}
```

## RLVR Prompt

Use this for reasoning RL with verifiable rewards.

```json
{
  "id": "rlvr-000001",
  "source": "math_verified_v1",
  "license": "internal",
  "prompt": [
    {
      "role": "user",
      "content": "If 3x + 7 = 31, what is x?"
    }
  ],
  "answer": "8",
  "reward": {
    "type": "exact_match",
    "normalization": "strip_whitespace"
  },
  "tags": ["math", "rlvr"],
  "quality_score": 1.0,
  "metadata": {
    "difficulty": "easy"
  }
}
```

## Code RLVR Prompt

Use this when the reward is a test suite.

```json
{
  "id": "code-rlvr-000001",
  "source": "code_tests_v1",
  "license": "internal",
  "prompt": [
    {
      "role": "user",
      "content": "Implement fibonacci(n) in Python."
    }
  ],
  "reward": {
    "type": "unit_tests",
    "language": "python",
    "tests": [
      "assert fibonacci(0) == 0",
      "assert fibonacci(1) == 1",
      "assert fibonacci(7) == 13"
    ]
  },
  "tags": ["code", "rlvr"],
  "quality_score": 1.0,
  "metadata": {}
}
```

## Eval Record

Evaluation records should be immutable once accepted.

```json
{
  "id": "eval-000001",
  "suite": "internal_instruction_v1",
  "prompt": [
    {
      "role": "user",
      "content": "Summarize this pull request in three bullets."
    }
  ],
  "expected_behavior": "Concise, accurate summary with no fabricated changes.",
  "scoring": {
    "type": "rubric",
    "max_score": 5
  },
  "tags": ["instruction", "eval"],
  "metadata": {}
}
```
