# Oumi Tool Calls Training Support - Bug Fixes and Improvements

This document describes the issues identified when training models with tool/function calling data in Oumi, and the proposed fixes.

## Overview

When fine-tuning LLMs (like Llama 3.2) with tool-calling training data, several critical issues prevent proper learning:

1. **Tool calls and tool response not parsed** - The `tool_calls` field in assistant messages is silently dropped
2. **Tool responses not masked** - Content from tool/ipython roles is incorrectly trained on
3. **End-of-turn tokens masked** - `<|eot_id|>` tokens are masked when `pad_token == eos_token`

These issues result in models that:
- Cannot generate proper tool calls
- Output garbage like `<|python_tag|>user<|end_header_id|>` repeatedly
- Only output `<|eot_id|>` without any content
- Don't know when to stop generating

---

## Issue 1: Tool Calls Not Parsed in Conversation Model

### Problem

The `Message` class in `oumi/core/types/conversation.py` does not have fields for `tool_calls` or `tool_call_id`. When loading training data in the standard OpenAI format:

```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"New York\"}"
    }
  }],
  "content": ""
}
```

The `tool_calls` field is silently dropped during Pydantic validation.

### Solution

Add `ToolCall` and `ToolCallFunction` models and corresponding fields to the `Message` class:

```python
# In oumi/core/types/conversation.py

import json
from typing import Any, Optional, Union

class ToolCallFunction(pydantic.BaseModel):
    """Represents the function details within a tool call."""
    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    """The name of the function to call."""

    arguments: Union[str, dict[str, Any]]
    """The arguments to pass to the function. Can be JSON string or dict."""

    @pydantic.field_serializer("arguments")
    def _serialize_arguments(self, value: Union[str, dict[str, Any]]) -> str:
        """Ensures arguments are always a JSON string for serialization."""
        if isinstance(value, dict):
            return json.dumps(value)
        return value


class ToolCall(pydantic.BaseModel):
    """Represents a tool/function call made by the assistant."""
    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    """Unique identifier for the tool call."""

    type: str = "function"
    """The type of tool call (currently only 'function' is supported)."""

    function: ToolCallFunction
    """The function to call with its arguments."""


class Message(pydantic.BaseModel):
    # ... existing fields ...

    tool_calls: Optional[list[ToolCall]] = None
    """Optional list of tool calls for assistant messages."""

    tool_call_id: Optional[str] = None
    """For tool role messages, the ID of the tool call being responded to."""
```

### Files to Modify

- `oumi/core/types/conversation.py`




---

## Training Data Format

The expected format for tool-call training data (OpenAI-compatible):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to tools."
    },
    {
      "role": "user",
      "content": "What's the weather in New York?"
    },
    {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"New York\"}"
        }
      }],
      "content": ""
    },
    {
      "role": "tool",
      "tool_call_id": "call_0",
      "content": "{\"temperature\": 72, \"conditions\": \"sunny\"}"
    },
    {
      "role": "assistant",
      "content": "The weather in New York is sunny with a temperature of 72°F."
    }
  ]
}
```

---

## Expected Masking Behavior

After applying all fixes:

| Content | Masked/Trained |
|---------|----------------|
| System prompt | MASKED |
| User message | MASKED |
| Assistant tool call JSON | **TRAINED** |
| `<\|eot_id\|>` after tool call | **TRAINED** |
| ipython/tool response | MASKED |
| `<\|eot_id\|>` after tool response | MASKED |
| Assistant final response | **TRAINED** |
| `<\|eot_id\|>` after final response | **TRAINED** |

---

## References

- [Llama 3 Tool Use Documentation](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [TRL DataCollatorForCompletionOnlyLM](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only)
