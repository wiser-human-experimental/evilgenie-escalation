"""Model registry for the cross-model runs (paper §5).

All models route through OpenRouter, for a single billing/auth path. Model IDs
and pricing were verified live against https://openrouter.ai/api/v1/models at
the time of the runs (prices in the inline comments are per 1M tokens,
input/output).

Per-model fields:
  model_id            OpenRouter model string
  judge               judge model for the LLM-judge scorer (GPT-4o throughout)
  include_text_editor whether to expose Inspect's text_editor tool. False for
                      the GPT-5 family: their strict function schema rejects
                      text_editor's signature ('file_text' missing from
                      'required'). True elsewhere.
  reasoning_history   set to "none" for the Gemini models: their thinking-mode
                      "thought signature" does not round-trip reliably through
                      OpenRouter's OpenAI-compatible tool-calling API across a
                      multi-turn agentic loop, producing intermittent
                      "400 - Corrupted thought signature" errors. Setting it to
                      "none" keeps thinking on (reasoning tokens are still
                      generated) but stops the corruptible signature being sent
                      back. See paper §7.3.
  family              model family, for grouping in the analysis
  notes               free-text caveats

For the OpenAI "Luna"/"Sol" entries the base (non-Pro) tier is used; the -pro
variants are identically priced on OpenRouter, so swap the model_id if that
tier is wanted.
"""

MODELS = {
    # OpenAI
    "gpt-5.3-codex": {
        "model_id": "openrouter/openai/gpt-5.3-codex",   # $1.75 / $14
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": False,  # GPT-5 strict schema rejects text_editor
        "family": "openai",
        "notes": "Stage 1 foundation model + negative-control agent (paper §6, §7.1).",
    },
    "gpt-5.6-luna": {
        "model_id": "openrouter/openai/gpt-5.6-luna",   # $1.00 / $6.00
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": False,  # assumed GPT-5-family strict schema
        "family": "openai",
        "notes": "luna-pro is identically priced -- swap model_id to "
                 "openrouter/openai/gpt-5.6-luna-pro for that tier.",
    },
    "gpt-5.6-sol": {
        "model_id": "openrouter/openai/gpt-5.6-sol",    # $5.00 / $30.00
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": False,  # assumed GPT-5-family strict schema
        "family": "openai",
        "notes": "sol-pro is identically priced -- swap model_id to "
                 "openrouter/openai/gpt-5.6-sol-pro for that tier.",
    },
    # Anthropic
    "claude-opus-4.8": {
        "model_id": "openrouter/anthropic/claude-opus-4.8",   # $5 / $25
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "anthropic",
    },
    "claude-sonnet-5": {
        "model_id": "openrouter/anthropic/claude-sonnet-5",   # $2 / $10
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "anthropic",
    },
    "claude-fable-5": {
        "model_id": "openrouter/anthropic/claude-fable-5",    # $10 / $50
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "anthropic",
        "notes": "Most expensive model in the roster.",
    },
    # Google (see reasoning_history note in the module docstring)
    "gemini-3.1-pro-preview": {
        "model_id": "openrouter/google/gemini-3.1-pro-preview",  # $2 / $12
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "reasoning_history": "none",
        "family": "google",
    },
    "gemini-3.5-flash": {
        "model_id": "openrouter/google/gemini-3.5-flash",  # $1.50 / $9
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "reasoning_history": "none",
        "family": "google",
    },
    # Qwen (Alibaba)
    "qwen-3.7-max": {
        "model_id": "openrouter/qwen/qwen3.7-max",   # $1.25 / $3.75
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "qwen",
        "notes": "Pilot before a full run -- tool-call reliability unverified for this harness.",
    },
    "qwen3.7-plus": {
        "model_id": "openrouter/qwen/qwen3.7-plus",  # $0.32 / $1.28
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "qwen",
        "notes": "Pilot before a full run (see qwen-3.7-max).",
    },
    # DeepSeek
    "deepseek-v4-pro": {
        "model_id": "openrouter/deepseek/deepseek-v4-pro",    # $0.43 / $0.87
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "deepseek",
        "notes": "DeepSeek-R1 emitted ~11% of tool calls as plain text in Stage 1 "
                 "(never reached the sandbox). Check the tool-call failure rate in a pilot.",
    },
    "deepseek-v4-flash": {
        "model_id": "openrouter/deepseek/deepseek-v4-flash",  # $0.10 / $0.20
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "deepseek",
        "notes": "Same tool-call risk as deepseek-v4-pro -- pilot first.",
    },
    # Kwaipilot
    "kwaipilot-kat-coder-pro-v2.5": {
        "model_id": "openrouter/kwaipilot/kat-coder-pro-v2.5",  # $0.74 / $2.96
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "kwaipilot",
    },
    # xAI
    "grok-4.5": {
        "model_id": "openrouter/x-ai/grok-4.5",   # $2 / $6
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "xai",
    },
    # MoonshotAI
    "kimi-k2.7-code": {
        "model_id": "openrouter/moonshotai/kimi-k2.7-code",   # $0.72 / $3.49
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "moonshotai",
    },
    "kimi-k3": {
        "model_id": "openrouter/moonshotai/kimi-k3",   # $3.00 / $15.00
        "judge": "openrouter/openai/gpt-4o",
        "include_text_editor": True,
        "family": "moonshotai",
    },
}
