"""DeepSeek provider — models and factory.

DeepSeek uses an OpenAI-compatible API, so we reuse openai_completions.
The compat settings handle DeepSeek-specific quirks:
- thinking_format: "deepseek" (uses {thinking: {type: "enabled"/"disabled"}})
- max_tokens field: "max_tokens" (not max_completion_tokens)
- requires reasoning_content on assistant messages for continuity
"""

from rho.ai.auth import env_api_key_auth, ProviderAuth
from rho.ai.provider import create_provider
from rho.ai.types import Model, ModelCostRates
from rho.ai.api.openai_completions import stream, stream_simple

# ═══════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════

# Pricing: $/million tokens (DeepSeek official pricing as of 2026)
_COST_V3 = ModelCostRates(input=0.27, output=1.10, cache_read=0.07, cache_write=0.27)
_COST_R1 = ModelCostRates(input=0.55, output=2.19, cache_read=0.14, cache_write=0.55)
_COST_V3_1 = ModelCostRates(input=0.27, output=1.10, cache_read=0.07, cache_write=0.27)

DEEPSEEK_COMPAT = {
    "thinking_format": "deepseek",
    "requires_reasoning_content_on_assistant_messages": True,
    "max_tokens_field": "max_tokens",
    "supports_developer_role": False,
    "supports_store": False,
    "supports_reasoning_effort": False,
}

DEEPSEEK_CHAT = Model(
    id="deepseek-chat",
    name="DeepSeek-V3.1",
    api="openai-completions",
    provider="deepseek",
    base_url="https://api.deepseek.com",
    reasoning=False,
    input=["text"],
    cost=_COST_V3_1,
    context_window=128_000,
    max_tokens=8_192,
    compat=DEEPSEEK_COMPAT,
)

DEEPSEEK_REASONER = Model(
    id="deepseek-reasoner",
    name="DeepSeek-R1",
    api="openai-completions",
    provider="deepseek",
    base_url="https://api.deepseek.com",
    reasoning=True,
    input=["text"],
    cost=_COST_R1,
    context_window=128_000,
    max_tokens=8_192,
    thinking_level_map={
        "off": None,
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    },
    compat=DEEPSEEK_COMPAT,
)

DEEPSEEK_MODELS = [DEEPSEEK_CHAT, DEEPSEEK_REASONER]


# ═══════════════════════════════════════════════════════════════════
# Provider factory
# ═══════════════════════════════════════════════════════════════════

def deepseek_provider() -> create_provider:
    """Create the DeepSeek provider."""
    return create_provider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        auth=ProviderAuth(
            api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"]),
        ),
        models=DEEPSEEK_MODELS,
        api=stream,
        stream_simple=stream_simple,
    )
