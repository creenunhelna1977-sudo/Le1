"""OpenAI provider — models and factory."""

from rho.ai.auth import env_api_key_auth, ProviderAuth
from rho.ai.provider import create_provider
from rho.ai.types import Model, ModelCostRates
from rho.ai.api.openai_completions import stream, stream_simple

# ═══════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════

# Pricing: $/million tokens
_COST_GPT4O = ModelCostRates(input=2.50, output=10.00, cache_read=1.25, cache_write=0.0)
_COST_GPT4O_MINI = ModelCostRates(input=0.15, output=0.60, cache_read=0.075, cache_write=0.0)
_COST_O3_MINI = ModelCostRates(input=1.10, output=4.40, cache_read=0.55, cache_write=0.0)
_COST_O4_MINI = ModelCostRates(input=1.10, output=4.40, cache_read=0.55, cache_write=0.0)
_COST_GPT5 = ModelCostRates(input=2.50, output=15.00, cache_read=1.25, cache_write=0.0)

GPT_4O = Model(
    id="gpt-4o",
    name="GPT-4o",
    api="openai-completions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text", "image"],
    cost=_COST_GPT4O,
    context_window=128_000,
    max_tokens=16_384,
)

GPT_4O_MINI = Model(
    id="gpt-4o-mini",
    name="GPT-4o Mini",
    api="openai-completions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=False,
    input=["text", "image"],
    cost=_COST_GPT4O_MINI,
    context_window=128_000,
    max_tokens=16_384,
)

O3_MINI = Model(
    id="o3-mini",
    name="o3-mini",
    api="openai-completions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=_COST_O3_MINI,
    context_window=200_000,
    max_tokens=100_000,
    thinking_level_map={
        "off": None,
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    },
)

O4_MINI = Model(
    id="o4-mini",
    name="o4-mini",
    api="openai-completions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=_COST_O4_MINI,
    context_window=200_000,
    max_tokens=100_000,
    thinking_level_map={
        "off": None,
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    },
)

OPENAI_MODELS = [GPT_4O, GPT_4O_MINI, O3_MINI, O4_MINI]


# ═══════════════════════════════════════════════════════════════════
# Provider factory
# ═══════════════════════════════════════════════════════════════════

def openai_provider() -> create_provider:
    """Create the OpenAI provider."""
    return create_provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"]),
        ),
        models=OPENAI_MODELS,
        api=stream,
        stream_simple=stream_simple,
    )
