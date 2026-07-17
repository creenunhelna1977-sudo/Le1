"""rho AI layer — unified, typed, streaming-first LLM provider abstraction.

Usage:
    from rho.ai import Models
    from rho.ai.providers.openai import openai_provider
    from rho.ai.types import Context, UserMessage, DoneEvent, TextDelta

    models = Models()
    models.register(openai_provider())

    ctx = Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="Hello!")],
    )

    model = models.get_model("openai", "gpt-4o-mini")

    async for event in models.stream(model, ctx):
        match event:
            case TextDelta(delta=text):
                print(text, end="")
            case DoneEvent(message=msg):
                print(f"\\nDone. Usage: {msg.usage.total_tokens} tokens")
"""

from rho.ai.models import Models, ModelsError
from rho.ai.provider import Provider, create_provider

__all__ = [
    "Models",
    "ModelsError",
    "Provider",
    "create_provider",
]
