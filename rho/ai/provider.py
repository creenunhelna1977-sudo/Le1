"""Provider abstraction for rho AI layer.

Matches PI's Provider<TApi> interface — a provider owns its id, name,
base URL, auth methods, model catalog, and API implementations.

Uses async generators instead of PI's EventStream class.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable

from rho.ai.auth import ProviderAuth
from rho.ai.types import (
    AssistantMessageEvent,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)


# Type alias for a stream function
StreamFunc = Callable[
    [Model, Context, StreamOptions | None],
    AsyncGenerator[AssistantMessageEvent, None],
]

SimpleStreamFunc = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AsyncGenerator[AssistantMessageEvent, None],
]


class Provider:
    """A runtime provider that owns models and streams.

    Each provider:
    - Has an identity (id, name, base_url)
    - Has auth methods (api_key, future: oauth)
    - Knows its models
    - Can stream responses via an API implementation
    """

    def __init__(
        self,
        id: str,
        name: str,
        base_url: str,
        auth: ProviderAuth,
        models: list[Model],
        api: StreamFunc,
        stream_simple_func: SimpleStreamFunc | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.id = id
        self.name = name
        self.base_url = base_url
        self.auth = auth
        self._models = list(models)
        self._stream_impl = api
        self._stream_simple_impl = stream_simple_func or api
        self.headers = headers or {}

    def get_models(self) -> list[Model]:
        """Return the current model catalog."""
        return list(self._models)

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        """Stream a response from the provider.

        Delegates to the API implementation.
        """
        async for event in self._stream_impl(model, context, options):
            yield event

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        """Stream with simplified options (reasoning level)."""
        async for event in self._stream_simple_impl(model, context, options):
            yield event


def create_provider(
    id: str,
    name: str,
    base_url: str,
    auth: ProviderAuth,
    models: list[Model],
    api: StreamFunc,
    stream_simple: SimpleStreamFunc | None = None,
    headers: dict[str, str] | None = None,
) -> Provider:
    """Create a Provider from parts.

    Matches PI's createProvider() factory.

    Example:
        provider = create_provider(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            auth=ProviderAuth(api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"])),
            models=OPENAI_MODELS,
            api=stream,
            stream_simple=stream_simple,
        )
    """
    return Provider(
        id=id,
        name=name,
        base_url=base_url,
        auth=auth,
        models=models,
        api=api,
        stream_simple_func=stream_simple,
        headers=headers,
    )
