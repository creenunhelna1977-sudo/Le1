"""Models registry for rho AI layer.

Matches PI's Models collection — registers providers, resolves auth,
and delegates streaming to the owning provider.

The central hub that consumers interact with.
"""

from __future__ import annotations

from typing import AsyncGenerator

from rho.ai.auth import DefaultAuthContext, ProviderAuth, resolve_provider_auth
from rho.ai.provider import Provider
from rho.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    StreamOptions,
)


class ModelsError(Exception):
    """Error from the Models registry (auth, provider lookup, etc.)."""
    pass


class Models:
    """Registry of AI providers with auth resolution.

    Usage:
        models = Models()
        models.register(openai_provider)
        models.register(deepseek_provider)

        # Stream
        async for event in models.stream(model, ctx, opts):
            match event:
                case TextDelta(delta=text): print(text, end="")
                case Done(message=msg): return msg

        # Complete (non-streaming)
        msg = await models.complete(model, ctx, opts)
    """

    def __init__(self):
        self._providers: dict[str, Provider] = {}
        self._auth_ctx = DefaultAuthContext()

    # ── Provider management ──────────────────────────────────────

    def register(self, provider: Provider) -> None:
        """Register a provider."""
        self._providers[provider.id] = provider

    def unregister(self, provider_id: str) -> None:
        """Remove a provider."""
        self._providers.pop(provider_id, None)

    @property
    def providers(self) -> list[Provider]:
        """All registered providers."""
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> Provider | None:
        """Look up a provider by id."""
        return self._providers.get(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        """Look up a model by provider and model id."""
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        for model in provider.get_models():
            if model.id == model_id:
                return model
        return None

    def get_all_models(self) -> list[Model]:
        """All models from all registered providers."""
        models: list[Model] = []
        for provider in self._providers.values():
            models.extend(provider.get_models())
        return models

    # ── Auth ─────────────────────────────────────────────────────

    async def resolve_auth(self, model: Model, options: StreamOptions | None = None) -> StreamOptions:
        """Resolve auth for a model and return enriched StreamOptions.

        Merges: provider auth → model headers → explicit options.
        """
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError(f"Unknown provider: {model.provider}")

        opts = options or StreamOptions()

        # If API key already provided explicitly, use it
        if opts.api_key:
            return opts

        # Resolve from provider auth
        result = await resolve_provider_auth(provider.auth, self._auth_ctx)
        if result is None:
            raise ModelsError(f"Provider is not configured: {model.provider}")

        # Merge headers: provider → model → options (last wins)
        merged_headers: dict[str, str | None] = {}
        if result.auth.headers:
            merged_headers.update(result.auth.headers)
        if model.headers:
            merged_headers.update(model.headers)
        if opts.headers:
            merged_headers.update(opts.headers)

        return StreamOptions(
            api_key=result.auth.api_key,
            headers=merged_headers,
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            timeout_ms=opts.timeout_ms,
            max_retries=opts.max_retries,
            max_retry_delay_ms=opts.max_retry_delay_ms,
            cache_retention=opts.cache_retention,
            session_id=opts.session_id,
            metadata=opts.metadata,
            env={**result.env, **opts.env},
        )

    # ── Streaming ────────────────────────────────────────────────

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        """Stream a response from the provider that owns the model.

        Resolves auth, then delegates to the provider.
        """
        provider = self._get_provider_for(model)
        resolved_opts = await self.resolve_auth(model, options)

        async for event in provider.stream(model, context, resolved_opts):
            yield event

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        """Collect a stream into a single AssistantMessage (non-streaming)."""
        last_message: AssistantMessage | None = None

        async for event in self.stream(model, context, options):
            if isinstance(event, DoneEvent):
                last_message = event.message
            elif isinstance(event, ErrorEvent):
                return event.error

        if last_message is None:
            raise ModelsError("Stream ended without a done event")

        return last_message

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        """Stream with simplified options."""
        provider = self._get_provider_for(model)
        resolved_opts = await self.resolve_auth(model, options)

        async for event in provider.stream_simple(model, context, resolved_opts):
            yield event

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        """Non-streaming with simplified options."""
        last_message: AssistantMessage | None = None

        async for event in self.stream_simple(model, context, options):
            if isinstance(event, DoneEvent):
                last_message = event.message
            elif isinstance(event, ErrorEvent):
                return event.error

        if last_message is None:
            raise ModelsError("Stream ended without a done event")

        return last_message

    def _get_provider_for(self, model: Model) -> Provider:
        """Get the provider that owns a model, or raise."""
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError(f"Unknown provider: {model.provider}")
        return provider
