"""Authentication layer for rho AI providers.

Matches PI's auth architecture:
- ApiKeyAuth: resolve from stored credential or env vars
- OAuth slot reserved for future
- AuthContext: injectable env var access for testing

Single file for v1; will extract to ai/auth/ directory when OAuth arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelAuth:
    """Resolved per-request auth: api_key, headers, optional base_url override."""
    api_key: str | None = None
    headers: dict[str, str | None] = field(default_factory=dict)
    base_url: str | None = None


@dataclass
class ApiKeyCredential:
    """Stored api-key credential."""
    type: Literal["api_key"] = "api_key"
    key: str | None = None
    env: dict[str, str] = field(default_factory=dict)


Credential = ApiKeyCredential  # will become: ApiKeyCredential | OAuthCredential


@dataclass
class AuthResult:
    """Result of resolving provider auth."""
    auth: ModelAuth
    env: dict[str, str] = field(default_factory=dict)
    source: str = ""


@dataclass
class AuthCheck:
    """Side-effect-free auth availability check."""
    source: str = ""
    type: Literal["api_key"] = "api_key"


# ═══════════════════════════════════════════════════════════════════
# AuthContext — injectable env var access for testing
# ═══════════════════════════════════════════════════════════════════

class AuthContext(Protocol):
    """Injectable environment access for auth resolution."""

    def env(self, name: str) -> str | None: ...


class DefaultAuthContext:
    """Default AuthContext reading from os.environ."""

    def env(self, name: str) -> str | None:
        import os
        return os.environ.get(name)


# ═══════════════════════════════════════════════════════════════════
# ApiKeyAuth
# ═══════════════════════════════════════════════════════════════════

class ApiKeyAuth:
    """Standard API key authentication.

    Resolution order: stored credential key → env vars.
    Includes an optional login() for interactive key entry.
    """

    def __init__(self, name: str, env_vars: list[str]):
        self.name = name
        self.env_vars = env_vars

    async def resolve(
        self,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
    ) -> AuthResult | None:
        """Resolve auth from stored credential or env vars.

        Returns None if the provider is not configured.
        """
        if credential and credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=credential.key),
                env=credential.env,
                source="stored credential",
            )
        for var in self.env_vars:
            value = ctx.env(var)
            if value:
                return AuthResult(
                    auth=ModelAuth(api_key=value),
                    source=var,
                )
        return None

    async def check(
        self,
        ctx: AuthContext,
        credential: ApiKeyCredential | None = None,
    ) -> AuthCheck | None:
        """Side-effect-free availability check."""
        result = await self.resolve(ctx, credential)
        if result is None:
            return None
        return AuthCheck(source=result.source, type="api_key")


# ═══════════════════════════════════════════════════════════════════
# ProviderAuth
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProviderAuth:
    """Auth configuration for a provider.

    At least one of api_key or oauth must be present.
    oauth slot reserved for future use.
    """
    api_key: ApiKeyAuth | None = None
    oauth: None = None  # future: OAuthAuth


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def env_api_key_auth(name: str, env_vars: list[str]) -> ApiKeyAuth:
    """Create a standard env-var-based API key auth.

    Matches PI's envApiKeyAuth().

    Example:
        auth = env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"])
    """
    return ApiKeyAuth(name=name, env_vars=env_vars)


# ═══════════════════════════════════════════════════════════════════
# Resolve
# ═══════════════════════════════════════════════════════════════════

async def resolve_provider_auth(
    provider_auth: ProviderAuth,
    ctx: AuthContext,
    credential: Credential | None = None,
) -> AuthResult | None:
    """Resolve auth for a provider.

    Returns None if the provider is not configured.
    """
    if provider_auth.api_key is None:
        return None
    return await provider_auth.api_key.resolve(ctx, credential)
