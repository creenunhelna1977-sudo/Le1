"""Shared HTTP client for rho AI providers.

Wraps httpx.AsyncClient for:
- Connection pooling and timeout management
- SSE (Server-Sent Events) line streaming
- Provider error normalization

No provider SDKs — raw HTTP with full control.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

import httpx


class ProviderError(Exception):
    """Normalized provider HTTP error."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: Any = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body


def normalize_provider_error(error: Exception) -> ProviderError:
    """Probe known HTTP error shapes and normalize.

    Handles httpx.HTTPStatusError (has .response) and httpx.RequestError.
    """
    if isinstance(error, ProviderError):
        return error

    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        status = response.status_code
        try:
            body = response.json()
        except Exception:
            try:
                body = response.text
            except Exception:
                body = None
        return ProviderError(
            message=f"{status}: {_format_body(body)}",
            status=status,
            body=body,
        )

    if isinstance(error, httpx.RequestError):
        return ProviderError(message=str(error))

    # Generic fallback
    return ProviderError(message=str(error))


def _format_body(body: Any) -> str:
    """Format an error body for display."""
    if isinstance(body, dict):
        if "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                return err.get("message", json.dumps(err))
            return str(err)
        return json.dumps(body)
    if isinstance(body, str):
        # Truncate long error bodies
        return body[:500] if len(body) > 500 else body
    return str(body)[:500]


class HttpClient:
    """Shared HTTP client for provider API calls.

    Usage:
        async with HttpClient() as client:
            async for event in client.stream_sse(url, json=payload, headers={...}):
                ...
    """

    def __init__(
        self,
        timeout: float = 120.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def request(
        self,
        method: str,
        url: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request.

        Raises ProviderError on HTTP errors.
        """
        try:
            response = await self._client.request(
                method=method,
                url=url,
                json=json_data,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise normalize_provider_error(e) from e

    async def stream_sse(
        self,
        url: str,
        *,
        json_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        method: str = "POST",
        timeout_ms: int | None = None,
        max_retries: int = 0,
        max_retry_delay_ms: int = 60_000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a Server-Sent Events response.

        Yields parsed JSON objects from `data:` lines.
        Handles both standard SSE (event:/data: fields) and
        OpenAI-style (bare data: JSON) formats.

        Args:
            url: Full endpoint URL.
            json_data: Request body as JSON.
            headers: Request headers.
            method: HTTP method (default POST).
        """
        merged_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(headers or {}),
        }

        attempts = 0
        while True:
            yielded = False
            try:
                request_timeout = (
                    httpx.Timeout(timeout_ms / 1000, read=timeout_ms / 1000)
                    if timeout_ms is not None
                    else httpx.Timeout(300.0, read=300.0)
                )
                async with self._client.stream(
                    method=method,
                    url=url,
                    json=json_data,
                    headers=merged_headers,
                    timeout=request_timeout,
                ) as response:
                    if response.is_error:
                        # Streaming responses must be read before their body is inspected.
                        await response.aread()
                        response.raise_for_status()

                    event_type: str | None = None
                    data_lines: list[str] = []

                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].lstrip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                        elif line == "":
                            if data_lines:
                                yielded = True
                                yield self._parse_sse_event(
                                    "\n".join(data_lines), event_type
                                )
                                data_lines = []
                                event_type = None

                    if data_lines:
                        yielded = True
                        yield self._parse_sse_event("\n".join(data_lines), event_type)
                return
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                error = normalize_provider_error(exc)
                retryable = (
                    error.status in {408, 409, 429, 500, 502, 503, 504}
                    or isinstance(exc, httpx.RequestError)
                )
                if not retryable or yielded or attempts >= max_retries:
                    raise error from exc

                delay_ms = 250 * (2 ** attempts)
                if max_retry_delay_ms > 0:
                    delay_ms = min(delay_ms, max_retry_delay_ms)
                attempts += 1
                await asyncio.sleep(delay_ms / 1000)

    @staticmethod
    def _parse_sse_event(data: str, event_type: str | None) -> dict[str, Any]:
        """Parse a single SSE data payload as JSON."""
        if data.strip() == "[DONE]":
            return {"_done": True}
        try:
            parsed = json.loads(data)
            if event_type:
                parsed["_sse_event"] = event_type
            return parsed
        except json.JSONDecodeError:
            return {"_raw": data, "_sse_event": event_type}
