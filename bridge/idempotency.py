"""Idempotency support for Marble FDS write endpoints.

Per ``spec.md §7`` (platform idempotency spec):

    All write APIs accept idempotency_key.

Why this matters (regulatory angle):
    Without idempotency, a retrying client can trigger duplicate STR drafts,
    duplicate audit entries, and duplicate compliance-queue items.
    Regulators treat duplicate STRs as a serious data-integrity issue in
    audits. This module ensures retries with the same key return the same
    response without re-executing the work.

How it works:
    1. Client sends ``Idempotency-Key: <uuid>`` header on a write request.
    2. First call with that key:
       - Compute a lookup key = hash(method + path + key + body).
       - Do the work, cache the response, return it.
    3. Subsequent call with the same key + same body → return cached response.
    4. Subsequent call with same key but DIFFERENT body → reject with
       ``DOC-415`` (client bug — reusing keys across different payloads).
    5. Keys expire after ``DEFAULT_TTL_SECONDS`` (24h).

Storage:
    ``InMemoryIdempotencyStore`` — fine for single-instance deployments.
    Protocol is ``IdempotencyStore`` so production can swap in Redis or
    Postgres without touching the middleware. For multi-instance /
    production-grade deployments, consider a Redis-backed implementation
    (see ``docs/deployment/redis-idempotency.md`` — future).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from .errors import InvalidDocument


DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours
DEFAULT_MAX_ENTRIES = 10_000         # LRU eviction cap (in-memory store only)


# ---------------------------------------------------------------------------
# Protocol + in-memory implementation
# ---------------------------------------------------------------------------

@dataclass
class CachedResponse:
    """A response previously computed for a given idempotency key."""
    status_code: int
    body: dict[str, Any]
    body_hash: str       # hash of the ORIGINAL request body; for mismatch detection
    expires_at: float    # unix epoch


class IdempotencyStore(Protocol):
    """Storage backend for idempotency records.

    Implementations must be safe for concurrent access from async handlers.
    """

    async def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for ``key``, or None if missing/expired."""
        ...

    async def put(self, key: str, value: CachedResponse) -> None:
        """Store a cached response for ``key``."""
        ...


class InMemoryIdempotencyStore:
    """TTL + LRU-bounded cache. Fine for single-instance deployments.

    Uses ``OrderedDict`` so eviction order tracks insertion/access order.
    Not suitable for multi-instance — use Redis or Postgres in production.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        from collections import OrderedDict
        self._ttl = ttl_seconds
        self._max = max_entries
        self._data: "OrderedDict[str, CachedResponse]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CachedResponse | None:
        async with self._lock:
            v = self._data.get(key)
            if v is None:
                return None
            if v.expires_at < time.time():
                del self._data[key]
                return None
            self._data.move_to_end(key)  # mark as recently used
            return v

    async def put(self, key: str, value: CachedResponse) -> None:
        async with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            # Sweep expired entries periodically (cheap amortized).
            if len(self._data) % 64 == 0:
                now = time.time()
                expired = [k for k, v in self._data.items() if v.expires_at < now]
                for k in expired:
                    del self._data[k]
            # Enforce LRU cap.
            while len(self._data) > self._max:
                self._data.popitem(last=False)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _hash_body(body: bytes) -> str:
    """SHA-256 of the raw request body — used for body-mismatch detection."""
    return hashlib.sha256(body).hexdigest()


def _cache_key(method: str, path: str, idempotency_key: str) -> str:
    """Build the cache lookup key. Note: body_hash is checked SEPARATELY
    (stored in CachedResponse) so we can return DOC-415 on mismatch.
    """
    raw = f"{method}:{path}:{idempotency_key}".encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Per-key locks — prevent concurrent duplicate work
# ---------------------------------------------------------------------------

class _PerKeyLocks:
    """Give each idempotency key its own asyncio.Lock so two concurrent
    requests with the same key serialize without blocking unrelated keys.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()

    async def get(self, key: str) -> asyncio.Lock:
        async with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def release(self, key: str) -> None:
        """Clean up a lock once no-one is waiting on it."""
        async with self._meta_lock:
            lock = self._locks.get(key)
            if lock and not lock.locked():
                self._locks.pop(key, None)


# ---------------------------------------------------------------------------
# Middleware entry point
# ---------------------------------------------------------------------------

class IdempotencyMiddleware:
    """Wraps an async handler with idempotency-key semantics.

    Usage from ``bridge/app.py``::

        idem = IdempotencyMiddleware(InMemoryIdempotencyStore())

        @app.post("/decide")
        async def decide(request: Request, body: DecideRequest):
            async def do_work():
                # existing handler body
                return response_dict
            return await idem.process(request, do_work)

    If no ``Idempotency-Key`` header is present the handler runs normally
    (idempotency is opt-in per request, per spec). Clients that care about
    retry safety MUST send the header; clients that don't accept the risk
    of duplicate side effects.
    """

    HEADER_NAME = "Idempotency-Key"

    def __init__(
        self,
        store: IdempotencyStore | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self._store = store or InMemoryIdempotencyStore(ttl_seconds)
        self._ttl = ttl_seconds
        self._locks = _PerKeyLocks()

    async def process(
        self,
        request,  # fastapi.Request
        do_work: Callable[[], Awaitable[dict[str, Any]]],
        status_code_on_replay: int = 200,
    ) -> dict[str, Any]:
        """Run ``do_work()`` under idempotency semantics.

        Returns the response dict (either fresh from do_work() or cached).

        Raises:
            InvalidDocument (DOC-415): client reused an Idempotency-Key with
                a different body (integration bug — see module docstring).
        """
        idem_key = request.headers.get(self.HEADER_NAME)
        if not idem_key:
            # No idempotency requested → run normally.
            return await do_work()

        body = await request.body()
        body_hash = _hash_body(body)
        cache_key = _cache_key(request.method, request.url.path, idem_key)

        # Fast path: check cache without locking.
        cached = await self._store.get(cache_key)
        if cached is not None:
            if cached.body_hash != body_hash:
                raise InvalidDocument(
                    message="Idempotency-Key reused with a different request body",
                    details={
                        "idempotency_key": idem_key,
                        "hint": "Generate a fresh Idempotency-Key for each distinct request",
                    },
                )
            return cached.body

        key_lock = await self._locks.get(cache_key)
        try:
            async with key_lock:
                # Re-check after acquiring the lock — another coroutine may
                # have populated the cache while we were waiting.
                cached = await self._store.get(cache_key)
                if cached is not None:
                    if cached.body_hash != body_hash:
                        raise InvalidDocument(
                            message="Idempotency-Key reused with a different request body",
                            details={"idempotency_key": idem_key},
                        )
                    return cached.body

                result = await do_work()

                await self._store.put(
                    cache_key,
                    CachedResponse(
                        status_code=status_code_on_replay,
                        body=result,
                        body_hash=body_hash,
                        expires_at=time.time() + self._ttl,
                    ),
                )
                return result
        finally:
            # Always release — keeps _PerKeyLocks bounded even on exception paths.
            await self._locks.release(cache_key)


__all__ = [
    "CachedResponse",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "IdempotencyMiddleware",
    "DEFAULT_TTL_SECONDS",
]
