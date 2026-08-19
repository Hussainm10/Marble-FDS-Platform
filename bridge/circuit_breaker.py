"""Marble FDS circuit breaker — the platform's compliance hard-lock policy §13 (M11).

When Marble is unavailable or returning errors, the bridge must NOT block
legitimate customer transactions. Per the document's fail-safe policy,
transactions continue under the base rule engine while Marble recovers.

Default config (the platform's compliance hard-lock policy §13.3):
    failure_threshold = 5       open circuit after 5 consecutive failures
    success_threshold = 2       close after 2 successes in half-open
    timeout_seconds   = 30      half-open probe window after this many seconds
    fallback_mode     = FAIL_SAFE_BASE_RULES

States:
    closed     — normal operation; calls pass through
    open       — Marble considered down; calls short-circuit to fail-safe
    half_open  — single trial call permitted to probe recovery

Prometheus exposure (registered in bridge/app.py):
    marble_fds_circuit_state              0=closed 1=open 2=half_open
    marble_fds_decide_errors_total        per error_type
    marble_fds_fallback_activations_total bumped on every fail-safe response
"""
from __future__ import annotations

import os
import threading
import time
from enum import IntEnum


class CircuitState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    """Thread-safe circuit breaker for the bridge → Marble call path.

    The bridge holds a single shared instance (see bridge/app.py). The
    breaker is process-local; in a multi-instance deployment each replica
    has its own state, which is acceptable per the platform's compliance hard-lock policy §13.1
    (FAIL-SAFE default — over-protection is fine, under-protection is not).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        open_timeout_seconds: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.open_timeout_seconds = open_timeout_seconds

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state

    def can_proceed(self) -> bool:
        """True if a call should be attempted; False if the circuit is
        fully open and not yet ready for a half-open probe."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_successes = 0
                    self._consecutive_failures = 0
                    self._opened_at = None
            else:
                # Closed: any success resets the failure counter.
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def _maybe_transition_to_half_open_locked(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (time.monotonic() - self._opened_at) >= self.open_timeout_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            self._consecutive_successes = 0


# Module-level singleton, configured from env. Imported by bridge/app.py.
marble_circuit = CircuitBreaker(
    failure_threshold=int(os.getenv("MARBLE_BREAKER_FAILURE_THRESHOLD", "5")),
    success_threshold=int(os.getenv("MARBLE_BREAKER_SUCCESS_THRESHOLD", "2")),
    open_timeout_seconds=float(os.getenv("MARBLE_BREAKER_OPEN_TIMEOUT_S", "30")),
)
