"""Canonical error model for the Marble FDS bridge service.

Per ``spec.md §7`` (platform error-model spec), all write endpoints return errors in a consistent
``{code, message, details}`` JSON shape with a small set of canonical codes:

    AGT-401  permission denied
    AGT-402  geo-scope violation
    AGT-429  velocity / rate limit exceeded
    FDS-451  manual review required (per ISO 6265, a "legally restricted"
             status — the transaction can't be auto-processed and must be
             reviewed by a compliance officer before completion)
    DOC-415  invalid document / payload schema

Other failures (unknown, transient infrastructure issues) fall through to
a generic ``FDS-500`` internal error response — still in the same shape
so clients have exactly one error-handling code path.

Why this matters:
    - Mobile apps + partner integrations can key off ``code`` to show
      user-facing messages or trigger retries selectively.
    - Regulators expect consistent error telemetry for audit reviews.
    - Prevents leaking stack traces or framework noise into responses.

Usage in route handlers::

    from .errors import GeoScopeViolation

    @app.post("/decide")
    async def decide(...):
        if not agent_in_zone(...):
            raise GeoScopeViolation(
                message="Agent is outside permitted GPS zone",
                details={"zone": "north_kabul", "gps": "34.1,69.2"},
            )

The FastAPI app then renders this as::

    HTTP 403
    {
      "code": "AGT-402",
      "message": "Agent is outside permitted GPS zone",
      "details": {"zone": "north_kabul", "gps": "34.1,69.2"}
    }
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class FDSError(Exception):
    """Base class for all canonical Marble FDS errors.

    Subclasses set ``code`` (e.g. ``AGT-402``) and ``http_status`` (e.g. 403).
    ``details`` is an arbitrary dict merged into the JSON response for
    structured client-side handling (e.g. ``{"zone": "north_kabul"}``).
    """

    code: str = "FDS-500"
    http_status: int = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Canonical codes (GNN PDF §7)
# ---------------------------------------------------------------------------

class PermissionDenied(FDSError):
    """AGT-401 — the caller isn't authorized for this resource/action."""
    code = "AGT-401"
    http_status = 403


class GeoScopeViolation(FDSError):
    """AGT-402 — request originates from outside the agent's permitted zone."""
    code = "AGT-402"
    http_status = 403


class VelocityLimitExceeded(FDSError):
    """AGT-429 — too many requests / velocity limit exceeded."""
    code = "AGT-429"
    http_status = 429


class ManualReviewRequired(FDSError):
    """FDS-451 — can't be auto-processed; compliance officer must review.

    Typically raised when the risk score plus context crosses a threshold
    that, per the operator's jurisdiction pack, mandates human review
    before the transaction proceeds. The response is NOT a failure —
    it's a correct outcome that happens to require human action.
    """
    code = "FDS-451"
    http_status = 451  # "Unavailable For Legal Reasons"


class InvalidDocument(FDSError):
    """DOC-415 — request body/document failed schema validation or was malformed."""
    code = "DOC-415"
    http_status = 415


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------

async def fds_error_handler(request: Request, exc: FDSError) -> JSONResponse:
    """Render an FDSError as canonical JSON."""
    return JSONResponse(status_code=exc.http_status, content=exc.to_payload())


async def validation_error_handler(
    request: Request, exc: ValidationError,
) -> JSONResponse:
    """Turn pydantic ValidationError into a canonical DOC-415 response.

    Pydantic's default error shape is fine for development but exposes
    internals (`type`, `loc`, `ctx`, etc.) that partner integrations
    shouldn't couple to. We flatten it into our canonical shape.
    """
    err = InvalidDocument(
        message="Request payload failed schema validation",
        details={
            "errors": [
                {
                    "field": ".".join(str(p) for p in e.get("loc", [])),
                    "type": e.get("type", ""),
                    "msg": e.get("msg", ""),
                }
                for e in exc.errors()
            ],
        },
    )
    return JSONResponse(status_code=err.http_status, content=err.to_payload())


async def generic_error_handler(
    request: Request, exc: Exception,
) -> JSONResponse:
    """Last-resort: any unhandled exception becomes a canonical FDS-500.

    In production, full stack traces should go to logs (via structlog —
    see Phase 7) but NEVER to the response body.
    """
    err = FDSError(
        message="Internal server error",
        details={"type": type(exc).__name__},
    )
    return JSONResponse(status_code=err.http_status, content=err.to_payload())


def register_exception_handlers(app) -> None:
    """Wire the canonical error handlers into a FastAPI app instance.

    Call from ``bridge/app.py`` after the app is constructed. Order
    matters: more specific exceptions first.
    """
    # Match our canonical errors first (they all derive from FDSError)
    app.add_exception_handler(FDSError, fds_error_handler)
    # Pydantic validation errors → DOC-415
    app.add_exception_handler(ValidationError, validation_error_handler)
    # Fallback for anything unexpected → FDS-500
    # (commented-out: enabling this swallows framework errors that FastAPI's
    # default handlers display helpfully in dev. Enable in production via
    # env flag.)
    # app.add_exception_handler(Exception, generic_error_handler)


__all__ = [
    "FDSError",
    "PermissionDenied",
    "GeoScopeViolation",
    "VelocityLimitExceeded",
    "ManualReviewRequired",
    "InvalidDocument",
    "fds_error_handler",
    "validation_error_handler",
    "generic_error_handler",
    "register_exception_handlers",
]
