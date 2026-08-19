"""Checkmarble AST expression helpers.

Shared between the platform provisioner (``setup.py``) and jurisdiction
rule packs (``rules_library/*.py``).

Checkmarble represents rule conditions as a JSON AST — these helpers
build well-formed nodes so rule definitions stay readable:

    gte(payload("amount"), const(100000))
    eq(fds("velocity_spike_detected"), const(True))
    and_(eq(...), gt(...))
"""

from __future__ import annotations

from typing import Any


def payload(field_name: str) -> dict:
    return {"name": "Payload", "children": [{"constant": field_name}]}


def fds(field: str) -> dict:
    """Pre-computed FDS feature reference.

    Architectural note: Marble v0.59's ``DatabaseAccess`` only navigates
    child→parent (many→one). From a ``transactions`` trigger we can't
    DatabaseAccess down to ``fds_input_features`` (the child/"many" side),
    so the operator backend flattens fds fields into the transaction
    trigger payload before ``/decide`` and rules read them as plain Payload.
    See ``bridge/app.py /decide`` and ``spec.md`` for the full story.
    """
    return payload(field)


def const(value: Any) -> dict:
    return {"constant": value}


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

def gt(left: dict, right: dict) -> dict:
    return {"name": ">", "children": [left, right]}


def gte(left: dict, right: dict) -> dict:
    return {"name": ">=", "children": [left, right]}


def lt(left: dict, right: dict) -> dict:
    return {"name": "<", "children": [left, right]}


def lte(left: dict, right: dict) -> dict:
    return {"name": "<=", "children": [left, right]}


def eq(left: dict, right: dict) -> dict:
    return {"name": "=", "children": [left, right]}


def neq(left: dict, right: dict) -> dict:
    return {"name": "≠", "children": [left, right]}


# ---------------------------------------------------------------------------
# Boolean operators
# ---------------------------------------------------------------------------

def and_(*conditions: dict) -> dict:
    return {"name": "And", "children": list(conditions)}


def or_(*conditions: dict) -> dict:
    return {"name": "Or", "children": list(conditions)}


def not_(condition: dict) -> dict:
    return {"name": "Not", "children": [condition]}


# ---------------------------------------------------------------------------
# Null / empty checks
# ---------------------------------------------------------------------------

def is_empty(node: dict) -> dict:
    return {"name": "IsEmpty", "children": [node]}


def is_not_empty(node: dict) -> dict:
    return {"name": "IsNotEmpty", "children": [node]}


# ---------------------------------------------------------------------------
# String operations
# ---------------------------------------------------------------------------

def string_contains(haystack: dict, needle: dict) -> dict:
    return {"name": "StringContains", "children": [haystack, needle]}


# ---------------------------------------------------------------------------
# Arithmetic (Checkmarble AST supports +, -, *, /)
# ---------------------------------------------------------------------------

def add(a: dict, b: dict) -> dict:
    return {"name": "+", "children": [a, b]}


def sub(a: dict, b: dict) -> dict:
    return {"name": "-", "children": [a, b]}


def mul(a: dict, b: dict) -> dict:
    return {"name": "*", "children": [a, b]}


def div(a: dict, b: dict) -> dict:
    return {"name": "/", "children": [a, b]}


__all__ = [
    "payload", "fds", "const",
    "gt", "gte", "lt", "lte", "eq", "neq",
    "and_", "or_", "not_",
    "is_empty", "is_not_empty",
    "string_contains",
    "add", "sub", "mul", "div",
]
