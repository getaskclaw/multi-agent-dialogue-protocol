"""Adapter boundary: how one worker turn is prepared, run, and observed.

An adapter turns an actor's declared transport plus non-secret settings
into a :class:`CommandPacket` (the exact real-CLI argv/env for one
turn, plus its declared lifecycle follow-ups) and, on explicit launch,
executes that one turn and derives the runtime-evidence record from
EXTERNAL records — session manifests, audit JSON, state databases, or
an explicit verifier command — never from worker-authored files.
Adapters never run by themselves, never loop, and never decide protocol
state. Selection is by the actor's declared ``transport`` field only;
role text is never consulted.
"""

from .base import (
    Adapter,
    AdapterError,
    CleanupUnprovenError,
    CommandPacket,
    PrepareContext,
    adapter_for,
    get_adapter,
)

__all__ = [
    "Adapter",
    "AdapterError",
    "CleanupUnprovenError",
    "CommandPacket",
    "PrepareContext",
    "adapter_for",
    "get_adapter",
]
