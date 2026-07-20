"""Shared helpers for the contract-faithful fake runtimes.

These fakes stand in for the real ``fable-session`` 0.3.0b1 and Hermes
Agent CLIs plus a generic worker/verifier pair. They accept the REAL
invocation shapes (and reject the invented flags older versions of this
repo used), produce the REAL external records the adapters consume
(run manifest + structured stream + audit JSON; ``state.db`` with
``sessions``/``messages``/``session_model_usage``), and are honest
about being fakes: manifests carry a fake tool name and verifier
reports carry ``"fake": true``.

Determinism: content, session IDs, and run IDs derive only from
arguments and environment (timestamps in the Hermes database use the
real clock because the adapter checks the invocation window).

Environment knobs (all optional):

- ``FAKE_SPAWN_MARKER``: file to append one tagged line to per
  invocation, so tests can prove which processes ran and how often.
- ``FAKE_EXIT``: force a nonzero exit before doing anything.
- ``FAKE_PROVIDER`` / ``FAKE_MODEL``: identity the fixture serves.
- ``FAKE_OUTCOME``: emulate a failed terminal state.
- ``FAKE_WORD_COUNT``: approximate body length.
"""

from __future__ import annotations

import hashlib
import os
import sys


def mark_spawn(tag: str) -> None:
    marker = os.environ.get("FAKE_SPAWN_MARKER")
    if marker:
        with open(marker, "a", encoding="utf-8") as handle:
            handle.write(tag + "\n")


def forced_exit() -> None:
    forced = os.environ.get("FAKE_EXIT")
    if forced:
        print(f"fake runtime forced exit {forced}", file=sys.stderr)
        raise SystemExit(int(forced))


def seed(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def body(seed_value: str, words: int | None = None) -> str:
    count = words if words is not None else int(os.environ.get("FAKE_WORD_COUNT", "40"))
    return " ".join(f"w{seed_value}{i}" for i in range(max(1, count)))


def parse_task_sections(text: str, path: str) -> None:
    """Enforce the real fable-session task format: exactly the required
    ``## Goal``/``## Checks``/``## Boundaries``/``## Report`` sections
    plus optional ``## Docs``; unknown sections are errors."""
    required = ("goal", "checks", "boundaries", "report")
    known = required + ("docs",)
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            name = line[3:].strip().lower()
            if name in sections:
                raise SystemExit(f"fake-fable-session: duplicate section '## {name}' in {path}")
            if name not in known:
                print(
                    f"fake-fable-session run: error: task file {path}: unknown "
                    f"section '## {name}' (known sections: {', '.join(known)})",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            current = sections.setdefault(name, [])
            continue
        if current is not None:
            current.append(line)
    for name in required:
        if name not in sections or not "".join(sections[name]).strip():
            print(
                f"fake-fable-session run: error: task file {path}: missing "
                f"required section '## {name}'",
                file=sys.stderr,
            )
            raise SystemExit(2)
