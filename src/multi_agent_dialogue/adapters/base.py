"""Adapter interface and registry.

An adapter owns one worker turn against a real external CLI contract:

- :meth:`Adapter.prepare` builds the non-secret :class:`CommandPacket`
  (primary argv plus any declared lifecycle follow-ups) without starting
  a process — this is what ``madp run --dry-run`` prints;
- :meth:`Adapter.execute` runs the transport's real lifecycle for
  exactly one turn, writes the turn artifact (where the transport
  produces the turn text itself), and returns a normalized
  runtime-evidence record derived from EXTERNAL records (session
  manifests, audit JSON, state databases, or an explicit verifier
  command) — never from worker-authored evidence files.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import Actor, TurnSpec

EVIDENCE_VERSION = 1


class AdapterError(RuntimeError):
    """Adapter settings are invalid or a turn cannot be prepared/proven."""


class CleanupUnprovenError(AdapterError):
    """A turn failed after launch and the adapter could not PROVE the
    external worker lane stopped. The claim must not be released: a
    retry could start a second worker beside the surviving one, so the
    runner locks the dialogue instead."""


@dataclass(frozen=True)
class PrepareContext:
    actor: Actor
    turn: TurnSpec
    dialogue_dir: Path
    work_dir: Path
    task_file: Path
    turn_file: Path
    evidence_file: Path
    substitution_reason: str | None = None

    def placeholders(self) -> dict[str, str]:
        return {
            "task_file": str(self.task_file),
            "turn_file": str(self.turn_file),
            "evidence_file": str(self.evidence_file),
            "dialogue_dir": str(self.dialogue_dir),
            "work_dir": str(self.work_dir),
            "round_id": self.turn.round_id,
            "actor_id": self.actor.actor_id,
        }


@dataclass(frozen=True)
class CommandPacket:
    adapter: str
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 120
    description: str = ""
    # Additional real-CLI steps this adapter will run around the primary
    # argv (e.g. fable-session's --dry-run preflight, the single
    # watch --follow, and the JSON audit). Purely descriptive at
    # prepare time; execute() runs them.
    lifecycle: tuple[dict, ...] = ()

    def as_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "argv": list(self.argv),
            "env": dict(self.env),
            "timeout_seconds": self.timeout_seconds,
            "description": self.description,
            "lifecycle": [
                {"purpose": step["purpose"], "argv": list(step["argv"])}
                for step in self.lifecycle
            ],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def substitute(template: str, placeholders: dict[str, str], where: str) -> str:
    try:
        return template.format_map(placeholders)
    except (KeyError, IndexError, ValueError) as exc:
        raise AdapterError(
            f"{where}: unknown or malformed placeholder in {template!r} "
            f"(allowed: {sorted(placeholders)})"
        ) from exc


def substitute_env(env_setting: object, placeholders: dict[str, str], where: str) -> dict[str, str]:
    if env_setting is None:
        return {}
    if not isinstance(env_setting, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env_setting.items()
    ):
        raise AdapterError(f"{where}: env must map string names to string values")
    return {
        key: substitute(value, placeholders, where)
        for key, value in env_setting.items()
    }


def read_timeout(settings: dict, where: str) -> int:
    timeout = settings.get("timeout_seconds", 120)
    if not isinstance(timeout, int) or timeout <= 0:
        raise AdapterError(f"{where}: timeout_seconds must be a positive integer")
    return timeout


def require_str_setting(settings: dict, key: str, where: str, default: str | None = None) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(
            f"{where}: settings.{key} is required and must be a non-empty "
            "string; it is never inferred from the actor's role name"
        )
    return value


def run_command(
    argv: list[str] | tuple[str, ...],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: int,
    what: str,
) -> subprocess.CompletedProcess:
    """Run one external command; OS/timeout failures become AdapterError."""
    full_env = dict(os.environ)
    full_env.update(env)
    try:
        return subprocess.run(
            list(argv),
            env=full_env,
            cwd=str(cwd),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError(f"{what} failed to run: {exc}") from exc


def command_failed(result: subprocess.CompletedProcess, what: str) -> AdapterError:
    detail = (result.stderr or result.stdout or "").strip()[:500]
    return AdapterError(f"{what} exited {result.returncode}: {detail}")


class Adapter(ABC):
    name: str = "abstract"
    transport: str = ""

    @abstractmethod
    def prepare(self, context: PrepareContext) -> CommandPacket:
        """Build the non-secret command packet for exactly one turn."""

    @abstractmethod
    def execute(self, context: PrepareContext, packet: CommandPacket,
                timeout: int | None = None) -> dict:
        """Run exactly one turn and return the normalized evidence record.

        The turn artifact must exist at ``context.turn_file`` on return.
        Evidence identity fields must come from external records; a
        worker's self-description is never sufficient.
        """

    def output_contract(self, context: PrepareContext) -> tuple[str, ...]:
        """Report-section bullets telling the worker how the turn is captured."""
        return (
            f"Write the complete {context.turn.artifact_kind} to: "
            f"{context.turn_file}",
        )

    def base_evidence(self, context: PrepareContext, *, provider: str, model: str,
                      session_id: str, exit_status: int, artifact_sha256: str,
                      proof: dict) -> dict:
        return {
            "evidence_version": EVIDENCE_VERSION,
            "actor_id": context.actor.actor_id,
            "scheduled_actor_id": context.turn.actor_id,
            "actor_selection": (
                "primary"
                if context.actor.actor_id == context.turn.actor_id
                else "substitute"
            ),
            "substitution_reason": context.substitution_reason,
            "round_id": context.turn.round_id,
            "adapter": self.name,
            "transport": self.transport,
            "provider": provider,
            "model": model,
            "session_id": session_id,
            "outcome": "success",
            "exit_status": exit_status,
            "artifact_path": str(context.turn_file),
            "artifact_sha256": artifact_sha256,
            "captured_at": utc_now(),
            "proof": proof,
        }


def get_adapter(transport: str) -> Adapter:
    from .claude_fable import ClaudeFableAdapter
    from .command import CommandAdapter
    from .hermes import HermesAdapter

    registry: dict[str, Adapter] = {
        CommandAdapter.transport: CommandAdapter(),
        ClaudeFableAdapter.transport: ClaudeFableAdapter(),
        HermesAdapter.transport: HermesAdapter(),
    }
    try:
        return registry[transport]
    except KeyError as exc:
        raise AdapterError(
            f"unknown transport {transport!r}; declared transports must be "
            f"one of {sorted(registry)}"
        ) from exc


def adapter_for(actor: Actor) -> Adapter:
    return get_adapter(actor.transport)
