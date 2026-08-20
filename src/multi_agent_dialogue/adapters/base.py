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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import artifacts
from ..config import Actor, TurnSpec
from ..evidence import EVIDENCE_VERSION


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
    # The manifest that gated this launch, when the actor declares
    # required_capabilities; runner attaches it via dataclasses.replace
    # so the evidence records exactly what gated — never a re-probe.
    capability_manifest: dict | None = None
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


# Bounded probe for the adapter CLI's own version string. The README
# warns that real adapters depend on the exact installed CLI versions,
# so accepted-turn evidence records the probed version verbatim.
VERSION_PROBE_TIMEOUT_SECONDS = 15


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

    def version_probe_argv(self, context: PrepareContext) -> list[str] | None:
        """Argv that prints the adapter CLI's version, or None if unknown.

        Engine-probed, never adapter self-report: the output of THIS
        command is what lands in evidence.
        """
        return None

    def capability_probes(
        self, context: PrepareContext
    ) -> dict[str, tuple[list[str], Callable[[str, int], bool]]]:
        """Extra CLI capability probes: name -> (argv, check).

        ``check`` receives (full_output, returncode) and returns bool.
        Every probe is run by the engine against the real CLI — the
        adapter only names WHAT to probe, never reports its own support.
        """
        return {}

    def _run_capability_probe(
        self,
        argv: list[str],
        check,
        context: PrepareContext,
        where: str,
    ) -> dict:
        """Run one bounded capability probe under the actor's env.

        cwd is the dialogue directory (always exists) so the pre-launch
        gate can probe before the turn's work directory is created.
        """
        try:
            probe_env = substitute_env(
                context.actor.settings.get("env"), context.placeholders(), where
            )
            result = run_command(
                argv, env=probe_env, cwd=context.dialogue_dir,
                timeout=VERSION_PROBE_TIMEOUT_SECONDS,
                what=f"{where}: capability probe {' '.join(argv[:2])}",
            )
        except AdapterError as exc:
            return {"ok": False, "error": str(exc), "probe": {"argv": list(argv)}}
        raw_output = (result.stdout or result.stderr or "")
        try:
            ok = bool(check(raw_output, result.returncode))
        except Exception as exc:  # a broken check must fail closed, not crash
            return {
                "ok": False,
                "error": f"capability check failed: {exc}",
                "probe": {"argv": list(argv)},
            }
        stripped = raw_output.strip()
        probe_record: dict = {
            "argv": list(argv),
            "exit_status": result.returncode,
            "output": stripped[:500],
            # The hash attests the FULL probe output, computed before
            # the 500-char storage truncation (cli_version parity).
            "output_sha256": artifacts.sha256_bytes(raw_output.encode("utf-8")),
        }
        if len(stripped) > 500:
            probe_record["output_truncated"] = True
        return {"ok": ok, "probe": probe_record}

    def probe_capabilities(self, context: PrepareContext) -> dict:
        """Build the capability manifest by probing the CLI itself.

        ``cli-version`` is always probed when the adapter names a version
        argv; adapter-specific probes come from ``capability_probes``.
        """
        where = f"actor {context.actor.actor_id!r} ({self.name})"
        capabilities: dict[str, dict] = {}
        try:
            version_argv = self.version_probe_argv(context)
        except AdapterError as exc:
            capabilities["cli-version"] = {"ok": False, "error": str(exc)}
        else:
            if version_argv is not None:
                capabilities["cli-version"] = self._run_capability_probe(
                    version_argv,
                    lambda output, rc: rc == 0,
                    context,
                    where,
                )
        try:
            extra_probes = self.capability_probes(context)
        except AdapterError as exc:
            # A broken hook must not crash the gate: its capabilities
            # simply never show up as ok, and the error is recorded.
            extra_probes = {}
            hook_error = str(exc)
        else:
            hook_error = None
        for name, (argv, check) in extra_probes.items():
            capabilities[name] = self._run_capability_probe(
                argv, check, context, where
            )
        manifest: dict = {
            "capabilities": capabilities,
            "probed_at": utc_now(),
        }
        if hook_error is not None:
            manifest["hook_error"] = hook_error
        return manifest

    def cli_version_evidence(self, context: PrepareContext) -> dict | None:
        """Probe the adapter CLI version for the evidence record.

        Informational only: a failed probe is recorded, never fatal —
        the turn's acceptance still rests on the identity evidence.
        """
        where = f"actor {context.actor.actor_id!r} ({self.name})"
        argv: list[str] | None = None
        try:
            # Inside the try: a subclass hook raising AdapterError (bad
            # settings, malformed env) must degrade to a recorded error,
            # never fail the accepted turn.
            argv = self.version_probe_argv(context)
            if argv is None:
                return None
            # Probe under the actor's own settings env (same substitution
            # as the turn packet) so PATH- or env-dependent CLIs resolve
            # the binary the turn actually used.
            probe_env = substitute_env(
                context.actor.settings.get("env"), context.placeholders(), where
            )
            result = run_command(
                argv, env=probe_env, cwd=context.work_dir,
                timeout=VERSION_PROBE_TIMEOUT_SECONDS,
                what=f"{where}: version probe",
            )
        except AdapterError as exc:
            record: dict = {"error": str(exc)}
            if argv is not None:
                record["argv"] = list(argv)
            return record
        raw_output = (result.stdout or result.stderr or "").strip()
        record = {
            "argv": list(argv),
            "exit_status": result.returncode,
            "output": raw_output[:500],
            # The hash attests the FULL probe output, computed before the
            # 500-char storage truncation; output_truncated flags when the
            # stored output is a prefix of what the hash covers.
            "output_sha256": artifacts.sha256_bytes(raw_output.encode("utf-8")),
            "probed_at": utc_now(),
        }
        if len(raw_output) > 500:
            record["output_truncated"] = True
        if result.returncode != 0:
            record["error"] = f"version probe exited {result.returncode}"
        return record

    def base_evidence(self, context: PrepareContext, *, provider: str, model: str,
                      session_id: str, exit_status: int, artifact_sha256: str,
                      proof: dict) -> dict:
        record = {
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
        cli_version = self.cli_version_evidence(context)
        if cli_version is not None:
            record["cli_version"] = cli_version
        if context.actor.required_capabilities:
            # The evidence records the SAME manifest that gated the
            # launch (attached to the context by the runner); a fresh
            # probe only happens if execute ran without the gate.
            manifest = context.capability_manifest
            if manifest is None:
                manifest = self.probe_capabilities(context)
            record["capability_manifest"] = manifest
        return record


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
