"""Generic command adapter: run one external worker, verify identity externally.

A generic arbitrary command cannot prove its own provider/model merely
by printing JSON — self-reported output is not identity proof. This
adapter therefore FAILS CLOSED unless the actor configures an explicit
external ``identity_verifier_argv`` command. After the worker writes
the turn artifact, the adapter (never the worker) runs that verifier
and builds the evidence record from the verifier's report; the proof is
labelled ``external-command-verifier`` and preserves the verifier's own
honesty flags (a fake verifier that says ``"fake": true`` stays fake in
the proof — it is never upgraded to real identity proof).

The actor's settings provide an ``argv`` template; ``{task_file}``,
``{turn_file}``, ``{evidence_file}``, ``{dialogue_dir}``, ``{work_dir}``,
``{round_id}``, and ``{actor_id}`` are substituted at prepare time.
"""

from __future__ import annotations

import json

from .. import artifacts
from .base import (
    Adapter,
    AdapterError,
    CommandPacket,
    PrepareContext,
    command_failed,
    read_timeout,
    run_command,
    substitute,
    substitute_env,
)

REPORT_FIELDS = ("provider", "model", "session_id", "outcome")


def _argv_setting(settings: dict, key: str, where: str, required: bool) -> list[str]:
    value = settings.get(key)
    if value is None and not required:
        return []
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise AdapterError(f"{where}: settings.{key} must be a non-empty list of strings")
    return value


class CommandAdapter(Adapter):
    name = "command"
    transport = "command"

    def version_probe_argv(self, context: PrepareContext) -> list[str]:
        settings = context.actor.settings
        where = f"actor {context.actor.actor_id!r} (command)"
        argv_setting = _argv_setting(settings, "argv", where, required=True)
        return [argv_setting[0], "--version"]

    def prepare(self, context: PrepareContext) -> CommandPacket:
        settings = context.actor.settings
        where = f"actor {context.actor.actor_id!r} (command)"
        argv_setting = _argv_setting(settings, "argv", where, required=True)
        verifier_setting = settings.get("identity_verifier_argv")
        if verifier_setting is None:
            raise AdapterError(
                f"{where}: a generic command cannot prove its own runtime "
                "identity; configure settings.identity_verifier_argv with an "
                "external identity verifier command, or use an "
                "identity-proving transport (fable-session, hermes-cli). "
                "Refusing identity-sensitive completion by default."
            )
        verifier_argv_setting = _argv_setting(
            settings, "identity_verifier_argv", where, required=True
        )
        placeholders = context.placeholders()
        argv = tuple(substitute(item, placeholders, where) for item in argv_setting)
        verifier_argv = tuple(
            substitute(item, placeholders, where) for item in verifier_argv_setting
        )
        env = substitute_env(settings.get("env"), placeholders, where)
        return CommandPacket(
            adapter=self.name,
            argv=argv,
            env=env,
            timeout_seconds=read_timeout(settings, where),
            description=(
                f"one {context.turn.round_id} turn via external command; "
                "identity is collected by the configured external verifier, "
                "never taken from the worker's own output"
            ),
            lifecycle=(
                {"purpose": "external identity verifier", "argv": verifier_argv},
            ),
        )

    def execute(self, context: PrepareContext, packet: CommandPacket,
                timeout: int | None = None) -> dict:
        where = f"actor {context.actor.actor_id!r} (command)"
        step_timeout = timeout or packet.timeout_seconds

        worker = run_command(
            packet.argv, env=packet.env, cwd=context.work_dir,
            timeout=step_timeout, what=f"{where}: worker command",
        )
        if worker.returncode != 0:
            raise command_failed(worker, f"{where}: worker command")
        try:
            turn_data = artifacts.read_bytes_nofollow(context.turn_file, "turn artifact")
        except artifacts.ArtifactError as exc:
            raise AdapterError(
                f"{where}: the worker produced no readable turn artifact: {exc}"
            ) from exc

        verifier_argv = list(packet.lifecycle[0]["argv"])
        verifier = run_command(
            verifier_argv, env=packet.env, cwd=context.work_dir,
            timeout=step_timeout, what=f"{where}: identity verifier",
        )
        if verifier.returncode != 0:
            raise command_failed(verifier, f"{where}: identity verifier")
        try:
            report = json.loads(verifier.stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"{where}: identity verifier printed no valid JSON report: {exc}"
            ) from exc
        if not isinstance(report, dict):
            raise AdapterError(f"{where}: identity verifier report must be a JSON object")
        missing = [key for key in REPORT_FIELDS if not report.get(key)]
        if missing:
            raise AdapterError(
                f"{where}: identity verifier report is missing {missing}; "
                "identity is unproven and completion is refused"
            )
        if report["outcome"] != "success":
            raise AdapterError(
                f"{where}: identity verifier reports terminal outcome "
                f"{report['outcome']!r}, not success"
            )

        proof = {
            "kind": "external-command-verifier",
            "worker_argv": list(packet.argv),
            "verifier_argv": verifier_argv,
            # The verifier's own report, verbatim — including any honesty
            # flags like {"fake": true}. This adapter never upgrades an
            # external report into a stronger identity claim.
            "report": report,
        }
        return self.base_evidence(
            context,
            provider=str(report["provider"]),
            model=str(report["model"]),
            session_id=str(report["session_id"]),
            exit_status=worker.returncode,
            artifact_sha256=artifacts.sha256_bytes(turn_data),
            proof=proof,
        )
