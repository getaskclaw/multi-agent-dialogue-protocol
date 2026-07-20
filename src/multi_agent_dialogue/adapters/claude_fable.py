"""Claude Code turns through the real ``fable-session`` 0.3.0b1 lifecycle.

Verified contract (REAL-RUNTIME-CONTRACTS.md, checked against the
installed 0.3.0b1 source):

.. code-block:: text

    fable-session run --project NAME --task ABS --registry ABS --state-dir ABS --dry-run
    fable-session run --project NAME --task ABS --registry ABS --state-dir ABS --launch --tmux PREFIX-UNIQUE
    fable-session watch --manifest ABS --follow      (exactly one watcher)
    fable-session audit --manifest ABS --format json

There is no ``--profile``, ``--turn-output``, ``--evidence-output``,
``--round``, or ``--actor`` option on ``fable-session run``. The launch
prints ``run manifest: /abs/manifest.json (pending)``; the audit JSON is
the model/runtime authority; the manifest names the structured stream;
the completed turn is the FINAL text-bearing assistant event in that
stream. The worker is never asked to write its own evidence — this
adapter derives the evidence record from manifest + audit JSON +
terminal watch result + the artifact hash it computes itself.
"""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path

from .. import artifacts
from .base import (
    Adapter,
    AdapterError,
    CleanupUnprovenError,
    CommandPacket,
    PrepareContext,
    command_failed,
    read_timeout,
    require_str_setting,
    run_command,
    substitute_env,
)

# Full tmux session-name allowlist enforced by fable-session itself.
_TMUX_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

_MANIFEST_LINE_RE = re.compile(r"run manifest:\s+(\S+)\s+\(pending\)")

# fable-session launches the Claude Code CLI; the provider family is a
# property of the transport, recorded honestly as such in the proof.
PROVIDER = "anthropic"


def _resolve(setting: str, dialogue_dir: Path) -> Path:
    path = Path(setting)
    if not path.is_absolute():
        path = dialogue_dir / path
    return Path(str(path)).absolute()


class ClaudeFableAdapter(Adapter):
    name = "claude-fable"
    transport = "fable-session"

    # -- packet ------------------------------------------------------------

    def _settings(self, context: PrepareContext) -> dict:
        settings = context.actor.settings
        where = f"actor {context.actor.actor_id!r} (claude-fable)"
        command_name = require_str_setting(settings, "command_name", where, "fable-session")
        project = require_str_setting(settings, "project", where)
        registry = _resolve(require_str_setting(settings, "registry", where), context.dialogue_dir)
        state_dir = _resolve(require_str_setting(settings, "state_dir", where), context.dialogue_dir)
        tmux_prefix = require_str_setting(settings, "tmux_prefix", where)
        if not _TMUX_NAME_RE.fullmatch(tmux_prefix):
            raise AdapterError(
                f"{where}: tmux_prefix {tmux_prefix!r} must match "
                f"{_TMUX_NAME_RE.pattern}"
            )
        tmux_command_name = require_str_setting(
            settings, "tmux_command_name", where, "tmux"
        )
        return {
            "where": where,
            "command_name": command_name,
            "project": project,
            "registry": registry,
            "state_dir": state_dir,
            "tmux_prefix": tmux_prefix,
            "tmux_command_name": tmux_command_name,
        }

    def _run_argv(self, resolved: dict, context: PrepareContext) -> list[str]:
        return [
            resolved["command_name"],
            "run",
            "--project", resolved["project"],
            "--task", str(Path(context.task_file).absolute()),
            "--registry", str(resolved["registry"]),
            "--state-dir", str(resolved["state_dir"]),
        ]

    def _tmux_name(self, resolved: dict, context: PrepareContext) -> str:
        name = f"{resolved['tmux_prefix']}{context.turn.round_id}-{secrets.token_hex(4)}"
        if not _TMUX_NAME_RE.fullmatch(name):
            raise AdapterError(
                f"{resolved['where']}: generated tmux name {name!r} violates "
                f"the {_TMUX_NAME_RE.pattern} allowlist; use a shorter prefix"
            )
        return name

    def prepare(self, context: PrepareContext) -> CommandPacket:
        resolved = self._settings(context)
        settings = context.actor.settings
        base = self._run_argv(resolved, context)
        tmux_name = self._tmux_name(resolved, context)
        command = resolved["command_name"]
        env = substitute_env(settings.get("env"), context.placeholders(), resolved["where"])
        return CommandPacket(
            adapter=self.name,
            argv=(*base, "--launch", "--tmux", tmux_name),
            env=env,
            timeout_seconds=read_timeout(settings, resolved["where"]),
            description=(
                f"one {context.turn.round_id} turn via the real fable-session "
                f"0.3.0b1 lifecycle for project {resolved['project']!r}: "
                "dry-run preflight, one launch, exactly one watch --follow, "
                "JSON audit; evidence is derived from manifest + audit + stream"
            ),
            lifecycle=(
                {"purpose": "no-side-effect preflight", "argv": (*base, "--dry-run")},
                {
                    "purpose": "single terminal watcher",
                    "argv": (command, "watch", "--manifest", "{manifest}", "--follow"),
                },
                {
                    "purpose": "model/runtime audit (identity authority)",
                    "argv": (command, "audit", "--manifest", "{manifest}", "--format", "json"),
                },
                {
                    "purpose": (
                        "post-launch failure only: stop exactly this "
                        "generated lane (never any other session)"
                    ),
                    "argv": (
                        resolved["tmux_command_name"], "kill-session",
                        "-t", f"={tmux_name}",
                    ),
                },
                {
                    "purpose": (
                        "post-launch failure only: prove the lane is gone "
                        "(nonzero exit) before the claim may be released"
                    ),
                    "argv": (
                        resolved["tmux_command_name"], "has-session",
                        "-t", f"={tmux_name}",
                    ),
                },
            ),
        )

    def output_contract(self, context: PrepareContext) -> tuple[str, ...]:
        return (
            f"End with the complete {context.turn.artifact_kind} as your "
            "final assistant message; the adapter publishes that final "
            "text-bearing message verbatim as the turn.",
            "Do not write turn or evidence files yourself: runtime evidence "
            "is derived externally from the session manifest, audit JSON, "
            "and structured stream.",
        )

    # -- execution ---------------------------------------------------------

    def execute(self, context: PrepareContext, packet: CommandPacket,
                timeout: int | None = None) -> dict:
        resolved = self._settings(context)
        where = resolved["where"]
        command = resolved["command_name"]
        step_timeout = timeout or packet.timeout_seconds
        base = self._run_argv(resolved, context)

        def run(argv: list[str] | tuple[str, ...], what: str):
            result = run_command(
                argv, env=packet.env, cwd=context.work_dir,
                timeout=step_timeout, what=what,
            )
            return result

        # 1. Real dry-run preflight: validates task/registry, starts nothing.
        # A failure here needs no lane cleanup — no launch was attempted.
        preflight = run((*base, "--dry-run"), f"{where}: fable-session dry-run preflight")
        if preflight.returncode != 0:
            raise command_failed(preflight, f"{where}: fable-session dry-run preflight")

        # The generated lane name is known BEFORE the launch, so cleanup
        # never depends on the manifest (a crashed launch prints none).
        argv = list(packet.argv)
        try:
            tmux_name = argv[argv.index("--tmux") + 1]
        except (ValueError, IndexError) as exc:
            raise AdapterError(
                f"{where}: launch packet names no --tmux lane; refusing to "
                "start an unstoppable session"
            ) from exc

        # From the launch attempt onward, ANY failure must stop the exact
        # generated tmux lane and prove it is gone before the error may
        # propagate (and the runner release the claim). Unproven cleanup
        # escalates to CleanupUnprovenError so the dialogue locks instead.
        try:
            return self._launched_lifecycle(context, packet, resolved, run)
        except CleanupUnprovenError:
            raise
        except BaseException as failure:
            self._stop_exact_lane(resolved, packet, context, tmux_name,
                                  step_timeout, failure)
            raise

    def _launched_lifecycle(self, context: PrepareContext, packet: CommandPacket,
                            resolved: dict, run) -> dict:
        """Steps 2–4 of the real lifecycle: launch, one watch, one audit."""
        where = resolved["where"]
        command = resolved["command_name"]

        # 2. One explicit launch. The packet's argv already carries a fresh
        # unique tmux name.
        launch = run(packet.argv, f"{where}: fable-session launch")
        if launch.returncode != 0:
            raise command_failed(launch, f"{where}: fable-session launch")
        match = _MANIFEST_LINE_RE.search(launch.stdout or "")
        if not match:
            raise AdapterError(
                f"{where}: launch output did not print "
                "'run manifest: <path> (pending)'; cannot locate the lane manifest"
            )
        manifest_path = Path(match.group(1))
        if not manifest_path.is_absolute():
            raise AdapterError(f"{where}: manifest path {manifest_path} is not absolute")

        # 3. Exactly ONE watcher; it exits only at the lane's terminal state.
        watch = run(
            (command, "watch", "--manifest", str(manifest_path), "--follow"),
            f"{where}: fable-session watch --follow",
        )
        if watch.returncode != 0:
            raise AdapterError(
                f"{where}: the lane did not reach a successful terminal state "
                f"(watch exit {watch.returncode}): "
                f"{(watch.stdout or watch.stderr or '').strip()[:300]}"
            )

        # 4. JSON audit: the model/runtime authority.
        audit_run = run(
            (command, "audit", "--manifest", str(manifest_path), "--format", "json"),
            f"{where}: fable-session audit",
        )
        if audit_run.returncode != 0:
            raise AdapterError(
                f"{where}: model audit did not report PURE "
                f"(exit {audit_run.returncode}); refusing unproven identity"
            )
        try:
            audit = json.loads(audit_run.stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{where}: audit output is not JSON: {exc}") from exc
        self._require_clean_audit(audit, where)

        manifest = self._load_json(manifest_path, f"{where}: run manifest")
        session_id = manifest.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise AdapterError(f"{where}: manifest has no session_id")
        if audit.get("session_ids") != [session_id]:
            raise AdapterError(
                f"{where}: audited session ids {audit.get('session_ids')!r} do "
                f"not identify exactly the launched session {session_id!r}"
            )

        stream_raw = manifest.get("expected_transcript_path")
        if not isinstance(stream_raw, str) or not stream_raw:
            raise AdapterError(f"{where}: manifest names no expected_transcript_path")
        stream_path = Path(stream_raw)
        try:
            stream_data = artifacts.read_bytes_nofollow(stream_path, "structured stream")
        except artifacts.ArtifactError as exc:
            raise AdapterError(f"{where}: {exc}") from exc

        turn_text = self._final_assistant_text(stream_data, session_id, where)
        if not turn_text.endswith("\n"):
            turn_text += "\n"
        try:
            artifacts.write_bytes_exclusive(
                context.turn_file, turn_text.encode("utf-8"), "turn artifact"
            )
        except artifacts.ArtifactError as exc:
            raise AdapterError(f"{where}: {exc}") from exc

        proof = {
            "kind": "fable-session",
            "tool": manifest.get("tool"),
            "run_id": manifest.get("run_id"),
            "manifest_path": str(manifest_path),
            "stream_path": str(stream_path),
            "stream_sha256": artifacts.sha256_bytes(stream_data),
            "watch_exit": watch.returncode,
            "watch_invocations": 1,
            "audit": {
                "verdict": audit.get("verdict"),
                "evidence_complete": audit.get("evidence_complete"),
                "observed_message_models": audit.get("observed_message_models"),
                "final_response_model": audit.get("final_response_model"),
                "fallback_event_count": audit.get("fallback_event_count"),
                "reason_codes": audit.get("reason_codes"),
                "result": audit.get("result"),
                "session_ids": audit.get("session_ids"),
            },
            "provider_basis": (
                "fable-session launches the Claude Code CLI; the provider is "
                "the transport family, the model is the audited "
                "final_response_model"
            ),
        }
        return self.base_evidence(
            context,
            provider=PROVIDER,
            model=audit["final_response_model"],
            session_id=session_id,
            exit_status=watch.returncode,
            artifact_sha256=artifacts.sha256_bytes(turn_text.encode("utf-8")),
            proof=proof,
        )

    # -- post-launch lane cleanup --------------------------------------------

    def _stop_exact_lane(self, resolved: dict, packet: CommandPacket,
                         context: PrepareContext, tmux_name: str,
                         timeout: int, failure: BaseException) -> None:
        """Stop ONLY the lane this packet generated and prove it is gone.

        ``=name`` is tmux's exact-match target form: it can never
        prefix-match or fuzzy-match another session, so no unrelated
        tmux session is ever inspected or killed. ``kill-session``'s
        exit status is not trusted (the lane may not exist, or the kill
        may lie/fail); the proof of death is ``has-session`` exiting
        nonzero. If the tmux commands cannot run, or the lane is still
        alive afterwards, cleanup is UNPROVEN and the failure escalates
        to :class:`CleanupUnprovenError` — the runner must then keep the
        claim and lock the dialogue rather than allow a retry to start a
        duplicate worker beside the surviving lane.
        """
        where = resolved["where"]
        tmux = resolved["tmux_command_name"]
        target = f"={tmux_name}"
        try:
            run_command(
                (tmux, "kill-session", "-t", target),
                env=packet.env, cwd=context.work_dir, timeout=timeout,
                what=f"{where}: tmux kill-session -t {target}",
            )
            alive = run_command(
                (tmux, "has-session", "-t", target),
                env=packet.env, cwd=context.work_dir, timeout=timeout,
                what=f"{where}: tmux has-session -t {target}",
            )
        except AdapterError as exc:
            raise CleanupUnprovenError(
                f"{where}: cannot prove tmux lane {tmux_name!r} stopped after "
                f"a failed launch lifecycle ({exc}); original failure: {failure}"
            ) from failure
        if alive.returncode == 0:
            raise CleanupUnprovenError(
                f"{where}: tmux lane {tmux_name!r} is still alive after "
                f"kill-session; refusing to release the turn claim; "
                f"original failure: {failure}"
            ) from failure

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path, what: str) -> dict:
        try:
            data = artifacts.read_bytes_nofollow(path, what)
        except artifacts.ArtifactError as exc:
            raise AdapterError(str(exc)) from exc
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError(f"{what} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise AdapterError(f"{what} is not a JSON object")
        return payload

    @staticmethod
    def _require_clean_audit(audit: dict, where: str) -> None:
        problems = []
        if audit.get("verdict") != "PURE":
            problems.append(f"verdict is {audit.get('verdict')!r}, not PURE")
        if audit.get("evidence_complete") is not True:
            problems.append("audit evidence is incomplete")
        result = audit.get("result") or {}
        if result.get("subtype") != "success" or result.get("is_error") is not False:
            problems.append(f"terminal result is not a success ({result!r})")
        model = audit.get("final_response_model")
        if not isinstance(model, str) or not model:
            problems.append("final response model is unproven")
        if problems:
            raise AdapterError(
                f"{where}: audit rejected the lane: " + "; ".join(problems)
            )

    @staticmethod
    def _final_assistant_text(stream_data: bytes, session_id: str, where: str) -> str:
        """Text of the FINAL text-bearing assistant event in the stream."""
        final_text: str | None = None
        for line in stream_data.decode("utf-8", errors="strict").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"{where}: malformed stream line: {exc}") from exc
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            event_session = event.get("session_id") or event.get("sessionId")
            if event_session is not None and event_session != session_id:
                raise AdapterError(
                    f"{where}: stream event belongs to session "
                    f"{event_session!r}, not the launched lane {session_id!r}"
                )
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            text = "".join(
                block.get("text", "")
                for block in message.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text.strip():
                final_text = text
        if final_text is None:
            raise AdapterError(
                f"{where}: the structured stream contains no text-bearing "
                "assistant event; there is no turn to publish"
            )
        return final_text
