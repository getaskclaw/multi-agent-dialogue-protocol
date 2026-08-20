"""Hermes turns through the real ``hermes chat`` one-shot contract.

Version-sensitive adapter contract:

.. code-block:: text

    HERMES_HOME=/abs/actor/home hermes chat -q PROMPT -Q \
        --source UNIQUE_SOURCE --pass-session-id

There is no ``hermes one-shot`` subcommand and no ``--task``,
``--turn-output``, ``--evidence-output``, ``--round``, or ``--actor``
option. Identity is never taken from worker-printed text: after the
command exits, this adapter opens the actor-specific
``${HERMES_HOME}/state.db`` read-only and derives the session ID,
provider, observed model set, and the final ACTIVE assistant message
from the ``sessions``, ``messages``, and ``session_model_usage`` tables
— exactly one session with this turn's unique ``--source`` started
inside the invocation window.

Compatible Hermes session records may leave ``sessions.ended_at`` and
``sessions.end_reason`` NULL after a successful one-shot. Completion is
therefore proven by the clean
subprocess exit + unique source attribution + final active assistant
message + positive API-call usage, and the evidence labels that basis
(``terminal_basis``) instead of claiming a DB terminal state. When the
terminal fields ARE present they must be
consistent (``ended_at`` inside the invocation window, never an
``end_reason`` without an ``ended_at``) and clean (``end_reason`` NULL,
empty, ``completed``, or ``cli_close`` — the normal one-shot exit reason
written by Hermes v0.20+); anything else fails closed.

Two Hermes actors are only independent if they use two different
``hermes_home`` directories; the setting is mandatory and never derived
from the actor's role or ID.
"""

from __future__ import annotations

import secrets
import sqlite3
import time
from pathlib import Path

from .. import artifacts
from .base import (
    Adapter,
    AdapterError,
    CommandPacket,
    PrepareContext,
    command_failed,
    read_timeout,
    require_str_setting,
    run_command,
    substitute_env,
)

PROMPT_PLACEHOLDER = "{task_briefing}"

# Clock skew allowance when matching sessions.started_at to the
# invocation window (same host, same clock; generous margin).
WINDOW_SLACK_SECONDS = 5.0

# Persisted end_reason values that count as a clean completion.
# Both terminal fields may be NULL; an unset reason is handled
# separately from these values. "cli_close" is how Hermes v0.20+
# finalizes one-shot (-q/-Q) sessions on normal CLI exit
# (_flush_one_shot_session_store in cli.py); it carries the same
# meaning as "completed" for the one-shot contract and is still gated
# by ended_at consistency plus the process-exit terminal basis below.
CLEAN_END_REASONS = frozenset({"completed", "cli_close"})

# Proof labels for what actually established the turn's completion.
TERMINAL_BASIS_DB = "state-db-ended"
TERMINAL_BASIS_PROCESS = (
    "process-exit+unique-source+final-active-assistant-message"
    "+positive-api-usage"
)


class HermesAdapter(Adapter):
    name = "hermes"
    transport = "hermes-cli"

    def _home(self, context: PrepareContext) -> Path:
        where = f"actor {context.actor.actor_id!r} (hermes)"
        home_setting = context.actor.settings.get("hermes_home")
        if not isinstance(home_setting, str) or not home_setting.strip():
            raise AdapterError(
                f"{where}: settings.hermes_home is required and is never "
                "inferred from the role name"
            )
        home = Path(home_setting)
        if not home.is_absolute():
            home = (context.dialogue_dir / home).resolve()
        return home

    def _source(self, context: PrepareContext) -> str:
        # A unique source per turn: the post-run state.db lookup must match
        # exactly one session, even across retries of the same round.
        return (
            f"madp-{context.turn.round_id}-{context.actor.actor_id}-"
            f"{secrets.token_hex(4)}"
        )

    def _argv(self, command_name: str, prompt: str, source: str) -> tuple[str, ...]:
        return (
            command_name,
            "chat",
            "-q", prompt,
            "-Q",
            "--source", source,
            "--pass-session-id",
        )

    def prepare(self, context: PrepareContext) -> CommandPacket:
        settings = context.actor.settings
        where = f"actor {context.actor.actor_id!r} (hermes)"
        command_name = require_str_setting(settings, "command_name", where, "hermes")
        home = self._home(context)
        source = self._source(context)
        env = substitute_env(settings.get("env"), context.placeholders(), where)
        env["HERMES_HOME"] = str(home)
        return CommandPacket(
            adapter=self.name,
            # The prompt slot carries a placeholder at prepare time; execute()
            # substitutes the task briefing text read from the task file.
            argv=self._argv(command_name, PROMPT_PLACEHOLDER, source),
            env=env,
            timeout_seconds=read_timeout(settings, where),
            description=(
                f"one {context.turn.round_id} turn via `hermes chat -q ... -Q "
                f"--source {source} --pass-session-id` under "
                f"HERMES_HOME={home}; identity and the final active assistant "
                "message are derived from that home's state.db"
            ),
        )

    def output_contract(self, context: PrepareContext) -> tuple[str, ...]:
        return (
            f"End your reply with the complete {context.turn.artifact_kind}; "
            "the final active assistant message recorded in this actor's "
            "Hermes session database is published verbatim as the turn.",
            "Do not write turn or evidence files yourself: runtime evidence "
            "is derived externally from the actor profile's state.db.",
        )

    # -- execution ---------------------------------------------------------

    def execute(self, context: PrepareContext, packet: CommandPacket,
                timeout: int | None = None) -> dict:
        where = f"actor {context.actor.actor_id!r} (hermes)"
        home = Path(packet.env["HERMES_HOME"])
        argv = list(packet.argv)
        try:
            source = argv[argv.index("--source") + 1]
        except (ValueError, IndexError) as exc:
            raise AdapterError(f"{where}: packet lost its --source argument") from exc

        try:
            prompt = artifacts.read_bytes_nofollow(
                context.task_file, "task briefing"
            ).decode("utf-8")
        except (artifacts.ArtifactError, UnicodeDecodeError) as exc:
            raise AdapterError(f"{where}: {exc}") from exc
        argv[argv.index(PROMPT_PLACEHOLDER)] = prompt

        started_before = time.time()
        result = run_command(
            argv, env=packet.env, cwd=context.work_dir,
            timeout=timeout or packet.timeout_seconds,
            what=f"{where}: hermes chat",
        )
        ended_after = time.time()
        if result.returncode != 0:
            raise command_failed(result, f"{where}: hermes chat")

        db_path = home / "state.db"
        if not db_path.is_file():
            raise AdapterError(
                f"{where}: no state.db under HERMES_HOME {home}; identity "
                "cannot be derived, completion is refused"
            )
        observed = self._observe_session(
            db_path, source, started_before, ended_after, where
        )

        turn_text = observed["final_message"]
        if not turn_text.endswith("\n"):
            turn_text += "\n"
        try:
            artifacts.write_bytes_exclusive(
                context.turn_file, turn_text.encode("utf-8"), "turn artifact"
            )
        except artifacts.ArtifactError as exc:
            raise AdapterError(f"{where}: {exc}") from exc

        proof = {
            "kind": "hermes-state-db",
            "state_db": str(db_path),
            "hermes_home": str(home),
            "source": source,
            "session_id": observed["session_id"],
            "started_at": observed["started_at"],
            "ended_at": observed["ended_at"],
            "end_reason": observed["end_reason"],
            "observed_models": observed["models"],
            "observed_providers": observed["providers"],
            "api_call_count": observed["api_call_count"],
            "invocation_window": [started_before, ended_after],
            # The proof names what actually established completion instead
            # of claiming a DB terminal state when none was persisted.
            "terminal_basis": (
                TERMINAL_BASIS_DB
                if observed["ended_at"] is not None
                else TERMINAL_BASIS_PROCESS
            ),
        }
        return self.base_evidence(
            context,
            provider=observed["providers"][0],
            model=observed["models"][0],
            session_id=observed["session_id"],
            exit_status=result.returncode,
            artifact_sha256=artifacts.sha256_bytes(turn_text.encode("utf-8")),
            proof=proof,
        )

    # -- state.db observation ------------------------------------------------

    @staticmethod
    def _observe_session(db_path: Path, source: str, started_before: float,
                         ended_after: float, where: str) -> dict:
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise AdapterError(f"{where}: cannot open {db_path}: {exc}") from exc
        try:
            try:
                sessions = con.execute(
                    "SELECT id, model, started_at, ended_at, end_reason, "
                    "billing_provider FROM sessions WHERE source = ?",
                    (source,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise AdapterError(f"{where}: cannot query sessions: {exc}") from exc
            if len(sessions) != 1:
                raise AdapterError(
                    f"{where}: expected exactly one session with source "
                    f"{source!r}, found {len(sessions)}; identity is ambiguous "
                    "and completion is refused"
                )
            session_id, model, started_at, ended_at, end_reason, provider = sessions[0]
            if not isinstance(session_id, str) or not session_id:
                raise AdapterError(f"{where}: session has no usable id")
            if started_at is None or not (
                started_before - WINDOW_SLACK_SECONDS
                <= float(started_at)
                <= ended_after + WINDOW_SLACK_SECONDS
            ):
                raise AdapterError(
                    f"{where}: session {session_id} started outside this "
                    "invocation window; refusing to attribute it to this turn"
                )
            # Both-NULL is a valid compatibility state. When terminal
            # fields are present, they must be consistent and clean.
            if end_reason is not None and end_reason != "":
                if ended_at is None or not isinstance(end_reason, str):
                    raise AdapterError(
                        f"{where}: session {session_id} has end_reason "
                        f"{end_reason!r} without a usable ended_at; terminal "
                        "fields are inconsistent and completion is refused"
                    )
                if end_reason not in CLEAN_END_REASONS:
                    raise AdapterError(
                        f"{where}: session {session_id} ended with reason "
                        f"{end_reason!r}, not a clean completion"
                    )
            if ended_at is not None:
                try:
                    ended_value = float(ended_at)
                except (TypeError, ValueError):
                    ended_value = None
                if (
                    ended_value is None
                    or ended_value < float(started_at)
                    or ended_value > ended_after + WINDOW_SLACK_SECONDS
                ):
                    raise AdapterError(
                        f"{where}: session {session_id} has ended_at "
                        f"{ended_at!r} before its start or outside this "
                        "invocation window; terminal fields are inconsistent "
                        "and completion is refused"
                    )

            models = {model} if isinstance(model, str) and model else set()
            providers = (
                {provider} if isinstance(provider, str) and provider else set()
            )
            api_calls = 0
            try:
                # Only main-conversation calls (task ''/NULL) prove the
                # actor's model identity. Auxiliary Hermes calls (title
                # generation, vision, compression, ...) are recorded in the
                # same table with a non-empty task and a different model;
                # they say nothing about which model reasoned the turn.
                # Older Hermes state.db schemas have no task column; there
                # every usage row is a main-conversation call.
                # Column names are matched case-insensitively (SQLite
                # resolves identifiers the same way) so an oddly-cased
                # "Task" column still triggers the filter instead of
                # silently reverting to all-rows counting. Consequence for
                # consumers: on task-column schemas, api_call_count covers
                # main-conversation calls only; auxiliary calls are
                # excluded from the count by design.
                cols = {
                    str(row[1]).lower()
                    for row in con.execute(
                        "PRAGMA table_info(session_model_usage)"
                    ).fetchall()
                }
                task_filter = (
                    " AND (task IS NULL OR task = '')" if "task" in cols else ""
                )
                usage_rows = con.execute(
                    "SELECT model, billing_provider, api_call_count "
                    "FROM session_model_usage WHERE session_id = ?"
                    + task_filter,
                    (session_id,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise AdapterError(f"{where}: cannot query model usage: {exc}") from exc
            for usage_model, usage_provider, count in usage_rows:
                if isinstance(usage_model, str) and usage_model:
                    models.add(usage_model)
                if isinstance(usage_provider, str) and usage_provider:
                    providers.add(usage_provider)
                api_calls += int(count or 0)
            if len(models) != 1:
                raise AdapterError(
                    f"{where}: session {session_id} observed model set "
                    f"{sorted(models)!r} is not exactly one model; identity "
                    "is unproven"
                )
            if len(providers) != 1:
                raise AdapterError(
                    f"{where}: session {session_id} observed provider set "
                    f"{sorted(providers)!r} is not exactly one provider"
                )
            if api_calls <= 0:
                raise AdapterError(
                    f"{where}: session {session_id} records no API calls; "
                    "no model demonstrably served this turn and completion "
                    "is refused"
                )

            try:
                final_row = con.execute(
                    "SELECT content FROM messages WHERE session_id = ? "
                    "AND role = 'assistant' AND active = 1 "
                    "AND content IS NOT NULL AND content != '' "
                    "ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise AdapterError(f"{where}: cannot query messages: {exc}") from exc
            if final_row is None or not str(final_row[0]).strip():
                raise AdapterError(
                    f"{where}: session {session_id} has no final active "
                    "assistant message; there is no turn to publish"
                )
            return {
                "session_id": session_id,
                "started_at": float(started_at),
                "ended_at": None if ended_at is None else float(ended_at),
                "end_reason": end_reason if end_reason else None,
                "models": sorted(models),
                "providers": sorted(providers),
                "api_call_count": api_calls,
                "final_message": str(final_row[0]),
            }
        finally:
            con.close()
