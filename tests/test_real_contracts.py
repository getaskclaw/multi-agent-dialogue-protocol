"""RED → GREEN: adapters must follow pinned, version-sensitive CLI contracts.

These tests encode the invocation shapes documented in
``docs/technical-reference.md``:

- fable-session 0.3.0b1:
  ``run --project NAME --task ABS --registry ABS --state-dir ABS``
  with ``--dry-run`` (default-safe preflight) or ``--launch --tmux
  PREFIX-UNIQUE``; then exactly ONE ``watch --manifest ABS --follow``;
  then ``audit --manifest ABS --format json``. There is no
  ``--profile``, ``--turn-output``, ``--evidence-output``, ``--round``,
  or ``--actor`` option on ``fable-session run``. The launch output
  prints ``run manifest: /abs/manifest.json (pending)``; the audit JSON
  is the model/runtime authority; the turn is the final text-bearing
  assistant event in the manifest's structured stream.

- Hermes Agent compatibility contract:
  ``HERMES_HOME=/abs/actor/home hermes chat -q PROMPT -Q --source
  UNIQUE_SOURCE --pass-session-id``. There is no ``hermes one-shot``
  subcommand. Session ID, provider, observed model set, and the final
  active assistant message come from ``${HERMES_HOME}/state.db``, never
  from worker-printed text. The compatibility fixture records exactly
  one source-matched session, one final active assistant message, one
  model/provider, and positive ``session_model_usage.api_call_count``,
  while leaving ``sessions.ended_at`` / ``sessions.end_reason`` NULL.
  A persisted DB terminal state is therefore not required; when those
  fields are present they must be consistent and clean.

- A generic command cannot prove its own runtime identity by printing
  JSON: without an explicit external identity verifier the adapter must
  fail closed.

All runtime proofs below use the contract-faithful fake executables in
examples/fakes/bin — no real Claude, Hermes, tmux, or network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import support

from multi_agent_dialogue import adapters, config, engine, runner
from multi_agent_dialogue.adapters import hermes as hermes_adapter

FABLE = str(support.FAKE_BIN / "fake-fable-session")
HERMES = str(support.FAKE_BIN / "fake-hermes")
WORKER = str(support.FAKE_BIN / "fake-worker")
VERIFIER = str(support.FAKE_BIN / "fake-verifier")
TMUX = str(support.FAKE_BIN / "fake-tmux")

# None of these exist on `fable-session run` 0.3.0b1 or `hermes chat`.
INVENTED_FLAGS = ("--profile", "--turn-output", "--evidence-output", "--round", "--actor")

TMUX_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def write_registry(path: Path, projects: dict[str, tuple[str, str]]) -> None:
    """Write a fable-session host registry: [project.NAME] tables."""
    lines: list[str] = []
    for name, (model, prefix) in projects.items():
        lines += [
            f"[project.{name}]",
            f'repo = "{path.parent}"',
            f'profile = "{support.REPO_ROOT / "examples" / "fakes" / "fable-profile.toml"}"',
            f'model = "{model}"',
            'effort = "high"',
            'fallback = "stop"',
            'permission_mode = "auto"',
            f'tmux_prefix = "{prefix}"',
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def packet_argvs(packet: adapters.CommandPacket) -> list[list[str]]:
    """Primary argv plus every declared lifecycle argv."""
    argvs = [list(packet.argv)]
    for step in packet.as_dict().get("lifecycle", []):
        argvs.append(list(step["argv"]))
    return argvs


class FableTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self.marker = self.base / "marker.log"
        self.registry = self.base / "registry.toml"
        # Fake tmux live-session registry: one file per live session.
        self.tmux_dir = self.base / "tmux-sessions"
        self.tmux_dir.mkdir()
        write_registry(
            self.registry,
            {"proj-a": ("claude-fable-5", "madpa-"), "proj-b": ("claude-opus-4-8", "madpb-")},
        )

    def markers(self) -> list[str]:
        if not self.marker.exists():
            return []
        return self.marker.read_text(encoding="utf-8").splitlines()

    def live_lanes(self) -> list[str]:
        return sorted(path.name for path in self.tmux_dir.iterdir())

    def tmux_marks(self) -> list[str]:
        return [mark for mark in self.markers() if mark.startswith("tmux:")]

    def fable_settings(
        self,
        project: str,
        prefix: str,
        extra_env: dict | None = None,
        extra_settings: dict | None = None,
    ) -> dict:
        env = {
            "FAKE_SPAWN_MARKER": str(self.marker),
            "FAKE_TMUX_DIR": str(self.tmux_dir),
        }
        env.update(extra_env or {})
        settings = {
            "command_name": FABLE,
            "project": project,
            "registry": str(self.registry),
            "state_dir": str(self.base / f"state-{project}"),
            "tmux_prefix": prefix,
            "tmux_command_name": TMUX,
            "env": env,
        }
        settings.update(extra_settings or {})
        return settings

    def definition_raw(
        self, env_a: dict | None = None, settings_a: dict | None = None
    ) -> dict:
        return {
            "protocol_id": "real-fable-contract",
            "version": 1,
            "owner": "owner-human",
            "source_sha": "0000000000000000000000000000000000000000",
            "evidence_roots": [],
            "actors": [
                {
                    "actor_id": "fable-a",
                    "role": "proposer",
                    "transport": "fable-session",
                    "expected_provider": "anthropic",
                    "expected_model": "claude-fable-5",
                    "settings": self.fable_settings("proj-a", "madpa-", env_a, settings_a),
                },
                {
                    "actor_id": "fable-b",
                    "role": "challenger",
                    "transport": "fable-session",
                    "expected_provider": "anthropic",
                    "expected_model": "claude-opus-4-8",
                    "settings": self.fable_settings("proj-b", "madpb-"),
                },
            ],
            "schedule": [
                {"round_id": "R01", "actor_id": "fable-a", "purpose": "propose", "artifact_kind": "proposal"},
                {"round_id": "R02", "actor_id": "fable-b", "purpose": "challenge", "artifact_kind": "challenge"},
            ],
            "final_round_id": "R02",
        }

    def make_dialogue(
        self, env_a: dict | None = None, settings_a: dict | None = None
    ) -> engine.Dialogue:
        definition = config.parse_definition(self.definition_raw(env_a, settings_a))
        return engine.init_dialogue(definition, self.base / "dialogue")

    def context(self) -> adapters.PrepareContext:
        definition = config.parse_definition(self.definition_raw())
        work = self.base / "dialogue" / "work" / "R01"
        return adapters.PrepareContext(
            actor=definition.actor("fable-a"),
            turn=definition.schedule[0],
            dialogue_dir=self.base / "dialogue",
            work_dir=work,
            task_file=work / "task.md",
            turn_file=work / "turn.md",
            evidence_file=work / "evidence.json",
        )


class FablePacketContractTests(FableTestCase):
    def test_packet_uses_real_run_launch_shape(self) -> None:
        packet = adapters.get_adapter("fable-session").prepare(self.context())
        argv = list(packet.argv)
        self.assertEqual(argv[1], "run")
        for flag in ("--project", "--task", "--registry", "--state-dir", "--launch", "--tmux"):
            self.assertIn(flag, argv, f"real fable-session flag {flag} missing")
        self.assertIn("proj-a", argv)
        task_arg = argv[argv.index("--task") + 1]
        self.assertTrue(Path(task_arg).is_absolute(), "fable-session requires an absolute task path")
        tmux_name = argv[argv.index("--tmux") + 1]
        self.assertTrue(tmux_name.startswith("madpa-"), tmux_name)
        self.assertIsNotNone(TMUX_NAME_RE.fullmatch(tmux_name), tmux_name)
        for invented in INVENTED_FLAGS:
            self.assertNotIn(invented, argv, f"invented flag {invented} must not be emitted")

    def test_packet_declares_dry_run_watch_and_audit_lifecycle(self) -> None:
        packet = adapters.get_adapter("fable-session").prepare(self.context())
        lifecycle = packet.as_dict()["lifecycle"]
        flat = [list(step["argv"]) for step in lifecycle]
        self.assertTrue(
            any("--dry-run" in argv and "run" in argv for argv in flat),
            "the real fable-session --dry-run preflight must be part of the lifecycle",
        )
        watch_steps = [argv for argv in flat if "watch" in argv]
        self.assertEqual(len(watch_steps), 1, "exactly one watch per lane")
        self.assertIn("--manifest", watch_steps[0])
        self.assertIn("--follow", watch_steps[0])
        audit_steps = [argv for argv in flat if "audit" in argv]
        self.assertEqual(len(audit_steps), 1)
        self.assertIn("--format", audit_steps[0])
        self.assertIn("json", audit_steps[0])
        for argv in [list(packet.argv), *flat]:
            for invented in INVENTED_FLAGS:
                self.assertNotIn(invented, argv)

    def test_missing_project_or_registry_fails_closed(self) -> None:
        for missing in ("project", "registry", "state_dir", "tmux_prefix"):
            settings = self.fable_settings("proj-a", "madpa-")
            del settings[missing]
            actor = config.Actor(
                actor_id="fable-a",
                role="proposer",
                transport="fable-session",
                expected_provider="anthropic",
                expected_model="claude-fable-5",
                settings=settings,
            )
            context = self.context()
            context = adapters.PrepareContext(
                actor=actor,
                turn=context.turn,
                dialogue_dir=context.dialogue_dir,
                work_dir=context.work_dir,
                task_file=context.task_file,
                turn_file=context.turn_file,
                evidence_file=context.evidence_file,
            )
            with self.assertRaises(adapters.AdapterError, msg=missing) as ctx:
                adapters.get_adapter("fable-session").prepare(context)
            self.assertIn(missing, str(ctx.exception))


class FableLaunchLifecycleTests(FableTestCase):
    def test_launch_runs_real_lifecycle_and_derives_turn_from_stream(self) -> None:
        dialogue = self.make_dialogue()
        result = runner.launch(dialogue, "fable-a")
        self.assertTrue(result["executed"])

        # Real lifecycle order, with exactly one watcher.
        marks = self.markers()
        self.assertEqual(
            marks,
            ["fable-run:dry-run", "fable-run:launch", "fable-watch:follow", "fable-audit:json"],
        )

        state = dialogue.state()
        record = state["completed_turns"][0]
        evidence_record = json.loads(
            (dialogue.directory / record["evidence_file"]).read_text(encoding="utf-8")
        )
        proof = evidence_record["proof"]
        self.assertEqual(proof["kind"], "fable-session")

        # Identity comes from the manifest + audit JSON, not worker claims.
        manifest = json.loads(Path(proof["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(evidence_record["session_id"], manifest["session_id"])
        self.assertEqual(evidence_record["model"], "claude-fable-5")
        self.assertEqual(evidence_record["provider"], "anthropic")
        self.assertEqual(proof["audit"]["verdict"], "PURE")
        self.assertIn(manifest["session_id"], proof["audit"]["session_ids"])

        # The published turn is the final text-bearing assistant event of
        # the structured stream — not any earlier draft.
        stream_path = Path(proof["stream_path"])
        final_text = None
        for line in stream_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") != "assistant":
                continue
            text = "".join(
                block.get("text", "")
                for block in event.get("message", {}).get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text.strip():
                final_text = text
        self.assertIsNotNone(final_text)
        published = (dialogue.directory / record["artifact_file"]).read_text(encoding="utf-8")
        self.assertEqual(published.rstrip("\n"), final_text.rstrip("\n"))
        self.assertNotIn("DRAFT", published)

    def test_impersonating_serving_model_fails_closed(self) -> None:
        # The fake serves a different model than the registry pins; the
        # audit is the authority, so the launch must not complete.
        dialogue = self.make_dialogue(env_a={"FAKE_MODEL": "claude-haiku-4-5"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(dialogue.state()["turn_index"], 0)

    def test_failed_terminal_result_fails_closed(self) -> None:
        dialogue = self.make_dialogue(env_a={"FAKE_OUTCOME": "error"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(dialogue.state()["turn_index"], 0)


class FableLaneCleanupTests(FableTestCase):
    """RED → GREEN: any fable failure AFTER launch must stop the exact
    generated tmux lane and prove it is gone before the claim is
    released; unprovable cleanup must leave the dialogue locked so a
    second launch can never create a duplicate worker. Only the lane
    this packet generated may ever be targeted."""

    def assert_claim_released(self, dialogue: engine.Dialogue) -> None:
        state = dialogue.state()
        self.assertEqual(state["turn_index"], 0)
        self.assertIsNone(state["claim"])
        self.assertEqual(state["status"], "OPEN")
        self.assertFalse(dialogue.lock_path.exists())

    def assert_exact_lane_stopped(self) -> str:
        """The generated lane is gone, killed by exactly one exact-match
        kill-session whose death was verified by exactly one has-session."""
        marks = self.tmux_marks()
        kills = [m for m in marks if m.startswith("tmux:kill-session:")]
        checks = [m for m in marks if m.startswith("tmux:has-session:")]
        self.assertEqual(len(kills), 1, marks)
        self.assertEqual(len(checks), 1, marks)
        self.assertEqual(len(marks), 2, f"unexpected tmux calls: {marks}")
        kill_target = kills[0].split(":", 2)[2]
        check_target = checks[0].split(":", 2)[2]
        self.assertEqual(kill_target, check_target)
        self.assertTrue(
            kill_target.startswith("="),
            f"cleanup must use tmux's '=' exact-match target, got {kill_target!r}",
        )
        name = kill_target[1:]
        self.assertTrue(name.startswith("madpa-R01-"), name)
        self.assertIsNotNone(TMUX_NAME_RE.fullmatch(name), name)
        self.assertNotIn(name, self.live_lanes(), "the lane must really be gone")
        return name

    def test_watch_failure_stops_exact_lane_before_claim_release(self) -> None:
        # The lane reaches a failed terminal state and its tmux session
        # stays alive; the adapter must stop that exact lane, prove it is
        # gone, and only then allow the claim release.
        dialogue = self.make_dialogue(env_a={"FAKE_OUTCOME": "error"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(self.live_lanes(), [], "the launched lane is still alive")
        self.assert_exact_lane_stopped()
        self.assert_claim_released(dialogue)
        # Still exactly one watcher and no audit on the failed lane.
        lifecycle = [m for m in self.markers() if not m.startswith("tmux:")]
        self.assertEqual(
            lifecycle, ["fable-run:dry-run", "fable-run:launch", "fable-watch:follow"]
        )

    def test_watch_timeout_with_live_lane_stops_lane_before_release(self) -> None:
        # The lane never reaches a terminal state: watch --follow blocks
        # until the adapter's timeout kills it. The still-running tmux
        # lane must be stopped and proven gone before the claim releases.
        dialogue = self.make_dialogue(
            env_a={"FAKE_LANE_STUCK": "1"}, settings_a={"timeout_seconds": 1}
        )
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(self.live_lanes(), [], "the stuck lane is still alive")
        self.assert_exact_lane_stopped()
        self.assert_claim_released(dialogue)

    def test_launch_crash_after_lane_creation_still_cleans_up(self) -> None:
        # fable-session crashed after creating the tmux session but before
        # printing the manifest line: no manifest exists, yet the lane is
        # alive. Cleanup must not depend on the manifest.
        dialogue = self.make_dialogue(env_a={"FAKE_LAUNCH_EXIT": "7"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(self.live_lanes(), [], "the crashed launch left its lane alive")
        self.assert_exact_lane_stopped()
        self.assert_claim_released(dialogue)

    def test_unprovable_cleanup_locks_dialogue_nonretryable(self) -> None:
        # kill-session claims success but the lane survives: cleanup is
        # unproven, so the claim must NOT be released — the dialogue locks
        # (BLOCKED, claim + lock retained) and a second launch must not
        # create a duplicate worker.
        dialogue = self.make_dialogue(
            env_a={"FAKE_OUTCOME": "error", "FAKE_TMUX_STUCK": "1"}
        )
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(len(self.live_lanes()), 1, "the unkillable lane is still alive")
        state = dialogue.state()
        self.assertEqual(state["status"], "BLOCKED")
        self.assertIsNotNone(state["claim"], "the claim must be retained")
        self.assertTrue(dialogue.lock_path.exists())
        self.assertIn("madpa-R01-", state.get("blocked_reason", ""))
        self.assertEqual(self.markers().count("fable-run:launch"), 1)
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(
            self.markers().count("fable-run:launch"),
            1,
            "a second launch created a duplicate worker lane",
        )
        with self.assertRaises(engine.ProtocolError):
            dialogue.release("fable-a")
        self.assertEqual(dialogue.validate()["recorded_status"], "BLOCKED")

    def test_cleanup_never_targets_unrelated_sessions(self) -> None:
        # Unrelated live sessions — including a lookalike with the same
        # prefix — must never be inspected or killed; the generated lane
        # name from this packet is the only allowed target.
        (self.tmux_dir / "user-session").write_text("keep\n", encoding="utf-8")
        (self.tmux_dir / "madpa-R01-lookalike").write_text("keep\n", encoding="utf-8")
        dialogue = self.make_dialogue(env_a={"FAKE_OUTCOME": "error"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "fable-a")
        self.assertEqual(
            self.live_lanes(),
            ["madpa-R01-lookalike", "user-session"],
            "only the generated lane may disappear",
        )
        name = self.assert_exact_lane_stopped()
        self.assertNotIn(name, ("user-session", "madpa-R01-lookalike"))
        self.assert_claim_released(dialogue)


class HermesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self.marker = self.base / "marker.log"

    def hermes_settings(self, home: str, extra_env: dict | None = None) -> dict:
        env = {"FAKE_SPAWN_MARKER": str(self.marker)}
        env.update(extra_env or {})
        return {"command_name": HERMES, "hermes_home": home, "env": env}

    def definition_raw(self, env_north: dict | None = None) -> dict:
        return {
            "protocol_id": "real-hermes-contract",
            "version": 1,
            "owner": "owner-human",
            "source_sha": "1111111111111111111111111111111111111111",
            "evidence_roots": [],
            "actors": [
                {
                    "actor_id": "hermes-north",
                    "role": "proposer",
                    "transport": "hermes-cli",
                    "expected_provider": "nousresearch",
                    "expected_model": "hermes-4-405b",
                    "settings": self.hermes_settings("homes/north", env_north),
                },
                {
                    "actor_id": "hermes-south",
                    "role": "challenger",
                    "transport": "hermes-cli",
                    "expected_provider": "nousresearch",
                    "expected_model": "hermes-4-70b",
                    "settings": self.hermes_settings(
                        "homes/south", {"FAKE_MODEL": "hermes-4-70b"}
                    ),
                },
            ],
            "schedule": [
                {"round_id": "R01", "actor_id": "hermes-north", "purpose": "propose", "artifact_kind": "proposal"},
                {"round_id": "R02", "actor_id": "hermes-south", "purpose": "challenge", "artifact_kind": "challenge"},
            ],
            "final_round_id": "R02",
        }

    def make_dialogue(self, env_north: dict | None = None) -> engine.Dialogue:
        definition = config.parse_definition(self.definition_raw(env_north))
        return engine.init_dialogue(definition, self.base / "dialogue")

    def evidence_for(self, dialogue: engine.Dialogue, index: int) -> dict:
        record = dialogue.state()["completed_turns"][index]
        return json.loads(
            (dialogue.directory / record["evidence_file"]).read_text(encoding="utf-8")
        )


class HermesPacketContractTests(HermesTestCase):
    def test_packet_uses_real_chat_shape(self) -> None:
        definition = config.parse_definition(self.definition_raw())
        work = self.base / "dialogue" / "work" / "R01"
        context = adapters.PrepareContext(
            actor=definition.actor("hermes-north"),
            turn=definition.schedule[0],
            dialogue_dir=self.base / "dialogue",
            work_dir=work,
            task_file=work / "task.md",
            turn_file=work / "turn.md",
            evidence_file=work / "evidence.json",
        )
        packet = adapters.get_adapter("hermes-cli").prepare(context)
        argv = list(packet.argv)
        self.assertEqual(argv[1], "chat", "the real one-shot shape is `hermes chat -q ... -Q`")
        self.assertNotIn("one-shot", argv, "`hermes one-shot` does not exist")
        for flag in ("-q", "-Q", "--source", "--pass-session-id"):
            self.assertIn(flag, argv)
        for invented in ("--task", *INVENTED_FLAGS):
            self.assertNotIn(invented, argv)
        source = argv[argv.index("--source") + 1]
        self.assertIn("R01", source)
        self.assertIn("hermes-north", source)
        home = packet.env["HERMES_HOME"]
        self.assertTrue(Path(home).is_absolute())
        self.assertTrue(home.endswith("north"))


class HermesStateDbTests(HermesTestCase):
    def test_identity_and_turn_come_from_state_db_not_stdout(self) -> None:
        # FAKE_STDOUT_LIE makes the fake print different text to stdout
        # than it records in state.db; only the database may be trusted.
        dialogue = self.make_dialogue(env_north={"FAKE_STDOUT_LIE": "1"})
        runner.launch(dialogue, "hermes-north")

        record = dialogue.state()["completed_turns"][0]
        evidence_record = self.evidence_for(dialogue, 0)
        proof = evidence_record["proof"]
        self.assertEqual(proof["kind"], "hermes-state-db")

        db_path = Path(proof["state_db"])
        self.assertTrue(db_path.is_file())
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT id, model, billing_provider, ended_at, end_reason "
                "FROM sessions WHERE source = ?",
                (proof["source"],),
            ).fetchall()
            self.assertEqual(len(rows), 1, "exactly one session per unique source")
            session_id, model, provider, ended_at, end_reason = rows[0]
            self.assertEqual(evidence_record["session_id"], session_id)
            self.assertEqual(evidence_record["model"], model)
            self.assertEqual(evidence_record["provider"], provider)
            # The compatibility fixture persists no terminal state.
            self.assertIsNone(ended_at)
            self.assertIsNone(end_reason)
            final_row = con.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' "
                "AND active = 1 AND content IS NOT NULL AND content != '' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        finally:
            con.close()
        self.assertIsNotNone(final_row)
        published = (dialogue.directory / record["artifact_file"]).read_text(encoding="utf-8")
        self.assertEqual(published.rstrip("\n"), final_row[0].rstrip("\n"))
        self.assertNotIn("STDOUT-ONLY", published)
        self.assertNotIn("GHOST", published, "inactive assistant rows are never the turn")

    def test_two_hermes_actors_use_two_state_dbs(self) -> None:
        dialogue = self.make_dialogue()
        runner.launch(dialogue, "hermes-north")
        runner.launch(dialogue, "hermes-south")
        first = self.evidence_for(dialogue, 0)["proof"]
        second = self.evidence_for(dialogue, 1)["proof"]
        self.assertNotEqual(first["state_db"], second["state_db"])
        self.assertNotEqual(first["hermes_home"], second["hermes_home"])

    def test_ambiguous_source_fails_closed(self) -> None:
        dialogue = self.make_dialogue(env_north={"FAKE_EXTRA_SESSION": "1"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 0)

    def test_mixed_observed_models_fail_closed(self) -> None:
        dialogue = self.make_dialogue(env_north={"FAKE_MULTI_MODEL": "1"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 0)

    def test_incomplete_session_fails_closed(self) -> None:
        dialogue = self.make_dialogue(env_north={"FAKE_OUTCOME": "error"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 0)

    def test_late_assistant_message_fails_closed(self) -> None:
        # Canary from the design review: a later ACTIVE assistant message
        # outside the invocation window → the turn is rejected and nothing
        # commits.
        dialogue = self.make_dialogue(env_north={"FAKE_LATE_ASSISTANT": "1"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 0)
        self.assertEqual(dialogue.state()["completed_turns"], [])

    def test_followup_user_message_fails_closed(self) -> None:
        dialogue = self.make_dialogue(env_north={"FAKE_FOLLOWUP_USER": "1"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 0)

    def test_message_boundary_recorded_in_proof(self) -> None:
        dialogue = self.make_dialogue()
        runner.launch(dialogue, "hermes-north")
        proof = self.evidence_for(dialogue, 0)["proof"]
        boundary = proof["message_boundary"]
        # user prompt + interim draft + final + inactive ghost tail.
        self.assertEqual(boundary["message_count"], 4)
        self.assertLess(
            boundary["first_message_id"], boundary["final_message_id"]
        )
        # The final id is the max ACTIVE text-bearing assistant row; the
        # inactive ghost tail never counts.
        con = sqlite3.connect(proof["state_db"])
        try:
            row = con.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? "
                "AND role = 'assistant' AND active = 1 "
                "AND content IS NOT NULL AND content != ''",
                (proof["session_id"],),
            ).fetchone()
        finally:
            con.close()
        self.assertEqual(boundary["final_message_id"], row[0])


class HermesOneShotTerminalTests(HermesTestCase):
    """Compatibility when a clean one-shot leaves terminal fields NULL."""

    def test_null_terminal_row_completes(self) -> None:
        dialogue = self.make_dialogue()
        runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 1)
        proof = self.evidence_for(dialogue, 0)["proof"]
        con = sqlite3.connect(proof["state_db"])
        try:
            row = con.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE source = ?",
                (proof["source"],),
            ).fetchone()
        finally:
            con.close()
        self.assertEqual(
            row, (None, None),
            "the fixture must exercise the no-persisted-terminal-state path",
        )
        self.assertIsNone(proof["ended_at"])
        self.assertIsNone(proof["end_reason"])
        self.assertGreaterEqual(proof["api_call_count"], 1)
        self.assertEqual(
            proof["terminal_basis"],
            "process-exit+unique-source+final-active-assistant-message"
            "+positive-api-usage",
            "the proof must name what actually proved completion, not "
            "claim a DB terminal state Hermes never wrote",
        )

    def test_persisted_clean_terminal_still_accepted_and_labeled(self) -> None:
        dialogue = self.make_dialogue(env_north={"FAKE_ENDED": "1"})
        runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 1)
        proof = self.evidence_for(dialogue, 0)["proof"]
        self.assertIsNotNone(proof["ended_at"])
        self.assertEqual(proof["end_reason"], "completed")
        self.assertEqual(proof["terminal_basis"], "state-db-ended")

    def test_zero_api_calls_fail_closed(self) -> None:
        dialogue = self.make_dialogue(env_north={"FAKE_API_CALLS": "0"})
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "hermes-north")
        self.assertEqual(dialogue.state()["turn_index"], 0)


class HermesTerminalFieldMatrixTests(unittest.TestCase):
    """Exact accept/reject matrix for the state.db row one turn produces.

    Accepted: exactly one source match started in the window, one final
    active assistant message, one model/provider, api_call_count > 0,
    AND terminal fields that are either both NULL or consistent-and-clean
    (ended_at inside the window with
    end_reason NULL/''/'completed'). Everything else fails closed.
    """

    SCHEMA = """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT,
        model_config TEXT, started_at REAL NOT NULL, ended_at REAL,
        end_reason TEXT, billing_provider TEXT, profile_name TEXT
    );
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
        timestamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1,
        compacted INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE session_model_usage (
        session_id TEXT NOT NULL, model TEXT NOT NULL,
        billing_provider TEXT NOT NULL DEFAULT '',
        api_call_count INTEGER NOT NULL DEFAULT 0
    );
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "state.db"
        self.before = 1_000_000.0
        self.after = self.before + 30.0

    def build_db(self, *, started_at: float | None = None,
                 ended_at: float | None = None, end_reason: str | None = None,
                 api_calls: int = 3, usage_row: bool = True,
                 final_message: str | None = "The final answer.") -> None:
        started = self.before + 1.0 if started_at is None else started_at
        con = sqlite3.connect(self.db)
        try:
            con.executescript(self.SCHEMA)
            con.execute(
                "INSERT INTO sessions (id, source, model, model_config, "
                "started_at, ended_at, end_reason, billing_provider, "
                "profile_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("sess-1", "src-1", "hermes-4-405b", "{}", started,
                 ended_at, end_reason, "nousresearch", "north"),
            )
            if final_message is not None:
                con.execute(
                    "INSERT INTO messages (session_id, role, content, "
                    "timestamp, active) VALUES (?, ?, ?, ?, ?)",
                    ("sess-1", "assistant", final_message, started + 0.5, 1),
                )
            if usage_row:
                con.execute(
                    "INSERT INTO session_model_usage (session_id, model, "
                    "billing_provider, api_call_count) VALUES (?, ?, ?, ?)",
                    ("sess-1", "hermes-4-405b", "nousresearch", api_calls),
                )
            con.commit()
        finally:
            con.close()

    def observe(self) -> dict:
        return hermes_adapter.HermesAdapter._observe_session(
            self.db, "src-1", self.before, self.after, "actor 'north' (hermes)"
        )

    def assert_refused(self, pattern: str) -> None:
        with self.assertRaisesRegex(adapters.AdapterError, pattern):
            self.observe()

    # -- accepted terminal states -----------------------------------------

    def test_null_null_is_the_observed_success_state(self) -> None:
        self.build_db(ended_at=None, end_reason=None)
        observed = self.observe()
        self.assertIsNone(observed["ended_at"])
        self.assertIsNone(observed["end_reason"])
        self.assertEqual(observed["api_call_count"], 3)
        self.assertEqual(observed["final_message"], "The final answer.")

    def test_ended_with_completed_reason_is_accepted(self) -> None:
        self.build_db(ended_at=self.before + 2.0, end_reason="completed")
        observed = self.observe()
        self.assertEqual(observed["ended_at"], self.before + 2.0)
        self.assertEqual(observed["end_reason"], "completed")

    def test_ended_with_cli_close_reason_is_accepted(self) -> None:
        # Hermes v0.20+ finalizes one-shot (-q/-Q) sessions with
        # end_reason "cli_close" on normal CLI exit; it must count as a
        # clean completion exactly like "completed".
        self.build_db(ended_at=self.before + 2.0, end_reason="cli_close")
        observed = self.observe()
        self.assertEqual(observed["ended_at"], self.before + 2.0)
        self.assertEqual(observed["end_reason"], "cli_close")

    def test_ended_with_unset_reason_is_accepted(self) -> None:
        for reason in (None, ""):
            with self.subTest(reason=reason):
                self.db.unlink(missing_ok=True)
                self.build_db(ended_at=self.before + 2.0, end_reason=reason)
                self.assertEqual(self.observe()["ended_at"], self.before + 2.0)

    # -- rejected terminal states ------------------------------------------

    def test_end_reason_without_ended_at_is_inconsistent(self) -> None:
        self.build_db(ended_at=None, end_reason="completed")
        self.assert_refused("inconsistent")

    def test_dirty_end_reasons_are_rejected(self) -> None:
        for reason in ("error", "canceled", "timeout", "completed\n", " completed"):
            with self.subTest(reason=reason):
                self.db.unlink(missing_ok=True)
                self.build_db(ended_at=self.before + 2.0, end_reason=reason)
                self.assert_refused("not a clean completion")

    def test_dirty_reason_without_ended_at_is_rejected(self) -> None:
        self.build_db(ended_at=None, end_reason="error")
        self.assert_refused("inconsistent")

    def test_ended_before_started_is_inconsistent(self) -> None:
        self.build_db(ended_at=self.before - 5.0)
        self.assert_refused("inconsistent")

    def test_ended_far_after_window_is_inconsistent(self) -> None:
        self.build_db(ended_at=self.after + 120.0)
        self.assert_refused("inconsistent")

    # -- rejected usage / message / window states ---------------------------

    def test_zero_api_call_count_is_rejected(self) -> None:
        self.build_db(api_calls=0)
        self.assert_refused("no API calls")

    def test_missing_usage_row_is_rejected(self) -> None:
        self.build_db(usage_row=False)
        self.assert_refused("no API calls")

    def test_missing_final_active_message_is_rejected(self) -> None:
        self.build_db(final_message=None)
        self.assert_refused("no final active")

    def test_start_outside_window_is_rejected_even_with_null_terminal(self) -> None:
        self.build_db(started_at=self.before - 120.0)
        self.assert_refused("outside this")

    # -- message boundary -------------------------------------------------

    def add_message_row(self, role: str, content: str, timestamp: float,
                        active: int = 1) -> None:
        con = sqlite3.connect(self.db)
        try:
            con.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, "
                "active) VALUES ('sess-1', ?, ?, ?, ?)",
                (role, content, timestamp, active),
            )
            con.commit()
        finally:
            con.close()

    def test_message_outside_invocation_window_is_rejected(self) -> None:
        self.build_db()
        self.add_message_row("assistant", "late write", self.after + 120.0)
        self.assert_refused("outside this invocation window")

    def test_user_message_after_final_assistant_is_rejected(self) -> None:
        self.build_db()
        self.add_message_row("user", "follow-up prompt", self.before + 2.0)
        self.assert_refused("message boundary")

    def test_message_without_usable_timestamp_is_rejected(self) -> None:
        self.build_db()
        # REAL affinity stores unconvertible text as-is; float() then
        # fails at observation time.
        self.add_message_row("assistant", "x", "not-a-number")  # type: ignore[arg-type]
        self.assert_refused("no usable timestamp")

    def test_message_boundary_is_recorded(self) -> None:
        self.build_db()
        boundary = self.observe()["message_boundary"]
        self.assertEqual(boundary["message_count"], 1)
        self.assertEqual(
            boundary["first_message_id"], boundary["final_message_id"]
        )


class HermesUsageTaskFilterTests(HermesTerminalFieldMatrixTests):
    """Auxiliary usage rows (title generation, vision, compression, ...)
    share session_model_usage with the main turn under a non-empty task;
    they must not pollute the actor's model/provider identity."""

    SCHEMA = """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT,
        model_config TEXT, started_at REAL NOT NULL, ended_at REAL,
        end_reason TEXT, billing_provider TEXT, profile_name TEXT
    );
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
        timestamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1,
        compacted INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE session_model_usage (
        session_id TEXT NOT NULL, model TEXT NOT NULL,
        billing_provider TEXT NOT NULL DEFAULT '',
        api_call_count INTEGER NOT NULL DEFAULT 0,
        task TEXT DEFAULT ''
    );
    """

    def add_usage_row(self, model: str, provider: str, calls: int,
                      task: str) -> None:
        con = sqlite3.connect(self.db)
        try:
            con.execute(
                "INSERT INTO session_model_usage (session_id, model, "
                "billing_provider, api_call_count, task) "
                "VALUES (?, ?, ?, ?, ?)",
                ("sess-1", model, provider, calls, task),
            )
            con.commit()
        finally:
            con.close()

    def test_auxiliary_task_rows_do_not_pollute_identity(self) -> None:
        self.build_db()
        self.add_usage_row("kimi-k2.7-code", "kimi", 1, "title_generation")
        observed = self.observe()
        self.assertEqual(observed["models"], ["hermes-4-405b"])
        self.assertEqual(observed["api_call_count"], 3)

    def test_null_task_row_counts_as_main_conversation(self) -> None:
        self.build_db()
        con = sqlite3.connect(self.db)
        try:
            con.execute(
                "INSERT INTO session_model_usage (session_id, model, "
                "billing_provider, api_call_count, task) "
                "VALUES ('sess-1', 'hermes-4-405b', 'nousresearch', 1, NULL)"
            )
            con.commit()
        finally:
            con.close()
        self.assertEqual(self.observe()["api_call_count"], 4)

    def test_two_main_task_models_are_still_rejected(self) -> None:
        self.build_db()
        self.add_usage_row("other-model", "other-provider", 1, "")
        self.assert_refused("not exactly one model")


class HermesUsageOddlyCasedTaskColumnTests(HermesUsageTaskFilterTests):
    """SQLite resolves identifiers case-insensitively; a schema declaring
    the column as ``Task`` must still trigger the main-conversation
    filter instead of silently reverting to all-rows counting."""

    SCHEMA = HermesUsageTaskFilterTests.SCHEMA.replace(
        "task TEXT DEFAULT ''", '"Task" TEXT DEFAULT \'\''
    )


class CommandFixtureBase(unittest.TestCase):
    """Shared command-adapter dialogue fixture (worker/verifier fakes,
    definition builder, dialogue factory); no test methods itself."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self.marker = self.base / "marker.log"

    def worker_argv(self) -> list[str]:
        return [
            WORKER,
            "--task", "{task_file}",
            "--turn-output", "{turn_file}",
            "--round", "{round_id}",
            "--actor", "{actor_id}",
        ]

    def verifier_argv(self) -> list[str]:
        return [
            VERIFIER,
            "--turn", "{turn_file}",
            "--round", "{round_id}",
            "--actor", "{actor_id}",
        ]

    def definition_raw(self, settings_a: dict) -> dict:
        settings_b = {
            "argv": self.worker_argv(),
            "identity_verifier_argv": self.verifier_argv(),
            "env": {
                "FAKE_PROVIDER": "fake-provider-b",
                "FAKE_MODEL": "fake-model-b",
                "FAKE_SPAWN_MARKER": str(self.marker),
            },
        }
        return {
            "protocol_id": "command-identity",
            "version": 1,
            "owner": "owner-human",
            "actors": [
                {
                    "actor_id": "worker-a",
                    "role": "proposer",
                    "transport": "command",
                    "expected_provider": "fake-provider-a",
                    "expected_model": "fake-model-a",
                    "settings": settings_a,
                },
                {
                    "actor_id": "worker-b",
                    "role": "challenger",
                    "transport": "command",
                    "expected_provider": "fake-provider-b",
                    "expected_model": "fake-model-b",
                    "settings": settings_b,
                },
            ],
            "schedule": [
                {"round_id": "R01", "actor_id": "worker-a", "purpose": "p", "artifact_kind": "proposal"},
                {"round_id": "R02", "actor_id": "worker-b", "purpose": "c", "artifact_kind": "challenge"},
            ],
            "final_round_id": "R02",
        }

    def make_dialogue(self, settings_a: dict) -> engine.Dialogue:
        definition = config.parse_definition(self.definition_raw(settings_a))
        return engine.init_dialogue(definition, self.base / "dialogue")


class CommandIdentityTests(CommandFixtureBase):
    """Identity acceptance/refusal for the command transport."""

    def test_command_without_external_verifier_fails_closed(self) -> None:
        # Self-reported output is not identity proof: with no external
        # identity verifier configured, identity-sensitive completion is
        # refused before any process starts.
        settings = {
            "argv": self.worker_argv(),
            "env": {
                "FAKE_PROVIDER": "fake-provider-a",
                "FAKE_MODEL": "fake-model-a",
                "FAKE_SPAWN_MARKER": str(self.marker),
            },
        }
        dialogue = self.make_dialogue(settings)
        with self.assertRaises((engine.ProtocolError, adapters.AdapterError)) as ctx:
            runner.launch(dialogue, "worker-a")
        self.assertIn("identity", str(ctx.exception).lower())
        self.assertEqual(dialogue.state()["turn_index"], 0)
        self.assertFalse(self.marker.exists(), "no worker process may start")
        with self.assertRaises((engine.ProtocolError, adapters.AdapterError)):
            runner.dry_run(dialogue, "worker-a")

    def test_external_verifier_builds_untrusted_labelled_evidence(self) -> None:
        settings = {
            "argv": self.worker_argv(),
            "identity_verifier_argv": self.verifier_argv(),
            "env": {
                "FAKE_PROVIDER": "fake-provider-a",
                "FAKE_MODEL": "fake-model-a",
                "FAKE_SPAWN_MARKER": str(self.marker),
            },
        }
        dialogue = self.make_dialogue(settings)
        result = runner.launch(dialogue, "worker-a")
        self.assertTrue(result["executed"])
        record = dialogue.state()["completed_turns"][0]
        evidence_record = json.loads(
            (dialogue.directory / record["evidence_file"]).read_text(encoding="utf-8")
        )
        proof = evidence_record["proof"]
        self.assertEqual(proof["kind"], "external-command-verifier")
        self.assertEqual(proof["verifier_argv"][0], VERIFIER)
        # The fake verifier reports itself as a fake; the proof must keep
        # that honesty instead of upgrading it to real identity proof.
        self.assertTrue(proof["report"].get("fake"))
        self.assertEqual(evidence_record["provider"], "fake-provider-a")
        self.assertEqual(evidence_record["model"], "fake-model-a")
        # Both worker and verifier actually ran.
        marks = self.marker.read_text(encoding="utf-8").splitlines()
        self.assertEqual(marks, ["worker", "verifier"])

    def test_verifier_identity_mismatch_fails_closed(self) -> None:
        settings = {
            "argv": self.worker_argv(),
            "identity_verifier_argv": self.verifier_argv(),
            "env": {
                "FAKE_PROVIDER": "fake-provider-a",
                "FAKE_MODEL": "impersonated-model",
                "FAKE_SPAWN_MARKER": str(self.marker),
            },
        }
        dialogue = self.make_dialogue(settings)
        with self.assertRaises(engine.ProtocolError) as ctx:
            runner.launch(dialogue, "worker-a")
        self.assertIn("model", str(ctx.exception))
        self.assertEqual(dialogue.state()["turn_index"], 0)


class CliVersionEvidenceTests(HermesTestCase):
    """Accepted-turn evidence records the engine-probed adapter CLI
    version (verbatim output + hash) — the README ties adapter behavior
    to the exact installed CLI versions, so the record must name them."""

    def _version_stub(self, name: str, version_line: str,
                      fail: bool = False) -> str:
        stub = self.base / name
        if fail:
            body = (
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then exit 1; fi\n'
                f'exec "{HERMES}" "$@"\n'
            )
        else:
            body = (
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then\n'
                f'    echo "{version_line}"\n'
                "    exit 0\n"
                "fi\n"
                f'exec "{HERMES}" "$@"\n'
            )
        stub.write_text(body, encoding="utf-8")
        os.chmod(stub, 0o755)
        return str(stub)

    def _dialogue_with_command(self, command_name: str,
                               directory: str) -> engine.Dialogue:
        raw = self.definition_raw()
        raw["actors"][0]["settings"]["command_name"] = command_name
        definition = config.parse_definition(raw)
        return engine.init_dialogue(definition, self.base / directory)

    def test_hermes_turn_records_probed_cli_version(self) -> None:
        dialogue = self.make_dialogue()
        runner.launch(dialogue, "hermes-north")
        cli = self.evidence_for(dialogue, 0)["cli_version"]
        self.assertEqual(cli["argv"], [HERMES, "--version"])
        self.assertEqual(cli["exit_status"], 0)
        self.assertEqual(cli["output"], "fake-hermes 1.1.0 (fixture)")
        self.assertEqual(
            cli["output_sha256"],
            hashlib.sha256(cli["output"].encode("utf-8")).hexdigest(),
        )
        self.assertNotIn("error", cli)

    def test_two_cli_versions_produce_distinct_evidence(self) -> None:
        # The canary shape: same turn under two stubbed CLI versions
        # yields different version records; both turns still validate.
        stub_a = self._version_stub("hermes-a", "fake-hermes 1.0.0 (canary-a)")
        stub_b = self._version_stub("hermes-b", "fake-hermes 2.0.0 (canary-b)")
        dialogue_a = self._dialogue_with_command(stub_a, "dialogue-a")
        dialogue_b = self._dialogue_with_command(stub_b, "dialogue-b")
        runner.launch(dialogue_a, "hermes-north")
        runner.launch(dialogue_b, "hermes-north")
        out_a = self.evidence_for(dialogue_a, 0)["cli_version"]["output"]
        out_b = self.evidence_for(dialogue_b, 0)["cli_version"]["output"]
        self.assertEqual(out_a, "fake-hermes 1.0.0 (canary-a)")
        self.assertEqual(out_b, "fake-hermes 2.0.0 (canary-b)")
        self.assertNotEqual(out_a, out_b)
        self.assertEqual(dialogue_a.state()["turn_index"], 1)
        self.assertEqual(dialogue_b.state()["turn_index"], 1)

    def test_failed_probe_is_recorded_not_fatal(self) -> None:
        stub = self._version_stub("hermes-broken-version", "", fail=True)
        dialogue = self._dialogue_with_command(stub, "dialogue-broken")
        runner.launch(dialogue, "hermes-north")
        cli = self.evidence_for(dialogue, 0)["cli_version"]
        self.assertIn("error", cli)
        self.assertEqual(cli["exit_status"], 1)
        self.assertEqual(dialogue.state()["turn_index"], 1)

    def test_long_output_hash_covers_full_untruncated_output(self) -> None:
        # output_sha256 attests what the CLI actually printed; the stored
        # output is a 500-char prefix flagged by output_truncated.
        version_line = "fake-hermes " + ("9" * 600) + " (long)"
        stub = self._version_stub("hermes-long-version", version_line)
        dialogue = self._dialogue_with_command(stub, "dialogue-long")
        runner.launch(dialogue, "hermes-north")
        cli = self.evidence_for(dialogue, 0)["cli_version"]
        self.assertEqual(cli["output"], version_line[:500])
        self.assertTrue(cli["output_truncated"])
        self.assertEqual(
            cli["output_sha256"],
            hashlib.sha256(version_line.encode("utf-8")).hexdigest(),
        )

    def test_probe_argv_hook_error_is_recorded_not_fatal(self) -> None:
        # A subclass hook raising AdapterError (bad settings/env) must
        # degrade to a recorded error, never fail the accepted turn.
        dialogue = self.make_dialogue()
        with mock.patch.object(
            hermes_adapter.HermesAdapter,
            "version_probe_argv",
            side_effect=adapters.AdapterError("settings broke"),
        ):
            runner.launch(dialogue, "hermes-north")
        cli = self.evidence_for(dialogue, 0)["cli_version"]
        self.assertIn("settings broke", cli["error"])
        self.assertNotIn("argv", cli)
        self.assertEqual(dialogue.state()["turn_index"], 1)

    def test_probe_runs_under_actor_settings_env(self) -> None:
        # PATH- or env-dependent CLIs must probe the binary the turn used:
        # the probe inherits the actor's substituted settings env.
        stub = self.base / "hermes-env-version"
        stub.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            '    echo "fake-hermes ${FAKE_VERSION_TAG:-unset} (env)"\n'
            "    exit 0\n"
            "fi\n"
            f'exec "{HERMES}" "$@"\n',
            encoding="utf-8",
        )
        os.chmod(stub, 0o755)
        raw = self.definition_raw({"FAKE_VERSION_TAG": "from-settings-env"})
        raw["actors"][0]["settings"]["command_name"] = str(stub)
        definition = config.parse_definition(raw)
        dialogue = engine.init_dialogue(definition, self.base / "dialogue-env")
        runner.launch(dialogue, "hermes-north")
        cli = self.evidence_for(dialogue, 0)["cli_version"]
        self.assertEqual(cli["output"], "fake-hermes from-settings-env (env)")


class FableCliVersionTests(FableTestCase):
    def test_fable_turn_records_probed_cli_version(self) -> None:
        dialogue = self.make_dialogue()
        runner.launch(dialogue, "fable-a")
        record = dialogue.state()["completed_turns"][0]
        evidence_record = json.loads(
            (dialogue.directory / record["evidence_file"]).read_text(
                encoding="utf-8"
            )
        )
        cli = evidence_record["cli_version"]
        self.assertEqual(cli["argv"], [FABLE, "--version"])
        self.assertEqual(cli["exit_status"], 0)
        self.assertEqual(cli["output"], "fake-fable-session 0.3.0b1 (fixture)")


class CommandCliVersionTests(CommandFixtureBase):
    def _settings(self, argv0: str) -> dict:
        argv = [argv0, *self.worker_argv()[1:]]
        return {
            "argv": argv,
            "identity_verifier_argv": self.verifier_argv(),
            "env": {
                "FAKE_PROVIDER": "fake-provider-a",
                "FAKE_MODEL": "fake-model-a",
                "FAKE_SPAWN_MARKER": str(self.marker),
            },
        }

    def _read_evidence(self, dialogue: engine.Dialogue) -> dict:
        record = dialogue.state()["completed_turns"][0]
        return json.loads(
            (dialogue.directory / record["evidence_file"]).read_text(
                encoding="utf-8"
            )
        )

    def test_worker_version_recorded(self) -> None:
        dialogue = self.make_dialogue(self._settings(WORKER))
        runner.launch(dialogue, "worker-a")
        cli = self._read_evidence(dialogue)["cli_version"]
        self.assertEqual(cli["argv"], [WORKER, "--version"])
        self.assertEqual(cli["exit_status"], 0)
        self.assertEqual(cli["output"], "fake-worker 1.0.0 (fixture)")

    def test_worker_without_version_flag_is_recorded_not_fatal(self) -> None:
        wrapper = self.base / "worker-no-version"
        wrapper.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then exit 1; fi\n'
            f'exec "{WORKER}" "$@"\n',
            encoding="utf-8",
        )
        os.chmod(wrapper, 0o755)
        dialogue = self.make_dialogue(self._settings(str(wrapper)))
        runner.launch(dialogue, "worker-a")
        cli = self._read_evidence(dialogue)["cli_version"]
        self.assertIn("error", cli)
        self.assertEqual(cli["exit_status"], 1)
        self.assertEqual(dialogue.state()["turn_index"], 1)


if __name__ == "__main__":
    unittest.main()
