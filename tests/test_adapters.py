"""Task 4: adapter boundary — packets, dry-run default, one-turn launch."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import adapters, config, engine, runner


def _actor(transport: str, settings: dict) -> config.Actor:
    return config.Actor(
        actor_id="worker-a",
        role="proposer",
        transport=transport,
        expected_provider="prov",
        expected_model="model",
        settings=settings,
    )


def _turn() -> config.TurnSpec:
    return config.TurnSpec(
        round_id="R01", actor_id="worker-a", purpose="propose", artifact_kind="proposal"
    )


def _context(actor: config.Actor, base: Path) -> adapters.PrepareContext:
    return adapters.PrepareContext(
        actor=actor,
        turn=_turn(),
        dialogue_dir=base / "dialogue",
        work_dir=base / "work",
        task_file=base / "work" / "task.md",
        turn_file=base / "work" / "turn.md",
        evidence_file=base / "work" / "evidence.json",
    )


class AdapterRegistryTests(unittest.TestCase):
    def test_transport_selects_adapter(self) -> None:
        self.assertEqual(adapters.get_adapter("command").name, "command")
        self.assertEqual(adapters.get_adapter("fable-session").name, "claude-fable")
        self.assertEqual(adapters.get_adapter("hermes-cli").name, "hermes")

    def test_unknown_transport_rejected(self) -> None:
        with self.assertRaises(adapters.AdapterError):
            adapters.get_adapter("telepathy")

    def test_role_text_never_selects_adapter(self) -> None:
        # An actor whose role mentions Claude still uses its declared transport.
        actor = config.Actor(
            actor_id="x",
            role="claude-code reviewer",
            transport="command",
            expected_provider="p",
            expected_model="m",
            settings={"argv": ["fake-worker"], "identity_verifier_argv": ["fake-verifier"]},
        )
        self.assertEqual(adapters.adapter_for(actor).name, "command")


class HermesSubstituteIsolationTests(unittest.TestCase):
    def definition_with_homes(self, primary_home: Path, substitute_home: Path):
        raw = support.two_actor_definition()
        raw["actors"][1]["role"] = raw["actors"][0]["role"]
        raw["actors"][0].update(
            transport="hermes-cli",
            settings={"command_name": "hermes", "hermes_home": str(primary_home)},
        )
        raw["actors"][1].update(
            transport="hermes-cli",
            settings={"command_name": "hermes", "hermes_home": str(substitute_home)},
        )
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        return config.parse_definition(raw)

    def test_init_rejects_symlinked_profile_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_home = base / "profile-real"
            real_home.mkdir()
            alias_home = base / "profile-alias"
            alias_home.symlink_to(real_home, target_is_directory=True)
            definition = self.definition_with_homes(real_home, alias_home)
            repo = support.init_git_repo(base / "repo")
            with self.assertRaises(engine.ProtocolError) as ctx:
                engine.init_dialogue(definition, repo / "dialogue")
            self.assertIn("same HERMES_HOME", str(ctx.exception))

    def test_engine_rechecks_alias_drift_at_validation_and_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            primary_home = base / "profile-primary"
            substitute_home = base / "profile-substitute"
            primary_home.mkdir()
            substitute_home.mkdir()
            definition = self.definition_with_homes(primary_home, substitute_home)
            repo = support.init_git_repo(base / "repo")
            dialogue = engine.init_dialogue(definition, repo / "dialogue")
            substitute_home.rmdir()
            substitute_home.symlink_to(primary_home, target_is_directory=True)
            report = dialogue.validate()
            self.assertFalse(report["ok"])
            self.assertTrue(
                any("same HERMES_HOME" in item for item in report["errors"]),
                report["errors"],
            )
            with self.assertRaises(engine.ProtocolError):
                dialogue.claim(
                    "worker-b", substitution_reason="provider_cooldown"
                )

    def test_engine_rechecks_alias_drift_before_direct_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            primary_home = base / "profile-primary"
            substitute_home = base / "profile-substitute"
            primary_home.mkdir()
            substitute_home.mkdir()
            definition = self.definition_with_homes(primary_home, substitute_home)
            repo = support.init_git_repo(base / "repo")
            dialogue = engine.init_dialogue(definition, repo / "dialogue")
            dialogue.claim("worker-b", substitution_reason="provider_cooldown")
            substitute_home.rmdir()
            substitute_home.symlink_to(primary_home, target_is_directory=True)
            with self.assertRaises(engine.ProtocolError) as ctx:
                dialogue.complete(
                    "worker-b", base / "turn.md", base / "evidence.json"
                )
            self.assertIn("same HERMES_HOME", str(ctx.exception))

    def test_engine_rechecks_alias_drift_before_primary_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            primary_home = base / "profile-primary"
            substitute_home = base / "profile-substitute"
            primary_home.mkdir()
            substitute_home.mkdir()
            definition = self.definition_with_homes(primary_home, substitute_home)
            repo = support.init_git_repo(base / "repo")
            dialogue = engine.init_dialogue(definition, repo / "dialogue")
            substitute_home.rmdir()
            substitute_home.symlink_to(primary_home, target_is_directory=True)
            with self.assertRaises(engine.ProtocolError) as ctx:
                dialogue.claim("worker-a")
            self.assertIn("same HERMES_HOME", str(ctx.exception))

    def test_engine_rechecks_alias_drift_before_primary_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            primary_home = base / "profile-primary"
            substitute_home = base / "profile-substitute"
            primary_home.mkdir()
            substitute_home.mkdir()
            definition = self.definition_with_homes(primary_home, substitute_home)
            repo = support.init_git_repo(base / "repo")
            dialogue = engine.init_dialogue(definition, repo / "dialogue")
            dialogue.claim("worker-a")
            substitute_home.rmdir()
            substitute_home.symlink_to(primary_home, target_is_directory=True)
            with self.assertRaises(engine.ProtocolError) as ctx:
                dialogue.complete(
                    "worker-a", base / "turn.md", base / "evidence.json"
                )
            self.assertIn("same HERMES_HOME", str(ctx.exception))


class CommandAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def command_settings(self, argv: list[str], **extra) -> dict:
        settings = {
            "argv": argv,
            "identity_verifier_argv": ["fake-verifier", "--turn", "{turn_file}"],
        }
        settings.update(extra)
        return settings

    def test_substitutes_placeholders(self) -> None:
        actor = _actor(
            "command",
            self.command_settings(
                ["fake-worker", "--task", "{task_file}", "--round", "{round_id}"],
                env={"WORKER_ACTOR": "{actor_id}"},
            ),
        )
        packet = adapters.adapter_for(actor).prepare(_context(actor, self.base))
        self.assertEqual(packet.argv[0], "fake-worker")
        self.assertIn(str(self.base / "work" / "task.md"), packet.argv)
        self.assertIn("R01", packet.argv)
        self.assertEqual(packet.env["WORKER_ACTOR"], "worker-a")
        self.assertEqual(packet.adapter, "command")
        # The external verifier is part of the declared lifecycle.
        verifier_argv = packet.as_dict()["lifecycle"][0]["argv"]
        self.assertEqual(verifier_argv[0], "fake-verifier")
        self.assertIn(str(self.base / "work" / "turn.md"), verifier_argv)

    def test_requires_argv(self) -> None:
        actor = _actor("command", {"identity_verifier_argv": ["fake-verifier"]})
        with self.assertRaises(adapters.AdapterError):
            adapters.adapter_for(actor).prepare(_context(actor, self.base))

    def test_requires_external_identity_verifier(self) -> None:
        # Self-reported worker output is not identity proof; without an
        # external verifier the command transport fails closed.
        actor = _actor("command", {"argv": ["fake-worker"]})
        with self.assertRaises(adapters.AdapterError) as ctx:
            adapters.adapter_for(actor).prepare(_context(actor, self.base))
        self.assertIn("identity", str(ctx.exception).lower())

    def test_rejects_unknown_placeholder(self) -> None:
        actor = _actor("command", self.command_settings(["fake-worker", "{secret_token}"]))
        with self.assertRaises(adapters.AdapterError):
            adapters.adapter_for(actor).prepare(_context(actor, self.base))


class ClaudeFableAdapterTests(unittest.TestCase):
    """Packet-shape checks; the full real lifecycle is proven in
    test_real_contracts.py against the contract-faithful fake."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def settings(self) -> dict:
        return {
            "command_name": "fake-fable-session",
            "project": "worker-a-project",
            "registry": "fable/registry.toml",
            "state_dir": "fable/state",
            "tmux_prefix": "madp-",
        }

    def test_prepares_real_fable_session_packet(self) -> None:
        actor = _actor("fable-session", self.settings())
        packet = adapters.adapter_for(actor).prepare(_context(actor, self.base))
        argv = list(packet.argv)
        self.assertEqual(argv[:2], ["fake-fable-session", "run"])
        for flag in ("--project", "--task", "--registry", "--state-dir", "--launch", "--tmux"):
            self.assertIn(flag, argv)
        self.assertEqual(packet.adapter, "claude-fable")
        # Relative registry/state-dir resolve against the dialogue directory.
        registry_arg = argv[argv.index("--registry") + 1]
        self.assertEqual(
            registry_arg, str((self.base / "dialogue" / "fable/registry.toml").absolute())
        )
        # No invented flags.
        for invented in ("--profile", "--turn-output", "--evidence-output", "--round", "--actor"):
            self.assertNotIn(invented, argv)

    def test_requires_project(self) -> None:
        settings = self.settings()
        del settings["project"]
        actor = _actor("fable-session", settings)
        with self.assertRaises(adapters.AdapterError) as ctx:
            adapters.adapter_for(actor).prepare(_context(actor, self.base))
        self.assertIn("project", str(ctx.exception))


class HermesAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_prepares_hermes_chat_packet_with_home(self) -> None:
        actor = _actor(
            "hermes-cli",
            {"command_name": "fake-hermes", "hermes_home": "hermes-homes/north"},
        )
        packet = adapters.adapter_for(actor).prepare(_context(actor, self.base))
        argv = list(packet.argv)
        self.assertEqual(argv[:2], ["fake-hermes", "chat"])
        for flag in ("-q", "-Q", "--source", "--pass-session-id"):
            self.assertIn(flag, argv)
        expected_home = str((self.base / "dialogue" / "hermes-homes/north").resolve())
        self.assertEqual(packet.env["HERMES_HOME"], expected_home)

    def test_requires_explicit_hermes_home(self) -> None:
        # Even a role literally named after a profile must not infer a home.
        actor = config.Actor(
            actor_id="hermes-north",
            role="hermes-north",
            transport="hermes-cli",
            expected_provider="p",
            expected_model="m",
            settings={"command_name": "fake-hermes"},
        )
        with self.assertRaises(adapters.AdapterError) as ctx:
            adapters.adapter_for(actor).prepare(_context(actor, self.base))
        self.assertIn("hermes_home", str(ctx.exception))


class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self.marker = self.base / "spawned.log"
        raw = support.two_actor_definition()
        for actor in raw["actors"]:
            actor["settings"] = support.command_worker_settings(
                self.marker, actor["expected_provider"], actor["expected_model"]
            )
        self.definition = config.parse_definition(raw)
        self.dialogue = engine.init_dialogue(self.definition, self.base / "dialogue")

    def spawn_count(self) -> int:
        """Worker processes started (the external verifier is not a worker)."""
        if not self.marker.exists():
            return 0
        lines = self.marker.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line == "worker")


class DryRunTests(RunnerTestCase):
    def test_dry_run_starts_no_process_and_changes_nothing(self) -> None:
        before = self.dialogue.state()
        result = runner.dry_run(self.dialogue, "worker-a")
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["packet"]["argv"][0], str(support.FAKE_BIN / "fake-worker"))
        self.assertEqual(self.spawn_count(), 0)
        after = self.dialogue.state()
        self.assertEqual(before["revision"], after["revision"])
        self.assertEqual(before["status"], after["status"])
        self.assertFalse((self.dialogue.directory / "work").exists())
        self.assertFalse(self.dialogue.lock_path.exists())

    def test_dry_run_rejects_wrong_actor(self) -> None:
        with self.assertRaises(engine.ProtocolError):
            runner.dry_run(self.dialogue, "worker-b")


class LaunchTests(RunnerTestCase):
    def substitute_dialogue(self) -> engine.Dialogue:
        raw = support.two_actor_definition()
        for actor in raw["actors"]:
            actor["settings"] = support.command_worker_settings(
                self.marker, actor["expected_provider"], actor["expected_model"]
            )
        raw["actors"].append(
            {
                "actor_id": "worker-c",
                "role": "proposer",
                "transport": "command",
                "expected_provider": "fake-provider-c",
                "expected_model": "fake-model-c",
                "settings": support.command_worker_settings(
                    self.marker, "fake-provider-c", "fake-model-c"
                ),
            }
        )
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-c"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        definition = config.parse_definition(raw)
        return engine.init_dialogue(definition, self.base / "dialogue-substitute")

    def test_launch_preapproved_substitute_preserves_actual_identity(self) -> None:
        dialogue = self.substitute_dialogue()
        dry = runner.dry_run(
            dialogue, "worker-c", substitution_reason="provider_cooldown"
        )
        self.assertEqual(dry["actor_id"], "worker-c")
        self.assertEqual(dry["scheduled_actor_id"], "worker-a")
        self.assertEqual(dry["actor_selection"], "substitute")
        self.assertEqual(dry["substitution_reason"], "provider_cooldown")
        prepared_path = self.base / "substitute-task.md"
        prepared = runner.prepare(
            dialogue,
            "worker-c",
            prepared_path,
            substitution_reason="provider_cooldown",
        )
        self.assertEqual(prepared["substitution_reason"], "provider_cooldown")
        self.assertIn(
            "substitution_reason: provider_cooldown",
            prepared_path.read_text(encoding="utf-8"),
        )

        result = runner.launch(
            dialogue, "worker-c", substitution_reason="provider_cooldown"
        )
        self.assertEqual(result["actor_id"], "worker-c")
        self.assertEqual(result["scheduled_actor_id"], "worker-a")
        self.assertEqual(result["actor_selection"], "substitute")
        record = dialogue.state()["completed_turns"][0]
        self.assertEqual(record["actor_id"], "worker-c")
        self.assertEqual(record["scheduled_actor_id"], "worker-a")
        self.assertEqual(record["actor_selection"], "substitute")
        self.assertEqual(record["substitution_reason"], "provider_cooldown")
        self.assertEqual(record["artifact_file"], "turns/R01-worker-c.md")
        runtime_record = json.loads(
            (dialogue.directory / record["evidence_file"]).read_text(encoding="utf-8")
        )
        self.assertEqual(runtime_record["actor_id"], "worker-c")
        self.assertEqual(runtime_record["scheduled_actor_id"], "worker-a")
        self.assertEqual(runtime_record["actor_selection"], "substitute")
        self.assertEqual(
            runtime_record["substitution_reason"], "provider_cooldown"
        )
        task = dialogue.directory / "work" / "R01" / "task.md"
        briefing = task.read_text(encoding="utf-8")
        self.assertIn("actor_id: worker-c", briefing)
        self.assertIn("scheduled_actor_id: worker-a", briefing)
        self.assertIn("actor_selection: substitute", briefing)
        self.assertIn("substitution_reason: provider_cooldown", briefing)
        self.assertIn("never write as, claim to be, or impersonate", briefing)
        report = dialogue.validate(
            require_git=True, require_runner_completion=True
        )
        self.assertTrue(report["ok"], report["errors"])
        proven = report["provenance"]["turn_commits"][0]
        self.assertEqual(proven["actor_id"], "worker-c")
        self.assertEqual(proven["scheduled_actor_id"], "worker-a")
        self.assertEqual(proven["actor_selection"], "substitute")

    def test_launch_executes_exactly_one_turn(self) -> None:
        result = runner.launch(self.dialogue, "worker-a")
        self.assertTrue(result["executed"])
        self.assertEqual(result["completed_round"], "R01")
        self.assertEqual(self.spawn_count(), 1)
        state = self.dialogue.state()
        self.assertEqual(state["turn_index"], 1)
        self.assertEqual(state["status"], "OPEN")
        # The same launch call never runs a second turn; the next turn
        # belongs to worker-b and a repeat launch by worker-a fails closed.
        with self.assertRaises(engine.ProtocolError):
            runner.launch(self.dialogue, "worker-a")
        self.assertEqual(self.spawn_count(), 1)

    def test_launch_failure_does_not_advance(self) -> None:
        # Re-init with failing worker-a.
        dialogue_dir = self.base / "dialogue-fail"
        raw = support.two_actor_definition()
        raw["actors"][0]["settings"] = support.command_worker_settings(
            None, "fake-provider-a", "fake-model-a", {"FAKE_EXIT": "3"}
        )
        definition = config.parse_definition(raw)
        dialogue = engine.init_dialogue(definition, dialogue_dir)
        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "worker-a")
        state = dialogue.state()
        self.assertEqual(state["turn_index"], 0)
        self.assertIsNone(state["claim"])
        self.assertFalse(dialogue.lock_path.exists())

    def test_launch_failed_outcome_evidence_does_not_advance(self) -> None:
        # Worker exits 0 but the external verifier reports a failed
        # terminal outcome: still blocked.
        raw = support.two_actor_definition()
        raw["actors"][0]["settings"] = support.command_worker_settings(
            None, "fake-provider-a", "fake-model-a",
            {"FAKE_VERIFIER_OUTCOME": "failure"},
        )
        definition = config.parse_definition(raw)
        dialogue = engine.init_dialogue(definition, self.base / "dialogue-outcome")
        with self.assertRaises(engine.ProtocolError) as ctx:
            runner.launch(dialogue, "worker-a")
        self.assertIn("outcome", str(ctx.exception))
        self.assertEqual(dialogue.state()["turn_index"], 0)

    def test_launch_with_impersonating_runtime_does_not_advance(self) -> None:
        # The external verifier observes a model that violates the actor
        # constraint.
        raw = support.two_actor_definition()
        raw["actors"][0]["settings"] = support.command_worker_settings(
            None, "fake-provider-a", "some-other-model"
        )
        definition = config.parse_definition(raw)
        dialogue = engine.init_dialogue(definition, self.base / "dialogue-imposter")
        with self.assertRaises(engine.ProtocolError) as ctx:
            runner.launch(dialogue, "worker-a")
        self.assertIn("model", str(ctx.exception))
        self.assertEqual(dialogue.state()["turn_index"], 0)


class PriorTurnBriefingTests(RunnerTestCase):
    """RED → GREEN: a later actor's briefing must hand the worker USABLE
    prior-turn context — absolute artifact and evidence paths rooted in
    the dialogue directory, their immutable digests, and an explicit
    requirement to read every prior turn before producing the reply.
    A bare relative name like ``turns/R01-worker-a.md`` is unusable from
    the worker's own working directory."""

    def r02_briefing(self) -> tuple[str, dict]:
        runner.launch(self.dialogue, "worker-a")
        output = self.base / "R02-TASK.md"
        runner.prepare(self.dialogue, "worker-b", output)
        record = self.dialogue.state()["completed_turns"][0]
        return output.read_text(encoding="utf-8"), record

    def test_r02_briefing_names_prior_turn_and_evidence_with_absolute_paths(self) -> None:
        text, record = self.r02_briefing()
        turn_abs = self.dialogue.directory.absolute() / record["artifact_file"]
        evidence_abs = self.dialogue.directory.absolute() / record["evidence_file"]
        self.assertTrue(turn_abs.is_absolute())
        self.assertIn(
            str(turn_abs), text,
            "the prior turn must be identified by an absolute path rooted "
            "in the dialogue directory, not a bare relative name",
        )
        self.assertIn(
            str(evidence_abs), text,
            "the prior turn's evidence record must be identified by an "
            "absolute path rooted in the dialogue directory",
        )

    def test_r02_briefing_binds_prior_artifacts_with_immutable_digests(self) -> None:
        text, record = self.r02_briefing()
        self.assertIn(record["artifact_sha256"], text)
        self.assertIn(
            record["evidence_sha256"], text,
            "the evidence record's digest must bind what the worker reads",
        )

    def test_r02_briefing_requires_reading_prior_turns_before_producing(self) -> None:
        text, _ = self.r02_briefing()
        self.assertRegex(
            text,
            r"(?is)read\b.*\bbefore producing",
            "the briefing must explicitly require reading each prior turn "
            "before producing the challenge or response",
        )

    def test_first_turn_briefing_states_there_is_nothing_to_read(self) -> None:
        output = self.base / "R01-TASK.md"
        runner.prepare(self.dialogue, "worker-a", output)
        text = output.read_text(encoding="utf-8")
        self.assertIn("nothing to read", text)

    def test_launched_worker_receives_absolute_prior_paths_in_task_file(self) -> None:
        runner.launch(self.dialogue, "worker-a")
        record = self.dialogue.state()["completed_turns"][0]
        runner.launch(self.dialogue, "worker-b")
        task = self.dialogue.directory / "work" / "R02" / "task.md"
        text = task.read_text(encoding="utf-8")
        self.assertIn(
            str(self.dialogue.directory.absolute() / record["artifact_file"]), text
        )
        self.assertIn(
            str(self.dialogue.directory.absolute() / record["evidence_file"]), text
        )


class TaskBriefingTests(RunnerTestCase):
    def test_prepare_writes_nonsecret_briefing(self) -> None:
        output = self.base / "TASK.md"
        result = runner.prepare(self.dialogue, "worker-a", output)
        text = output.read_text(encoding="utf-8")
        self.assertIn("R01", text)
        self.assertIn("propose a design", text)
        self.assertIn("proposer", text)
        self.assertIn(self.definition.source_sha, text)
        self.assertEqual(result["packet"]["adapter"], "command")
        # Preparing must not claim or advance anything.
        self.assertEqual(self.dialogue.state()["status"], "OPEN")


if __name__ == "__main__":
    unittest.main()
