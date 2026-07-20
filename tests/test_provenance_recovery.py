"""RED → GREEN: the converged v1 provenance and recovery contract.

The public CLI exposes exactly the five-command production path plus the
read-only ``next`` helper; the recovery verbs (``claim``, ``prepare``,
``complete``, ``release``) are quarantined behind
``python -m multi_agent_dialogue.unverified`` and cannot manufacture
adapter provenance:

- every completed-turn state record and turn commit carries
  ``completed_via`` (``runner-launch`` from ``runner.launch()`` only;
  ``caller-supplied`` from every other completion door);
- ``validate --require-git`` cross-checks ``completed_via`` between the
  current state and the original turn commit (the source of truth), so a
  later state edit cannot launder ``caller-supplied`` into
  ``runner-launch``; missing or unknown values fail closed;
- a structurally valid caller-supplied recovery may stay ``ok`` but must
  warn, naming every affected round;
- ``validate --require-git --require-runner-completion`` is the
  production gate: it fails unless every Git-proven completed turn is
  ``runner-launch``, without pretending an incomplete dialogue is done;
- ``status`` exposes ``blocked_reason``, ``next_legal_action``, and the
  recovered-turn count;
- ``BLOCKED`` stays non-releasable and non-retryable through the
  recovery namespace too.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import config, engine, runner

RUNNER_LAUNCH = "runner-launch"
CALLER_SUPPLIED = "caller-supplied"


def run_cli(
    *args: str, module: str = "multi_agent_dialogue", cwd: Path | None = None
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(support.SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or support.REPO_ROOT,
        check=False,
    )


def subcommands(help_text: str) -> set[str]:
    """The argparse subcommand set from a ``--help`` usage line."""
    match = re.search(r"\{([^{}]+)\}", help_text)
    if match is None:
        raise AssertionError(f"no subcommand list in help output:\n{help_text}")
    return {item.strip() for item in match.group(1).split(",")}


def two_round_definition(runnable: bool = False) -> dict:
    raw = support.two_actor_definition()
    raw["protocol_id"] = "converged-provenance-demo"
    raw["schedule"] = raw["schedule"][:2]
    raw["final_round_id"] = "R02"
    if runnable:
        for actor in raw["actors"]:
            actor["settings"] = support.command_worker_settings(
                None, actor["expected_provider"], actor["expected_model"]
            )
    return raw


class GitFixture(unittest.TestCase):
    """An engine-level dialogue inside an isolated temporary repository."""

    runnable = False

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.repo = support.init_git_repo(self.base / "repo")
        self.scratch = self.base / "scratch"
        self.scratch.mkdir()
        self.definition = config.parse_definition(
            two_round_definition(runnable=self.runnable)
        )
        self.dialogue_dir = self.repo / "dialogues" / "demo"
        self.rel = "dialogues/demo"

    def init_dialogue(self) -> engine.Dialogue:
        return engine.init_dialogue(self.definition, self.dialogue_dir)

    def turn_inputs(self, actor_id: str, round_id: str) -> tuple[Path, Path]:
        turn_path = self.scratch / f"{round_id}-{actor_id}.md"
        turn_path.write_text(
            f"# {round_id}\n\nbody for {round_id} by {actor_id}\n", encoding="utf-8"
        )
        actor = self.definition.actor(actor_id)
        record = support.make_evidence(
            actor_id=actor_id,
            round_id=round_id,
            artifact_path=turn_path,
            provider=actor.expected_provider,
            model=actor.expected_model,
        )
        evidence_path = self.scratch / f"{round_id}-{actor_id}.json"
        evidence_path.write_text(json.dumps(record), encoding="utf-8")
        return turn_path, evidence_path

    def complete_turn(
        self, dialogue: engine.Dialogue, actor_id: str, round_id: str, **kwargs
    ) -> dict:
        dialogue.claim(actor_id)
        turn_path, evidence_path = self.turn_inputs(actor_id, round_id)
        return dialogue.complete(actor_id, turn_path, evidence_path, **kwargs)

    def decision_file(self) -> Path:
        path = self.scratch / "decision.md"
        path.write_text("Decision: APPROVE\n\nRationale.\n", encoding="utf-8")
        return path

    def head_message(self) -> str:
        return support.git(self.repo, "log", "-1", "--format=%B").stdout

    def trailers(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.head_message().splitlines():
            if line.startswith("Madp-") and ": " in line:
                key, value = line.split(": ", 1)
                result[key] = value.strip()
        return result

    def rewrite_state(self, mutate) -> None:
        """Apply ``mutate`` to the on-disk state and commit the cover-up."""
        state_path = self.dialogue_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        mutate(state)
        state_path.write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8"
        )
        support.git(self.repo, "add", "-f", "--", f"{self.rel}/state.json")
        support.git(self.repo, "commit", "-q", "-m", "cover-up")


class TopLevelNamespaceTests(unittest.TestCase):
    def test_top_level_help_has_exactly_the_public_commands(self) -> None:
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            subcommands(result.stdout),
            {"init", "status", "next", "run", "validate", "owner-decide"},
            "top-level madp must expose exactly the public path "
            "(claim/prepare/complete are recovery-only)",
        )

    def test_top_level_recovery_verbs_are_rejected(self) -> None:
        for verb in ("claim", "prepare", "complete"):
            result = run_cli(verb, "somedir", "--actor", "worker-a")
            self.assertNotEqual(result.returncode, 0, verb)
            self.assertIn("invalid choice", result.stderr, verb)


class UnverifiedNamespaceTests(unittest.TestCase):
    def test_unverified_namespace_has_exactly_the_recovery_verbs(self) -> None:
        result = run_cli("--help", module="multi_agent_dialogue.unverified")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            subcommands(result.stdout),
            {"claim", "prepare", "complete", "release"},
        )

    def test_unverified_help_is_honest_about_provenance(self) -> None:
        result = run_cli("--help", module="multi_agent_dialogue.unverified")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stdout.lower()
        self.assertIn("unverified", text)
        self.assertIn("provenance", text)
        self.assertIn("caller-supplied", text)


class CompletedViaTests(GitFixture):
    def test_complete_defaults_to_caller_supplied_in_state_and_trailer(self) -> None:
        dialogue = self.init_dialogue()
        state = self.complete_turn(dialogue, "worker-a", "R01")
        record = state["completed_turns"][0]
        self.assertEqual(record.get("completed_via"), CALLER_SUPPLIED)
        self.assertEqual(
            self.trailers().get("Madp-Completed-Via"), CALLER_SUPPLIED
        )

    def test_unknown_completed_via_fails_closed(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        turn_path, evidence_path = self.turn_inputs("worker-a", "R01")
        with self.assertRaises(engine.ProtocolError):
            dialogue.complete(
                "worker-a", turn_path, evidence_path, completed_via="hand-audited"
            )
        self.assertEqual(dialogue.state()["completed_turns"], [])


class RunnerLaunchProvenanceTests(GitFixture):
    runnable = True

    def test_runner_launch_records_runner_launch_provenance(self) -> None:
        dialogue = self.init_dialogue()
        runner.launch(dialogue, "worker-a")
        record = dialogue.state()["completed_turns"][0]
        self.assertEqual(record.get("completed_via"), RUNNER_LAUNCH)
        self.assertEqual(self.trailers().get("Madp-Completed-Via"), RUNNER_LAUNCH)


class ValidationLevelTests(GitFixture):
    runnable = True

    def test_require_runner_completion_requires_require_git(self) -> None:
        dialogue = self.init_dialogue()
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.validate(require_runner_completion=True)
        self.assertIn("require", str(ctx.exception).lower())

    def test_caller_supplied_recovery_warns_per_round_but_stays_ok(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        self.complete_turn(dialogue, "worker-b", "R02")
        report = dialogue.validate(require_git=True)
        self.assertTrue(report["ok"], report["errors"])
        warned = "\n".join(report["warnings"])
        self.assertIn(CALLER_SUPPLIED, warned)
        self.assertIn("R01", warned)
        self.assertIn("R02", warned)

    def test_caller_supplied_recovery_fails_production_gate_naming_rounds(
        self,
    ) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        report = dialogue.validate(require_git=True, require_runner_completion=True)
        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("R01", errors)
        self.assertIn(RUNNER_LAUNCH, errors)

    def test_mixed_provenance_names_only_recovered_rounds(self) -> None:
        dialogue = self.init_dialogue()
        runner.launch(dialogue, "worker-a")
        self.complete_turn(dialogue, "worker-b", "R02")
        report = dialogue.validate(require_git=True)
        self.assertTrue(report["ok"], report["errors"])
        warned = "\n".join(report["warnings"])
        self.assertIn("R02", warned)
        self.assertNotIn("R01", warned)
        production = dialogue.validate(
            require_git=True, require_runner_completion=True
        )
        self.assertFalse(production["ok"])
        errors = "\n".join(production["errors"])
        self.assertIn("R02", errors)
        self.assertNotIn("R01", errors)

    def test_all_runner_dialogue_passes_both_levels_at_ready_and_decided(
        self,
    ) -> None:
        dialogue = self.init_dialogue()
        runner.launch(dialogue, "worker-a")
        runner.launch(dialogue, "worker-b")
        for require_runner in (False, True):
            report = dialogue.validate(
                require_git=True, require_runner_completion=require_runner
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["status"], engine.STATUS_READY_FOR_OWNER)
        dialogue.owner_decide(self.decision_file())
        for require_runner in (False, True):
            report = dialogue.validate(
                require_git=True, require_runner_completion=require_runner
            )
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["status"], engine.STATUS_OWNER_DECIDED)

    def test_incomplete_all_runner_dialogue_passes_provenance_reports_open(
        self,
    ) -> None:
        dialogue = self.init_dialogue()
        runner.launch(dialogue, "worker-a")
        report = dialogue.validate(require_git=True, require_runner_completion=True)
        self.assertTrue(report["ok"], report["errors"])
        # Provenance never implies progress: the dialogue is still OPEN
        # with one scheduled turn left, and the report must say so.
        self.assertEqual(report["status"], engine.STATUS_OPEN)
        self.assertEqual(report["recorded_status"], engine.STATUS_OPEN)
        self.assertEqual(report["completed_turns"], 1)


class LaunderingTests(GitFixture):
    def test_state_edit_cannot_launder_caller_supplied_into_runner_launch(
        self,
    ) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")

        def launder(state: dict) -> None:
            state["completed_turns"][0]["completed_via"] = RUNNER_LAUNCH

        self.rewrite_state(launder)
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertIn("completed_via", "\n".join(report["errors"]))
        production = dialogue.validate(
            require_git=True, require_runner_completion=True
        )
        self.assertFalse(production["ok"])

    def test_unknown_completed_via_value_fails_closed(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")

        def corrupt(state: dict) -> None:
            state["completed_turns"][0]["completed_via"] = "hand-edited"

        self.rewrite_state(corrupt)
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertIn("completed_via", "\n".join(report["errors"]))

    def test_missing_completed_via_fails_closed(self) -> None:
        """A legacy turn without ``completed_via`` anywhere fails closed."""
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        state_path = self.dialogue_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["completed_turns"][0].pop("completed_via", None)
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        support.git(self.repo, "add", "-f", "--", f"{self.rel}/state.json")
        # Amend the original turn commit so neither the current state nor
        # the committed source of truth carries a completed_via value.
        support.git(self.repo, "commit", "-q", "--amend", "--no-edit")
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertIn("completed_via", "\n".join(report["errors"]))


class StatusSurfaceTests(GitFixture):
    runnable = True

    def status(self) -> dict:
        result = run_cli("status", str(self.dialogue_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_status_open_reports_next_action_without_breaking_shape(self) -> None:
        self.init_dialogue()
        payload = self.status()
        # Existing consumers keep their fields.
        self.assertEqual(payload["status"], "OPEN")
        self.assertEqual(payload["turn_index"], 0)
        self.assertIsNone(payload["claim"])
        self.assertEqual(payload["next"]["round_id"], "R01")
        # New surface: why nothing is blocked, what may happen next.
        self.assertIsNone(payload["blocked_reason"])
        self.assertEqual(payload["recovered_turns"], 0)
        action = payload["next_legal_action"]
        self.assertIn("run", action)
        self.assertIn("worker-a", action)
        self.assertIn("--launch", action)

    def test_status_claimed_names_the_claim_holder(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        action = self.status()["next_legal_action"]
        # Names the holder and the recovery namespace.
        self.assertIn("worker-a", action)
        self.assertIn("python -m multi_agent_dialogue.unverified", action)
        # Wait-first: the healthy CLAIMED state is a live run --launch, so
        # the guidance must lead with waiting, not complete/release.
        self.assertTrue(action.startswith("wait"), action)
        # The duplicate-worker hazard and its precondition must come before
        # any mention of completing or releasing the claim.
        self.assertIn("run --launch", action)
        self.assertIn("duplicate worker", action)
        self.assertIn("dead", action)
        hazard_at = action.index("duplicate worker")
        for verb in ("complet", "releas"):
            self.assertIn(verb, action)
            self.assertGreater(action.index(verb), action.index("wait"))
        self.assertGreater(hazard_at, action.index("wait"))
        # Recovery is conditional on the launching process being proven dead.
        self.assertLess(action.index("dead"), action.index("unverified"))

    def test_status_blocked_reports_reason_and_human_recovery(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.block("unproven worker-lane cleanup: lane demo-lane survived")
        payload = self.status()
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertIn("demo-lane", payload["blocked_reason"])
        action = payload["next_legal_action"]
        self.assertTrue(action.startswith("none"), action)
        self.assertIn("human", action)

    def test_status_terminal_states_report_owner_actions(self) -> None:
        dialogue = self.init_dialogue()
        runner.launch(dialogue, "worker-a")
        runner.launch(dialogue, "worker-b")
        self.assertIn("owner-decide", self.status()["next_legal_action"])
        dialogue.owner_decide(self.decision_file())
        payload = self.status()
        self.assertEqual(payload["status"], "OWNER_DECIDED")
        self.assertTrue(payload["next_legal_action"].startswith("none"))

    def test_status_counts_recovered_turns(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        self.assertEqual(self.status()["recovered_turns"], 1)
        runner.launch(dialogue, "worker-b")
        payload = self.status()
        self.assertEqual(payload["recovered_turns"], 1)
        self.assertEqual(payload["status"], "READY_FOR_OWNER")


class RecoveryCliTests(unittest.TestCase):
    """The unverified namespace end to end, over the real CLI."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self.definition_path = self.base / "protocol.json"
        self.dialogue_dir = self.base / "dialogue"
        self.definition_path.write_text(
            json.dumps(two_round_definition(runnable=True)), encoding="utf-8"
        )
        result = run_cli(
            "init",
            "--definition", str(self.definition_path),
            "--dialogue", str(self.dialogue_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def recover(self, *args: str) -> subprocess.CompletedProcess:
        return run_cli(*args, module="multi_agent_dialogue.unverified")

    def turn_inputs(self, actor_id: str, round_id: str) -> tuple[Path, Path]:
        turn_path = self.base / f"{round_id}.md"
        turn_path.write_text(f"# {round_id}\n\nrecovered body\n", encoding="utf-8")
        provider, model = {
            "worker-a": ("fake-provider-a", "fake-model-a"),
            "worker-b": ("fake-provider-b", "fake-model-b"),
        }[actor_id]
        record = support.make_evidence(
            actor_id=actor_id,
            round_id=round_id,
            artifact_path=turn_path,
            provider=provider,
            model=model,
        )
        evidence_path = self.base / f"{round_id}.json"
        evidence_path.write_text(json.dumps(record), encoding="utf-8")
        return turn_path, evidence_path

    def test_recovery_round_trip_and_validation_levels(self) -> None:
        claim = self.recover("claim", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertEqual(claim.returncode, 0, claim.stderr)
        turn_path, evidence_path = self.turn_inputs("worker-a", "R01")
        complete = self.recover(
            "complete", str(self.dialogue_dir),
            "--actor", "worker-a",
            "--turn", str(turn_path),
            "--runtime-evidence", str(evidence_path),
        )
        self.assertEqual(complete.returncode, 0, complete.stderr)
        payload = json.loads(complete.stdout)
        self.assertEqual(payload.get("completed_via"), CALLER_SUPPLIED)
        self.assertTrue(
            (self.dialogue_dir / "turns" / "R01-worker-a.md").is_file()
        )
        structural = run_cli("validate", str(self.dialogue_dir), "--require-git")
        self.assertEqual(structural.returncode, 0, structural.stderr)
        report = json.loads(structural.stdout)
        self.assertTrue(report["ok"], report["errors"])
        warned = "\n".join(report["warnings"])
        self.assertIn("R01", warned)
        self.assertIn(CALLER_SUPPLIED, warned)
        production = run_cli(
            "validate", str(self.dialogue_dir),
            "--require-git", "--require-runner-completion",
        )
        self.assertEqual(production.returncode, 1, production.stderr)
        report = json.loads(production.stdout)
        self.assertFalse(report["ok"])
        self.assertIn("R01", "\n".join(report["errors"]))

    def test_unverified_prepare_writes_briefing(self) -> None:
        output = self.base / "recovery-task.md"
        result = self.recover(
            "prepare", str(self.dialogue_dir),
            "--actor", "worker-a",
            "--output", str(output),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("## Goal", output.read_text(encoding="utf-8"))

    def test_unverified_release_recovers_a_stray_claim(self) -> None:
        claim = self.recover("claim", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertEqual(claim.returncode, 0, claim.stderr)
        release = self.recover(
            "release", str(self.dialogue_dir), "--actor", "worker-a"
        )
        self.assertEqual(release.returncode, 0, release.stderr)
        status = json.loads(run_cli("status", str(self.dialogue_dir)).stdout)
        self.assertEqual(status["status"], "OPEN")
        self.assertIsNone(status["claim"])
        again = self.recover("claim", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertEqual(again.returncode, 0, again.stderr)

    def test_require_runner_completion_without_require_git_is_an_error(
        self,
    ) -> None:
        result = run_cli(
            "validate", str(self.dialogue_dir), "--require-runner-completion"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--require-git", result.stderr)

    def test_blocked_refuses_every_recovery_operation(self) -> None:
        engine.Dialogue(self.dialogue_dir).block("manual block for the test")
        turn_path, evidence_path = self.turn_inputs("worker-a", "R01")
        for args in (
            ("claim", str(self.dialogue_dir), "--actor", "worker-a"),
            ("release", str(self.dialogue_dir), "--actor", "worker-a"),
            (
                "complete", str(self.dialogue_dir),
                "--actor", "worker-a",
                "--turn", str(turn_path),
                "--runtime-evidence", str(evidence_path),
            ),
        ):
            result = self.recover(*args)
            self.assertNotEqual(result.returncode, 0, args[0])
            self.assertIn("BLOCKED", result.stderr, args[0])


if __name__ == "__main__":
    unittest.main()
