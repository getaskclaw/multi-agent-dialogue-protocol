"""Task 5: the madp CLI via subprocess (python -m multi_agent_dialogue)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import config


def run_cli(
    *args: str, cwd: Path | None = None, module: str = "multi_agent_dialogue"
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


def run_recovery(*args: str) -> subprocess.CompletedProcess:
    """The quarantined recovery namespace (claim/prepare/complete/release)."""
    return run_cli(*args, module="multi_agent_dialogue.unverified")


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self.marker = self.base / "spawned.log"
        self.definition_path = self.base / "protocol.json"
        self.dialogue_dir = self.base / "dialogue"
        raw = support.two_actor_definition()
        for actor in raw["actors"]:
            actor["settings"] = support.command_worker_settings(
                self.marker, actor["expected_provider"], actor["expected_model"]
            )
        self.raw = raw
        self.definition_path.write_text(json.dumps(raw), encoding="utf-8")

    def init(self) -> None:
        result = run_cli(
            "init",
            "--definition", str(self.definition_path),
            "--dialogue", str(self.dialogue_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def spawn_count(self) -> int:
        """Worker processes started (the external verifier is not a worker)."""
        if not self.marker.exists():
            return 0
        lines = self.marker.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line == "worker")


class InitStatusNextTests(CliTestCase):
    def test_init_status_next(self) -> None:
        self.init()
        status = run_cli("status", str(self.dialogue_dir))
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertEqual(payload["status"], "OPEN")
        self.assertEqual(payload["next"]["round_id"], "R01")
        self.assertEqual(payload["next"]["actor_id"], "worker-a")

        next_result = run_cli("next", str(self.dialogue_dir))
        self.assertEqual(next_result.returncode, 0)
        next_payload = json.loads(next_result.stdout)
        self.assertEqual(next_payload["round_id"], "R01")

    def test_init_twice_fails(self) -> None:
        self.init()
        result = run_cli(
            "init",
            "--definition", str(self.definition_path),
            "--dialogue", str(self.dialogue_dir),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.strip())

    def test_invalid_definition_fails(self) -> None:
        self.raw["actors"] = self.raw["actors"][:1]
        self.definition_path.write_text(json.dumps(self.raw), encoding="utf-8")
        result = run_cli(
            "init",
            "--definition", str(self.definition_path),
            "--dialogue", str(self.dialogue_dir),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("two actors", result.stderr)


class ClaimTests(CliTestCase):
    """claim moved to the unverified recovery namespace; same fail-closed rules."""

    def test_claim_and_wrong_actor(self) -> None:
        self.init()
        good = run_recovery("claim", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)["status"], "CLAIMED")

        duplicate = run_recovery("claim", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertNotEqual(duplicate.returncode, 0)

    def test_wrong_actor_claim_fails(self) -> None:
        self.init()
        result = run_recovery("claim", str(self.dialogue_dir), "--actor", "worker-b")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker-b", result.stderr)


class RunTests(CliTestCase):
    def test_run_defaults_to_dry_run_no_process(self) -> None:
        self.init()
        before = (self.dialogue_dir / "state.json").read_text(encoding="utf-8")
        result = run_cli("run", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["executed"])
        self.assertEqual(self.spawn_count(), 0)
        self.assertFalse((self.dialogue_dir / "work").exists())
        after = (self.dialogue_dir / "state.json").read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_run_launch_executes_one_turn(self) -> None:
        self.init()
        result = run_cli("run", str(self.dialogue_dir), "--actor", "worker-a", "--launch")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["executed"])
        self.assertEqual(payload["completed_round"], "R01")
        self.assertEqual(self.spawn_count(), 1)
        status = json.loads(run_cli("status", str(self.dialogue_dir)).stdout)
        self.assertEqual(status["turn_index"], 1)

    def test_run_launch_wrong_actor_fails(self) -> None:
        self.init()
        result = run_cli("run", str(self.dialogue_dir), "--actor", "worker-b", "--launch")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.spawn_count(), 0)

    def test_run_rejects_conflicting_flags(self) -> None:
        self.init()
        result = run_cli(
            "run", str(self.dialogue_dir), "--actor", "worker-a", "--launch", "--dry-run"
        )
        self.assertNotEqual(result.returncode, 0)


class CompleteTests(CliTestCase):
    """complete moved to the unverified recovery namespace; same evidence gates."""

    def test_manual_complete_roundtrip(self) -> None:
        self.init()
        claim = run_recovery("claim", str(self.dialogue_dir), "--actor", "worker-a")
        self.assertEqual(claim.returncode, 0)
        turn_path = self.base / "turn.md"
        turn_path.write_text("# R01\n\nmanual turn\n", encoding="utf-8")
        record = support.make_evidence(
            actor_id="worker-a",
            round_id="R01",
            artifact_path=turn_path,
            provider="fake-provider-a",
            model="fake-model-a",
        )
        evidence_path = self.base / "evidence.json"
        evidence_path.write_text(json.dumps(record), encoding="utf-8")
        result = run_recovery(
            "complete", str(self.dialogue_dir),
            "--actor", "worker-a",
            "--turn", str(turn_path),
            "--runtime-evidence", str(evidence_path),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["turn_index"], 1)

    def test_complete_with_fake_identity_fails(self) -> None:
        self.init()
        run_recovery("claim", str(self.dialogue_dir), "--actor", "worker-a")
        turn_path = self.base / "turn.md"
        turn_path.write_text("# R01\n\nmanual turn\n", encoding="utf-8")
        record = support.make_evidence(
            actor_id="worker-a",
            round_id="R01",
            artifact_path=turn_path,
            provider="fake-provider-a",
            model="wrong-model",
        )
        evidence_path = self.base / "evidence.json"
        evidence_path.write_text(json.dumps(record), encoding="utf-8")
        result = run_recovery(
            "complete", str(self.dialogue_dir),
            "--actor", "worker-a",
            "--turn", str(turn_path),
            "--runtime-evidence", str(evidence_path),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model", result.stderr)


class ValidateAndOwnerTests(CliTestCase):
    def run_all_turns(self) -> None:
        for actor in ("worker-a", "worker-b", "worker-a", "worker-b"):
            result = run_cli("run", str(self.dialogue_dir), "--actor", actor, "--launch")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_validate_clean_and_tampered(self) -> None:
        self.init()
        self.run_all_turns()
        ok = run_cli("validate", str(self.dialogue_dir))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue(json.loads(ok.stdout)["ok"])

        published = next((self.dialogue_dir / "turns").glob("R01-*.md"))
        published.write_text("tampered\n", encoding="utf-8")
        bad = run_cli("validate", str(self.dialogue_dir))
        self.assertNotEqual(bad.returncode, 0)
        self.assertFalse(json.loads(bad.stdout)["ok"])

    def test_owner_decide_flow_and_post_final_stop(self) -> None:
        self.init()
        decision = self.base / "decision.md"
        decision.write_text("Decision: APPROVE\n\nRationale.\n", encoding="utf-8")
        early = run_cli("owner-decide", str(self.dialogue_dir), "--decision", str(decision))
        self.assertNotEqual(early.returncode, 0)

        self.run_all_turns()
        post_final = run_cli("run", str(self.dialogue_dir), "--actor", "worker-a", "--launch")
        self.assertNotEqual(post_final.returncode, 0)
        self.assertEqual(self.spawn_count(), 4)

        decide = run_cli("owner-decide", str(self.dialogue_dir), "--decision", str(decision))
        self.assertEqual(decide.returncode, 0, decide.stderr)
        payload = json.loads(decide.stdout)
        self.assertEqual(payload["status"], "OWNER_DECIDED")

        again = run_cli("owner-decide", str(self.dialogue_dir), "--decision", str(decision))
        self.assertNotEqual(again.returncode, 0)


class CanaryTests(unittest.TestCase):
    """`madp canary` exercises the real acceptance path end to end."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def run_canary(self, adapter: str, *extra: str,
                   dirname: str = "canary") -> subprocess.CompletedProcess:
        return run_cli(
            "canary", "--adapter", adapter,
            "--dialogue", str(self.base / dirname), *extra,
        )

    def run_cli_dir(self, path: Path) -> subprocess.CompletedProcess:
        return run_cli("canary", "--adapter", "command", "--dialogue", str(path))

    def test_canary_command_adapter_passes(self) -> None:
        result = self.run_canary("command", "--no-push")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["validation_ok"])
        self.assertEqual(payload["turn"]["completed_via"], "runner-launch")
        self.assertTrue(payload["local_only"])
        cli = payload["cli_version"]
        self.assertEqual(cli["exit_status"], 0)
        self.assertIn("fake-worker", cli["output"])

    def test_canary_hermes_adapter_passes(self) -> None:
        result = self.run_canary("hermes-cli")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertIn("fake-hermes", payload["cli_version"]["output"])

    def test_canary_fable_adapter_passes(self) -> None:
        result = self.run_canary("fable-session")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertIn("fake-fable-session", payload["cli_version"]["output"])

    def test_sabotaged_turn_fails_validation_afterwards(self) -> None:
        # The canary dialogue is a real Git-backed dialogue: tampering
        # with the published turn must flip the production gate to
        # non-zero, with no new commit and nothing pushed anywhere.
        result = self.run_canary("command")
        self.assertEqual(result.returncode, 0, result.stderr)
        dialogue = self.base / "canary"
        turn_files = list((dialogue / "turns").glob("*.md"))
        self.assertEqual(len(turn_files), 1)
        before = subprocess.run(
            ["git", "-C", str(dialogue), "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        with turn_files[0].open("a", encoding="utf-8") as handle:
            handle.write("\nsabotage\n")
        validation = run_cli(
            "validate", str(dialogue),
            "--require-git", "--require-runner-completion",
        )
        self.assertEqual(validation.returncode, 1)
        after = subprocess.run(
            ["git", "-C", str(dialogue), "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(before, after, "validation never commits")

    def test_non_empty_dialogue_dir_is_rejected(self) -> None:
        target = self.base / "occupied"
        target.mkdir()
        (target / "stale.txt").write_text("x", encoding="utf-8")
        result = self.run_cli_dir(target)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not empty", result.stderr)

    def test_dialogue_path_that_is_a_file_is_rejected(self) -> None:
        target = self.base / "a-file"
        target.write_text("x", encoding="utf-8")
        result = self.run_cli_dir(target)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a directory", result.stderr)

    def test_stale_scratch_dir_is_rejected(self) -> None:
        (self.base / "canary.canary-scratch").mkdir()
        result = self.run_canary("command")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def test_expected_identity_flags_require_command_name(self) -> None:
        result = self.run_canary(
            "hermes-cli", "--expected-model", "some-model"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only meaningful", result.stderr)

    def test_real_binary_override_requires_explicit_identity(self) -> None:
        result = self.run_canary(
            "hermes-cli", "--command-name", "/bin/true"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("never guesses identity", result.stderr)


if __name__ == "__main__":
    unittest.main()
