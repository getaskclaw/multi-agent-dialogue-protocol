"""RED → GREEN: the Git transaction model.

Git must be the dialogue's transaction log, not an after-the-fact
cleanliness check:

- a dialogue initializes only inside a real Git worktree/repository and
  the initialization (definition + initial state + dialogue-local ignore
  rules) is one commit;
- every successful worker turn is exactly one local commit containing
  the updated state, the published turn, and its runtime evidence, with
  trailers naming protocol/round/actor/transport/provider/model/session
  and the artifact/evidence digests;
- scratch ``work/``, the live claim lock, prompts, and transport scratch
  never enter history;
- a commit failure after publication (or after the owner decision)
  transitions the dialogue to non-retryable ``BLOCKED``;
- ``validate --require-git`` proves artifact-to-commit provenance and
  exposes the proven commit SHAs.

These tests prove the current gap first (RED): completion writes files
and advances state but creates no commit at all.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import config, engine, gitops, runner

SHA_HEX = frozenset("0123456789abcdef")


def head(repo: Path) -> str:
    return support.git(repo, "rev-parse", "HEAD").stdout.strip()


def commit_count(repo: Path) -> int:
    result = support.git(repo, "rev-list", "--count", "HEAD", check=False)
    return int(result.stdout.strip()) if result.returncode == 0 else 0


def commit_message(repo: Path, ref: str = "HEAD") -> str:
    return support.git(repo, "log", "-1", "--format=%B", ref).stdout


def commit_paths(repo: Path, ref: str = "HEAD") -> set[str]:
    out = support.git(repo, "show", "--name-only", "--format=", ref).stdout
    return {line for line in out.splitlines() if line.strip()}


def all_committed_paths(repo: Path) -> set[str]:
    out = support.git(repo, "log", "--all", "--name-only", "--format=").stdout
    return {line for line in out.splitlines() if line.strip()}


def parse_trailers(message: str) -> dict[str, str]:
    trailers: dict[str, str] = {}
    for line in message.splitlines():
        if line.startswith("Madp-") and ": " in line:
            key, value = line.split(": ", 1)
            trailers[key] = value.strip()
    return trailers


def two_round_definition() -> dict:
    raw = support.two_actor_definition()
    raw["protocol_id"] = "git-transaction-demo"
    raw["schedule"] = raw["schedule"][:2]
    raw["final_round_id"] = "R02"
    return raw


class GitDialogueTestCase(unittest.TestCase):
    """A dialogue nested below the root of an isolated temporary repo."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.repo = support.init_git_repo(self.base / "repo")
        self.scratch = self.base / "scratch"
        self.scratch.mkdir()
        self.definition = config.parse_definition(two_round_definition())
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

    def complete_turn(self, dialogue: engine.Dialogue, actor_id: str, round_id: str) -> dict:
        dialogue.claim(actor_id)
        turn_path, evidence_path = self.turn_inputs(actor_id, round_id)
        return dialogue.complete(actor_id, turn_path, evidence_path)

    def finish_dialogue(self, dialogue: engine.Dialogue) -> None:
        self.complete_turn(dialogue, "worker-a", "R01")
        self.complete_turn(dialogue, "worker-b", "R02")

    def decision_file(self, text: str = "Decision: APPROVE\n\nRationale.\n") -> Path:
        path = self.scratch / "decision.md"
        path.write_text(text, encoding="utf-8")
        return path

    def sabotage_object_store(self) -> None:
        """Make every Git object write fail, including when tests run as root."""
        blocker = self.base / "not-a-git-object-directory"
        blocker.write_text("this is a file, not an object directory\n", encoding="utf-8")
        previous = os.environ.get("GIT_OBJECT_DIRECTORY")

        def restore() -> None:
            if previous is None:
                os.environ.pop("GIT_OBJECT_DIRECTORY", None)
            else:
                os.environ["GIT_OBJECT_DIRECTORY"] = previous

        self.addCleanup(restore)
        os.environ["GIT_OBJECT_DIRECTORY"] = str(blocker)


class InitTransactionTests(GitDialogueTestCase):
    def test_init_outside_git_repository_fails_closed(self) -> None:
        outside = self.base / "no-repo" / "dialogue"
        with self.assertRaises(engine.ProtocolError) as ctx:
            engine.init_dialogue(self.definition, outside)
        self.assertIn("git", str(ctx.exception).lower())
        self.assertFalse((outside / engine.STATE_FILE).exists())
        self.assertFalse((outside / engine.DEFINITION_FILE).exists())

    def test_init_creates_one_initialization_commit(self) -> None:
        self.init_dialogue()
        self.assertEqual(commit_count(self.repo), 1)
        self.assertEqual(
            commit_paths(self.repo),
            {
                f"{self.rel}/definition.json",
                f"{self.rel}/state.json",
                f"{self.rel}/.gitignore",
            },
        )
        trailers = parse_trailers(commit_message(self.repo))
        self.assertEqual(trailers.get("Madp-Event"), "init")
        self.assertEqual(trailers.get("Madp-Protocol"), self.definition.protocol_id)
        self.assertEqual(
            trailers.get("Madp-Definition-Digest"), self.definition.digest()
        )
        author = support.git(
            self.repo, "log", "-1", "--format=%an <%ae>"
        ).stdout.strip()
        self.assertEqual(
            author, f"{support.TEST_GIT_NAME} <{support.TEST_GIT_EMAIL}>"
        )

    def test_init_ignore_rules_cover_scratch_lock_and_temp(self) -> None:
        self.init_dialogue()
        for transient in (
            f"{self.rel}/work/R01/task.md",
            f"{self.rel}/work/fable/registry.toml",
            f"{self.rel}/.dialogue-lock",
            f"{self.rel}/turns/leftover.tmp",
        ):
            check = support.git(self.repo, "check-ignore", "-q", transient, check=False)
            self.assertEqual(check.returncode, 0, f"{transient} must be ignored")
        for owned in (
            f"{self.rel}/state.json",
            f"{self.rel}/turns/R01-worker-a.md",
            f"{self.rel}/evidence/R01-worker-a.json",
        ):
            check = support.git(self.repo, "check-ignore", "-q", owned, check=False)
            self.assertNotEqual(check.returncode, 0, f"{owned} must not be ignored")


class TurnCommitTests(GitDialogueTestCase):
    def test_commit_api_rejects_trailer_value_injection(self) -> None:
        path = self.repo / "safe.txt"
        path.write_text("safe\n", encoding="utf-8")
        with self.assertRaises(gitops.GitError):
            gitops.commit_paths(
                self.repo,
                [path],
                "unsafe trailer fixture",
                {"Madp-Actor": "worker-a\nMadp-Event: owner-decision"},
            )
        self.assertEqual(commit_count(self.repo), 0)

    def test_complete_creates_exactly_one_turn_commit(self) -> None:
        dialogue = self.init_dialogue()
        before = commit_count(self.repo)
        self.complete_turn(dialogue, "worker-a", "R01")
        # The files and state exist (this already holds today)...
        self.assertTrue((self.dialogue_dir / "turns" / "R01-worker-a.md").is_file())
        self.assertTrue((self.dialogue_dir / "evidence" / "R01-worker-a.json").is_file())
        self.assertEqual(dialogue.state()["turn_index"], 1)
        # ...and the turn is exactly one local commit of exactly its paths.
        self.assertEqual(commit_count(self.repo), before + 1)
        self.assertEqual(
            commit_paths(self.repo),
            {
                f"{self.rel}/state.json",
                f"{self.rel}/turns/R01-worker-a.md",
                f"{self.rel}/evidence/R01-worker-a.json",
            },
        )
        clean = support.git(
            self.repo, "status", "--porcelain", "--", self.rel
        ).stdout.strip()
        self.assertEqual(clean, "", "the committed turn must leave the dialogue clean")

    def test_turn_commit_trailers_identify_runtime(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        record = dialogue.state()["completed_turns"][0]
        trailers = parse_trailers(commit_message(self.repo))
        self.assertEqual(trailers.get("Madp-Event"), "turn")
        self.assertEqual(trailers.get("Madp-Protocol"), self.definition.protocol_id)
        self.assertEqual(trailers.get("Madp-Round"), "R01")
        self.assertEqual(trailers.get("Madp-Actor"), "worker-a")
        self.assertEqual(trailers.get("Madp-Transport"), "command")
        self.assertEqual(trailers.get("Madp-Provider"), "fake-provider-a")
        self.assertEqual(trailers.get("Madp-Model"), "fake-model-a")
        self.assertEqual(trailers.get("Madp-Session"), record["session_id"])
        self.assertEqual(
            trailers.get("Madp-Artifact-Sha256"), record["artifact_sha256"]
        )
        self.assertEqual(
            trailers.get("Madp-Evidence-Sha256"), record["evidence_sha256"]
        )

    def test_unrelated_repository_files_are_never_staged(self) -> None:
        dialogue = self.init_dialogue()
        (self.repo / "unrelated.txt").write_text("untracked bystander\n", encoding="utf-8")
        (self.repo / "staged.txt").write_text("user-staged work\n", encoding="utf-8")
        support.git(self.repo, "add", "staged.txt")
        self.complete_turn(dialogue, "worker-a", "R01")
        committed = commit_paths(self.repo)
        self.assertNotIn("unrelated.txt", committed)
        self.assertNotIn("staged.txt", committed)
        status = support.git(self.repo, "status", "--porcelain").stdout
        self.assertIn("A  staged.txt", status, "user-staged work must stay staged")
        self.assertIn("?? unrelated.txt", status)

    def test_released_attempt_state_history_rides_next_turn_commit(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        dialogue.release("worker-a")  # transient state/revision churn only
        self.complete_turn(dialogue, "worker-a", "R01")
        self.assertIn(f"{self.rel}/state.json", commit_paths(self.repo))
        clean = support.git(
            self.repo, "status", "--porcelain", "--", self.rel
        ).stdout.strip()
        self.assertEqual(clean, "")
        committed_state = json.loads(
            support.git(self.repo, "show", f"HEAD:{self.rel}/state.json").stdout
        )
        self.assertEqual(committed_state["revision"], dialogue.state()["revision"])
        self.assertEqual(len(committed_state["completed_turns"]), 1)

    def test_global_git_config_is_never_mutated(self) -> None:
        import os

        sentinel = "[madp]\n\tsentinel = true\n"
        fake_global = self.base / "fake-global-gitconfig"
        fake_global.write_text(sentinel, encoding="utf-8")
        previous = os.environ.get("GIT_CONFIG_GLOBAL")
        os.environ["GIT_CONFIG_GLOBAL"] = str(fake_global)
        self.addCleanup(
            lambda: os.environ.__setitem__("GIT_CONFIG_GLOBAL", previous or os.devnull)
        )
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        self.assertEqual(fake_global.read_text(encoding="utf-8"), sentinel)
        identity = support.git(
            self.repo, "log", "-1", "--format=%an <%ae> %cn <%ce>"
        ).stdout.strip()
        expected = f"{support.TEST_GIT_NAME} <{support.TEST_GIT_EMAIL}>"
        self.assertEqual(identity, f"{expected} {expected}")


class LaunchCommitTests(GitDialogueTestCase):
    def test_launch_commits_turn_and_never_scratch_or_prompts(self) -> None:
        raw = two_round_definition()
        for actor in raw["actors"]:
            actor["settings"] = support.command_worker_settings(
                None, actor["expected_provider"], actor["expected_model"]
            )
        self.definition = config.parse_definition(raw)
        dialogue = self.init_dialogue()
        before = commit_count(self.repo)
        runner.launch(dialogue, "worker-a")
        self.assertEqual(commit_count(self.repo), before + 1)
        self.assertTrue((self.dialogue_dir / "work" / "R01" / "task.md").is_file())
        committed = all_committed_paths(self.repo)
        for path in committed:
            parts = Path(path).parts
            self.assertNotIn("work", parts, f"scratch/prompt path committed: {path}")
            self.assertNotIn(".dialogue-lock", parts, f"live lock committed: {path}")
        clean = support.git(
            self.repo, "status", "--porcelain", "--", self.rel
        ).stdout.strip()
        self.assertEqual(clean, "", "ignored scratch must leave the dialogue clean")


class CommitFailureTests(GitDialogueTestCase):
    def test_commit_failure_after_publication_blocks_non_retryably(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        turn_path, evidence_path = self.turn_inputs("worker-a", "R01")
        self.sabotage_object_store()
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.complete("worker-a", turn_path, evidence_path)
        self.assertIn("commit", str(ctx.exception).lower())
        state = dialogue.state()
        self.assertEqual(state["status"], engine.STATUS_BLOCKED)
        self.assertIn("commit", state["blocked_reason"].lower())
        # Publication happened; the turn must never be reported complete
        # without commit proof, and nothing may retry past BLOCKED.
        self.assertTrue((self.dialogue_dir / "turns" / "R01-worker-a.md").is_file())
        with self.assertRaises(engine.ProtocolError):
            dialogue.claim("worker-b")
        with self.assertRaises(engine.ProtocolError):
            dialogue.release("worker-a")

    def test_owner_decision_commit_failure_blocks_not_owner_decided(self) -> None:
        dialogue = self.init_dialogue()
        self.finish_dialogue(dialogue)
        self.sabotage_object_store()
        with self.assertRaises(engine.ProtocolError):
            dialogue.owner_decide(self.decision_file())
        state = dialogue.state()
        self.assertEqual(state["status"], engine.STATUS_BLOCKED)
        self.assertNotEqual(state["status"], engine.STATUS_OWNER_DECIDED)
        with self.assertRaises(engine.ProtocolError):
            dialogue.owner_decide(self.decision_file())


class OwnerDecisionCommitTests(GitDialogueTestCase):
    def test_owner_decision_creates_its_own_terminal_commit(self) -> None:
        dialogue = self.init_dialogue()
        self.finish_dialogue(dialogue)
        last_turn_commit = head(self.repo)
        before = commit_count(self.repo)
        dialogue.owner_decide(self.decision_file())
        self.assertEqual(commit_count(self.repo), before + 1)
        self.assertNotEqual(head(self.repo), last_turn_commit)
        self.assertEqual(
            commit_paths(self.repo),
            {f"{self.rel}/state.json", f"{self.rel}/OWNER-DECISION.md"},
        )
        trailers = parse_trailers(commit_message(self.repo))
        self.assertEqual(trailers.get("Madp-Event"), "owner-decision")
        self.assertEqual(trailers.get("Madp-Decision"), "APPROVE")
        self.assertEqual(trailers.get("Madp-Protocol"), self.definition.protocol_id)


class RequireGitValidationTests(GitDialogueTestCase):
    def assert_sha(self, value: object) -> None:
        self.assertIsInstance(value, str)
        assert isinstance(value, str)
        self.assertEqual(len(value), 40)
        self.assertTrue(set(value) <= SHA_HEX, value)

    def test_validate_proves_commit_provenance_and_exposes_shas(self) -> None:
        dialogue = self.init_dialogue()
        self.finish_dialogue(dialogue)
        dialogue.owner_decide(self.decision_file())
        report = dialogue.validate(require_git=True)
        self.assertTrue(report["ok"], report["errors"])
        provenance = report["provenance"]
        self.assert_sha(provenance.get("init_commit"))
        turns = provenance.get("turn_commits")
        self.assertIsInstance(turns, list)
        self.assertEqual(
            [(t["round_id"], t["actor_id"]) for t in turns],
            [("R01", "worker-a"), ("R02", "worker-b")],
        )
        for turn in turns:
            self.assert_sha(turn["commit"])
        self.assert_sha(provenance.get("owner_decision_commit"))
        shas = [
            provenance["init_commit"],
            *(t["commit"] for t in turns),
            provenance["owner_decision_commit"],
        ]
        self.assertEqual(len(shas), len(set(shas)), "each transition is its own commit")
        self.assertEqual(provenance["owner_decision_commit"], head(self.repo))

    def test_validate_rejects_amended_owner_decision_trailers(self) -> None:
        dialogue = self.init_dialogue()
        self.finish_dialogue(dialogue)
        dialogue.owner_decide(self.decision_file())
        message = commit_message(self.repo)
        message = message.replace("Madp-Decision: APPROVE", "Madp-Decision: REJECT")
        message = message.replace(
            "Madp-Caller-Identity: unverified",
            "Madp-Caller-Identity: externally-verified",
        )
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "owner-decision commit: Git trailer Madp-Decision" in item
                for item in report["errors"]
            ),
            report["errors"],
        )
        self.assertTrue(
            any(
                "owner-decision commit: Git trailer Madp-Caller-Identity" in item
                for item in report["errors"]
            ),
            report["errors"],
        )

    def test_validate_rejects_mixed_case_owner_decision_trailers(self) -> None:
        dialogue = self.init_dialogue()
        self.finish_dialogue(dialogue)
        dialogue.owner_decide(self.decision_file())
        message = commit_message(self.repo).replace(
            "Madp-Event: owner-decision",
            "Madp-Event: owner-decision\nmadp-event: turn\nMADP-Round: R99",
        )
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("duplicate Git trailer 'madp-event'" in item for item in report["errors"]),
            report["errors"],
        )
        self.assertTrue(
            any("unexpected Git trailer 'MADP-Round'" in item for item in report["errors"]),
            report["errors"],
        )

    def test_validate_rejects_unexpected_init_trailer(self) -> None:
        dialogue = self.init_dialogue()
        init_message = commit_message(self.repo).replace(
            "Madp-Event: init",
            "Madp-Event: init\nmadp-event: owner-decision\nMADP-Decision: APPROVE",
        )
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            init_message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "duplicate Git trailer 'madp-event'" in item
                for item in report["errors"]
            ),
            report["errors"],
        )
        self.assertTrue(
            any(
                "init commit: unexpected Git trailer 'MADP-Decision'" in item
                for item in report["errors"]
            ),
            report["errors"],
        )

    def test_validate_rejects_owner_trailer_on_turn(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        turn_message = commit_message(self.repo) + "\nMadp-Decision: APPROVE\n"
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            turn_message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "turn R01: unexpected Git trailer 'Madp-Decision'" in item
                for item in report["errors"]
            ),
            report["errors"],
        )

    def test_validate_rejects_committed_artifact_tampering(self) -> None:
        import hashlib

        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        # Tamper with the published artifact AND rewrite state + evidence to
        # match, then commit the cover-up so the working tree looks clean.
        artifact = self.dialogue_dir / "turns" / "R01-worker-a.md"
        artifact.write_text("# R01\n\nrewritten history\n", encoding="utf-8")
        new_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        evidence_path = self.dialogue_dir / "evidence" / "R01-worker-a.json"
        record = json.loads(evidence_path.read_text(encoding="utf-8"))
        record["artifact_sha256"] = new_sha
        evidence_bytes = json.dumps(record).encode("utf-8")
        evidence_path.write_bytes(evidence_bytes)
        state_path = self.dialogue_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["completed_turns"][0]["artifact_sha256"] = new_sha
        state["completed_turns"][0]["evidence_sha256"] = hashlib.sha256(
            evidence_bytes
        ).hexdigest()
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        support.git(
            self.repo, "add", "-f", "--",
            f"{self.rel}/turns/R01-worker-a.md",
            f"{self.rel}/evidence/R01-worker-a.json",
            f"{self.rel}/state.json",
        )
        support.git(self.repo, "commit", "-q", "-m", "cover-up")
        # Convention-only validation cannot see the rewrite...
        self.assertTrue(dialogue.validate()["ok"])
        # ...but commit provenance must.
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("git history" in error.lower() for error in report["errors"]),
            report["errors"],
        )

    def test_validate_rejects_contradictory_turn_trailers(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        message = commit_message(self.repo).replace(
            "Madp-Actor: worker-a", "Madp-Actor: worker-b"
        )
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("Git trailer Madp-Actor" in error for error in report["errors"]),
            report["errors"],
        )

    def test_validate_rejects_deleted_identity_trailers_on_new_primary_turn(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        removed = {
            "Madp-Scheduled-Actor",
            "Madp-Actor-Selection",
            "Madp-Substitution-Reason",
        }
        message = "\n".join(
            line
            for line in commit_message(self.repo).splitlines()
            if line.partition(":")[0] not in removed
        )
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("Git trailer Madp-Scheduled-Actor" in item for item in report["errors"]),
            report["errors"],
        )

    def test_validate_rejects_mixed_case_turn_trailers(self) -> None:
        dialogue = self.init_dialogue()
        self.complete_turn(dialogue, "worker-a", "R01")
        message = commit_message(self.repo).replace(
            "Madp-Event: turn",
            "Madp-Event: turn\nmadp-event: owner-decision\nMADP-Decision: APPROVE",
        )
        support.git(
            self.repo,
            "commit",
            "--amend",
            "--quiet",
            "--no-verify",
            "-m",
            message,
        )
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("duplicate Git trailer 'madp-event'" in item for item in report["errors"]),
            report["errors"],
        )
        self.assertTrue(
            any("unexpected Git trailer 'MADP-Decision'" in item for item in report["errors"]),
            report["errors"],
        )

    def test_validate_rejects_substitute_identity_deleted_from_original_commit(
        self,
    ) -> None:
        raw = two_round_definition()
        raw["actors"][1]["role"] = raw["actors"][0]["role"]
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.definition = config.parse_definition(raw)
        dialogue = self.init_dialogue()
        dialogue.claim("worker-b", substitution_reason="provider_cooldown")
        turn_path, evidence_path = self.turn_inputs("worker-b", "R01")
        evidence_record = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_record.update(
            {
                "scheduled_actor_id": "worker-a",
                "actor_selection": "substitute",
                "substitution_reason": "provider_cooldown",
            }
        )
        evidence_path.write_text(json.dumps(evidence_record), encoding="utf-8")
        dialogue.complete("worker-b", turn_path, evidence_path)

        state_path = self.dialogue_dir / "state.json"
        valid_state = json.loads(state_path.read_text(encoding="utf-8"))
        tampered = json.loads(json.dumps(valid_state))
        entry = tampered["completed_turns"][0]
        for key in ("scheduled_actor_id", "actor_selection", "substitution_reason"):
            entry.pop(key)
        engine.atomic_write_json(state_path, tampered)
        support.git(self.repo, "add", "-f", "--", f"{self.rel}/state.json")
        support.git(self.repo, "commit", "--amend", "--quiet", "--no-edit")
        amended_turn = head(self.repo)

        valid_state["last_commit"] = amended_turn
        if "commit" in valid_state["completed_turns"][0]:
            valid_state["completed_turns"][0]["commit"] = amended_turn
        engine.atomic_write_json(state_path, valid_state)
        support.git(self.repo, "add", "-f", "--", f"{self.rel}/state.json")
        support.git(self.repo, "commit", "-q", "-m", "restore current state only")

        self.assertTrue(dialogue.validate()["ok"])
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "turn R01 committed state" in error
                and "identity fields" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_validate_rejects_deleted_historical_artifact(self) -> None:
        dialogue = self.init_dialogue()
        self.finish_dialogue(dialogue)
        # Roll back R02 entirely: delete its artifacts, rewrite state to the
        # post-R01 shape, and commit — the working tree is clean afterwards.
        (self.dialogue_dir / "turns" / "R02-worker-b.md").unlink()
        (self.dialogue_dir / "evidence" / "R02-worker-b.json").unlink()
        state_path = self.dialogue_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["completed_turns"] = state["completed_turns"][:1]
        state["turn_index"] = 1
        state["status"] = engine.STATUS_OPEN
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        support.git(self.repo, "add", "-f", "-A", "--", self.rel)
        support.git(self.repo, "commit", "-q", "-m", "roll back R02")
        self.assertTrue(dialogue.validate()["ok"])
        report = dialogue.validate(require_git=True)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("git history" in error.lower() for error in report["errors"]),
            report["errors"],
        )


class PlacementTests(unittest.TestCase):
    """Dialogues at the repository root and in a linked worktree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.definition = config.parse_definition(two_round_definition())
        self.scratch = self.base / "scratch"
        self.scratch.mkdir()

    def run_dialogue(self, dialogue: engine.Dialogue) -> None:
        for actor_id, round_id in (("worker-a", "R01"), ("worker-b", "R02")):
            dialogue.claim(actor_id)
            turn_path = self.scratch / f"{round_id}.md"
            turn_path.write_text(f"# {round_id}\n\nbody\n", encoding="utf-8")
            actor = self.definition.actor(actor_id)
            record = support.make_evidence(
                actor_id=actor_id,
                round_id=round_id,
                artifact_path=turn_path,
                provider=actor.expected_provider,
                model=actor.expected_model,
            )
            evidence_path = self.scratch / f"{round_id}.json"
            evidence_path.write_text(json.dumps(record), encoding="utf-8")
            dialogue.complete(actor_id, turn_path, evidence_path)
        decision = self.scratch / "decision.md"
        decision.write_text("Decision: APPROVE\n\nDone.\n", encoding="utf-8")
        dialogue.owner_decide(decision)

    def test_dialogue_at_repository_root(self) -> None:
        repo = support.init_git_repo(self.base / "rootrepo")
        dialogue = engine.init_dialogue(self.definition, repo)
        self.run_dialogue(dialogue)
        self.assertEqual(commit_count(repo), 4)  # init + R01 + R02 + decision
        self.assertIn("state.json", commit_paths(repo, "HEAD"))
        report = dialogue.validate(require_git=True)
        self.assertTrue(report["ok"], report["errors"])

    def test_dialogue_in_linked_worktree(self) -> None:
        main = support.init_git_repo(self.base / "main")
        (main / "README.md").write_text("host repo\n", encoding="utf-8")
        support.git(main, "add", "README.md")
        support.git(main, "commit", "-q", "-m", "base")
        base_commit = head(main)
        worktree = self.base / "wt"
        support.git(main, "worktree", "add", "-q", str(worktree), "-b", "dialogue-run")
        dialogue = engine.init_dialogue(self.definition, worktree / "reviews" / "demo")
        self.run_dialogue(dialogue)
        self.assertEqual(commit_count(worktree), 5)  # base + init + 2 turns + decision
        report = dialogue.validate(require_git=True)
        self.assertTrue(report["ok"], report["errors"])
        # The linked worktree's branch advanced; the main checkout did not.
        self.assertEqual(head(main), base_commit)
        branch = support.git(
            worktree, "rev-parse", "--abbrev-ref", "HEAD"
        ).stdout.strip()
        self.assertEqual(branch, "dialogue-run")


if __name__ == "__main__":
    unittest.main()
