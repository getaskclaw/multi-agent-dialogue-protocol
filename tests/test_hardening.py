"""RED → GREEN: publication hardening and honest owner authority.

- Artifact/state/work publication must reject symlink and path-swap
  attacks with exclusive no-follow writes — planting a symlinked parent
  or swapping bytes between validation and publication must never place
  protocol-written bytes outside the dialogue directory or publish
  bytes that were not validated.
- ``owner-decide`` is a separate terminal transition, but the caller's
  identity is NOT authenticated unless an external owner-proof verifier
  is configured; the recorded decision must say so instead of implying
  cryptographic owner identity.
- Runtime-evidence proof objects must carry a structured ``kind``; a
  bare non-empty dict is not an adapter reference.
- scripts/verify.py's Git hygiene must keep working in normal clones
  AND linked Git worktrees (where ``.git`` is a file, not a directory).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import support

from multi_agent_dialogue import artifacts, config, engine, evidence

sys.path.insert(0, str(support.REPO_ROOT / "scripts"))
import verify  # noqa: E402


def make_evidence(*, actor_id: str, round_id: str, artifact_path: Path,
                  provider: str, model: str, **overrides) -> dict:
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    record = {
        "evidence_version": 1,
        "actor_id": actor_id,
        "round_id": round_id,
        "adapter": "command",
        "transport": "command",
        "provider": provider,
        "model": model,
        "session_id": f"run-{actor_id}-{round_id}",
        "outcome": "success",
        "exit_status": 0,
        "artifact_path": str(artifact_path),
        "artifact_sha256": digest,
        "captured_at": "2026-07-16T00:00:00Z",
        "proof": {
            "kind": "external-command-verifier",
            "verifier_argv": ["fake-verifier"],
            "report": {"fake": True},
        },
    }
    record.update(overrides)
    return record


TWO_TURN_DEFINITION = {
    "protocol_id": "hardening-demo",
    "version": 1,
    "owner": "owner-human",
    "actors": [
        {
            "actor_id": "worker-a",
            "role": "proposer",
            "transport": "command",
            "expected_provider": "prov-a",
            "expected_model": "model-a",
            "settings": {"argv": ["fake-worker"], "identity_verifier_argv": ["fake-verifier"]},
        },
        {
            "actor_id": "worker-b",
            "role": "challenger",
            "transport": "command",
            "expected_provider": "prov-b",
            "expected_model": "model-b",
            "settings": {"argv": ["fake-worker"], "identity_verifier_argv": ["fake-verifier"]},
        },
    ],
    "schedule": [
        {"round_id": "R01", "actor_id": "worker-a", "purpose": "p", "artifact_kind": "proposal"},
        {"round_id": "R02", "actor_id": "worker-b", "purpose": "c", "artifact_kind": "challenge"},
    ],
    "final_round_id": "R02",
}


class DialogueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = support.init_git_repo(Path(self._tmp.name))
        self.work = self.root / "scratch"
        self.work.mkdir()
        raw = json.loads(json.dumps(TWO_TURN_DEFINITION))
        self.definition = config.parse_definition(raw)
        self.dialogue = engine.init_dialogue(self.definition, self.root / "dialogue")

    def turn_and_evidence(self, actor_id: str, round_id: str) -> tuple[Path, Path]:
        turn_path = self.work / f"{round_id}.md"
        turn_path.write_text(f"# {round_id}\n\nbody for {round_id}\n", encoding="utf-8")
        actor = self.definition.actor(actor_id)
        record = make_evidence(
            actor_id=actor_id,
            round_id=round_id,
            artifact_path=turn_path,
            provider=actor.expected_provider,
            model=actor.expected_model,
        )
        evidence_path = self.work / f"{round_id}-evidence.json"
        evidence_path.write_text(json.dumps(record), encoding="utf-8")
        return turn_path, evidence_path

    def complete(self, actor_id: str, round_id: str) -> dict:
        self.dialogue.claim(actor_id)
        turn_path, evidence_path = self.turn_and_evidence(actor_id, round_id)
        return self.dialogue.complete(actor_id, turn_path, evidence_path)


class PublishHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src = self.root / "src.md"
        self.src.write_text("published body\n", encoding="utf-8")

    def test_symlinked_parent_directory_is_rejected(self) -> None:
        # Classic path swap: the publication directory itself is a symlink
        # pointing outside. Following it would write attacker-chosen paths.
        outside = self.root / "outside"
        outside.mkdir()
        turns = self.root / "turns"
        os.symlink(outside, turns)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.publish(self.src, turns / "R01.md")
        self.assertEqual(list(outside.iterdir()), [], "no bytes may escape through the symlink")

    def test_expected_sha_pins_published_bytes(self) -> None:
        # publish() must be able to pin the exact validated bytes so a file
        # swapped between hashing and publication fails closed.
        target = self.root / "out" / "R01.md"
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.publish(self.src, target, expected_sha="0" * 64)
        self.assertFalse(target.exists())
        good = hashlib.sha256(self.src.read_bytes()).hexdigest()
        published = artifacts.publish(self.src, target, expected_sha=good)
        self.assertEqual(published, good)


class CompletionPublicationTests(DialogueTestCase):
    def test_turns_directory_symlink_escape_is_rejected(self) -> None:
        outside = self.root / "outside-turns"
        outside.mkdir()
        turns = self.dialogue.directory / engine.TURNS_DIR
        turns.rmdir()
        os.symlink(outside, turns)
        with self.assertRaises(engine.ProtocolError):
            self.complete("worker-a", "R01")
        self.assertEqual(
            list(outside.iterdir()), [],
            "publication through a symlinked turns/ directory must not escape",
        )

    def test_complete_publishes_exactly_the_validated_bytes(self) -> None:
        # Adversarial race: the turn file is swapped after the engine hashed
        # and validated it but before publication. The published artifact
        # must be the validated bytes (or the completion must fail) — never
        # silently the swapped bytes.
        self.dialogue.claim("worker-a")
        turn_path, evidence_path = self.turn_and_evidence("worker-a", "R01")
        original = turn_path.read_text(encoding="utf-8")

        real_validate = evidence.validate_evidence

        def swap_then_validate(record, **kwargs):
            turn_path.write_text("ATTACKER SWAPPED CONTENT\n", encoding="utf-8")
            return real_validate(record, **kwargs)

        with mock.patch.object(engine.evidence, "validate_evidence", side_effect=swap_then_validate):
            state = self.dialogue.complete("worker-a", turn_path, evidence_path)

        record = state["completed_turns"][0]
        published = (self.dialogue.directory / record["artifact_file"]).read_text(encoding="utf-8")
        self.assertEqual(published, original)
        self.assertEqual(
            record["artifact_sha256"],
            hashlib.sha256(original.encode("utf-8")).hexdigest(),
        )


class WorkFileHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))

    def test_symlinked_task_file_is_never_followed(self) -> None:
        from multi_agent_dialogue import runner

        raw = json.loads(json.dumps(TWO_TURN_DEFINITION))
        worker = str(support.FAKE_BIN / "fake-worker")
        verifier = str(support.FAKE_BIN / "fake-verifier")
        for actor in raw["actors"]:
            actor["settings"] = {
                "argv": [worker, "--task", "{task_file}", "--turn-output", "{turn_file}",
                         "--round", "{round_id}", "--actor", "{actor_id}"],
                "identity_verifier_argv": [verifier, "--turn", "{turn_file}",
                                           "--round", "{round_id}", "--actor", "{actor_id}"],
                "env": {
                    "FAKE_PROVIDER": actor["expected_provider"],
                    "FAKE_MODEL": actor["expected_model"],
                },
            }
        definition = config.parse_definition(raw)
        dialogue = engine.init_dialogue(definition, self.base / "dialogue")

        outside = self.base / "outside-task.md"
        work_dir = dialogue.directory / "work" / "R01"
        work_dir.mkdir(parents=True)
        os.symlink(outside, work_dir / "task.md")

        with self.assertRaises(engine.ProtocolError):
            runner.launch(dialogue, "worker-a")
        self.assertFalse(outside.exists(), "the task briefing must not be written through a symlink")
        self.assertEqual(dialogue.state()["turn_index"], 0)


class OwnerAuthorityTests(DialogueTestCase):
    def finish(self) -> None:
        self.complete("worker-a", "R01")
        self.complete("worker-b", "R02")

    def decision_file(self, text: str) -> Path:
        path = self.work / "decision.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_decision_is_marked_unverified_without_owner_proof(self) -> None:
        self.finish()
        state = self.dialogue.owner_decide(self.decision_file("Decision: APPROVE\n"))
        decision = state["owner_decision"]
        self.assertEqual(decision["caller_identity"], "unverified")
        self.assertIsNone(decision["owner_proof"])

    def test_configured_owner_proof_verifier_marks_verified(self) -> None:
        raw = json.loads(json.dumps(TWO_TURN_DEFINITION))
        raw["protocol_id"] = "hardening-owner-proof"
        raw["owner_proof_argv"] = [
            sys.executable,
            "-c",
            "import sys; raise SystemExit(0 if 'APPROVE' in open(sys.argv[1], encoding='utf-8').read() else 1)",
            "{decision_file}",
        ]
        definition = config.parse_definition(raw)
        self.assertEqual(tuple(definition.owner_proof_argv), tuple(raw["owner_proof_argv"]))
        dialogue = engine.init_dialogue(definition, self.root / "dialogue-proof")
        self.dialogue = dialogue
        self.definition = definition
        self.finish()
        state = dialogue.owner_decide(self.decision_file("Decision: APPROVE\n"))
        decision = state["owner_decision"]
        self.assertEqual(decision["caller_identity"], "externally-verified")
        self.assertTrue(decision["owner_proof"])

    def test_failing_owner_proof_verifier_blocks_decision(self) -> None:
        raw = json.loads(json.dumps(TWO_TURN_DEFINITION))
        raw["protocol_id"] = "hardening-owner-proof-fail"
        raw["owner_proof_argv"] = [
            sys.executable,
            "-c",
            "import sys; raise SystemExit(0 if 'APPROVE' in open(sys.argv[1], encoding='utf-8').read() else 1)",
            "{decision_file}",
        ]
        definition = config.parse_definition(raw)
        dialogue = engine.init_dialogue(definition, self.root / "dialogue-proof-fail")
        self.dialogue = dialogue
        self.definition = definition
        self.finish()
        with self.assertRaises(engine.ProtocolError):
            dialogue.owner_decide(self.decision_file("Decision: REJECT\n"))
        state = dialogue.state()
        self.assertEqual(state["status"], engine.STATUS_READY_FOR_OWNER)
        self.assertFalse((dialogue.directory / engine.OWNER_DECISION_FILE).exists())


class EvidenceProofKindTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.definition = config.parse_definition(json.loads(json.dumps(TWO_TURN_DEFINITION)))
        self.artifact = self.root / "turn.md"
        self.artifact.write_text("# turn\n\nbody\n", encoding="utf-8")

    def test_proof_without_kind_is_rejected(self) -> None:
        record = make_evidence(
            actor_id="worker-a",
            round_id="R01",
            artifact_path=self.artifact,
            provider="prov-a",
            model="model-a",
            proof={"argv": ["fake-worker"]},
        )
        errors = evidence.validate_evidence(
            record,
            actor=self.definition.actor("worker-a"),
            turn=self.definition.schedule[0],
            artifact_sha256=hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
        )
        self.assertTrue(any("kind" in error for error in errors), errors)


class VerifyGitWorktreeTests(unittest.TestCase):
    def test_git_backed_handles_clone_and_linked_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(verify.git_backed(root))
            clone = root / "clone"
            (clone / ".git").mkdir(parents=True)
            self.assertTrue(verify.git_backed(clone))
            worktree = root / "worktree"
            worktree.mkdir()
            (worktree / ".git").write_text(
                "gitdir: /somewhere/.git/worktrees/x\n", encoding="utf-8"
            )
            self.assertTrue(verify.git_backed(worktree))


if __name__ == "__main__":
    unittest.main()
