"""Task 3: runtime-evidence validation and immutable completion gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import config, engine, evidence


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.definition = config.parse_definition(support.two_actor_definition())
        self.actor = self.definition.actor("worker-a")
        self.turn = self.definition.schedule[0]
        self.artifact = self.root / "turn.md"
        self.artifact.write_text("# Proposal\n\nBody words here.\n", encoding="utf-8")

    def valid_evidence(self, **overrides) -> dict:
        base = {
            "actor_id": "worker-a",
            "round_id": "R01",
            "artifact_path": self.artifact,
            "provider": "fake-provider-a",
            "model": "fake-model-a",
        }
        return support.make_evidence(**{**base, **overrides})

    def check(self, record: dict) -> list[str]:
        return evidence.validate_evidence(
            record,
            actor=self.actor,
            turn=self.turn,
            artifact_sha256=evidence.sha256_file(self.artifact),
        )

    def test_valid_evidence_passes(self) -> None:
        self.assertEqual(self.check(self.valid_evidence()), [])

    def test_missing_field_rejected(self) -> None:
        record = self.valid_evidence()
        del record["session_id"]
        self.assertTrue(any("session_id" in e for e in self.check(record)))

    def test_wrong_actor_rejected(self) -> None:
        errors = self.check(self.valid_evidence(actor_id="worker-b"))
        self.assertTrue(any("actor" in e for e in errors))

    def test_wrong_round_rejected(self) -> None:
        errors = self.check(self.valid_evidence(round_id="R02"))
        self.assertTrue(any("round" in e for e in errors))

    def test_provider_mismatch_rejected(self) -> None:
        errors = self.check(self.valid_evidence(provider="someone-else"))
        self.assertTrue(any("provider" in e for e in errors))

    def test_model_mismatch_rejected(self) -> None:
        errors = self.check(self.valid_evidence(model="other-model"))
        self.assertTrue(any("model" in e for e in errors))

    def test_transport_mismatch_rejected(self) -> None:
        errors = self.check(self.valid_evidence(transport="hermes-cli"))
        self.assertTrue(any("transport" in e for e in errors))

    def test_failed_outcome_rejected(self) -> None:
        errors = self.check(self.valid_evidence(outcome="failure"))
        self.assertTrue(any("outcome" in e for e in errors))

    def test_nonzero_exit_rejected(self) -> None:
        errors = self.check(self.valid_evidence(exit_status=3))
        self.assertTrue(any("exit" in e for e in errors))

    def test_empty_session_rejected(self) -> None:
        errors = self.check(self.valid_evidence(session_id=""))
        self.assertTrue(any("session" in e for e in errors))

    def test_artifact_sha_mismatch_rejected(self) -> None:
        errors = self.check(self.valid_evidence(artifact_sha256="0" * 64))
        self.assertTrue(any("sha" in e.lower() for e in errors))

    def test_bad_timestamp_rejected(self) -> None:
        errors = self.check(self.valid_evidence(captured_at="yesterday"))
        self.assertTrue(any("captured_at" in e for e in errors))

    def test_empty_proof_rejected(self) -> None:
        errors = self.check(self.valid_evidence(proof={}))
        self.assertTrue(any("proof" in e for e in errors))

    def test_markdown_labels_are_not_evidence(self) -> None:
        # A record consisting only of frontmatter-style claims must fail.
        record = {"actor": "worker-a", "model": "fake-model-a", "round": "R01"}
        errors = self.check(record)
        self.assertTrue(errors)

    def test_load_evidence_strict_json(self) -> None:
        path = self.root / "evidence.json"
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(evidence.EvidenceError):
            evidence.load_evidence(path)
        with self.assertRaises(evidence.EvidenceError):
            evidence.load_evidence(self.root / "missing.json")


class CompletionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = support.init_git_repo(Path(self._tmp.name))
        self.definition = config.parse_definition(support.two_actor_definition())
        self.dialogue = engine.init_dialogue(self.definition, self.root / "dialogue")
        self.work = self.root / "work"
        self.work.mkdir()

    def provider_model(self, actor_id: str) -> tuple[str, str]:
        actor = self.definition.actor(actor_id)
        return actor.expected_provider, actor.expected_model

    def write_turn(self, round_id: str, text: str | None = None) -> Path:
        path = self.work / f"{round_id}-turn.md"
        path.write_text(text or f"# {round_id}\n\ncontent for {round_id}\n", encoding="utf-8")
        return path

    def write_evidence(self, record: dict, name: str) -> Path:
        path = self.work / name
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def complete_turn(self, actor_id: str, round_id: str, **overrides):
        self.dialogue.claim(actor_id)
        turn_path = self.write_turn(round_id)
        provider, model = self.provider_model(actor_id)
        if "evidence_round_id" in overrides:
            overrides["round_id"] = overrides.pop("evidence_round_id")
        base = {
            "actor_id": actor_id,
            "round_id": round_id,
            "artifact_path": turn_path,
            "provider": provider,
            "model": model,
        }
        record = support.make_evidence(**{**base, **overrides})
        evidence_path = self.write_evidence(record, f"{round_id}-evidence.json")
        return self.dialogue.complete(actor_id, turn_path, evidence_path)


class CompletionTests(CompletionTestCase):
    def test_successful_completion_advances(self) -> None:
        state = self.complete_turn("worker-a", "R01")
        self.assertEqual(state["turn_index"], 1)
        self.assertEqual(state["status"], "OPEN")
        self.assertIsNone(state["claim"])
        record = state["completed_turns"][0]
        self.assertEqual(record["round_id"], "R01")
        self.assertEqual(record["actor_id"], "worker-a")
        published = self.dialogue.directory / record["artifact_file"]
        self.assertTrue(published.is_file())
        self.assertEqual(len(record["artifact_sha256"]), 64)
        self.assertTrue((self.dialogue.directory / record["evidence_file"]).is_file())
        self.assertFalse(self.dialogue.lock_path.exists())

    def test_final_completion_is_ready_for_owner(self) -> None:
        self.complete_turn("worker-a", "R01")
        self.complete_turn("worker-b", "R02")
        self.complete_turn("worker-a", "R03")
        state = self.complete_turn("worker-b", "R04")
        self.assertEqual(state["status"], "READY_FOR_OWNER")
        self.assertEqual(state["turn_index"], 4)

    def test_complete_requires_claim(self) -> None:
        turn_path = self.write_turn("R01")
        provider, model = self.provider_model("worker-a")
        record = support.make_evidence(
            actor_id="worker-a", round_id="R01", artifact_path=turn_path,
            provider=provider, model=model,
        )
        evidence_path = self.write_evidence(record, "e.json")
        with self.assertRaises(engine.ProtocolError) as ctx:
            self.dialogue.complete("worker-a", turn_path, evidence_path)
        self.assertIn("claim", str(ctx.exception).lower())

    def test_complete_by_other_actor_rejected(self) -> None:
        self.dialogue.claim("worker-a")
        turn_path = self.write_turn("R01")
        provider, model = self.provider_model("worker-b")
        record = support.make_evidence(
            actor_id="worker-b", round_id="R01", artifact_path=turn_path,
            provider=provider, model=model,
        )
        evidence_path = self.write_evidence(record, "e.json")
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.complete("worker-b", turn_path, evidence_path)

    def test_missing_evidence_blocks_completion(self) -> None:
        self.dialogue.claim("worker-a")
        turn_path = self.write_turn("R01")
        with self.assertRaises(engine.ProtocolError) as ctx:
            self.dialogue.complete("worker-a", turn_path, self.work / "nope.json")
        self.assertIn("evidence", str(ctx.exception).lower())

    def test_fake_identity_blocks_completion(self) -> None:
        with self.assertRaises(engine.ProtocolError) as ctx:
            self.complete_turn("worker-a", "R01", model="impersonated-model")
        self.assertIn("model", str(ctx.exception))
        # State must not advance.
        state = self.dialogue.state()
        self.assertEqual(state["turn_index"], 0)

    def test_failed_terminal_outcome_blocks_completion(self) -> None:
        with self.assertRaises(engine.ProtocolError):
            self.complete_turn("worker-a", "R01", outcome="failure", exit_status=1)

    def test_wrong_round_evidence_blocks_completion(self) -> None:
        with self.assertRaises(engine.ProtocolError):
            self.complete_turn("worker-a", "R01", evidence_round_id="R03")

    def test_word_limit_enforced(self) -> None:
        self.dialogue.claim("worker-a")
        turn_path = self.write_turn("R01", "word " * 800)
        provider, model = self.provider_model("worker-a")
        record = support.make_evidence(
            actor_id="worker-a", round_id="R01", artifact_path=turn_path,
            provider=provider, model=model,
        )
        evidence_path = self.write_evidence(record, "e.json")
        with self.assertRaises(engine.ProtocolError) as ctx:
            self.dialogue.complete("worker-a", turn_path, evidence_path)
        self.assertIn("word", str(ctx.exception).lower())

    def test_reused_session_id_rejected(self) -> None:
        self.complete_turn("worker-a", "R01", session_id="same-session")
        with self.assertRaises(engine.ProtocolError) as ctx:
            self.complete_turn("worker-b", "R02", session_id="same-session")
        self.assertIn("session", str(ctx.exception).lower())

    def test_mutated_prior_turn_blocks_dialogue(self) -> None:
        state = self.complete_turn("worker-a", "R01")
        published = self.dialogue.directory / state["completed_turns"][0]["artifact_file"]
        published.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(engine.ProtocolError) as ctx:
            self.complete_turn("worker-b", "R02")
        self.assertIn("immutab", str(ctx.exception).lower())
        self.assertEqual(self.dialogue._read_state()["status"], "BLOCKED")

    def test_post_final_completion_fails_closed(self) -> None:
        for actor_id, round_id in (
            ("worker-a", "R01"), ("worker-b", "R02"),
            ("worker-a", "R03"), ("worker-b", "R04"),
        ):
            self.complete_turn(actor_id, round_id)
        with self.assertRaises(engine.ProtocolError):
            self.complete_turn("worker-a", "R05")


class OwnerDecisionTests(CompletionTestCase):
    def finish_dialogue(self) -> None:
        for actor_id, round_id in (
            ("worker-a", "R01"), ("worker-b", "R02"),
            ("worker-a", "R03"), ("worker-b", "R04"),
        ):
            self.complete_turn(actor_id, round_id)

    def decision_file(self, text: str) -> Path:
        path = self.work / "decision.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_owner_decide_requires_ready_state(self) -> None:
        path = self.decision_file("Decision: APPROVE\n\nLooks good.\n")
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.owner_decide(path)

    def test_owner_decide_records_decision(self) -> None:
        self.finish_dialogue()
        path = self.decision_file("Decision: APPROVE\n\nShip it.\n")
        state = self.dialogue.owner_decide(path)
        self.assertEqual(state["status"], "OWNER_DECIDED")
        self.assertEqual(state["owner_decision"]["decision"], "APPROVE")
        self.assertTrue((self.dialogue.directory / "OWNER-DECISION.md").is_file())

    def test_owner_decide_rejects_unknown_decision(self) -> None:
        self.finish_dialogue()
        path = self.decision_file("Decision: MAYBE\n")
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.owner_decide(path)

    def test_owner_decide_rejects_missing_decision_line(self) -> None:
        self.finish_dialogue()
        path = self.decision_file("I approve of this.\n")
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.owner_decide(path)

    def test_owner_decide_is_terminal(self) -> None:
        self.finish_dialogue()
        self.dialogue.owner_decide(self.decision_file("Decision: APPROVE\n"))
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.owner_decide(self.decision_file("Decision: REJECT\n"))
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.claim("worker-a")

    def test_owner_decide_blocks_on_tampered_artifacts(self) -> None:
        self.finish_dialogue()
        state = self.dialogue.state()
        published = self.dialogue.directory / state["completed_turns"][2]["artifact_file"]
        published.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.owner_decide(self.decision_file("Decision: APPROVE\n"))


class ValidateTests(CompletionTestCase):
    def test_validate_passes_clean_dialogue(self) -> None:
        self.complete_turn("worker-a", "R01")
        report = self.dialogue.validate()
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["status"], "OPEN")
        self.assertTrue(report["ok"])

    def test_validate_reports_mutated_turn(self) -> None:
        state = self.complete_turn("worker-a", "R01")
        published = self.dialogue.directory / state["completed_turns"][0]["artifact_file"]
        published.write_text("tampered\n", encoding="utf-8")
        report = self.dialogue.validate()
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("sha" in e.lower() or "immutab" in e.lower() for e in report["errors"]))

    def test_validate_reports_mutated_evidence(self) -> None:
        state = self.complete_turn("worker-a", "R01")
        evidence_file = self.dialogue.directory / state["completed_turns"][0]["evidence_file"]
        record = json.loads(evidence_file.read_text(encoding="utf-8"))
        record["model"] = "revised-history-model"
        evidence_file.write_text(json.dumps(record), encoding="utf-8")
        report = self.dialogue.validate()
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
