"""Task 2: deterministic state, atomic claims, hard final stop."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import config, engine


class EngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = support.init_git_repo(Path(self._tmp.name))
        self.definition = config.parse_definition(support.two_actor_definition())
        self.dialogue_dir = self.root / "dialogue"

    def init_dialogue(self) -> engine.Dialogue:
        return engine.init_dialogue(self.definition, self.dialogue_dir)


class InitTests(EngineTestCase):
    def continuation_definition(self) -> tuple[config.ProtocolDefinition, Path]:
        artifact = self.root / "prior" / "R00-worker.md"
        artifact.parent.mkdir()
        artifact.write_text("# R00\n\nPublished prior turn.\n", encoding="utf-8")
        support.git(self.root, "add", "prior/R00-worker.md")
        support.git(self.root, "commit", "-q", "-m", "publish prior turn")
        published = support.git(self.root, "rev-parse", "HEAD").stdout.strip()
        marker = self.root / "prior" / "marker.txt"
        marker.write_text("original dialogue head\n", encoding="utf-8")
        support.git(self.root, "add", "prior/marker.txt")
        support.git(self.root, "commit", "-q", "-m", "close prior dialogue")
        original_head = support.git(self.root, "rev-parse", "HEAD").stdout.strip()
        raw = support.two_actor_definition()
        raw["continuation"] = {
            "protocol_id": "prior-dialogue",
            "round_id": "R00",
            "artifact_path": str(artifact),
            "artifact_sha256": engine.artifacts.sha256_file(artifact),
            "published_commit": published,
            "original_dialogue_head": original_head,
            "start_round": "R01",
        }
        return config.parse_definition(raw), artifact

    def test_continuation_anchor_is_rechecked_after_init(self) -> None:
        self.definition, artifact = self.continuation_definition()
        dialogue = self.init_dialogue()
        self.assertEqual(dialogue.state()["status"], "OPEN")
        artifact.write_text("tampered prior turn\n", encoding="utf-8")
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.state()
        self.assertIn("continuation artifact hash mismatch", str(ctx.exception))

    def test_init_creates_state_and_definition(self) -> None:
        dialogue = self.init_dialogue()
        self.assertTrue((self.dialogue_dir / "definition.json").is_file())
        self.assertTrue((self.dialogue_dir / "state.json").is_file())
        state = dialogue.state()
        self.assertEqual(state["status"], "OPEN")
        self.assertEqual(state["turn_index"], 0)
        self.assertEqual(state["revision"], 0)
        self.assertIsNone(state["claim"])
        self.assertEqual(state["definition_digest"], self.definition.digest())
        self.assertEqual(state["completed_turns"], [])

    def test_init_refuses_existing_dialogue(self) -> None:
        self.init_dialogue()
        with self.assertRaises(engine.ProtocolError):
            self.init_dialogue()

    def test_init_refuses_symlinked_directory(self) -> None:
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        os.symlink(real, link)
        with self.assertRaises(engine.ProtocolError):
            engine.init_dialogue(self.definition, link / "dialogue")

    def test_open_refuses_definition_tampering(self) -> None:
        self.init_dialogue()
        definition_path = self.dialogue_dir / "definition.json"
        raw = json.loads(definition_path.read_text(encoding="utf-8"))
        raw["owner"] = "impostor"
        definition_path.write_text(json.dumps(raw), encoding="utf-8")
        dialogue = engine.Dialogue(self.dialogue_dir)
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.state()
        self.assertIn("digest", str(ctx.exception))

    def test_no_temp_files_left_behind(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        leftovers = [p.name for p in self.dialogue_dir.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class NextTurnTests(EngineTestCase):
    def test_next_turn_is_first_scheduled(self) -> None:
        dialogue = self.init_dialogue()
        turn = dialogue.next_turn()
        assert turn is not None
        self.assertEqual(turn.round_id, "R01")
        self.assertEqual(turn.actor_id, "worker-a")

    def test_next_turn_none_after_final(self) -> None:
        dialogue = self.init_dialogue()
        self._force_state(turn_index=4, status="READY_FOR_OWNER")
        self.assertIsNone(dialogue.next_turn())

    def _force_state(self, **overrides) -> None:
        path = self.dialogue_dir / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(overrides)
        path.write_text(json.dumps(state), encoding="utf-8")


class ClaimTests(EngineTestCase):
    def enable_substitute_for_r01(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"].append(
            {
                "actor_id": "worker-c",
                "role": "proposer",
                "transport": "command",
                "expected_provider": "prov-c",
                "expected_model": "model-c",
                "settings": {
                    "argv": ["fake-worker"],
                    "identity_verifier_argv": ["fake-verifier"],
                },
            }
        )
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-c"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.definition = config.parse_definition(raw)

    def test_correct_claim_locks_turn(self) -> None:
        dialogue = self.init_dialogue()
        state = dialogue.claim("worker-a")
        self.assertEqual(state["status"], "CLAIMED")
        self.assertEqual(state["claim"]["actor_id"], "worker-a")
        self.assertTrue(state["claim"]["nonce"])
        self.assertEqual(state["revision"], 1)
        self.assertTrue((self.dialogue_dir / engine.LOCK_FILE).exists())

    def test_wrong_actor_cannot_claim(self) -> None:
        dialogue = self.init_dialogue()
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.claim("worker-b")
        self.assertIn("not an allowed actor", str(ctx.exception))
        self.assertEqual(dialogue.state()["status"], "OPEN")
        self.assertFalse((self.dialogue_dir / engine.LOCK_FILE).exists())

    def test_preapproved_substitute_can_claim_without_impersonating_primary(self) -> None:
        self.enable_substitute_for_r01()
        dialogue = self.init_dialogue()
        state = dialogue.claim("worker-c", substitution_reason="provider_cooldown")
        self.assertEqual(state["claim"]["actor_id"], "worker-c")
        self.assertEqual(state["claim"]["scheduled_actor_id"], "worker-a")
        self.assertEqual(state["claim"]["actor_selection"], "substitute")
        self.assertEqual(state["claim"]["substitution_reason"], "provider_cooldown")

    def test_substitute_requires_frozen_reason_code(self) -> None:
        self.enable_substitute_for_r01()
        dialogue = self.init_dialogue()
        with self.assertRaises(engine.ProtocolError) as missing:
            dialogue.claim("worker-c")
        self.assertIn("substitution reason", str(missing.exception))
        with self.assertRaises(engine.ProtocolError) as unknown:
            dialogue.claim("worker-c", substitution_reason="operator_preference")
        self.assertIn("not allowed", str(unknown.exception))

    def test_primary_actor_rejects_substitution_reason(self) -> None:
        dialogue = self.init_dialogue()
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.claim("worker-a", substitution_reason="provider_cooldown")
        self.assertIn("primary actor", str(ctx.exception))
        self.assertIsNone(dialogue.state()["claim"])

    def test_active_substitute_claim_requires_explicit_identity_fields(self) -> None:
        self.enable_substitute_for_r01()
        dialogue = self.init_dialogue()
        dialogue.claim("worker-c", substitution_reason="provider_cooldown")
        state = json.loads(dialogue.state_path.read_text(encoding="utf-8"))
        for key in (
            "scheduled_actor_id",
            "actor_selection",
            "substitution_reason",
        ):
            del state["claim"][key]
        dialogue.state_path.write_text(json.dumps(state), encoding="utf-8")
        report = dialogue.validate()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("explicit actor identity fields" in item for item in report["errors"]),
            report["errors"],
        )

    def test_primary_claim_on_substitute_capable_turn_is_explicit(self) -> None:
        self.enable_substitute_for_r01()
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        state = json.loads(dialogue.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["claim"]["scheduled_actor_id"], "worker-a")
        self.assertEqual(state["claim"]["actor_selection"], "primary")
        self.assertIsNone(state["claim"]["substitution_reason"])
        for key in (
            "scheduled_actor_id",
            "actor_selection",
            "substitution_reason",
        ):
            del state["claim"][key]
        dialogue.state_path.write_text(json.dumps(state), encoding="utf-8")
        report = dialogue.validate()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("explicit actor identity fields" in item for item in report["errors"]),
            report["errors"],
        )

    def test_unknown_actor_cannot_claim(self) -> None:
        dialogue = self.init_dialogue()
        with self.assertRaises(engine.ProtocolError):
            dialogue.claim("ghost")

    def test_duplicate_claim_rejected(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.claim("worker-a")
        self.assertIn("claim", str(ctx.exception).lower())

    def test_second_writer_rejected_by_lock_even_with_stale_state(self) -> None:
        # Simulate a racing writer that saw OPEN state but lost the lock race:
        # the lock file already exists, so the claim must fail closed.
        dialogue = self.init_dialogue()
        (self.dialogue_dir / engine.LOCK_FILE).write_text("{}", encoding="utf-8")
        with self.assertRaises(engine.ProtocolError):
            dialogue.claim("worker-a")

    def test_stale_revision_rejected(self) -> None:
        dialogue = self.init_dialogue()
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.claim("worker-a", expected_revision=7)
        self.assertIn("revision", str(ctx.exception))

    def test_matching_revision_accepted(self) -> None:
        dialogue = self.init_dialogue()
        state = dialogue.claim("worker-a", expected_revision=0)
        self.assertEqual(state["status"], "CLAIMED")

    def test_release_requires_claiming_actor(self) -> None:
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        with self.assertRaises(engine.ProtocolError):
            dialogue.release("worker-b")
        state = dialogue.release("worker-a")
        self.assertEqual(state["status"], "OPEN")
        self.assertIsNone(state["claim"])
        self.assertFalse((self.dialogue_dir / engine.LOCK_FILE).exists())


class FinalStopTests(EngineTestCase):
    def test_final_agent_status_is_distinct_from_owner_decision(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"] = raw["schedule"][:1]
        raw["final_round_id"] = "R01"
        raw["agent_final_statuses"] = [
            "READY_FOR_OWNER",
            "AGENT_NEEDS_MORE_EVIDENCE",
            "SPLIT_QUESTION",
        ]
        self.definition = config.parse_definition(raw)
        dialogue = self.init_dialogue()
        dialogue.claim("worker-a")
        scratch = self.root / "scratch"
        scratch.mkdir()
        turn_path = scratch / "R01.md"
        evidence_path = scratch / "R01.json"

        def write_body(body: str) -> None:
            turn_path.write_text(body, encoding="utf-8")
            record = support.make_evidence(
                actor_id="worker-a",
                round_id="R01",
                artifact_path=turn_path,
                provider="fake-provider-a",
                model="fake-model-a",
            )
            evidence_path.write_text(json.dumps(record), encoding="utf-8")

        def write_attempt(status: str) -> None:
            write_body(f"# R01\n\nFinal analysis.\n\nStatus: {status}\n")

        for malformed in (
            "# R01\n\nFinal analysis.\n\nStatus:\nREADY_FOR_OWNER\n",
            "# R01\n\nFinal analysis.\n\nStatus:\n\nREADY_FOR_OWNER\n",
        ):
            write_body(malformed)
            with self.assertRaises(engine.ProtocolError) as ctx:
                dialogue.complete("worker-a", turn_path, evidence_path)
            self.assertIn("exactly one line", str(ctx.exception))
            self.assertEqual(dialogue.state()["status"], "CLAIMED")

        write_attempt("APPROVE")
        with self.assertRaises(engine.ProtocolError) as ctx:
            dialogue.complete("worker-a", turn_path, evidence_path)
        self.assertIn("not allowed", str(ctx.exception))
        self.assertEqual(dialogue.state()["status"], "CLAIMED")

        write_attempt("READY_FOR_OWNER")
        state = dialogue.complete("worker-a", turn_path, evidence_path)
        self.assertEqual(state["status"], "READY_FOR_OWNER")
        self.assertIsNone(state["owner_decision"])

    def _finish_all_turns(self) -> None:
        path = self.dialogue_dir / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["turn_index"] = len(self.definition.schedule)
        state["status"] = "READY_FOR_OWNER"
        path.write_text(json.dumps(state), encoding="utf-8")

    def test_claim_after_final_turn_fails_closed(self) -> None:
        dialogue = self.init_dialogue()
        self._finish_all_turns()
        for actor in ("worker-a", "worker-b"):
            with self.assertRaises(engine.ProtocolError) as ctx:
                dialogue.claim(actor)
            self.assertIn("final", str(ctx.exception).lower())

    def test_claim_after_owner_decision_fails_closed(self) -> None:
        dialogue = self.init_dialogue()
        path = self.dialogue_dir / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["turn_index"] = len(self.definition.schedule)
        state["status"] = "OWNER_DECIDED"
        path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(engine.ProtocolError):
            dialogue.claim("worker-a")


if __name__ == "__main__":
    unittest.main()
