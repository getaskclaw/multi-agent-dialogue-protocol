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
        self.assertIn("not the scheduled actor", str(ctx.exception))
        self.assertEqual(dialogue.state()["status"], "OPEN")
        self.assertFalse((self.dialogue_dir / engine.LOCK_FILE).exists())

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
