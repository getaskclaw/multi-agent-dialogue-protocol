"""Task 6: full-dialogue proofs for two-Claude, two-Hermes, three-mixed.

Each topology runs against the committed example definition with the
deterministic fake runtimes on PATH. No external credentials, no real
agent, no network.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import support

from multi_agent_dialogue import config, engine, evidence, runner

EXAMPLES = support.REPO_ROOT / "examples"


class TopologyTestCase(unittest.TestCase):
    example: str = ""

    def setUp(self) -> None:
        if not self.example:
            self.skipTest("abstract topology case")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = support.init_git_repo(Path(self._tmp.name))
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(support.FAKE_BIN) + os.pathsep + self._old_path
        self.addCleanup(self._restore_path)
        self.definition = config.load_definition(
            EXAMPLES / self.example / "protocol.json"
        )
        self.dialogue = engine.init_dialogue(self.definition, self.base / "dialogue")
        self._write_fable_registry()

    def _write_fable_registry(self) -> None:
        """Mirror run.sh: the fable-session registry is host-local transport
        scratch, generated per run under the dialogue's ignored work/ area
        so it can never be committed."""
        entries = []
        for actor in self.definition.actors:
            if actor.transport != "fable-session":
                continue
            entries += [
                f"[project.{actor.settings['project']}]",
                f'repo = "{support.REPO_ROOT}"',
                f'profile = "{EXAMPLES / "fakes" / "fable-profile.toml"}"',
                f'model = "{actor.expected_model}"',
                'effort = "high"',
                'fallback = "stop"',
                'permission_mode = "auto"',
                f'tmux_prefix = "{actor.settings["tmux_prefix"]}"',
                "",
            ]
        if entries:
            registry = self.dialogue.directory / "work" / "fable" / "registry.toml"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text("\n".join(entries), encoding="utf-8")

    def _restore_path(self) -> None:
        os.environ["PATH"] = self._old_path

    def evidence_records(self) -> list[dict]:
        state = self.dialogue.state()
        return [
            evidence.load_evidence(self.dialogue.directory / record["evidence_file"])
            for record in state["completed_turns"]
        ]

    def run_full_dialogue(self) -> None:
        schedule = self.definition.schedule
        for position, turn in enumerate(schedule):
            # Proof: nobody else can claim this actor's turn.
            for actor in self.definition.actors:
                if actor.actor_id != turn.actor_id:
                    with self.assertRaises(engine.ProtocolError):
                        self.dialogue.claim(actor.actor_id)
            result = runner.launch(self.dialogue, turn.actor_id)
            self.assertEqual(result["completed_round"], turn.round_id)
            expected = (
                "READY_FOR_OWNER" if position == len(schedule) - 1 else "OPEN"
            )
            self.assertEqual(result["status"], expected)

    def assert_common_proofs(self) -> None:
        state = self.dialogue.state()
        # All turns occurred in configured order.
        self.assertEqual(
            [r["round_id"] for r in state["completed_turns"]],
            [t.round_id for t in self.definition.schedule],
        )
        self.assertEqual(
            [r["actor_id"] for r in state["completed_turns"]],
            [t.actor_id for t in self.definition.schedule],
        )
        self.assertEqual(state["status"], "READY_FOR_OWNER")

        # Every completion has independent, adapter-derived runtime evidence.
        records = self.evidence_records()
        sessions = [r["session_id"] for r in records]
        self.assertEqual(len(sessions), len(set(sessions)))
        for record, turn in zip(records, self.definition.schedule):
            actor = self.definition.actor(turn.actor_id)
            self.assertEqual(record["provider"], actor.expected_provider)
            self.assertEqual(record["model"], actor.expected_model)
            self.assertEqual(record["transport"], actor.transport)
            self.assert_honest_proof(record["proof"])

        # Another worker turn is rejected after the final round.
        for actor in self.definition.actors:
            with self.assertRaises(engine.ProtocolError):
                runner.launch(self.dialogue, actor.actor_id)

        # Validation passes before the decision — with full commit
        # provenance: one commit per turn, transport scratch ignored.
        report = self.dialogue.validate(require_git=True)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            [t["round_id"] for t in report["provenance"]["turn_commits"]],
            [t.round_id for t in self.definition.schedule],
        )

        # Only owner-decide reaches OWNER_DECIDED.
        decision = self.base / "decision.md"
        decision.write_text("Decision: APPROVE\n\nOwner rationale.\n", encoding="utf-8")
        final = self.dialogue.owner_decide(decision)
        self.assertEqual(final["status"], "OWNER_DECIDED")
        with self.assertRaises(engine.ProtocolError):
            self.dialogue.claim(self.definition.schedule[0].actor_id)

        # The decision is its own proven terminal commit.
        final_report = self.dialogue.validate(require_git=True)
        self.assertTrue(final_report["ok"], final_report["errors"])
        self.assertIn("owner_decision_commit", final_report["provenance"])

    def assert_honest_proof(self, proof: dict) -> None:
        """The proof names its external record family, and every fake fixture
        stays visibly fake — never upgraded to real identity proof."""
        kind = proof["kind"]
        self.assertIn(
            kind, {"fable-session", "hermes-state-db", "external-command-verifier"}
        )
        if kind == "fable-session":
            self.assertIn("fake-fable-session", proof["tool"])
            self.assertTrue(Path(proof["manifest_path"]).is_file())
        elif kind == "hermes-state-db":
            self.assertTrue(Path(proof["state_db"]).is_file())
        else:
            self.assertTrue(proof["report"]["fake"])


class TwoClaudeTopologyTests(TopologyTestCase):
    example = "two-claude"

    def test_full_dialogue(self) -> None:
        actors = {a.actor_id for a in self.definition.actors}
        self.assertEqual(len(actors), 2)
        self.assertTrue(
            all(a.transport == "fable-session" for a in self.definition.actors)
        )
        # claude-a owns odd rounds, claude-b owns even rounds.
        for position, turn in enumerate(self.definition.schedule):
            expected = "claude-a" if position % 2 == 0 else "claude-b"
            self.assertEqual(turn.actor_id, expected)
        self.run_full_dialogue()
        self.assert_common_proofs()
        records = self.evidence_records()
        # Four independent lanes across two registered projects.
        run_ids = {r["proof"]["run_id"] for r in records}
        self.assertEqual(len(run_ids), 4)
        projects = set()
        for record in records:
            manifest = json.loads(
                Path(record["proof"]["manifest_path"]).read_text(encoding="utf-8")
            )
            projects.add(manifest["project"])
        self.assertEqual(projects, {"claude-a", "claude-b"})


class TwoHermesTopologyTests(TopologyTestCase):
    example = "two-hermes"

    def test_full_dialogue(self) -> None:
        self.assertTrue(
            all(a.transport == "hermes-cli" for a in self.definition.actors)
        )
        self.run_full_dialogue()
        self.assert_common_proofs()
        records = self.evidence_records()
        homes = {r["proof"]["hermes_home"] for r in records}
        self.assertEqual(len(homes), 2, "two Hermes actors need two HERMES_HOMEs")
        databases = {r["proof"]["state_db"] for r in records}
        self.assertEqual(len(databases), 2, "identity comes from two separate state.dbs")


class ThreeMixedTopologyTests(TopologyTestCase):
    example = "three-mixed"

    def test_full_dialogue(self) -> None:
        transports = {a.transport for a in self.definition.actors}
        self.assertEqual(
            transports, {"fable-session", "hermes-cli", "command"},
            "three-mixed must span all three transports",
        )
        self.assertGreaterEqual(len(self.definition.actors), 3)
        self.run_full_dialogue()
        self.assert_common_proofs()
        records = self.evidence_records()
        self.assertEqual(
            {r["transport"] for r in records},
            {"fable-session", "hermes-cli", "command"},
        )


if __name__ == "__main__":
    unittest.main()
