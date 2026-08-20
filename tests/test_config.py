"""Task 1: protocol definition loading, validation, canonical digest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
import support

from multi_agent_dialogue import config


class ValidDefinitionTests(unittest.TestCase):
    def test_parses_two_actor_definition(self) -> None:
        definition = config.parse_definition(support.two_actor_definition())
        self.assertEqual(definition.protocol_id, "demo-two-worker")
        self.assertEqual(definition.owner, "owner-human")
        self.assertEqual(len(definition.actors), 2)
        self.assertEqual(len(definition.schedule), 4)
        self.assertEqual(definition.final_round_id, "R04")
        self.assertEqual(definition.schedule[0].actor_id, "worker-a")
        self.assertEqual(definition.schedule[0].word_limit, 700)
        self.assertIsNone(definition.schedule[2].word_limit)

    def test_parses_three_actor_definition(self) -> None:
        definition = config.parse_definition(support.three_actor_definition())
        self.assertEqual(len(definition.actors), 3)
        self.assertEqual(
            [turn.actor_id for turn in definition.schedule],
            ["lead", "critic", "scribe"],
        )

    def test_parses_owner_preapproved_substitute_actors(self) -> None:
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
        definition = config.parse_definition(raw)
        turn = definition.schedule[0]
        self.assertEqual(turn.actor_id, "worker-a")
        self.assertEqual(turn.substitute_actor_ids, ("worker-c",))
        self.assertEqual(turn.allowed_actor_ids, ("worker-a", "worker-c"))
        self.assertEqual(turn.substitution_reasons, ("provider_cooldown",))

    def test_actor_lookup_by_id(self) -> None:
        definition = config.parse_definition(support.two_actor_definition())
        actor = definition.actor("worker-b")
        self.assertEqual(actor.role, "challenger")
        self.assertEqual(actor.expected_model, "fake-model-b")
        with self.assertRaises(config.ConfigError):
            definition.actor("nobody")

    def test_load_definition_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(support.two_actor_definition()), encoding="utf-8")
            definition = config.load_definition(path)
            self.assertEqual(definition.protocol_id, "demo-two-worker")

    def test_digest_is_stable_and_order_insensitive(self) -> None:
        raw_a = support.two_actor_definition()
        definition_a = config.parse_definition(raw_a)
        # Same content serialized with different key order must hash equal.
        reordered = json.loads(json.dumps(raw_a, sort_keys=True))
        definition_b = config.parse_definition(reordered)
        self.assertEqual(definition_a.digest(), definition_b.digest())
        self.assertEqual(len(definition_a.digest()), 64)

    def test_digest_changes_when_content_changes(self) -> None:
        raw = support.two_actor_definition()
        definition_a = config.parse_definition(raw)
        raw["schedule"][0]["purpose"] = "something else"
        definition_b = config.parse_definition(raw)
        self.assertNotEqual(definition_a.digest(), definition_b.digest())

    def test_parses_structured_continuation_anchor(self) -> None:
        raw = support.two_actor_definition()
        raw["continuation"] = {
            "protocol_id": "prior-dialogue",
            "round_id": "R00",
            "artifact_path": "/tmp/prior/R00.md",
            "artifact_sha256": "a" * 64,
            "published_commit": "b" * 40,
            "original_dialogue_head": "c" * 40,
            "start_round": "R01",
        }
        definition = config.parse_definition(raw)
        self.assertIsNotNone(definition.continuation)
        assert definition.continuation is not None
        self.assertEqual(definition.continuation.start_round, "R01")


class InvalidDefinitionTests(unittest.TestCase):
    def assert_rejected(self, raw: dict, fragment: str) -> None:
        with self.assertRaises(config.ConfigError) as ctx:
            config.parse_definition(raw)
        self.assertIn(fragment, str(ctx.exception))

    def test_rejects_duplicate_actor_ids(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][1]["actor_id"] = "worker-a"
        self.assert_rejected(raw, "duplicate actor_id")

    def test_rejects_fewer_than_two_actors(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"] = raw["actors"][:1]
        raw["schedule"] = [t for t in raw["schedule"] if t["actor_id"] == "worker-a"]
        raw["final_round_id"] = raw["schedule"][-1]["round_id"]
        self.assert_rejected(raw, "at least two actors")

    def test_rejects_unknown_scheduled_actor(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][1]["actor_id"] = "ghost"
        self.assert_rejected(raw, "unknown actor")

    def test_rejects_unknown_substitute_actor(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitute_actor_ids"] = ["ghost"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.assert_rejected(raw, "unknown substitute actor")

    def test_rejects_primary_actor_repeated_as_substitute(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-a"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.assert_rejected(raw, "primary actor")

    def test_rejects_duplicate_substitute_actor(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b", "worker-b"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.assert_rejected(raw, "duplicate substitute actor")

    def test_rejects_substitute_with_different_protocol_role(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.assert_rejected(raw, "role must match primary actor")

    def test_none_reason_code_is_reserved(self) -> None:
        # "none" is the null sentinel written to Madp-Substitution-Reason
        # trailers; as a real code it would collide with it.
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
        raw["schedule"][0]["substitution_reasons"] = ["none"]
        self.assert_rejected(raw, "reserved")

    def test_rejects_substitute_using_same_hermes_home(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][1]["role"] = raw["actors"][0]["role"]
        for actor in raw["actors"]:
            actor["transport"] = "hermes-cli"
            actor["settings"] = {
                "command_name": "hermes",
                "hermes_home": "/profiles/shared",
            }
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.assert_rejected(raw, "distinct hermes_home")

    def test_rejects_substitutes_without_frozen_reason_codes(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        self.assert_rejected(raw, "substitution_reasons is required")

    def test_rejects_reason_codes_without_substitutes(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        self.assert_rejected(raw, "requires substitute_actor_ids")

    def test_rejects_unsafe_substitution_reason_code(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        raw["schedule"][0]["substitution_reasons"] = ["provider\ncooldown"]
        self.assert_rejected(raw, "safe reason code")

    def test_rejects_duplicate_round_ids(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][1]["round_id"] = "R01"
        self.assert_rejected(raw, "duplicate round_id")

    def test_rejects_empty_schedule(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"] = []
        self.assert_rejected(raw, "schedule must contain at least one turn")

    def test_rejects_final_round_mismatch(self) -> None:
        raw = support.two_actor_definition()
        raw["final_round_id"] = "R99"
        self.assert_rejected(raw, "final_round_id")

    def test_rejects_missing_final_round(self) -> None:
        raw = support.two_actor_definition()
        del raw["final_round_id"]
        self.assert_rejected(raw, "final_round_id")

    def test_rejects_unbounded_schedule_marker(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["repeat"] = "forever"
        self.assert_rejected(raw, "unbounded")

    def test_rejects_continuation_with_wrong_start_round(self) -> None:
        raw = support.two_actor_definition()
        raw["continuation"] = {
            "protocol_id": "prior-dialogue",
            "round_id": "R00",
            "artifact_path": "/tmp/prior/R00.md",
            "artifact_sha256": "a" * 64,
            "published_commit": "b" * 40,
            "original_dialogue_head": "c" * 40,
            "start_round": "R02",
        }
        self.assert_rejected(raw, "start_round")

    def test_rejects_owner_who_is_also_actor(self) -> None:
        raw = support.two_actor_definition()
        raw["owner"] = "worker-a"
        self.assert_rejected(raw, "owner")

    def test_rejects_missing_identity_constraints(self) -> None:
        raw = support.two_actor_definition()
        del raw["actors"][0]["expected_provider"]
        self.assert_rejected(raw, "expected_provider")
        raw = support.two_actor_definition()
        raw["actors"][0]["expected_model"] = ""
        self.assert_rejected(raw, "expected_model")

    def test_rejects_unknown_transport(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][0]["transport"] = "smoke-signals"
        self.assert_rejected(raw, "transport")

    def test_rejects_role_named_transport_inference(self) -> None:
        # A transport must be declared; it is never inferred from role text.
        raw = support.two_actor_definition()
        del raw["actors"][0]["transport"]
        raw["actors"][0]["role"] = "claude-code worker"
        self.assert_rejected(raw, "transport")

    def test_rejects_non_dict_document(self) -> None:
        with self.assertRaises(config.ConfigError):
            config.parse_definition(["not", "an", "object"])  # type: ignore[arg-type]

    def test_rejects_bad_word_limit(self) -> None:
        raw = support.two_actor_definition()
        raw["schedule"][0]["word_limit"] = -5
        self.assert_rejected(raw, "word_limit")

    def test_rejects_owner_decisions_empty(self) -> None:
        raw = support.two_actor_definition()
        raw["owner_decisions"] = []
        self.assert_rejected(raw, "owner_decisions")

    def test_rejects_agent_status_that_claims_owner_decision(self) -> None:
        raw = support.two_actor_definition()
        raw["agent_final_statuses"] = ["READY_FOR_OWNER", "APPROVE"]
        self.assert_rejected(raw, "must not overlap owner_decisions")

    def test_load_rejects_invalid_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(config.ConfigError):
                config.load_definition(path)

    def test_load_rejects_missing_file(self) -> None:
        with self.assertRaises(config.ConfigError):
            config.load_definition(Path("/nonexistent/protocol.json"))


class SchemaCrossCheckTests(unittest.TestCase):
    """The JSON schema documents must agree with the Python validator."""

    def test_protocol_schema_exists_and_matches_validator_fields(self) -> None:
        schema_path = support.REPO_ROOT / "schemas" / "protocol.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required = set(schema["required"])
        self.assertLessEqual(
            {"protocol_id", "version", "owner", "actors", "schedule", "final_round_id"},
            required,
        )
        actor_required = set(schema["properties"]["actors"]["items"]["required"])
        self.assertLessEqual(
            {"actor_id", "role", "transport", "expected_provider", "expected_model"},
            actor_required,
        )

    def test_schema_examples_pass_python_validator(self) -> None:
        schema_path = support.REPO_ROOT / "schemas" / "protocol.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        examples = schema.get("examples", [])
        self.assertGreaterEqual(len(examples), 1)
        for example in examples:
            definition = config.parse_definition(example)
            self.assertGreaterEqual(len(definition.actors), 2)

    def test_schema_enforces_substitute_reason_pairing(self) -> None:
        schema = json.loads(
            (support.REPO_ROOT / "schemas" / "protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        base = support.two_actor_definition()
        base["actors"][1]["role"] = base["actors"][0]["role"]
        substitutes_only = json.loads(json.dumps(base))
        substitutes_only["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        reasons_only = json.loads(json.dumps(base))
        reasons_only["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        validator = jsonschema.Draft202012Validator(schema)
        for raw in (substitutes_only, reasons_only):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(raw)

        valid = json.loads(json.dumps(base))
        valid["schedule"][0]["substitute_actor_ids"] = ["worker-b"]
        valid["schedule"][0]["substitution_reasons"] = ["provider_cooldown"]
        validator.validate(valid)


class EvidenceVersionsDefinitionTests(unittest.TestCase):
    """The definition pins the closed set of evidence-record versions it
    accepts; versions the engine cannot interpret are rejected at load
    even when a definition names them."""

    def test_default_binds_to_engine_supported_set(self) -> None:
        definition = config.parse_definition(support.two_actor_definition())
        self.assertEqual(
            definition.evidence_versions, config.SUPPORTED_EVIDENCE_VERSIONS
        )

    def test_explicit_supported_version_accepted(self) -> None:
        raw = support.two_actor_definition()
        raw["evidence_versions"] = [1]
        definition = config.parse_definition(raw)
        self.assertEqual(definition.evidence_versions, (1,))

    def test_engine_unknown_version_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["evidence_versions"] = [1, 2]
        with self.assertRaisesRegex(config.ConfigError, "not interpretable"):
            config.parse_definition(raw)

    def test_empty_list_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["evidence_versions"] = []
        with self.assertRaisesRegex(config.ConfigError, "evidence_versions"):
            config.parse_definition(raw)

    def test_non_integer_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["evidence_versions"] = ["1"]
        with self.assertRaisesRegex(config.ConfigError, "evidence_versions"):
            config.parse_definition(raw)

    def test_boolean_is_not_an_integer_here(self) -> None:
        raw = support.two_actor_definition()
        raw["evidence_versions"] = [True]
        with self.assertRaisesRegex(config.ConfigError, "evidence_versions"):
            config.parse_definition(raw)

    def test_duplicates_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["evidence_versions"] = [1, 1]
        with self.assertRaisesRegex(config.ConfigError, "duplicates"):
            config.parse_definition(raw)

    def test_schema_capability_enums_match_config_vocabulary(self) -> None:
        # The controlled vocabulary lives in config.KNOWN_CAPABILITIES;
        # both schema enums must derive from it, never drift.
        proto = json.loads(
            (support.REPO_ROOT / "schemas" / "protocol.schema.json").read_text(
                encoding="utf-8"
            )
        )
        actor_props = proto["properties"]["actors"]["items"]["properties"]
        schema_caps = actor_props["required_capabilities"]["items"]["enum"]
        self.assertEqual(
            sorted(schema_caps), sorted(config.KNOWN_CAPABILITIES)
        )
        ev_schema = json.loads(
            (support.REPO_ROOT / "schemas" / "runtime-evidence.schema.json").read_text(
                encoding="utf-8"
            )
        )
        ev_caps = ev_schema["properties"]["capability_manifest"]["properties"][
            "capabilities"
        ]["propertyNames"]["enum"]
        self.assertEqual(sorted(ev_caps), sorted(config.KNOWN_CAPABILITIES))


class RequiredCapabilitiesDefinitionTests(unittest.TestCase):
    """Actor required_capabilities use the controlled vocabulary only."""

    def test_default_is_empty(self) -> None:
        definition = config.parse_definition(support.two_actor_definition())
        self.assertEqual(definition.actor("worker-a").required_capabilities, ())

    def test_known_capability_accepted(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][0]["required_capabilities"] = ["cli-version"]
        definition = config.parse_definition(raw)
        self.assertEqual(
            definition.actor("worker-a").required_capabilities, ("cli-version",)
        )

    def test_unknown_capability_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][0]["required_capabilities"] = ["teleport"]
        with self.assertRaisesRegex(config.ConfigError, "unknown required_capabilities"):
            config.parse_definition(raw)

    def test_duplicate_capabilities_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][0]["required_capabilities"] = ["cli-version", "cli-version"]
        with self.assertRaisesRegex(config.ConfigError, "duplicates"):
            config.parse_definition(raw)

    def test_non_string_capability_rejected(self) -> None:
        raw = support.two_actor_definition()
        raw["actors"][0]["required_capabilities"] = [1]
        with self.assertRaisesRegex(config.ConfigError, "list of strings"):
            config.parse_definition(raw)


if __name__ == "__main__":
    unittest.main()
