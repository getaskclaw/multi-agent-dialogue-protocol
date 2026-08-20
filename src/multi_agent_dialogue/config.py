"""Strict protocol-definition loading, validation, and canonical digests.

A definition is plain JSON. Every actor declares its transport and its
expected provider/model explicitly; nothing is ever inferred from a role
name. The schedule is a finite, explicit list with an unambiguous final
round. Anything else fails closed with :class:`ConfigError`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_TRANSPORTS = ("command", "fable-session", "hermes-cli")

# Evidence-record versions this engine can interpret. A dialogue
# definition binds to a subset via `evidence_versions`; a version the
# engine cannot interpret must never be accepted, even if a definition
# names it. The latest entry is the version adapters write per turn.
SUPPORTED_EVIDENCE_VERSIONS = (1,)

# Controlled capability vocabulary for actor `required_capabilities`.
# Every entry names a capability the ENGINE probes from the CLI itself
# (never adapter self-report); the pre-launch gate refuses a turn whose
# probed manifest lacks a required capability. Add entries only with a
# probe verified against the real CLI.
KNOWN_CAPABILITIES = (
    # `<cli> --version` exits 0.
    "cli-version",
    # hermes-cli: `<cli> chat --help` shows the one-shot contract
    # surface (-q, --source, --pass-session-id).
    "one-shot-source-tagging",
)

DEFAULT_OWNER_DECISIONS = ("APPROVE", "REJECT", "NEED_MORE_EVIDENCE")

_ACTOR_KEYS = {
    "actor_id",
    "role",
    "transport",
    "expected_provider",
    "expected_model",
    "settings",
    "required_capabilities",
}
_TURN_KEYS = {"round_id", "actor_id", "purpose", "artifact_kind", "word_limit"}


class ConfigError(ValueError):
    """A protocol definition is invalid; the dialogue must not start."""


@dataclass(frozen=True)
class Actor:
    actor_id: str
    role: str
    transport: str
    expected_provider: str
    expected_model: str
    settings: dict[str, Any] = field(default_factory=dict)
    # Capabilities (controlled vocabulary, KNOWN_CAPABILITIES) the
    # engine must probe from the CLI before any runtime spawn.
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnSpec:
    round_id: str
    actor_id: str
    purpose: str
    artifact_kind: str
    word_limit: int | None = None


@dataclass(frozen=True)
class ProtocolDefinition:
    protocol_id: str
    version: int
    owner: str
    actors: tuple[Actor, ...]
    schedule: tuple[TurnSpec, ...]
    final_round_id: str
    source_sha: str
    evidence_roots: tuple[str, ...]
    owner_decisions: tuple[str, ...]
    # Closed set of evidence-record versions this dialogue accepts,
    # guaranteed ⊆ SUPPORTED_EVIDENCE_VERSIONS at parse time. A turn whose
    # evidence_version is outside this set fails closed.
    evidence_versions: tuple[int, ...] = SUPPORTED_EVIDENCE_VERSIONS
    # Optional external owner-proof verifier command. When empty, the
    # engine records the owner decision as caller-identity "unverified":
    # nothing authenticates WHO invoked owner-decide.
    owner_proof_argv: tuple[str, ...] = ()
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    def actor(self, actor_id: str) -> Actor:
        for actor in self.actors:
            if actor.actor_id == actor_id:
                return actor
        raise ConfigError(f"unknown actor: {actor_id}")

    def turn(self, index: int) -> TurnSpec:
        if not 0 <= index < len(self.schedule):
            raise ConfigError(f"turn index out of range: {index}")
        return self.schedule[index]

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.raw).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_str(raw: dict, key: str, where: str, errors: list[str]) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: {key} must be a non-empty string")
        return ""
    return value


def _parse_actor(raw: Any, position: int, errors: list[str]) -> Actor | None:
    where = f"actors[{position}]"
    if not isinstance(raw, dict):
        errors.append(f"{where}: must be an object")
        return None
    unknown = set(raw) - _ACTOR_KEYS
    if unknown:
        errors.append(f"{where}: unknown keys: {sorted(unknown)}")
    actor_id = _require_str(raw, "actor_id", where, errors)
    role = _require_str(raw, "role", where, errors)
    transport = _require_str(raw, "transport", where, errors)
    if transport and transport not in KNOWN_TRANSPORTS:
        errors.append(
            f"{where}: unknown transport {transport!r}; "
            f"declare one of {list(KNOWN_TRANSPORTS)} (never inferred from role)"
        )
    provider = _require_str(raw, "expected_provider", where, errors)
    model = _require_str(raw, "expected_model", where, errors)
    settings = raw.get("settings", {})
    if not isinstance(settings, dict):
        errors.append(f"{where}: settings must be an object")
        settings = {}
    capabilities_raw = raw.get("required_capabilities", [])
    capabilities: tuple[str, ...] = ()
    if (
        not isinstance(capabilities_raw, list)
        or not all(isinstance(item, str) for item in capabilities_raw)
    ):
        errors.append(f"{where}: required_capabilities must be a list of strings")
    else:
        unknown = [item for item in capabilities_raw if item not in KNOWN_CAPABILITIES]
        if unknown:
            errors.append(
                f"{where}: unknown required_capabilities {unknown}; the "
                f"controlled vocabulary is {list(KNOWN_CAPABILITIES)}"
            )
        if len(set(capabilities_raw)) != len(capabilities_raw):
            errors.append(f"{where}: required_capabilities contains duplicates")
        capabilities = tuple(capabilities_raw)
    if actor_id and _looks_secret(canonical_json(settings)):
        errors.append(f"{where}: settings must not embed credential material")
    if errors:
        # Still build a best-effort record so later checks can proceed.
        pass
    return Actor(
        actor_id=actor_id,
        role=role,
        transport=transport,
        expected_provider=provider,
        expected_model=model,
        settings=settings,
        required_capabilities=capabilities,
    )


def _parse_turn(raw: Any, position: int, errors: list[str]) -> TurnSpec | None:
    where = f"schedule[{position}]"
    if not isinstance(raw, dict):
        errors.append(f"{where}: must be an object")
        return None
    unknown = set(raw) - _TURN_KEYS
    if unknown:
        errors.append(
            f"{where}: unknown keys {sorted(unknown)} — the schedule is a finite "
            "explicit list; unbounded/repeat markers are rejected"
        )
    round_id = _require_str(raw, "round_id", where, errors)
    actor_id = _require_str(raw, "actor_id", where, errors)
    purpose = _require_str(raw, "purpose", where, errors)
    artifact_kind = _require_str(raw, "artifact_kind", where, errors)
    word_limit = raw.get("word_limit")
    if word_limit is not None and (not isinstance(word_limit, int) or word_limit <= 0):
        errors.append(f"{where}: word_limit must be a positive integer")
        word_limit = None
    return TurnSpec(
        round_id=round_id,
        actor_id=actor_id,
        purpose=purpose,
        artifact_kind=artifact_kind,
        word_limit=word_limit,
    )


_SECRET_MARKERS = (
    "sk-" + "ant-",
    "AKIA",
    "ghp" + "_",
    "xox" + "b-",
    "BEGIN " + "PRIVATE KEY",
    "api" + "_key",
)


def _looks_secret(text: str) -> bool:
    return any(marker in text for marker in _SECRET_MARKERS)


def parse_definition(raw: Any) -> ProtocolDefinition:
    if not isinstance(raw, dict):
        raise ConfigError("protocol definition must be a JSON object")
    errors: list[str] = []

    protocol_id = _require_str(raw, "protocol_id", "definition", errors)
    owner = _require_str(raw, "owner", "definition", errors)
    version = raw.get("version")
    if not isinstance(version, int) or version < 1:
        errors.append("definition: version must be a positive integer")
        version = 0
    source_sha = raw.get("source_sha", "")
    if not isinstance(source_sha, str):
        errors.append("definition: source_sha must be a string")
        source_sha = ""

    evidence_roots_raw = raw.get("evidence_roots", [])
    if not isinstance(evidence_roots_raw, list) or not all(
        isinstance(item, str) for item in evidence_roots_raw
    ):
        errors.append("definition: evidence_roots must be a list of strings")
        evidence_roots_raw = []

    owner_proof_raw = raw.get("owner_proof_argv", [])
    if not isinstance(owner_proof_raw, list) or not all(
        isinstance(item, str) and item for item in owner_proof_raw
    ):
        errors.append(
            "definition: owner_proof_argv must be a list of non-empty strings "
            "naming an external owner-proof verifier command"
        )
        owner_proof_raw = []

    decisions_raw = raw.get("owner_decisions", list(DEFAULT_OWNER_DECISIONS))
    if (
        not isinstance(decisions_raw, list)
        or not decisions_raw
        or not all(isinstance(item, str) and item for item in decisions_raw)
    ):
        errors.append("definition: owner_decisions must be a non-empty list of strings")
        decisions_raw = list(DEFAULT_OWNER_DECISIONS)

    versions_raw = raw.get("evidence_versions", list(SUPPORTED_EVIDENCE_VERSIONS))
    if (
        not isinstance(versions_raw, list)
        or not versions_raw
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 1
            for item in versions_raw
        )
    ):
        errors.append(
            "definition: evidence_versions must be a non-empty list of "
            "positive integers (the closed set of evidence-record versions "
            "this dialogue accepts)"
        )
        versions_raw = list(SUPPORTED_EVIDENCE_VERSIONS)
    else:
        unsupported = sorted(
            {item for item in versions_raw if item not in SUPPORTED_EVIDENCE_VERSIONS}
        )
        if unsupported:
            errors.append(
                f"definition: evidence_versions {unsupported} are not "
                f"interpretable by this engine (supported: "
                f"{list(SUPPORTED_EVIDENCE_VERSIONS)}); a version the engine "
                "cannot read must never be accepted"
            )
        if len(set(versions_raw)) != len(versions_raw):
            errors.append("definition: evidence_versions contains duplicates")

    actors_raw = raw.get("actors")
    actors: list[Actor] = []
    if not isinstance(actors_raw, list):
        errors.append("definition: actors must be a list")
    else:
        for position, item in enumerate(actors_raw):
            actor = _parse_actor(item, position, errors)
            if actor is not None:
                actors.append(actor)
        if len(actors) < 2:
            errors.append("definition: at least two actors are required")
        seen_actors: set[str] = set()
        for actor in actors:
            if actor.actor_id in seen_actors:
                errors.append(f"definition: duplicate actor_id {actor.actor_id!r}")
            seen_actors.add(actor.actor_id)

    if owner and any(actor.actor_id == owner for actor in actors):
        errors.append(
            "definition: owner must not be a scheduled actor; "
            "owner approval stays outside the worker set"
        )

    schedule_raw = raw.get("schedule")
    schedule: list[TurnSpec] = []
    if not isinstance(schedule_raw, list):
        errors.append("definition: schedule must be a finite list of turns")
    elif not schedule_raw:
        errors.append("definition: schedule must contain at least one turn")
    else:
        for position, item in enumerate(schedule_raw):
            turn = _parse_turn(item, position, errors)
            if turn is not None:
                schedule.append(turn)
        seen_rounds: set[str] = set()
        actor_ids = {actor.actor_id for actor in actors}
        for turn in schedule:
            if turn.round_id in seen_rounds:
                errors.append(f"definition: duplicate round_id {turn.round_id!r}")
            seen_rounds.add(turn.round_id)
            if turn.actor_id and turn.actor_id not in actor_ids:
                errors.append(
                    f"definition: schedule references unknown actor {turn.actor_id!r}"
                )

    final_round_id = raw.get("final_round_id")
    if not isinstance(final_round_id, str) or not final_round_id:
        errors.append(
            "definition: final_round_id is required so the hard stop is unambiguous"
        )
        final_round_id = ""
    elif schedule and final_round_id != schedule[-1].round_id:
        errors.append(
            f"definition: final_round_id {final_round_id!r} must equal the last "
            f"scheduled round {schedule[-1].round_id!r}"
        )

    if errors:
        raise ConfigError("invalid protocol definition:\n- " + "\n- ".join(errors))

    return ProtocolDefinition(
        protocol_id=protocol_id,
        version=version,
        owner=owner,
        actors=tuple(actors),
        schedule=tuple(schedule),
        final_round_id=final_round_id,
        source_sha=source_sha,
        evidence_roots=tuple(evidence_roots_raw),
        owner_decisions=tuple(decisions_raw),
        evidence_versions=tuple(versions_raw),
        owner_proof_argv=tuple(owner_proof_raw),
        # Snapshot so later caller mutations cannot change the digest.
        raw=json.loads(canonical_json(raw)),
    )


def load_definition(path: Path | str) -> ProtocolDefinition:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read definition {path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"definition {path} is not valid JSON: {exc}") from exc
    return parse_definition(raw)
