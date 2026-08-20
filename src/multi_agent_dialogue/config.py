"""Strict protocol-definition loading, validation, and canonical digests.

A definition is plain JSON. Every actor declares its transport and its
expected provider/model explicitly; nothing is ever inferred from a role
name. The schedule is a finite, explicit list with an unambiguous final
round. Anything else fails closed with :class:`ConfigError`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
SUBSTITUTION_REASON_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
AGENT_STATUS_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_ROOT_KEYS = {
    "protocol_id",
    "version",
    "owner",
    "source_sha",
    "evidence_roots",
    "owner_decisions",
    "agent_final_statuses",
    "owner_proof_argv",
    "evidence_versions",
    "actors",
    "schedule",
    "final_round_id",
    "continuation",
}
_CONTINUATION_KEYS = {
    "protocol_id",
    "round_id",
    "artifact_path",
    "artifact_sha256",
    "published_commit",
    "original_dialogue_head",
    "start_round",
}


def static_hermes_home_key(actor: "Actor") -> tuple[bool, str] | None:
    """Best-effort profile identity available before a dialogue path exists."""
    if actor.transport != "hermes-cli":
        return None
    value = actor.settings.get("hermes_home")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path.is_absolute(), os.path.normpath(value)

_ACTOR_KEYS = {
    "actor_id",
    "role",
    "transport",
    "expected_provider",
    "expected_model",
    "settings",
    "required_capabilities",
}
_TURN_KEYS = {
    "round_id",
    "actor_id",
    "substitute_actor_ids",
    "substitution_reasons",
    "purpose",
    "artifact_kind",
    "word_limit",
}


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
    # Ordered alternatives preauthorized by the frozen definition.json.
    # ``actor_id`` remains the primary scheduled actor; a substitute never
    # inherits that identity and is recorded under its own actor_id.
    substitute_actor_ids: tuple[str, ...] = ()
    substitution_reasons: tuple[str, ...] = ()

    @property
    def allowed_actor_ids(self) -> tuple[str, ...]:
        return (self.actor_id, *self.substitute_actor_ids)


@dataclass(frozen=True)
class ContinuationAnchor:
    protocol_id: str
    round_id: str
    artifact_path: str
    artifact_sha256: str
    published_commit: str
    original_dialogue_head: str
    start_round: str


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
    agent_final_statuses: tuple[str, ...] = ()
    # Optional external owner-proof verifier command. When empty, the
    # engine records the owner decision as caller-identity "unverified":
    # nothing authenticates WHO invoked owner-decide.
    owner_proof_argv: tuple[str, ...] = ()
    continuation: ContinuationAnchor | None = None
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
    substitutes_raw = raw.get("substitute_actor_ids", [])
    if not isinstance(substitutes_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in substitutes_raw
    ):
        errors.append(
            f"{where}: substitute_actor_ids must be a list of non-empty actor IDs"
        )
        substitutes_raw = []
    if actor_id and actor_id in substitutes_raw:
        errors.append(
            f"{where}: primary actor {actor_id!r} must not be repeated as a substitute"
        )
    if len(substitutes_raw) != len(set(substitutes_raw)):
        errors.append(f"{where}: duplicate substitute actor IDs are not allowed")
    reasons_raw = raw.get("substitution_reasons", [])
    if not isinstance(reasons_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in reasons_raw
    ):
        errors.append(
            f"{where}: substitution_reasons must be a list of non-empty reason codes"
        )
        reasons_raw = []
    if len(reasons_raw) != len(set(reasons_raw)):
        errors.append(f"{where}: duplicate substitution reason codes are not allowed")
    for reason in reasons_raw:
        if not SUBSTITUTION_REASON_RE.fullmatch(reason):
            errors.append(
                f"{where}: {reason!r} is not a safe reason code; use 1-64 "
                "lowercase letters, digits, underscores, or hyphens"
            )
        elif reason == "none":
            # "none" is the null sentinel written to Madp-Substitution-
            # Reason trailers; allowing it as a real code would collide.
            errors.append(
                f"{where}: 'none' is reserved as the null substitution "
                "sentinel and cannot be a reason code"
            )
    if substitutes_raw and not reasons_raw:
        errors.append(
            f"{where}: substitution_reasons is required when substitute actors exist"
        )
    if reasons_raw and not substitutes_raw:
        errors.append(
            f"{where}: substitution_reasons requires substitute_actor_ids"
        )
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
        substitute_actor_ids=tuple(substitutes_raw),
        substitution_reasons=tuple(reasons_raw),
    )


def _parse_continuation(raw: Any, errors: list[str]) -> ContinuationAnchor | None:
    if raw is None:
        return None
    where = "definition.continuation"
    if not isinstance(raw, dict):
        errors.append(f"{where}: must be an object")
        return None
    unknown = set(raw) - _CONTINUATION_KEYS
    if unknown:
        errors.append(f"{where}: unknown keys: {sorted(unknown)}")
    values = {key: _require_str(raw, key, where, errors) for key in _CONTINUATION_KEYS}
    artifact_path = values["artifact_path"]
    if artifact_path and not Path(artifact_path).is_absolute():
        errors.append(f"{where}: artifact_path must be absolute")
    if values["artifact_sha256"] and not SHA256_RE.fullmatch(
        values["artifact_sha256"]
    ):
        errors.append(f"{where}: artifact_sha256 must be 64 lowercase hex characters")
    for key in ("published_commit", "original_dialogue_head"):
        if values[key] and not GIT_SHA_RE.fullmatch(values[key]):
            errors.append(f"{where}: {key} must be a full 40-character Git SHA")
    return ContinuationAnchor(**values)


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
    unknown_root = set(raw) - _ROOT_KEYS
    if unknown_root:
        errors.append(f"definition: unknown keys: {sorted(unknown_root)}")

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

    continuation = _parse_continuation(raw.get("continuation"), errors)

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

    agent_statuses_raw = raw.get("agent_final_statuses", [])
    if not isinstance(agent_statuses_raw, list) or not all(
        isinstance(item, str) and AGENT_STATUS_RE.fullmatch(item)
        for item in agent_statuses_raw
    ):
        errors.append(
            "definition: agent_final_statuses must be a list of uppercase "
            "status tokens"
        )
        agent_statuses_raw = []
    if len(agent_statuses_raw) != len(set(agent_statuses_raw)):
        errors.append("definition: duplicate agent_final_statuses are not allowed")
    overlap = set(agent_statuses_raw) & set(decisions_raw)
    if overlap:
        errors.append(
            "definition: agent_final_statuses must not overlap owner_decisions: "
            f"{sorted(overlap)}"
        )

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
        actors_by_id = {actor.actor_id: actor for actor in actors}
        actor_ids = set(actors_by_id)
        for turn in schedule:
            if turn.round_id in seen_rounds:
                errors.append(f"definition: duplicate round_id {turn.round_id!r}")
            seen_rounds.add(turn.round_id)
            if turn.actor_id and turn.actor_id not in actor_ids:
                errors.append(
                    f"definition: schedule references unknown actor {turn.actor_id!r}"
                )
            for substitute_id in turn.substitute_actor_ids:
                if substitute_id not in actor_ids:
                    errors.append(
                        "definition: schedule references unknown substitute actor "
                        f"{substitute_id!r} in {turn.round_id}"
                    )
                    continue
                primary = actors_by_id.get(turn.actor_id)
                substitute = actors_by_id[substitute_id]
                if primary is not None and substitute.role != primary.role:
                    errors.append(
                        f"definition: substitute actor {substitute_id!r} role must "
                        f"match primary actor {turn.actor_id!r} role {primary.role!r} "
                        f"in {turn.round_id}"
                    )
                if primary is not None:
                    primary_home = static_hermes_home_key(primary)
                    substitute_home = static_hermes_home_key(substitute)
                    if primary_home is not None and primary_home == substitute_home:
                        errors.append(
                            f"definition: substitute actor {substitute_id!r} must use "
                            f"a distinct hermes_home from primary actor "
                            f"{turn.actor_id!r} in {turn.round_id}"
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
    if continuation is not None and schedule:
        if continuation.start_round != schedule[0].round_id:
            errors.append(
                "definition.continuation: start_round must equal the first "
                f"scheduled round {schedule[0].round_id!r}"
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
        agent_final_statuses=tuple(agent_statuses_raw),
        owner_proof_argv=tuple(owner_proof_raw),
        continuation=continuation,
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
