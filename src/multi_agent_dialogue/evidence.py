"""Runtime-evidence records: what actually ran, proven per turn.

Evidence is a strict JSON object captured by the adapter after a worker
turn. It records the runtime identity actually observed — transport,
provider, model, session/run ID, terminal outcome — plus the SHA-256 of
the produced artifact. The engine validates it against the actor's
declared constraints; a Markdown ``actor:`` or ``model:`` label is never
sufficient and never consulted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .artifacts import ArtifactError, read_bytes_nofollow, sha256_file
from .config import SUPPORTED_EVIDENCE_VERSIONS, Actor, TurnSpec

__all__ = [
    "EvidenceError",
    "load_evidence",
    "parse_evidence",
    "validate_evidence",
    "sha256_file",
]

# The evidence-record version adapters write per turn: the latest the
# engine supports. The canonical support list lives in config.py.
EVIDENCE_VERSION = SUPPORTED_EVIDENCE_VERSIONS[-1]

REQUIRED_FIELDS = (
    "evidence_version",
    "actor_id",
    "round_id",
    "adapter",
    "transport",
    "provider",
    "model",
    "session_id",
    "outcome",
    "exit_status",
    "artifact_path",
    "artifact_sha256",
    "captured_at",
    "proof",
)

_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(RuntimeError):
    """Evidence is missing, unreadable, or fails validation."""


def parse_evidence(data: bytes, origin: Path | str) -> dict:
    """Parse evidence from already-read bytes (read exactly once upstream)."""
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"evidence {origin} is not valid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise EvidenceError(f"evidence {origin} must be a JSON object")
    return record


def load_evidence(path: Path | str) -> dict:
    path = Path(path)
    try:
        data = read_bytes_nofollow(path, "evidence")
    except ArtifactError as exc:
        raise EvidenceError(str(exc)) from exc
    return parse_evidence(data, path)


def validate_evidence(
    record: dict,
    *,
    actor: Actor,
    turn: TurnSpec,
    artifact_sha256: str,
    accepted_versions: tuple[int, ...] | None = None,
) -> list[str]:
    """Return every reason this evidence fails; empty list means valid.

    ``accepted_versions`` is the closed set declared by the dialogue
    definition (``definition.evidence_versions``); None falls back to the
    engine-supported set. A record whose evidence_version sits outside
    the set fails closed even when every other field is perfect.
    """
    if accepted_versions is None:
        accepted_versions = SUPPORTED_EVIDENCE_VERSIONS
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["evidence must be a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing evidence field: {field}")
    if errors:
        return errors

    version = record["evidence_version"]
    if version not in accepted_versions:
        errors.append(
            f"evidence_version {version!r} is not accepted by this "
            f"dialogue definition (accepted: {sorted(accepted_versions)})"
        )
    elif version not in SUPPORTED_EVIDENCE_VERSIONS:
        # Defense in depth: even a caller that hand-passes a wider
        # accepted set cannot make the engine read a record layout it
        # does not understand.
        errors.append(f"unsupported evidence_version: {version!r}")
    if record["actor_id"] != actor.actor_id:
        errors.append(
            f"evidence actor_id {record['actor_id']!r} does not match "
            f"claimed actor {actor.actor_id!r}"
        )
    if record["round_id"] != turn.round_id:
        errors.append(
            f"evidence round_id {record['round_id']!r} does not match "
            f"current round {turn.round_id!r}"
        )
    if record["transport"] != actor.transport:
        errors.append(
            f"evidence transport {record['transport']!r} does not match "
            f"declared transport {actor.transport!r}"
        )
    if record["provider"] != actor.expected_provider:
        errors.append(
            f"observed provider {record['provider']!r} does not match "
            f"expected_provider {actor.expected_provider!r}"
        )
    if record["model"] != actor.expected_model:
        errors.append(
            f"observed model {record['model']!r} does not match "
            f"expected_model {actor.expected_model!r}"
        )
    session_id = record["session_id"]
    if not isinstance(session_id, str) or not session_id.strip():
        errors.append("session_id must be a non-empty string; a runtime "
                      "without a session/run ID is not proven")
    if record["outcome"] != "success":
        errors.append(
            f"terminal outcome is {record['outcome']!r}; only 'success' "
            "may complete a turn"
        )
    if record["exit_status"] != 0:
        errors.append(f"exit_status {record['exit_status']!r} is not 0")
    sha = record["artifact_sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        errors.append("artifact_sha256 must be a lowercase hex SHA-256")
    elif sha != artifact_sha256:
        errors.append(
            "artifact_sha256 does not match the submitted turn file; "
            "the artifact changed after the runtime produced it"
        )
    captured_at = record["captured_at"]
    if not isinstance(captured_at, str) or not _UTC_RE.fullmatch(captured_at):
        errors.append("captured_at must be UTC like 2026-07-16T00:00:00Z")
    proof = record["proof"]
    if not isinstance(proof, dict) or not proof:
        errors.append("proof must be a non-empty object of adapter references")
    elif not isinstance(proof.get("kind"), str) or not proof["kind"].strip():
        errors.append(
            "proof.kind must name the external record family that backs this "
            "evidence (e.g. fable-session, hermes-state-db, "
            "external-command-verifier); an unstructured dict is not proof"
        )
    return errors
