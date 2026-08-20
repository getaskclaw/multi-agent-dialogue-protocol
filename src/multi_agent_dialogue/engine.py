"""Deterministic dialogue state: init, claims, advancement, hard stop.

State lives in one JSON file per dialogue directory, written atomically
(temp file + ``os.replace``). Claims are made exclusive by an ``O_EXCL``
lock file plus a compare-and-swap revision counter. Every transition
fails closed: wrong actor, duplicate claim, stale revision, tampered
definition, and any turn after the configured final round are errors.

Git is the dialogue's transaction log: a dialogue initializes only
inside a real Git worktree, and initialization, every successful worker
turn, and the owner decision are each exactly one local commit of
exactly the dialogue-owned paths (see ``gitops``). A commit failure
after publication locks the dialogue ``BLOCKED`` — a turn or decision is
never reported complete without commit proof. Pushing, hosting, and
remote coordination stay external.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import artifacts, config, evidence, gitops

STATE_FILE = "state.json"
DEFINITION_FILE = "definition.json"
LOCK_FILE = ".dialogue-lock"
GITIGNORE_FILE = ".gitignore"
TURNS_DIR = "turns"
EVIDENCE_DIR = "evidence"
OWNER_DECISION_FILE = "OWNER-DECISION.md"

# Committed at init: scratch work (task briefings/prompts, transport
# scratch, raw worker output), the live claim lock, and temp files are
# dialogue-transient and never enter Git history.
GITIGNORE_CONTENT = """\
# multi-agent-dialogue transient paths (never committed)
/work/
/.dialogue-lock
*.tmp
"""

STATUS_OPEN = "OPEN"
STATUS_CLAIMED = "CLAIMED"
STATUS_READY_FOR_OWNER = "READY_FOR_OWNER"
STATUS_OWNER_DECIDED = "OWNER_DECIDED"
STATUS_BLOCKED = "BLOCKED"

_WORKER_STATUSES = {STATUS_OPEN, STATUS_CLAIMED}

# Completion provenance: how a completed turn entered the record. Only
# ``runner.launch`` may pass ``runner-launch`` (an adapter executed the
# real external lifecycle); every other completion door — including the
# unverified recovery namespace — is ``caller-supplied``. The value is
# stored in state, carried as a commit trailer, and cross-checked against
# the original turn commit, so it cannot be laundered by a later state
# edit; but it remains a local-code claim inside the documented
# local-code/rewritable-local-Git trust boundary, not cryptographic
# provenance.
COMPLETED_VIA_RUNNER_LAUNCH = "runner-launch"
COMPLETED_VIA_CALLER_SUPPLIED = "caller-supplied"
COMPLETED_VIA_VALUES = frozenset(
    {COMPLETED_VIA_RUNNER_LAUNCH, COMPLETED_VIA_CALLER_SUPPLIED}
)


class ProtocolError(RuntimeError):
    """A dialogue transition is not allowed; nothing was changed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reject_symlink(path: Path, what: str) -> None:
    if path.is_symlink():
        raise ProtocolError(f"{what} must not be a symlink: {path}")
    for parent in path.parents:
        if parent.is_symlink():
            raise ProtocolError(f"{what} must not sit behind a symlink: {parent}")


def atomic_write_json(path: Path, payload: dict) -> None:
    _reject_symlink(path, "state path")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # Bind the write to the parent directory's inode: the temp file is
    # created and renamed relative to one O_NOFOLLOW directory descriptor,
    # so a parent swapped for a symlink after the check cannot redirect
    # the state write.
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ProtocolError(f"cannot open state directory {path.parent}: {exc}") from exc
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                os.path.basename(tmp_name), path.name,
                src_dir_fd=dir_fd, dst_dir_fd=dir_fd,
            )
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    finally:
        os.close(dir_fd)


def init_dialogue(definition: config.ProtocolDefinition, directory: Path | str) -> "Dialogue":
    directory = Path(directory)
    _reject_symlink(directory, "dialogue directory")
    if (directory / STATE_FILE).exists() or (directory / DEFINITION_FILE).exists():
        raise ProtocolError(f"dialogue already initialized: {directory}")
    root = gitops.worktree_root(directory)
    if root is None:
        raise ProtocolError(
            f"dialogue directory {directory} is not inside a Git repository or "
            "worktree; initialization, every turn, and the owner decision must "
            "be local Git commits, so a real Git worktree is required"
        )
    gitignore_path = directory / GITIGNORE_FILE
    if gitignore_path.exists() or gitignore_path.is_symlink():
        raise ProtocolError(
            f"refusing to overwrite existing ignore rules: {gitignore_path}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TURNS_DIR).mkdir(exist_ok=True)
    (directory / EVIDENCE_DIR).mkdir(exist_ok=True)
    try:
        artifacts.publish_bytes(GITIGNORE_CONTENT.encode("utf-8"), gitignore_path)
    except artifacts.ArtifactError as exc:
        raise ProtocolError(str(exc)) from exc
    atomic_write_json(directory / DEFINITION_FILE, definition.raw)
    now = utc_now()
    state = {
        "protocol_id": definition.protocol_id,
        "definition_digest": definition.digest(),
        "revision": 0,
        "status": STATUS_OPEN,
        "turn_index": 0,
        "claim": None,
        "completed_turns": [],
        "owner_decision": None,
        "created_at": now,
        "updated_at": now,
    }
    atomic_write_json(directory / STATE_FILE, state)
    dialogue = Dialogue(directory)
    try:
        gitops.commit_paths(
            root,
            [directory / DEFINITION_FILE, directory / STATE_FILE, gitignore_path],
            subject=f"madp({definition.protocol_id}): init dialogue",
            trailers={
                "Madp-Protocol": definition.protocol_id,
                "Madp-Event": "init",
                "Madp-Definition-Digest": definition.digest(),
            },
        )
    except gitops.GitError as exc:
        reason = (
            f"dialogue files were written but the init commit failed ({exc}); "
            "the dialogue is BLOCKED without commit proof"
        )
        dialogue.block(reason)
        raise ProtocolError(reason) from exc
    return dialogue


class Dialogue:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        _reject_symlink(self.directory, "dialogue directory")
        if not self.directory.is_dir():
            raise ProtocolError(f"not a dialogue directory: {self.directory}")

    # -- loading ---------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILE

    @property
    def lock_path(self) -> Path:
        return self.directory / LOCK_FILE

    def definition(self) -> config.ProtocolDefinition:
        return config.load_definition(self.directory / DEFINITION_FILE)

    def _read_state(self) -> dict:
        _reject_symlink(self.state_path, "state file")
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProtocolError(f"missing {STATE_FILE} in {self.directory}") from exc
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"corrupt {STATE_FILE}: {exc}") from exc
        if not isinstance(state, dict):
            raise ProtocolError(f"corrupt {STATE_FILE}: not an object")
        return state

    def state(self) -> dict:
        state = self._read_state()
        definition = self.definition()
        if state.get("definition_digest") != definition.digest():
            raise ProtocolError(
                "definition digest mismatch: definition.json was modified after init"
            )
        return state

    def _write_state(self, state: dict) -> dict:
        state["revision"] = int(state.get("revision", 0)) + 1
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)
        return state

    # -- turn order ------------------------------------------------------

    def next_turn(self) -> config.TurnSpec | None:
        state = self.state()
        definition = self.definition()
        index = int(state["turn_index"])
        if index >= len(definition.schedule):
            return None
        return definition.schedule[index]

    def _require_worker_phase(self, state: dict, definition: config.ProtocolDefinition) -> None:
        if state["status"] == STATUS_OWNER_DECIDED:
            raise ProtocolError("dialogue is OWNER_DECIDED; no further turns are allowed")
        if state["status"] == STATUS_BLOCKED:
            raise ProtocolError("dialogue is BLOCKED; resolve validation errors first")
        if int(state["turn_index"]) >= len(definition.schedule):
            raise ProtocolError(
                f"final turn {definition.final_round_id} is already complete; "
                "the schedule is a hard stop and extra turns fail closed"
            )

    # -- claims ----------------------------------------------------------

    def claim(self, actor_id: str, expected_revision: int | None = None) -> dict:
        definition = self.definition()
        try:
            definition.actor(actor_id)
        except config.ConfigError as exc:
            raise ProtocolError(str(exc)) from exc
        state = self.state()
        self._require_worker_phase(state, definition)
        if expected_revision is not None and expected_revision != state["revision"]:
            raise ProtocolError(
                f"stale revision: expected {expected_revision}, "
                f"state is at {state['revision']}"
            )
        if state["status"] == STATUS_CLAIMED or state.get("claim"):
            claim = state.get("claim") or {}
            raise ProtocolError(
                f"turn already claimed by {claim.get('actor_id')!r} "
                f"at {claim.get('claimed_at')!r}"
            )
        turn = definition.schedule[int(state["turn_index"])]
        if turn.actor_id != actor_id:
            raise ProtocolError(
                f"{actor_id!r} is not the scheduled actor for {turn.round_id}; "
                f"next actor is {turn.actor_id!r}"
            )

        claim = {
            "actor_id": actor_id,
            "round_id": turn.round_id,
            "nonce": secrets.token_hex(16),
            "claimed_at": utc_now(),
        }
        # Exclusive lock: O_EXCL makes exactly one writer win the race.
        try:
            fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise ProtocolError(
                f"claim lock already held ({self.lock_path.name} exists); "
                "another writer owns this turn"
            ) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(claim, handle)
        except BaseException:
            self.lock_path.unlink(missing_ok=True)
            raise
        try:
            state["claim"] = claim
            state["status"] = STATUS_CLAIMED
            return self._write_state(state)
        except BaseException:
            self.lock_path.unlink(missing_ok=True)
            raise

    def release(self, actor_id: str) -> dict:
        state = self.state()
        if state["status"] == STATUS_BLOCKED:
            raise ProtocolError("dialogue is BLOCKED; a release cannot unblock it")
        claim = state.get("claim")
        if not claim:
            raise ProtocolError("no active claim to release")
        if claim.get("actor_id") != actor_id:
            raise ProtocolError(
                f"claim is held by {claim.get('actor_id')!r}, not {actor_id!r}"
            )
        state["claim"] = None
        state["status"] = STATUS_OPEN
        written = self._write_state(state)
        self.lock_path.unlink(missing_ok=True)
        return written

    # -- immutability ------------------------------------------------------

    def _verify_history(self, state: dict) -> list[str]:
        """Recompute digests of every published turn and evidence record."""
        errors: list[str] = []
        for record in state.get("completed_turns", []):
            for key, label in (("artifact_file", "turn"), ("evidence_file", "evidence")):
                rel = record.get(key)
                if not rel:
                    errors.append(f"{record.get('round_id')}: missing {key} in state")
                    continue
                path = self.directory / rel
                try:
                    digest = artifacts.sha256_file(path)
                except artifacts.ArtifactError as exc:
                    errors.append(str(exc))
                    continue
                expected = record.get(f"{key.split('_')[0]}_sha256")
                if digest != expected:
                    errors.append(
                        f"published {label} {rel} violates immutability: "
                        f"sha256 changed after completion"
                    )
        decision = state.get("owner_decision")
        if decision:
            path = self.directory / decision.get("artifact_file", OWNER_DECISION_FILE)
            try:
                if artifacts.sha256_file(path) != decision.get("artifact_sha256"):
                    errors.append("OWNER-DECISION.md violates immutability")
            except artifacts.ArtifactError as exc:
                errors.append(str(exc))
        return errors

    def _block(self, state: dict, reason: str) -> ProtocolError:
        state["status"] = STATUS_BLOCKED
        state["blocked_reason"] = reason
        self._write_state(state)
        return ProtocolError(reason)

    def _git_commit(
        self, subject: str, paths: list[Path], trailers: dict[str, str]
    ) -> str:
        """Commit exactly ``paths`` in the containing worktree.

        Raises ``gitops.GitError``; callers turn that into a BLOCKED
        dialogue because a transition without commit proof never counts
        as complete."""
        root = gitops.worktree_root(self.directory)
        if root is None:
            raise gitops.GitError(
                f"dialogue directory {self.directory} is no longer inside a "
                "Git worktree"
            )
        return gitops.commit_paths(root, paths, subject, trailers)

    def block(self, reason: str) -> dict:
        """Lock the dialogue BLOCKED without touching any active claim.

        Used when an external worker lane's shutdown cannot be proven
        after a failed launch: the claim and its lock file stay in place
        and ``claim``/``release``/``complete`` all refuse BLOCKED
        dialogues, so no retry can start a duplicate worker. Recovery is
        a human decision, outside the protocol."""
        state = self.state()
        state["status"] = STATUS_BLOCKED
        state["blocked_reason"] = reason
        return self._write_state(state)

    # -- completion --------------------------------------------------------

    def complete(
        self,
        actor_id: str,
        turn_path: Path | str,
        evidence_path: Path | str,
        completed_via: str = COMPLETED_VIA_CALLER_SUPPLIED,
    ) -> dict:
        if completed_via not in COMPLETED_VIA_VALUES:
            raise ProtocolError(
                f"unknown completed_via {completed_via!r}; allowed values are "
                f"{sorted(COMPLETED_VIA_VALUES)}"
            )
        turn_path = Path(turn_path)
        evidence_path = Path(evidence_path)
        definition = self.definition()
        state = self.state()
        self._require_worker_phase(state, definition)

        claim = state.get("claim")
        if not claim or state["status"] != STATUS_CLAIMED:
            raise ProtocolError("no active claim; run claim before complete")
        if claim.get("actor_id") != actor_id:
            raise ProtocolError(
                f"turn is claimed by {claim.get('actor_id')!r}, not {actor_id!r}"
            )
        turn = definition.schedule[int(state["turn_index"])]
        if turn.actor_id != actor_id:
            raise ProtocolError(
                f"{actor_id!r} is not the scheduled actor for {turn.round_id}"
            )

        history_errors = self._verify_history(state)
        if history_errors:
            raise self._block(
                state,
                "dialogue BLOCKED before completion: " + "; ".join(history_errors),
            )

        # Read both inputs exactly once through no-follow descriptors:
        # validation, the word count, and publication below all use these
        # same immutable bytes, so a file swapped after validation can
        # never be published.
        try:
            turn_data = artifacts.read_bytes_nofollow(turn_path, "turn artifact")
        except artifacts.ArtifactError as exc:
            raise ProtocolError(str(exc)) from exc
        artifact_sha = artifacts.sha256_bytes(turn_data)
        try:
            evidence_data = artifacts.read_bytes_nofollow(evidence_path, "evidence")
            record = evidence.parse_evidence(evidence_data, evidence_path)
        except (artifacts.ArtifactError, evidence.EvidenceError) as exc:
            raise ProtocolError(str(exc)) from exc

        actor = definition.actor(actor_id)
        evidence_errors = evidence.validate_evidence(
            record, actor=actor, turn=turn, artifact_sha256=artifact_sha,
            accepted_versions=definition.evidence_versions,
        )
        if evidence_errors:
            raise ProtocolError(
                "runtime evidence rejected:\n- " + "\n- ".join(evidence_errors)
            )

        used_sessions = {
            item.get("session_id") for item in state.get("completed_turns", [])
        }
        if record["session_id"] in used_sessions:
            raise ProtocolError(
                f"session_id {record['session_id']!r} was already used by a "
                "previous turn; each turn needs an independent runtime session"
            )

        if turn.word_limit is not None:
            try:
                count = artifacts.word_count(turn_data.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ProtocolError(f"turn artifact is not UTF-8: {exc}") from exc
            if count > turn.word_limit:
                raise ProtocolError(
                    f"turn body has {count} words, over the {turn.word_limit} "
                    f"word limit for {turn.round_id}"
                )

        artifact_rel = f"{TURNS_DIR}/{turn.round_id}-{actor_id}.md"
        evidence_rel = f"{EVIDENCE_DIR}/{turn.round_id}-{actor_id}.json"
        try:
            published_sha = artifacts.publish_bytes(turn_data, self.directory / artifact_rel)
            evidence_sha = artifacts.publish_bytes(evidence_data, self.directory / evidence_rel)
        except artifacts.ArtifactError as exc:
            raise ProtocolError(str(exc)) from exc

        state["completed_turns"].append(
            {
                "round_id": turn.round_id,
                "actor_id": actor_id,
                "artifact_file": artifact_rel,
                "artifact_sha256": published_sha,
                "evidence_file": evidence_rel,
                "evidence_sha256": evidence_sha,
                "session_id": record["session_id"],
                "completed_via": completed_via,
                "completed_at": utc_now(),
            }
        )
        state["turn_index"] = int(state["turn_index"]) + 1
        state["claim"] = None
        if state["turn_index"] >= len(definition.schedule):
            state["status"] = STATUS_READY_FOR_OWNER
        else:
            state["status"] = STATUS_OPEN
        written = self._write_state(state)
        self.lock_path.unlink(missing_ok=True)
        # One exact local commit per successful turn: the updated state,
        # the published turn, and its runtime evidence, with non-secret
        # runtime-identity trailers. Any transient state churn from an
        # earlier failed/released attempt rides along in state.json
        # without staging anything outside these three paths.
        try:
            self._git_commit(
                subject=(
                    f"madp({definition.protocol_id}): "
                    f"turn {turn.round_id} by {actor_id}"
                ),
                paths=[
                    self.state_path,
                    self.directory / artifact_rel,
                    self.directory / evidence_rel,
                ],
                trailers={
                    "Madp-Protocol": definition.protocol_id,
                    "Madp-Event": "turn",
                    "Madp-Round": turn.round_id,
                    "Madp-Actor": actor_id,
                    "Madp-Transport": actor.transport,
                    "Madp-Provider": record["provider"],
                    "Madp-Model": record["model"],
                    "Madp-Session": record["session_id"],
                    "Madp-Completed-Via": completed_via,
                    "Madp-Artifact-Sha256": published_sha,
                    "Madp-Evidence-Sha256": evidence_sha,
                },
            )
        except gitops.GitError as exc:
            raise self._block(
                written,
                f"turn {turn.round_id} was published but its Git commit failed "
                f"({exc}); the turn is not complete without commit proof and "
                "the dialogue is BLOCKED (non-retryable) pending human recovery",
            ) from exc
        return written

    # -- owner decision ------------------------------------------------------

    def _run_owner_proof(self, definition: config.ProtocolDefinition,
                         decision_path: Path) -> dict | None:
        """Run the configured external owner-proof verifier, if any.

        Owner authority is honest by design: ``owner-decide`` is a
        separate terminal transition, but nothing in this engine
        authenticates WHO invoked it. Only an external verifier command
        (``owner_proof_argv`` in the definition) can upgrade the recorded
        caller identity — and even then the claim is exactly "the
        configured verifier exited 0", never cryptographic proof.
        """
        if not definition.owner_proof_argv:
            return None
        import subprocess

        argv = [
            item.replace("{decision_file}", str(decision_path))
            for item in definition.owner_proof_argv
        ]
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProtocolError(f"owner-proof verifier failed to run: {exc}") from exc
        if result.returncode != 0:
            raise ProtocolError(
                "owner-proof verifier rejected the decision "
                f"(exit {result.returncode}): "
                f"{(result.stderr or result.stdout or '').strip()[:300]}"
            )
        return {"argv": argv, "exit_status": 0}

    def owner_decide(self, decision_path: Path | str) -> dict:
        import re

        decision_path = Path(decision_path)
        definition = self.definition()
        state = self.state()
        if state["status"] == STATUS_OWNER_DECIDED:
            raise ProtocolError("dialogue is already OWNER_DECIDED and closed")
        if state["status"] != STATUS_READY_FOR_OWNER:
            raise ProtocolError(
                f"owner decision requires READY_FOR_OWNER, not {state['status']}; "
                "workers cannot be skipped and workers cannot decide"
            )
        history_errors = self._verify_history(state)
        if history_errors:
            raise self._block(
                state, "dialogue BLOCKED before decision: " + "; ".join(history_errors)
            )
        try:
            decision_data = artifacts.read_bytes_nofollow(decision_path, "decision file")
        except artifacts.ArtifactError as exc:
            raise ProtocolError(f"cannot read decision file: {exc}") from exc
        try:
            text = decision_data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"decision file is not UTF-8: {exc}") from exc
        match = re.search(r"(?m)^Decision:\s*(\S+)\s*$", text)
        if not match:
            raise ProtocolError(
                "decision file must contain a line 'Decision: <VALUE>'"
            )
        decision = match.group(1)
        if decision not in definition.owner_decisions:
            raise ProtocolError(
                f"decision {decision!r} is not one of {list(definition.owner_decisions)}"
            )
        owner_proof = self._run_owner_proof(definition, decision_path)
        try:
            sha = artifacts.publish_bytes(
                decision_data, self.directory / OWNER_DECISION_FILE
            )
        except artifacts.ArtifactError as exc:
            raise ProtocolError(str(exc)) from exc
        state["owner_decision"] = {
            "decision": decision,
            "artifact_file": OWNER_DECISION_FILE,
            "artifact_sha256": sha,
            "decided_at": utc_now(),
            # Honest authority record: without an external verifier the
            # caller is whoever could run this command; the engine never
            # claims cryptographic owner identity.
            "caller_identity": (
                "externally-verified" if owner_proof else "unverified"
            ),
            "owner_proof": owner_proof,
        }
        state["status"] = STATUS_OWNER_DECIDED
        written = self._write_state(state)
        # The owner decision is its own terminal local commit; without it
        # the dialogue is BLOCKED, never OWNER_DECIDED.
        try:
            self._git_commit(
                subject=f"madp({definition.protocol_id}): owner decision {decision}",
                paths=[self.state_path, self.directory / OWNER_DECISION_FILE],
                trailers={
                    "Madp-Protocol": definition.protocol_id,
                    "Madp-Event": "owner-decision",
                    "Madp-Decision": decision,
                    "Madp-Artifact-Sha256": sha,
                    "Madp-Caller-Identity": written["owner_decision"][
                        "caller_identity"
                    ],
                },
            )
        except gitops.GitError as exc:
            raise self._block(
                written,
                f"owner decision {decision} was recorded on disk but its Git "
                f"commit failed ({exc}); without commit proof the dialogue is "
                "BLOCKED and is NOT OWNER_DECIDED",
            ) from exc
        return written

    # -- validation ------------------------------------------------------

    def validate(
        self, require_git: bool = False, require_runner_completion: bool = False
    ) -> dict:
        if require_runner_completion and not require_git:
            raise ProtocolError(
                "--require-runner-completion requires --require-git: runner "
                "completion provenance is proven from the turn commits, so it "
                "cannot be checked without Git provenance"
            )
        errors: list[str] = []
        warnings: list[str] = []
        definition = None
        state: dict = {}
        try:
            definition = self.definition()
        except config.ConfigError as exc:
            errors.append(str(exc))
        try:
            state = self.state()
        except ProtocolError as exc:
            errors.append(str(exc))

        if definition is not None and state:
            errors.extend(self._verify_history(state))
            completed = state.get("completed_turns", [])
            if int(state.get("turn_index", -1)) != len(completed):
                errors.append(
                    f"turn_index {state.get('turn_index')} does not match "
                    f"{len(completed)} completed turns"
                )
            for position, record in enumerate(completed):
                if position >= len(definition.schedule):
                    errors.append(
                        f"completed turn {record.get('round_id')!r} is beyond "
                        "the configured schedule"
                    )
                    continue
                spec = definition.schedule[position]
                if record.get("round_id") != spec.round_id:
                    errors.append(
                        f"completed turn {position} is {record.get('round_id')!r}; "
                        f"schedule requires {spec.round_id!r}"
                    )
                if record.get("actor_id") != spec.actor_id:
                    errors.append(
                        f"turn {spec.round_id} was completed by "
                        f"{record.get('actor_id')!r}, schedule requires {spec.actor_id!r}"
                    )
                evidence_path = self.directory / record.get("evidence_file", "")
                try:
                    evidence_record = evidence.load_evidence(evidence_path)
                    artifact_sha = record.get("artifact_sha256", "")
                    actor = definition.actor(record.get("actor_id", ""))
                    errors.extend(
                        f"{spec.round_id}: {item}"
                        for item in evidence.validate_evidence(
                            evidence_record,
                            actor=actor,
                            turn=spec,
                            artifact_sha256=artifact_sha,
                            accepted_versions=definition.evidence_versions,
                        )
                    )
                except (evidence.EvidenceError, config.ConfigError) as exc:
                    errors.append(f"{spec.round_id}: {exc}")
            sessions = [item.get("session_id") for item in completed]
            if len(sessions) != len(set(sessions)):
                errors.append("session_id values are not unique across turns")
            # A caller-supplied recovery can be structurally valid, but it
            # must never pass silently: every affected round is named.
            for record in completed:
                if record.get("completed_via") == COMPLETED_VIA_CALLER_SUPPLIED:
                    warnings.append(
                        f"turn {record.get('round_id')} was completed via "
                        "caller-supplied recovery (unverified), not "
                        "runner-launch: the artifact and evidence passed every "
                        "structural check, but no adapter proved the external "
                        "lifecycle for this round"
                    )
            if state.get("status") == STATUS_OWNER_DECIDED and not state.get("owner_decision"):
                errors.append("status is OWNER_DECIDED without an owner_decision record")
            # The lock file must exist exactly when a claim record does —
            # including a BLOCKED dialogue that retained its claim after
            # an unproven worker-lane cleanup.
            lock_exists = self.lock_path.exists()
            if state.get("claim") and not lock_exists:
                errors.append("state records an active claim but the claim lock file is missing")
            if not state.get("claim") and lock_exists:
                errors.append("claim lock file exists without an active claim")

        provenance = self._git_provenance(require_git, state)
        if require_git:
            errors.extend(provenance.pop("errors"))
            if require_runner_completion:
                # Production gate: every Git-proven completed turn must be
                # runner-launch. This checks provenance only — it never
                # implies the dialogue made progress or reached a terminal
                # state, and the actual status is reported unchanged.
                for entry in provenance.get("turn_commits") or []:
                    if entry.get("completed_via") != COMPLETED_VIA_RUNNER_LAUNCH:
                        errors.append(
                            f"turn {entry.get('round_id')}: Git-proven "
                            f"completed_via is {entry.get('completed_via')!r}, "
                            f"not {COMPLETED_VIA_RUNNER_LAUNCH!r}; production "
                            "validation requires runner-launch completion for "
                            "every completed turn"
                        )
        else:
            provenance.pop("errors", None)
            if not provenance.get("git_backed"):
                warnings.append(
                    "dialogue is not Git-backed; immutability is convention-only"
                )
            elif not provenance.get("committed"):
                warnings.append("dialogue has uncommitted changes; history is not frozen yet")

        status = state.get("status", "UNKNOWN") if state else "UNKNOWN"
        return {
            "ok": not errors,
            "status": STATUS_BLOCKED if errors else status,
            "recorded_status": status,
            "directory": str(self.directory),
            "completed_turns": len(state.get("completed_turns", [])) if state else 0,
            "provenance": provenance,
            "warnings": warnings,
            "errors": errors,
        }

    def _git_provenance(self, require_git: bool, state: dict) -> dict:
        import subprocess

        errors: list[str] = []
        root = gitops.worktree_root(self.directory)
        if root is None:
            if require_git:
                errors.append("dialogue directory is not inside a Git repository")
            return {"git_backed": False, "committed": False, "errors": errors}
        root = root.resolve()
        rel = self.directory.resolve().relative_to(root)
        tracked = (
            subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(rel)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        clean = not subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", str(rel)],
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        ).stdout.strip()
        if require_git and not tracked:
            errors.append("dialogue directory is not fully tracked by Git")
        if require_git and not clean:
            errors.append("dialogue directory has uncommitted changes")
        provenance = {
            "git_backed": True,
            "git_root": str(root),
            "tracked": tracked,
            "clean": clean,
            "committed": tracked and clean,
            "errors": errors,
        }
        if require_git and state:
            errors.extend(self._prove_commit_history(root, state, provenance))
        return provenance

    def _prove_commit_history(
        self, root: Path, state: dict, provenance: dict
    ) -> list[str]:
        """Prove artifact-to-commit provenance from the local Git history.

        Proven: init is committed (definition + initial state + ignore
        rules together); every completed turn's artifact and evidence
        first appear together in exactly one commit whose committed
        state already carries this exact round/actor/digest/session/
        completed_via record (the original turn commit is the source of
        truth for completion provenance, so a later state edit cannot
        launder ``caller-supplied`` into ``runner-launch``, and a
        missing or unknown value fails closed); commit order matches
        the schedule; the owner decision,
        if present, is its own later commit recording OWNER_DECIDED; and
        no commit ever modified or deleted published history. Proven
        commit SHAs are exposed in ``provenance`` — a commit cannot
        store its own SHA, so provenance is always derived, never
        self-reported."""
        errors: list[str] = []

        def rel(name: str) -> str:
            return gitops.rel_to_root(root, self.directory / name)

        state_rel = rel(STATE_FILE)
        init_commit = gitops.first_commit_adding(root, state_rel)
        if init_commit is None:
            errors.append("init was never committed: state.json has no Git history")
        else:
            for name in (DEFINITION_FILE, GITIGNORE_FILE):
                if gitops.first_commit_adding(root, rel(name)) != init_commit:
                    errors.append(
                        f"{name} was not committed together with the initial "
                        "state in one init commit"
                    )
            provenance["init_commit"] = init_commit

        def committed_state_at(commit: str) -> dict | None:
            raw = gitops.committed_bytes(root, commit, state_rel)
            if raw is None:
                return None
            try:
                loaded = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return loaded if isinstance(loaded, dict) else None

        turn_commits: list[dict] = []
        previous = init_commit
        for position, record in enumerate(state.get("completed_turns", [])):
            round_id = record.get("round_id")
            if not record.get("artifact_file") or not record.get("evidence_file"):
                previous = None
                continue  # already reported by _verify_history
            artifact_rel = rel(record["artifact_file"])
            evidence_rel = rel(record["evidence_file"])
            artifact_commit = gitops.first_commit_adding(root, artifact_rel)
            evidence_commit = gitops.first_commit_adding(root, evidence_rel)
            if artifact_commit is None or evidence_commit is None:
                errors.append(
                    f"turn {round_id}: published artifact/evidence was never "
                    "committed"
                )
                previous = None
                continue
            if artifact_commit != evidence_commit:
                errors.append(
                    f"turn {round_id}: artifact and evidence do not first "
                    "appear together in one commit"
                )
                previous = None
                continue
            commit = artifact_commit
            if gitops.committed_sha256(root, commit, artifact_rel) != record.get(
                "artifact_sha256"
            ):
                errors.append(
                    f"turn {round_id}: committed artifact bytes do not match "
                    "the recorded sha256"
                )
            if gitops.committed_sha256(root, commit, evidence_rel) != record.get(
                "evidence_sha256"
            ):
                errors.append(
                    f"turn {round_id}: committed evidence bytes do not match "
                    "the recorded sha256"
                )
            committed_state = committed_state_at(commit)
            turns = (committed_state or {}).get("completed_turns", [])
            entry = turns[position] if position < len(turns) else None
            committed_via = None
            if not isinstance(entry, dict):
                errors.append(
                    f"turn {round_id}: the turn commit's state has no entry "
                    "for this turn"
                )
            else:
                for key in (
                    "round_id",
                    "actor_id",
                    "artifact_sha256",
                    "evidence_sha256",
                    "session_id",
                    "completed_via",
                ):
                    if entry.get(key) != record.get(key):
                        errors.append(
                            f"turn {round_id}: committed state {key} does not "
                            "match the current state record"
                        )
                # Completion provenance fails closed: the original turn
                # commit — not the mutable current state — is the source of
                # truth, and a missing or unknown value on either side is a
                # structural Git-provenance error.
                committed_via = entry.get("completed_via")
                if committed_via not in COMPLETED_VIA_VALUES:
                    errors.append(
                        f"turn {round_id}: the turn commit records a missing "
                        f"or unknown completed_via ({committed_via!r}); "
                        "completion provenance fails closed"
                    )
                if record.get("completed_via") not in COMPLETED_VIA_VALUES:
                    errors.append(
                        f"turn {round_id}: the current state records a "
                        "missing or unknown completed_via "
                        f"({record.get('completed_via')!r}); completion "
                        "provenance fails closed"
                    )
            if previous is not None and (
                commit == previous or not gitops.is_ancestor(root, previous, commit)
            ):
                errors.append(
                    f"turn {round_id}: commit order does not match the schedule"
                )
            turn_commits.append(
                {
                    "round_id": round_id,
                    "actor_id": record.get("actor_id"),
                    "commit": commit,
                    # The Git-proven provenance value (from the committed
                    # state at the original turn commit), never the current
                    # state's claim.
                    "completed_via": committed_via,
                }
            )
            previous = commit
        provenance["turn_commits"] = turn_commits

        decision = state.get("owner_decision")
        if decision and state.get("status") == STATUS_OWNER_DECIDED:
            decision_rel = rel(decision.get("artifact_file", OWNER_DECISION_FILE))
            decision_commit = gitops.first_commit_adding(root, decision_rel)
            if decision_commit is None:
                errors.append("the owner decision was never committed")
            else:
                if gitops.committed_sha256(
                    root, decision_commit, decision_rel
                ) != decision.get("artifact_sha256"):
                    errors.append(
                        "committed owner-decision bytes do not match the "
                        "recorded sha256"
                    )
                if previous is not None and (
                    decision_commit == previous
                    or not gitops.is_ancestor(root, previous, decision_commit)
                ):
                    errors.append(
                        "the owner decision must be its own later commit after "
                        "the final turn commit"
                    )
                committed_state = committed_state_at(decision_commit)
                if (committed_state or {}).get("status") != STATUS_OWNER_DECIDED:
                    errors.append(
                        "the owner-decision commit does not record "
                        "OWNER_DECIDED state"
                    )
                provenance["owner_decision_commit"] = decision_commit

        mutated = gitops.history_mutations(
            root,
            [
                rel(TURNS_DIR),
                rel(EVIDENCE_DIR),
                rel(OWNER_DECISION_FILE),
                rel(DEFINITION_FILE),
                rel(GITIGNORE_FILE),
            ],
        )
        if mutated:
            errors.append(
                "published dialogue paths were modified or deleted in Git "
                f"history (commits: {', '.join(mutated[:3])}); published "
                "history is append-only"
            )
        return errors
