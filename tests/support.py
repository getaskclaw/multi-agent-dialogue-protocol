"""Shared fixtures for multi-agent dialogue protocol tests."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FAKE_BIN = REPO_ROOT / "examples" / "fakes" / "bin"

# Tests must never read or touch the developer's global/system Git config:
# every Git-backed fixture is an isolated temporary repository with a
# repo-local test identity, and the process environment masks the global
# and system config files for every git child process the tests (or the
# engine under test) spawn.
os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
os.environ["GIT_CONFIG_SYSTEM"] = os.devnull

TEST_GIT_NAME = "MADP Test Identity"
TEST_GIT_EMAIL = "madp-tests@example.invalid"


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git against an isolated test repository."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def init_git_repo(path: Path) -> Path:
    """Create an isolated temporary Git repository with a repo-local
    test identity. Global/system config is never consulted or written."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "--quiet")
    git(path, "config", "user.name", TEST_GIT_NAME)
    git(path, "config", "user.email", TEST_GIT_EMAIL)
    return path

TWO_ACTOR_DEFINITION = {
    "protocol_id": "demo-two-worker",
    "version": 1,
    "owner": "owner-human",
    "source_sha": "0000000000000000000000000000000000000000",
    "evidence_roots": ["docs/"],
    "actors": [
        {
            "actor_id": "worker-a",
            "role": "proposer",
            "transport": "command",
            "expected_provider": "fake-provider-a",
            "expected_model": "fake-model-a",
            "settings": {
                "argv": ["fake-worker", "--task", "{task_file}"],
                "identity_verifier_argv": ["fake-verifier", "--turn", "{turn_file}"],
            },
        },
        {
            "actor_id": "worker-b",
            "role": "challenger",
            "transport": "command",
            "expected_provider": "fake-provider-b",
            "expected_model": "fake-model-b",
            "settings": {
                "argv": ["fake-worker", "--task", "{task_file}"],
                "identity_verifier_argv": ["fake-verifier", "--turn", "{turn_file}"],
            },
        },
    ],
    "schedule": [
        {
            "round_id": "R01",
            "actor_id": "worker-a",
            "purpose": "propose a design",
            "artifact_kind": "proposal",
            "word_limit": 700,
        },
        {
            "round_id": "R02",
            "actor_id": "worker-b",
            "purpose": "challenge the design",
            "artifact_kind": "challenge",
            "word_limit": 700,
        },
        {
            "round_id": "R03",
            "actor_id": "worker-a",
            "purpose": "respond to the challenge",
            "artifact_kind": "response",
        },
        {
            "round_id": "R04",
            "actor_id": "worker-b",
            "purpose": "final recommendation",
            "artifact_kind": "recommendation",
        },
    ],
    "final_round_id": "R04",
    "owner_decisions": ["APPROVE", "REJECT", "NEED_MORE_EVIDENCE"],
}

THREE_ACTOR_DEFINITION = {
    "protocol_id": "demo-three-worker",
    "version": 1,
    "owner": "owner-human",
    "source_sha": "1111111111111111111111111111111111111111",
    "evidence_roots": [],
    "actors": [
        {
            "actor_id": "lead",
            "role": "proposer",
            "transport": "command",
            "expected_provider": "prov-1",
            "expected_model": "model-1",
            "settings": {
                "argv": ["fake-worker"],
                "identity_verifier_argv": ["fake-verifier"],
            },
        },
        {
            "actor_id": "critic",
            "role": "challenger",
            "transport": "command",
            "expected_provider": "prov-2",
            "expected_model": "model-2",
            "settings": {
                "argv": ["fake-worker"],
                "identity_verifier_argv": ["fake-verifier"],
            },
        },
        {
            "actor_id": "scribe",
            "role": "synthesizer",
            "transport": "command",
            "expected_provider": "prov-3",
            "expected_model": "model-3",
            "settings": {
                "argv": ["fake-worker"],
                "identity_verifier_argv": ["fake-verifier"],
            },
        },
    ],
    "schedule": [
        {"round_id": "R01", "actor_id": "lead", "purpose": "p", "artifact_kind": "proposal"},
        {"round_id": "R02", "actor_id": "critic", "purpose": "c", "artifact_kind": "challenge"},
        {"round_id": "R03", "actor_id": "scribe", "purpose": "s", "artifact_kind": "synthesis"},
    ],
    "final_round_id": "R03",
    "owner_decisions": ["APPROVE", "REJECT"],
}


def two_actor_definition() -> dict:
    return copy.deepcopy(TWO_ACTOR_DEFINITION)


def command_worker_settings(
    marker: Path | None,
    provider: str,
    model: str,
    extra_env: dict | None = None,
) -> dict:
    """Runnable command-actor settings: fake worker + external fake verifier."""
    env = {"FAKE_PROVIDER": provider, "FAKE_MODEL": model}
    if marker is not None:
        env["FAKE_SPAWN_MARKER"] = str(marker)
    env.update(extra_env or {})
    return {
        "argv": [
            str(FAKE_BIN / "fake-worker"),
            "--task", "{task_file}",
            "--turn-output", "{turn_file}",
            "--round", "{round_id}",
            "--actor", "{actor_id}",
        ],
        "identity_verifier_argv": [
            str(FAKE_BIN / "fake-verifier"),
            "--turn", "{turn_file}",
            "--round", "{round_id}",
            "--actor", "{actor_id}",
        ],
        "env": env,
    }


def make_evidence(
    *,
    actor_id: str,
    round_id: str,
    artifact_path: Path,
    provider: str,
    model: str,
    **overrides,
) -> dict:
    import hashlib

    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    evidence = {
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
            "verifier_argv": ["fake-verifier", "--round", round_id],
            "report": {"fake": True},
        },
    }
    evidence.update(overrides)
    return evidence


def three_actor_definition() -> dict:
    return copy.deepcopy(THREE_ACTOR_DEFINITION)
