"""The standard canary: one turn through the REAL acceptance path.

`madp canary` builds a tiny local protocol (two actors, one scheduled
turn) inside a fresh Git repository, launches the single turn through
the normal runner, and validates the result with the production gate
(``--require-git --require-runner-completion``). It operationalizes the
README's "harmless live smoke and bounded canary" advice: a pass means
the engine, the adapter, the Git transaction model, and the validation
gate all work end to end on this machine.

Default binaries are the contract-faithful fakes shipped under
``examples/fakes/bin`` (override with ``MADP_FAKE_BIN``). The hermes-cli
adapter additionally accepts a real binary via ``--command-name`` plus
explicit ``--expected-provider``/``--expected-model`` — the canary never
guesses an identity.

Everything is local-only: the engine has no push capability, and the
scratch runtime state (actor homes, registries, tmux dirs) stays in a
sibling directory ``<dialogue>.canary-scratch/`` OUTSIDE the dialogue's
Git repository, so the production gate's clean-tree check still passes.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import config, engine, runner

__all__ = ["CanaryError", "run_canary"]

CANARY_TRANSPORTS = ("command", "hermes-cli", "fable-session")

# Identities the shipped fakes produce by default. A canary against a
# real CLI must name its expectations explicitly instead.
_FAKE_IDENTITIES = {
    "command": ("madp-canary-provider", "madp-canary-model"),
    "hermes-cli": ("nousresearch", "hermes-4-405b"),
    "fable-session": ("anthropic", "claude-fable-5"),
}


class CanaryError(RuntimeError):
    """The canary could not run or the acceptance path failed."""


def _fake_bin() -> Path:
    override = os.environ.get("MADP_FAKE_BIN", "").strip()
    candidates = []
    if override:
        candidates.append(Path(override))
    # Source checkout: <repo>/examples/fakes/bin relative to this package.
    candidates.append(
        Path(__file__).resolve().parents[2] / "examples" / "fakes" / "bin"
    )
    for candidate in candidates:
        if (candidate / "fake-worker").is_file():
            return candidate
    raise CanaryError(
        "cannot find the shipped fake executables (looked for "
        + ", ".join(str(c) for c in candidates)
        + "); the fakes ship with the source repository under "
        "examples/fakes/bin — they are NOT part of the installed "
        "package, so an installed madp needs MADP_FAKE_BIN pointed at a "
        "source checkout's examples/fakes/bin"
    )


def _git(dirpath: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(dirpath), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CanaryError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _ensure_git_repo(dirpath: Path) -> bool:
    """git init + a repo-local canary identity when none resolves.

    Returns True when a fallback identity was installed (reported in the
    canary output; the engine itself never touches Git config).
    """
    _git(dirpath, "init", "--quiet")
    probe = subprocess.run(
        ["git", "-C", str(dirpath), "var", "GIT_AUTHOR_IDENT"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        return False
    # No global identity resolves: install a repo-local canary identity
    # so the engine's commits can land on identity-less machines.
    _git(dirpath, "config", "user.name", "MADP Canary")
    _git(dirpath, "config", "user.email", "madp-canary@example.invalid")
    return True


def _settings(
    transport: str,
    *,
    fakes: Path,
    scratch: Path,
    command_name: str | None,
    expected: tuple[str, str],
) -> dict:
    provider, model = expected
    if transport == "command":
        if command_name is not None:
            raise CanaryError(
                "the command transport proves identity through its "
                "identity_verifier_argv; overriding the worker binary is "
                "not a meaningful canary — use the shipped fakes"
            )
        worker = str(fakes / "fake-worker")
        verifier = str(fakes / "fake-verifier")
        return {
            "argv": [
                worker, "--task", "{task_file}",
                "--turn-output", "{turn_file}",
                "--round", "{round_id}", "--actor", "{actor_id}",
            ],
            "identity_verifier_argv": [
                verifier, "--turn", "{turn_file}",
                "--round", "{round_id}", "--actor", "{actor_id}",
            ],
            "env": {"FAKE_PROVIDER": provider, "FAKE_MODEL": model},
        }
    if transport == "hermes-cli":
        return {
            "command_name": command_name or str(fakes / "fake-hermes"),
            "hermes_home": str(scratch / "hermes-home"),
        }
    if transport == "fable-session":
        if command_name is not None:
            raise CanaryError(
                "fable-session canaries run against the shipped fake "
                "(a real launch needs a host-local profile/registry); "
                "run the live example for real-CLI smoke instead"
            )
        registry = scratch / "registry.toml"
        registry.write_text(
            "[project.canary]\n"
            f'repo = "{scratch.parent}"\n'
            f'profile = "{fakes.parent / "fable-profile.toml"}"\n'
            f'model = "{model}"\n'
            'effort = "high"\n'
            'fallback = "stop"\n'
            'permission_mode = "auto"\n'
            'tmux_prefix = "madpcanary-"\n',
            encoding="utf-8",
        )
        return {
            "command_name": str(fakes / "fake-fable-session"),
            "project": "canary",
            "registry": str(registry),
            "state_dir": str(scratch / "fable-state"),
            "tmux_prefix": "madpcanary-",
            "tmux_command_name": str(fakes / "fake-tmux"),
            "env": {"FAKE_TMUX_DIR": str(scratch / "tmux-sessions")},
        }
    raise CanaryError(
        f"unknown canary transport {transport!r}; "
        f"choose one of {list(CANARY_TRANSPORTS)}"
    )


def run_canary(
    dialogue_dir: Path,
    transport: str,
    *,
    command_name: str | None = None,
    expected_provider: str | None = None,
    expected_model: str | None = None,
) -> dict:
    """Run one turn through the real acceptance path; return the report."""
    dialogue_dir = Path(dialogue_dir)
    if transport not in CANARY_TRANSPORTS:
        raise CanaryError(
            f"unknown canary transport {transport!r}; "
            f"choose one of {list(CANARY_TRANSPORTS)}"
        )
    if command_name is not None and (
        expected_provider is None or expected_model is None
    ):
        raise CanaryError(
            "--command-name names a real CLI; pass --expected-provider and "
            "--expected-model explicitly — a canary never guesses identity"
        )
    if command_name is None and (
        expected_provider is not None or expected_model is not None
    ):
        raise CanaryError(
            "--expected-provider/--expected-model are only meaningful "
            "with --command-name; without it the shipped fake's identity "
            "is used and these flags would be silently ignored"
        )
    if dialogue_dir.exists():
        if not dialogue_dir.is_dir():
            raise CanaryError(
                f"canary dialogue path {dialogue_dir} exists and is not "
                "a directory"
            )
        if any(dialogue_dir.iterdir()):
            raise CanaryError(
                f"canary dialogue path {dialogue_dir} is not empty; "
                "a canary always starts from a fresh directory"
            )
    scratch = dialogue_dir.parent / (dialogue_dir.name + ".canary-scratch")
    if scratch.exists():
        raise CanaryError(
            f"canary scratch {scratch} already exists from a prior run; "
            "remove it first — a canary never reuses stale runtime state"
        )
    dialogue_dir.mkdir(parents=True, exist_ok=True)
    identity_fallback = _ensure_git_repo(dialogue_dir)
    # Scratch lives OUTSIDE the dialogue's Git repository: untracked
    # runtime state inside the repo would trip the production gate's
    # clean-tree check.
    scratch.mkdir()

    expected = (
        expected_provider or _FAKE_IDENTITIES[transport][0],
        expected_model or _FAKE_IDENTITIES[transport][1],
    )
    settings = _settings(
        transport,
        fakes=_fake_bin(),
        scratch=scratch,
        command_name=command_name,
        expected=expected,
    )
    challenger = dict(settings)
    if transport == "hermes-cli":
        challenger["hermes_home"] = str(scratch / "hermes-home-challenger")
    raw = {
        "protocol_id": f"madp-canary-{transport}",
        "version": 1,
        "owner": "canary-owner",
        "source_sha": "0" * 40,
        "evidence_roots": [],
        "actors": [
            {
                "actor_id": "canary-proposer",
                "role": "proposer",
                "transport": transport,
                "expected_provider": expected[0],
                "expected_model": expected[1],
                "settings": settings,
            },
            {
                # Never scheduled: the two-actor minimum is a definition
                # rule, not a second turn.
                "actor_id": "canary-challenger",
                "role": "challenger",
                "transport": transport,
                "expected_provider": expected[0],
                "expected_model": expected[1],
                "settings": challenger,
            },
        ],
        "schedule": [
            {
                "round_id": "C01",
                "actor_id": "canary-proposer",
                "purpose": "exercise the real acceptance path end to end",
                "artifact_kind": "proposal",
            }
        ],
        "final_round_id": "C01",
    }
    definition = config.parse_definition(raw)
    try:
        dialogue = engine.init_dialogue(definition, dialogue_dir)
    except (config.ConfigError, engine.ProtocolError) as exc:
        raise CanaryError(f"canary init failed: {exc}") from exc
    try:
        runner.launch(dialogue, "canary-proposer")
    except Exception as exc:
        raise CanaryError(
            f"canary turn REJECTED by the real acceptance path: {exc}"
        ) from exc
    report = dialogue.validate(
        require_git=True, require_runner_completion=True
    )
    record = dialogue.state()["completed_turns"][0]
    evidence_file = dialogue_dir / record["evidence_file"]
    turn_evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    ok = bool(report["ok"])
    return {
        "ok": ok,
        "adapter": transport,
        "dialogue": str(dialogue_dir),
        "local_only": True,
        "identity_fallback_installed": identity_fallback,
        "turn": {
            "round_id": record["round_id"],
            "actor_id": record["actor_id"],
            "completed_via": record.get("completed_via"),
            "evidence_file": record["evidence_file"],
        },
        "cli_version": turn_evidence.get("cli_version"),
        "validation_ok": ok,
        "validation": {k: report[k] for k in ("errors", "warnings") if k in report},
        "note": (
            "local-only: the engine never pushes; scratch runtime state "
            "stays in the .canary-scratch sibling outside the Git repo"
        ),
    }
