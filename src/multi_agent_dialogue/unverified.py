"""Explicitly unverified recovery operations.

``python -m multi_agent_dialogue.unverified {claim,prepare,complete,release}``

These verbs are quarantined out of the public ``madp`` CLI because they
cannot manufacture adapter provenance. A completion made here is
recorded as ``completed_via: caller-supplied`` in the turn's state
record and turn commit; ``madp validate --require-git`` warns about
every such round, and the production gate
(``madp validate --require-git --require-runner-completion``) rejects
it. They exist to:

- preserve adapter output after an orchestrator crash between external
  execution and committed completion;
- recover a stray live ``CLAIMED`` state (``release``);
- regenerate a task briefing for a manually controlled recovery.

Every operation remains subject to the actor, round, artifact,
evidence-shape, digest, session-uniqueness, and state checks, and no
recovery operation releases or completes a ``BLOCKED`` dialogue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import adapters, artifacts, config, engine, evidence, runner
from .cli import _emit


def cmd_claim(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    state = dialogue.claim(args.actor, expected_revision=args.revision)
    _emit({"claimed": True, "status": state["status"], "claim": state["claim"],
           "revision": state["revision"]})
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    _emit(runner.prepare(dialogue, args.actor, args.output))
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    # No completed_via argument exists here on purpose: this door always
    # records caller-supplied provenance (the engine default) and can
    # never claim a runner-launch completion.
    state = dialogue.complete(args.actor, args.turn, args.runtime_evidence)
    record = state["completed_turns"][-1]
    _emit(
        {
            "completed": True,
            "completed_via": record["completed_via"],
            "status": state["status"],
            "turn_index": state["turn_index"],
            "completed_turns": state["completed_turns"],
        }
    )
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    state = dialogue.release(args.actor)
    _emit({"released": True, "status": state["status"],
           "revision": state["revision"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m multi_agent_dialogue.unverified",
        description=(
            "Explicitly UNVERIFIED recovery operations for a dialogue. "
            "Completions made here are recorded with caller-supplied "
            "provenance and cannot manufacture runner-launch adapter "
            "provenance: structural validation warns about every recovered "
            "round, and the production gate (madp validate --require-git "
            "--require-runner-completion) rejects them. All actor, round, "
            "artifact, evidence, digest, session-uniqueness, and state "
            "checks still apply, and no operation releases or completes a "
            "BLOCKED dialogue."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("claim", help="atomically claim the next turn for an actor")
    p.add_argument("dialogue", type=Path)
    p.add_argument("--actor", required=True)
    p.add_argument("--revision", type=int, default=None,
                   help="compare-and-swap: fail if state revision differs")
    p.set_defaults(func=cmd_claim)

    p = sub.add_parser("prepare", help="write the non-secret task briefing for a turn")
    p.add_argument("dialogue", type=Path)
    p.add_argument("--actor", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser(
        "complete",
        help="publish one finished turn with runtime evidence "
             "(recorded as caller-supplied provenance)",
    )
    p.add_argument("dialogue", type=Path)
    p.add_argument("--actor", required=True)
    p.add_argument("--turn", required=True, type=Path)
    p.add_argument("--runtime-evidence", required=True, type=Path)
    p.set_defaults(func=cmd_complete)

    p = sub.add_parser("release", help="release a stray live claim held by an actor")
    p.add_argument("dialogue", type=Path)
    p.add_argument("--actor", required=True)
    p.set_defaults(func=cmd_release)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        config.ConfigError,
        engine.ProtocolError,
        evidence.EvidenceError,
        artifacts.ArtifactError,
        adapters.AdapterError,
    ) as exc:
        print(f"madp-unverified: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
