"""The madp command-line interface.

Machine-readable JSON goes to stdout; failures go to stderr with a
nonzero exit code. Every fail-closed protocol violation exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import adapters, artifacts, canary, config, engine, evidence, runner


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _next_legal_action(state: dict, turn) -> str:
    """One sentence naming what may legally happen next, and why not more."""
    status = state["status"]
    if status == engine.STATUS_BLOCKED:
        return (
            "none: the dialogue is BLOCKED (non-retryable); recovery is a "
            "human decision outside the protocol"
        )
    if status == engine.STATUS_OWNER_DECIDED:
        return "none: the dialogue is closed by the owner decision"
    if status == engine.STATUS_CLAIMED:
        holder = (state.get("claim") or {}).get("actor_id")
        return (
            f"wait: the claim is held by {holder!r}; if its run --launch is "
            "still executing, completing or releasing now can start a "
            "duplicate worker — only after the operator confirms the "
            "launching process is dead, recover via "
            "python -m multi_agent_dialogue.unverified "
            "(caller-supplied provenance)"
        )
    if status == engine.STATUS_READY_FOR_OWNER or turn is None:
        return "owner-decide --decision FILE"
    return f"run --actor {turn.actor_id} --launch"


def _status_payload(dialogue: engine.Dialogue) -> dict:
    state = dialogue.state()
    definition = dialogue.definition()
    turn = dialogue.next_turn()
    return {
        "protocol_id": state["protocol_id"],
        "status": state["status"],
        "revision": state["revision"],
        "turn_index": state["turn_index"],
        "total_turns": len(definition.schedule),
        "final_round_id": definition.final_round_id,
        "claim": state["claim"],
        "next": None
        if turn is None
        else {
            "round_id": turn.round_id,
            "actor_id": turn.actor_id,
            "role": definition.actor(turn.actor_id).role,
            "transport": definition.actor(turn.actor_id).transport,
            "purpose": turn.purpose,
        },
        "completed_turns": state["completed_turns"],
        "owner_decision": state["owner_decision"],
        "blocked_reason": state.get("blocked_reason"),
        "next_legal_action": _next_legal_action(state, turn),
        "recovered_turns": sum(
            1
            for record in state["completed_turns"]
            if record.get("completed_via") == engine.COMPLETED_VIA_CALLER_SUPPLIED
        ),
    }


def cmd_init(args: argparse.Namespace) -> int:
    definition = config.load_definition(args.definition)
    dialogue = engine.init_dialogue(definition, args.dialogue)
    _emit(_status_payload(dialogue))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _emit(_status_payload(engine.Dialogue(args.dialogue)))
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    definition = dialogue.definition()
    turn = dialogue.next_turn()
    if turn is None:
        _emit(
            {
                "done": True,
                "status": dialogue.state()["status"],
                "note": "all scheduled turns are complete; only the owner may act",
            }
        )
        return 0
    actor = definition.actor(turn.actor_id)
    _emit(
        {
            "done": False,
            "round_id": turn.round_id,
            "actor_id": turn.actor_id,
            "role": actor.role,
            "transport": actor.transport,
            "purpose": turn.purpose,
            "artifact_kind": turn.artifact_kind,
            "word_limit": turn.word_limit,
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    if args.launch:
        _emit(runner.launch(dialogue, args.actor, timeout=args.timeout))
    else:
        _emit(runner.dry_run(dialogue, args.actor))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = engine.Dialogue(args.dialogue).validate(
        require_git=args.require_git,
        require_runner_completion=args.require_runner_completion,
    )
    _emit(report)
    return 0 if report["ok"] else 1


def cmd_owner_decide(args: argparse.Namespace) -> int:
    dialogue = engine.Dialogue(args.dialogue)
    state = dialogue.owner_decide(args.decision)
    _emit({"status": state["status"], "owner_decision": state["owner_decision"]})
    return 0


def cmd_canary(args: argparse.Namespace) -> int:
    report = canary.run_canary(
        args.dialogue,
        args.adapter,
        command_name=args.command_name,
        expected_provider=args.expected_provider,
        expected_model=args.expected_model,
    )
    _emit(report)
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="madp",
        description=(
            "Transport-neutral, fail-closed multi-agent dialogue protocol. "
            "The production path is: init, status, run --launch, "
            "validate --require-git --require-runner-completion, owner-decide."
        ),
        epilog=(
            "Recovery verbs (claim, prepare, complete, release) are "
            "deliberately not here: they live in "
            "'python -m multi_agent_dialogue.unverified' and record "
            "caller-supplied completion provenance."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="initialize a dialogue from a protocol definition")
    p.add_argument("--definition", required=True, type=Path)
    p.add_argument("--dialogue", required=True, type=Path)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("status", help="show dialogue state as JSON")
    p.add_argument("dialogue", type=Path)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("next", help="show the next scheduled turn (read-only helper)")
    p.add_argument("dialogue", type=Path)
    p.set_defaults(func=cmd_next)

    p = sub.add_parser(
        "run",
        help="process at most one turn (dry-run by default; --launch executes once)",
    )
    p.add_argument("dialogue", type=Path)
    p.add_argument("--actor", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="show the command packet without starting any process (default)")
    group.add_argument("--launch", action="store_true",
                       help="execute exactly one scheduled turn through its adapter "
                            "and complete it (the only completion door that earns "
                            "runner-launch provenance)")
    p.add_argument("--timeout", type=int, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("validate", help="deterministically validate the whole dialogue")
    p.add_argument("dialogue", type=Path)
    p.add_argument("--require-git", action="store_true",
                   help="structural validation: also prove commit provenance, "
                        "including the committed completed_via of every turn")
    p.add_argument("--require-runner-completion", action="store_true",
                   help="production gate (needs --require-git): fail unless every "
                        "Git-proven completed turn is runner-launch")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("owner-decide", help="record the owner's final decision")
    p.add_argument("dialogue", type=Path)
    p.add_argument("--decision", required=True, type=Path)
    p.set_defaults(func=cmd_owner_decide)

    p = sub.add_parser(
        "canary",
        help="run one turn through the real acceptance path in a fresh local "
             "dialogue and validate it with the production gate",
    )
    p.add_argument("--adapter", required=True, choices=canary.CANARY_TRANSPORTS,
                   help="transport to exercise; binaries default to the "
                        "shipped fakes (MADP_FAKE_BIN overrides their location)")
    p.add_argument("--dialogue", required=True, type=Path,
                   help="fresh directory for the canary dialogue (must be empty)")
    p.add_argument("--command-name", default=None,
                   help="hermes-cli only: probe a REAL installed CLI instead of "
                        "the fake; requires --expected-provider/--expected-model")
    p.add_argument("--expected-provider", default=None)
    p.add_argument("--expected-model", default=None)
    p.add_argument("--no-push", action="store_true",
                   help="accepted for explicitness: canaries are always "
                        "local-only (the engine has no push capability)")
    p.set_defaults(func=cmd_canary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        canary.CanaryError,
        config.ConfigError,
        engine.ProtocolError,
        evidence.EvidenceError,
        artifacts.ArtifactError,
        adapters.AdapterError,
    ) as exc:
        print(f"madp: {exc}", file=sys.stderr)
        return 1
