"""One-turn execution: dry-run by default, one explicit launch at most.

``dry_run`` computes the exact command packet without creating a
process, a claim, a work directory, or any state change. ``launch``
claims the turn, hands the packet to the transport adapter — which runs
the real external lifecycle once and returns a normalized evidence
record derived from external runner/session records — writes that
evidence itself, and completes the turn. Workers never author their own
evidence files. On failure the dialogue never advances: the claim is
released only when the adapter has proven that no worker lane survived
the failure; if lane cleanup is unproven the dialogue locks ``BLOCKED``
with the claim retained, so no retry can start a duplicate worker.

The task briefing uses the strictest transport's task format
(``## Goal`` / ``## Checks`` / ``## Boundaries`` / ``## Report``, the
exact sections ``fable-session`` 0.3.0b1 requires), so one briefing
shape serves every transport.
"""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from . import adapters, artifacts, config, engine

WORK_DIR = "work"

# Run-attempt receipts live under work/ — the gitignored transient
# namespace — so they NEVER enter the ledger: they are crash forensics
# for the recovery path, not acceptance input. Nothing in the acceptance
# or validation logic reads them.
RUN_ATTEMPTS_DIR = "run-attempts"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _receipt_path(context: adapters.PrepareContext, token: str) -> Path:
    return (
        context.work_dir.parent
        / RUN_ATTEMPTS_DIR
        / f"{context.turn.round_id}-{context.actor.actor_id}-{token}.json"
    )


def _write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine.atomic_write_json(path, receipt)


def _start_receipt(
    context: adapters.PrepareContext, packet: adapters.CommandPacket
) -> tuple[Path, dict]:
    """Write the in-flight receipt BEFORE any process starts.

    If the orchestrator is killed mid-turn, this receipt is what names
    the attempt afterwards: round, actor, argv digest (never the
    plaintext argv — it embeds the briefing), runner pid, start time.
    A receipt that still says ``in_flight`` means the attempt never
    reported an outcome.
    """
    started = time.time()
    receipt = {
        "kind": "run-attempt-receipt",
        "round_id": context.turn.round_id,
        "actor_id": context.actor.actor_id,
        "adapter": context.actor.transport,
        "argv_sha256": artifacts.sha256_bytes(
            config.canonical_json(list(packet.argv)).encode("utf-8")
        ),
        "runner_pid": os.getpid(),
        "started_at": _utc_now(),
        "finalized_at": None,
        "outcome": "in_flight",
        "exit_status": None,
        "cleanup": "pending",
        "detail": None,
    }
    path = _receipt_path(
        context,
        f"{int(started * 1000):x}-{os.getpid():x}-{secrets.token_hex(2)}",
    )
    _write_receipt(path, receipt)
    return path, receipt


def _finalize_receipt(
    path: Path | None,
    receipt: dict | None,
    *,
    outcome: str,
    cleanup: str,
    exit_status: int | None = None,
    detail: str | None = None,
) -> None:
    """Best-effort outcome stamp; the ledger stays the authority.

    A failed finalize leaves the receipt ``in_flight`` — itself a
    truthful "no outcome was recorded" signal — and must never mask the
    real outcome of the turn.
    """
    if path is None or receipt is None:
        return
    receipt.update(
        {
            "finalized_at": _utc_now(),
            "outcome": outcome,
            "exit_status": exit_status,
            "cleanup": cleanup,
            "detail": detail,
        }
    )
    try:
        _write_receipt(path, receipt)
    except Exception:
        pass


def _context_for(
    dialogue: engine.Dialogue, actor_id: str
) -> tuple[config.ProtocolDefinition, config.TurnSpec, adapters.PrepareContext]:
    definition = dialogue.definition()
    try:
        actor = definition.actor(actor_id)
    except config.ConfigError as exc:
        raise engine.ProtocolError(str(exc)) from exc
    turn = dialogue.next_turn()
    if turn is None:
        raise engine.ProtocolError(
            f"final turn {definition.final_round_id} is already complete; "
            "no further worker turns exist"
        )
    if turn.actor_id != actor_id:
        raise engine.ProtocolError(
            f"{actor_id!r} is not the scheduled actor for {turn.round_id}; "
            f"next actor is {turn.actor_id!r}"
        )
    work_dir = dialogue.directory.absolute() / WORK_DIR / turn.round_id
    context = adapters.PrepareContext(
        actor=actor,
        turn=turn,
        dialogue_dir=dialogue.directory.absolute(),
        work_dir=work_dir,
        task_file=work_dir / "task.md",
        turn_file=work_dir / "turn.md",
        evidence_file=work_dir / "evidence.json",
    )
    return definition, turn, context


def build_task_briefing(
    definition: config.ProtocolDefinition,
    turn: config.TurnSpec,
    dialogue: engine.Dialogue,
    context: adapters.PrepareContext,
    output_contract: tuple[str, ...],
) -> str:
    actor = definition.actor(turn.actor_id)
    state = dialogue.state()
    word_limit = (
        f"{turn.word_limit} words" if turn.word_limit is not None else "none"
    )
    lines = [
        f"# Turn briefing: {turn.round_id} of {definition.protocol_id}",
        "",
        "## Goal",
        "",
        f"You are actor {actor.actor_id} with protocol role {actor.role!r} "
        f"in decision dialogue {definition.protocol_id}. Produce the "
        f"{turn.artifact_kind} for round {turn.round_id}: {turn.purpose}.",
        "",
        "## Checks",
        "",
        f"- round_id: {turn.round_id}",
        f"- actor_id: {actor.actor_id}",
        f"- protocol_role: {actor.role}",
        f"- artifact_kind: {turn.artifact_kind}",
        f"- word_limit: {word_limit}",
    ]
    completed = state.get("completed_turns", [])
    if completed:
        # Usable prior-turn context: absolute paths rooted in the dialogue
        # directory (a relative name is meaningless from the worker's own
        # working directory) plus the published sha256 digests that bind
        # exactly which bytes must be read.
        dialogue_dir = context.dialogue_dir
        lines.append(
            f"- REQUIRED READING: read each of the {len(completed)} prior "
            f"turn file(s) below in full, and their runtime-evidence "
            f"records, before producing this {turn.artifact_kind}; argue "
            "against what was actually written, never from memory of the "
            "schedule."
        )
        for record in completed:
            turn_path = dialogue_dir / record["artifact_file"]
            evidence_path = dialogue_dir / record["evidence_file"]
            lines.append(
                f"- prior turn {record['round_id']} by {record['actor_id']}: "
                f"read {turn_path} (sha256 {record['artifact_sha256']}); "
                f"evidence {evidence_path} "
                f"(sha256 {record['evidence_sha256']})"
            )
    else:
        lines.append(
            "- prior turns: none — this is the first turn; "
            "there is nothing to read"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        f"- Argue only from the frozen evidence boundary "
        f"source_sha {definition.source_sha or 'none'}.",
        f"- Allowed evidence roots: {', '.join(definition.evidence_roots) or 'none'}.",
        "- This is exactly one turn; do not continue past it and do not "
        "claim any other actor's turn.",
        "- Runtime identity (provider, model, session) is observed "
        "externally; a written label never proves it.",
        "",
        "## Report",
        "",
    ]
    lines += [f"- {item}" for item in output_contract]
    lines.append("")
    return "\n".join(lines)


def dry_run(dialogue: engine.Dialogue, actor_id: str) -> dict:
    definition, turn, context = _context_for(dialogue, actor_id)
    packet = adapters.adapter_for(context.actor).prepare(context)
    return {
        "dry_run": True,
        "executed": False,
        "round_id": turn.round_id,
        "actor_id": actor_id,
        "transport": context.actor.transport,
        "work_dir": str(context.work_dir),
        "packet": packet.as_dict(),
        "note": "no process was started, no claim was made, no state changed",
    }


def prepare(dialogue: engine.Dialogue, actor_id: str, output: Path | str) -> dict:
    definition, turn, context = _context_for(dialogue, actor_id)
    adapter = adapters.adapter_for(context.actor)
    packet = adapter.prepare(context)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    briefing = build_task_briefing(
        definition, turn, dialogue, context, adapter.output_contract(context)
    )
    try:
        artifacts.write_bytes_exclusive(
            output, briefing.encode("utf-8"), "task briefing"
        )
    except artifacts.ArtifactError as exc:
        raise engine.ProtocolError(str(exc)) from exc
    return {
        "prepared": True,
        "round_id": turn.round_id,
        "actor_id": actor_id,
        "task_file": str(output),
        "packet": packet.as_dict(),
    }


def launch(dialogue: engine.Dialogue, actor_id: str, timeout: int | None = None) -> dict:
    definition, turn, context = _context_for(dialogue, actor_id)
    adapter = adapters.adapter_for(context.actor)
    try:
        packet = adapter.prepare(context)
    except adapters.AdapterError as exc:
        raise engine.ProtocolError(str(exc)) from exc

    # Capability gate: when the definition requires capabilities, the
    # engine probes them from the CLI itself and refuses the launch
    # before any claim or runtime spawn.
    required = context.actor.required_capabilities
    if required:
        try:
            manifest = adapter.probe_capabilities(context)
        except adapters.AdapterError as exc:
            raise engine.ProtocolError(str(exc)) from exc
        missing = [
            name
            for name in required
            if not manifest["capabilities"].get(name, {}).get("ok")
        ]
        if missing:
            raise engine.ProtocolError(
                f"actor {actor_id!r} requires capabilities the probed CLI "
                f"does not show: {missing}; launch refused before any "
                "runtime spawn"
            )
        # The evidence must record exactly what gated the launch.
        context = dataclasses.replace(context, capability_manifest=manifest)

    dialogue.claim(actor_id)
    # Forensics are best-effort: a receipt that cannot be written must
    # never hold a claim hostage or block a turn the ledger can prove.
    try:
        receipt_path: Path | None
        receipt: dict | None
        receipt_path, receipt = _start_receipt(context, packet)
    except Exception:
        receipt_path, receipt = None, None
    try:
        context.work_dir.mkdir(parents=True, exist_ok=True)
        briefing = build_task_briefing(
            definition, turn, dialogue, context, adapter.output_contract(context)
        )
        try:
            artifacts.write_bytes_exclusive(
                context.task_file, briefing.encode("utf-8"), "task briefing"
            )
            # One turn, executed by the adapter against the real external
            # lifecycle; the returned evidence is adapter-derived from
            # external records, never worker-authored.
            record = adapter.execute(context, packet, timeout=timeout)
            evidence_bytes = (
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            artifacts.write_bytes_exclusive(
                context.evidence_file, evidence_bytes, "runtime evidence"
            )
        except adapters.CleanupUnprovenError:
            raise
        except (adapters.AdapterError, artifacts.ArtifactError) as exc:
            raise engine.ProtocolError(str(exc)) from exc
        # The single runner-launch completion door: only this call — an
        # adapter-executed external lifecycle — may claim runner-launch
        # provenance. Every other completion path stays caller-supplied.
        state = dialogue.complete(
            actor_id,
            context.turn_file,
            context.evidence_file,
            completed_via=engine.COMPLETED_VIA_RUNNER_LAUNCH,
        )
        _finalize_receipt(
            receipt_path, receipt,
            outcome="completed",
            cleanup="claim-consumed-by-completion",
            exit_status=record.get("exit_status"),
        )
    except adapters.CleanupUnprovenError as exc:
        # The external worker lane may still be alive. Releasing the claim
        # would let a retry start a duplicate worker beside it, so the
        # claim and its lock stay in place and the dialogue locks BLOCKED
        # (release() refuses BLOCKED dialogues) until a human resolves it.
        _finalize_receipt(
            receipt_path, receipt,
            outcome="failed",
            cleanup="claim-retained-blocked",
            detail=str(exc),
        )
        dialogue.block(f"unproven worker-lane cleanup: {exc}")
        raise engine.ProtocolError(str(exc)) from exc
    except BaseException as exc:
        # Fail closed but leave the turn claimable again: the adapter has
        # already proven that no worker lane survived this failure.
        cleanup = "claim-released"
        try:
            dialogue.release(actor_id)
        except engine.ProtocolError:
            cleanup = "release-failed"
        _finalize_receipt(
            receipt_path, receipt,
            outcome="failed",
            cleanup=cleanup,
            detail=str(exc),
        )
        raise
    return {
        "dry_run": False,
        "executed": True,
        "completed_round": turn.round_id,
        "actor_id": actor_id,
        "status": state["status"],
        "turn_index": state["turn_index"],
        "packet": packet.as_dict(),
    }
