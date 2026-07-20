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

import json
from pathlib import Path

from . import adapters, artifacts, config, engine

WORK_DIR = "work"


def _context_for(
    dialogue: engine.Dialogue,
    actor_id: str,
    substitution_reason: str | None = None,
) -> tuple[
    config.ProtocolDefinition,
    config.TurnSpec,
    adapters.PrepareContext,
    str,
    str | None,
]:
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
    if substitution_reason is None:
        claim = dialogue.state().get("claim") or {}
        if claim.get("actor_id") == actor_id and claim.get("round_id") == turn.round_id:
            substitution_reason = claim.get("substitution_reason")
    actor_selection, substitution_reason = engine.resolve_actor_selection(
        turn, actor_id, substitution_reason
    )
    isolation_errors = engine.hermes_profile_isolation_errors(
        definition, dialogue.directory.absolute(), turn
    )
    if isolation_errors:
        raise engine.ProtocolError("\n".join(isolation_errors))
    work_dir = dialogue.directory.absolute() / WORK_DIR / turn.round_id
    context = adapters.PrepareContext(
        actor=actor,
        turn=turn,
        dialogue_dir=dialogue.directory.absolute(),
        work_dir=work_dir,
        task_file=work_dir / "task.md",
        turn_file=work_dir / "turn.md",
        evidence_file=work_dir / "evidence.json",
        substitution_reason=substitution_reason,
    )
    return definition, turn, context, actor_selection, substitution_reason


def build_task_briefing(
    definition: config.ProtocolDefinition,
    turn: config.TurnSpec,
    dialogue: engine.Dialogue,
    context: adapters.PrepareContext,
    output_contract: tuple[str, ...],
    substitution_reason: str | None = None,
) -> str:
    # The selected runtime actor may be a frozen-definition substitute.
    # It keeps its own actor/provider/model identity; ``turn.actor_id`` is
    # always the primary scheduled actor and is never impersonated.
    actor = context.actor
    actor_selection = "primary" if actor.actor_id == turn.actor_id else "substitute"
    state = dialogue.state()
    if substitution_reason is None:
        claim = state.get("claim") or {}
        substitution_reason = claim.get("substitution_reason")
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
        f"- scheduled_actor_id: {turn.actor_id}",
        f"- actor_selection: {actor_selection}",
        f"- substitution_reason: {substitution_reason or 'none'}",
        f"- allowed_actor_ids: {', '.join(turn.allowed_actor_ids)}",
        f"- protocol_role: {actor.role}",
        f"- artifact_kind: {turn.artifact_kind}",
        f"- word_limit: {word_limit}",
    ]
    if turn.round_id == definition.final_round_id and definition.agent_final_statuses:
        lines.append(
            "- final agent status: include exactly one `Status: <VALUE>` line; "
            "allowed values: " + ", ".join(definition.agent_final_statuses)
        )
        lines.append(
            "- agent status is not an owner decision; do not emit any "
            f"owner-only token: {', '.join(definition.owner_decisions)}"
        )
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
    elif definition.continuation is not None:
        anchor = definition.continuation
        lines.append(
            "- REQUIRED READING: this is a bounded continuation; read the "
            f"anchored prior turn {anchor.round_id} at {anchor.artifact_path} "
            f"(sha256 {anchor.artifact_sha256}) in full before producing this "
            f"{turn.artifact_kind}."
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
        "- If actor_selection is substitute, keep your actual actor identity; "
        "never write as, claim to be, or impersonate the primary scheduled actor.",
        "- Runtime identity (provider, model, session) is observed "
        "externally; a written label never proves it.",
        "",
        "## Report",
        "",
    ]
    lines += [f"- {item}" for item in output_contract]
    lines.append("")
    return "\n".join(lines)


def dry_run(
    dialogue: engine.Dialogue,
    actor_id: str,
    substitution_reason: str | None = None,
) -> dict:
    definition, turn, context, actor_selection, substitution_reason = _context_for(
        dialogue, actor_id, substitution_reason
    )
    packet = adapters.adapter_for(context.actor).prepare(context)
    return {
        "dry_run": True,
        "executed": False,
        "round_id": turn.round_id,
        "actor_id": actor_id,
        "scheduled_actor_id": turn.actor_id,
        "actor_selection": actor_selection,
        "substitution_reason": substitution_reason,
        "transport": context.actor.transport,
        "work_dir": str(context.work_dir),
        "packet": packet.as_dict(),
        "note": "no process was started, no claim was made, no state changed",
    }


def prepare(
    dialogue: engine.Dialogue,
    actor_id: str,
    output: Path | str,
    substitution_reason: str | None = None,
) -> dict:
    definition, turn, context, actor_selection, substitution_reason = _context_for(
        dialogue, actor_id, substitution_reason
    )
    adapter = adapters.adapter_for(context.actor)
    packet = adapter.prepare(context)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    briefing = build_task_briefing(
        definition,
        turn,
        dialogue,
        context,
        adapter.output_contract(context),
        substitution_reason,
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
        "scheduled_actor_id": turn.actor_id,
        "actor_selection": actor_selection,
        "substitution_reason": substitution_reason,
        "task_file": str(output),
        "packet": packet.as_dict(),
    }


def launch(
    dialogue: engine.Dialogue,
    actor_id: str,
    timeout: int | None = None,
    substitution_reason: str | None = None,
) -> dict:
    definition, turn, context, actor_selection, substitution_reason = _context_for(
        dialogue, actor_id, substitution_reason
    )
    adapter = adapters.adapter_for(context.actor)
    try:
        packet = adapter.prepare(context)
    except adapters.AdapterError as exc:
        raise engine.ProtocolError(str(exc)) from exc

    dialogue.claim(actor_id, substitution_reason=substitution_reason)
    try:
        context.work_dir.mkdir(parents=True, exist_ok=True)
        briefing = build_task_briefing(
            definition,
            turn,
            dialogue,
            context,
            adapter.output_contract(context),
            substitution_reason,
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
    except adapters.CleanupUnprovenError as exc:
        # The external worker lane may still be alive. Releasing the claim
        # would let a retry start a duplicate worker beside it, so the
        # claim and its lock stay in place and the dialogue locks BLOCKED
        # (release() refuses BLOCKED dialogues) until a human resolves it.
        dialogue.block(f"unproven worker-lane cleanup: {exc}")
        raise engine.ProtocolError(str(exc)) from exc
    except BaseException:
        # Fail closed but leave the turn claimable again: the adapter has
        # already proven that no worker lane survived this failure.
        try:
            dialogue.release(actor_id)
        except engine.ProtocolError:
            pass
        raise
    return {
        "dry_run": False,
        "executed": True,
        "completed_round": turn.round_id,
        "actor_id": actor_id,
        "scheduled_actor_id": turn.actor_id,
        "actor_selection": actor_selection,
        "substitution_reason": substitution_reason,
        "status": state["status"],
        "turn_index": state["turn_index"],
        "packet": packet.as_dict(),
    }
