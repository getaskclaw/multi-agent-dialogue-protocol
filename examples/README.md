# Examples

Three complete topologies, each proven with deterministic fake runtimes.
Nothing here contacts a real agent, credential, or network.

| Example | Actors | Transports |
|---|---|---|
| `two-claude/` | claude-a, claude-b | fable-session × 2 |
| `two-hermes/` | hermes-north, hermes-south | hermes-cli × 2 (distinct `HERMES_HOME`s) |
| `three-mixed/` | claude-lead, hermes-critic, tool-scribe | fable-session + hermes-cli + command |

## Run one

```bash
bash examples/two-claude/run.sh          # fresh temp dialogue directory
bash examples/two-hermes/run.sh /tmp/my-dialogue   # or choose the directory
bash examples/three-mixed/run.sh
```

Each script creates an **isolated throwaway Git repository** for the
dialogue (a dialogue only initializes inside a Git worktree; the repo
gets a repo-local identity, global Git config is never touched), then
shows, in order: `init` (one initialization commit), a **dry run**
(packet printed, no process starts), a rejected wrong-actor claim, one
`--launch` per scheduled turn (each successful turn is exactly one
local commit), a rejected post-final turn, `owner-decide` (its own
terminal commit), `validate --require-git` (proves the commit
provenance and prints the proven SHAs), the dialogue's Git history, and
the final JSON status. Nothing is ever pushed.

## Step by step (two-claude)

```bash
export PATH="$PWD/examples/fakes/bin:$PATH"
export PYTHONPATH="$PWD/src"
D=$(mktemp -d)/dialogue
git init -q "$(dirname "$D")" && git -C "$(dirname "$D")" config user.name demo \
  && git -C "$(dirname "$D")" config user.email demo@example.invalid

python3 -m multi_agent_dialogue init --definition examples/two-claude/protocol.json --dialogue "$D"
python3 -m multi_agent_dialogue next "$D"                      # who is up: R01, claude-a
python3 -m multi_agent_dialogue run "$D" --actor claude-a      # dry-run (default): no process
python3 -m multi_agent_dialogue run "$D" --actor claude-a --launch   # exactly one turn
python3 -m multi_agent_dialogue status "$D"
```

## Fake runtimes

`fakes/bin` contains contract-faithful fakes of the real CLIs:

- `fake-fable-session` implements the verified 0.3.0b1 lifecycle
  (`run --project/--task/--registry/--state-dir` with `--dry-run` or
  `--launch --tmux`, one `watch --manifest --follow`, and
  `audit --manifest --format json`), writes a real-shaped run manifest
  plus stream-json transcript, and rejects the invented flags older
  revisions used. The two-claude and three-mixed `run.sh` scripts
  generate the host-local project registry under the dialogue's ignored
  `work/` scratch area (transport scratch is never committed).
- `fake-hermes` implements the real one-shot shape
  (`chat -q ... -Q --source ... --pass-session-id`) and records the
  session in `${HERMES_HOME}/state.db` with the real table/column
  names; stdout is deliberately not authoritative. Its compatibility
  fixture leaves `sessions.ended_at` and `end_reason` NULL — the proof
  of completion is the clean exit plus
  the source-matched session's final active assistant message and
  positive `api_call_count`.
- `fake-worker` writes only the turn artifact and prints self-claimed
  identity noise that adapters must ignore; `fake-verifier` is the
  external identity collector the command transport requires.

They are deterministic (content and session IDs derive only from
arguments/environment; only database timestamps use the clock), they
append a tagged line to `$FAKE_SPAWN_MARKER` per invocation so tests
can prove exactly which processes ran, and they stay honest about
being fakes: manifests carry a fake tool name and verifier reports
carry `"fake": true` — nothing they emit is ever presented as real
model identity evidence.
