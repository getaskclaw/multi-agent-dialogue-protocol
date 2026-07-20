#!/usr/bin/env bash
# Three mixed workers — Claude (fable-session), Hermes (hermes-cli), and a
# generic command worker — proven end to end with deterministic fakes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PATH="$ROOT/examples/fakes/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

source "$ROOT/examples/demo-common.sh"
madp() { python3 -m multi_agent_dialogue "$@"; }
madp_example_init_repo "$@"

echo "== init =="
madp init --definition "$HERE/protocol.json" --dialogue "$DIALOGUE"

echo "== register the fable project (host-local registry, generated per run) =="
mkdir -p "$DIALOGUE/work/fable"
cat > "$DIALOGUE/work/fable/registry.toml" <<EOF
[project.claude-lead]
repo = "$ROOT"
profile = "$ROOT/examples/fakes/fable-profile.toml"
model = "claude-fable-5"
effort = "high"
fallback = "stop"
permission_mode = "auto"
tmux_prefix = "madplead-"
EOF

echo "== dry run: packet only, no process starts =="
madp run "$DIALOGUE" --actor claude-lead

echo "== wrong-actor production launch is rejected without a spawn =="
madp_example_assert_wrong_actor tool-scribe claude-lead R01

echo "== six launches, one turn each =="
for actor in claude-lead hermes-critic tool-scribe claude-lead hermes-critic tool-scribe; do
  madp run "$DIALOGUE" --actor "$actor" --launch
done

echo "== post-final worker turn is rejected (expected failure) =="
if madp run "$DIALOGUE" --actor claude-lead --launch; then
  echo "ERROR: post-final turn was allowed"; exit 1
fi

echo "== owner decision =="
DECISION="$(mktemp)"
printf 'Decision: APPROVE\n\nExample owner decision.\n' > "$DECISION"
madp owner-decide "$DIALOGUE" --decision "$DECISION"

echo "== validate (with full Git commit provenance) =="
madp validate "$DIALOGUE" --require-git

echo "== dialogue Git history: init, one commit per turn, owner decision =="
git -C "$REPO_DIR" log --oneline

echo "== final status =="
madp status "$DIALOGUE"
