#!/usr/bin/env bash
# Two independent Hermes profiles (distinct HERMES_HOME per actor),
# proven end to end with deterministic fake runtimes.
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

echo "== dry run: packet only, no process starts =="
madp run "$DIALOGUE" --actor hermes-north

echo "== wrong-actor production launch is rejected without a spawn =="
madp_example_assert_wrong_actor hermes-south hermes-north R01

echo "== four launches, one turn each =="
for actor in hermes-north hermes-south hermes-north hermes-south; do
  madp run "$DIALOGUE" --actor "$actor" --launch
done

echo "== post-final worker turn is rejected (expected failure) =="
if madp run "$DIALOGUE" --actor hermes-north --launch; then
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
