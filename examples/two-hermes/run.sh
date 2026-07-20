#!/usr/bin/env bash
# Two independent Hermes profiles (distinct HERMES_HOME per actor),
# proven end to end with deterministic fake runtimes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PATH="$ROOT/examples/fakes/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DIALOGUE="${1:-$(mktemp -d)/dialogue}"
madp() { python3 -m multi_agent_dialogue "$@"; }

# Every dialogue transition is a local Git commit, so the dialogue must
# live inside a Git worktree. The default is an isolated throwaway repo
# with a repo-local identity; global Git config is never touched.
REPO_DIR="$(dirname "$DIALOGUE")"
mkdir -p "$REPO_DIR"
if ! git -C "$REPO_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "== create isolated example Git repo (repo-local identity) =="
  git init -q "$REPO_DIR"
  git -C "$REPO_DIR" config user.name "madp-example"
  git -C "$REPO_DIR" config user.email "madp-example@example.invalid"
fi

echo "== init =="
madp init --definition "$HERE/protocol.json" --dialogue "$DIALOGUE"

echo "== dry run: packet only, no process starts =="
madp run "$DIALOGUE" --actor hermes-north

echo "== wrong-actor claim is rejected (expected failure) =="
if madp claim "$DIALOGUE" --actor hermes-south; then
  echo "ERROR: hermes-south claimed hermes-north's turn"; exit 1
fi

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
