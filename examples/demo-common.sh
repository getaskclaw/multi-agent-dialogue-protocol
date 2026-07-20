#!/usr/bin/env bash
# Shared safety helpers for the deterministic public example runners.

madp_example_init_repo() {
  if [[ $# -gt 1 ]]; then
    echo "usage: $0 [NEW_OR_EMPTY_REPOSITORY_ROOT]" >&2
    return 2
  fi

  local requested="${1:-}"
  if [[ -n "$requested" ]]; then
    REPO_DIR="$requested"
    if [[ -L "$REPO_DIR" ]]; then
      echo "ERROR: example repository root must not be a symlink: $REPO_DIR" >&2
      return 2
    fi
    if [[ -e "$REPO_DIR" && ! -d "$REPO_DIR" ]]; then
      echo "ERROR: example repository root is not a directory: $REPO_DIR" >&2
      return 2
    fi
    mkdir -p -- "$REPO_DIR"
    if find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "ERROR: custom example repository root must be empty: $REPO_DIR" >&2
      return 2
    fi
  else
    REPO_DIR="$(mktemp -d)"
  fi

  REPO_DIR="$(cd "$REPO_DIR" && pwd -P)"
  if git -C "$REPO_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "ERROR: refusing to initialize an example inside an existing Git worktree: $REPO_DIR" >&2
    return 2
  fi

  echo "== create isolated example Git repo at exact root (repo-local identity) =="
  git init -q "$REPO_DIR"
  git -C "$REPO_DIR" config user.name "madp-example"
  git -C "$REPO_DIR" config user.email "madp-example@example.invalid"
  DIALOGUE="$REPO_DIR/dialogue"
}

madp_example_assert_wrong_actor() {
  if [[ $# -ne 3 ]]; then
    echo "ERROR: wrong-actor helper requires WRONG_ACTOR EXPECTED_ACTOR ROUND" >&2
    return 2
  fi

  local wrong_actor="$1"
  local expected_actor="$2"
  local round_id="$3"
  local work_dir="$DIALOGUE/work"
  local marker="$work_dir/wrong-actor-spawn-marker"
  local output="$work_dir/wrong-actor-output.txt"
  local expected_error="'$wrong_actor' is not the scheduled actor for $round_id; next actor is '$expected_actor'"
  local state_before
  local state_after

  mkdir -p "$work_dir"
  : > "$marker"
  state_before="$(git hash-object -- "$DIALOGUE/state.json")"
  if FAKE_SPAWN_MARKER="$marker" madp run "$DIALOGUE" --actor "$wrong_actor" --launch > "$output" 2>&1; then
    echo "ERROR: wrong actor $wrong_actor was allowed to launch" >&2
    return 1
  fi
  if ! grep -F -- "$expected_error" "$output" >/dev/null; then
    echo "ERROR: wrong-actor launch failed for an unexpected reason" >&2
    cat "$output" >&2
    return 1
  fi
  if [[ -s "$marker" ]]; then
    echo "ERROR: wrong-actor rejection spawned a fake runtime" >&2
    cat "$marker" >&2
    return 1
  fi
  state_after="$(git hash-object -- "$DIALOGUE/state.json")"
  if [[ "$state_after" != "$state_before" ]]; then
    echo "ERROR: wrong-actor rejection changed dialogue state" >&2
    return 1
  fi
  echo "wrong actor rejected before runtime spawn; state unchanged: $wrong_actor"
}
