"""Local Git transactions and provenance queries for dialogues.

Git is the dialogue's transaction log: initialization, every successful
worker turn, and the owner decision are each exactly one local commit of
exactly the dialogue-owned paths. This module only shells out to the
ambient ``git`` binary inside the repository that already contains the
dialogue directory; it never pushes, never talks to a remote, and never
reads or writes ``git config`` — the commit identity is whatever the
repository already resolves (a missing identity makes the commit fail,
which the engine turns into a ``BLOCKED`` dialogue).

Commits are made with ``git commit -- <paths>`` (a pathspec-limited
commit): unrelated staged or untracked files elsewhere in a larger
repository are never committed and user-staged work stays staged.
Hooks are skipped (``--no-verify``) so a host repository's commit hooks
cannot rewrite or veto protocol history.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """A required local Git operation failed."""


def _run(
    args: list[str], cwd: Path, check: bool = True
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise GitError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {detail}"
        )
    return result


def worktree_root(directory: Path | str) -> Path | None:
    """The root of the Git worktree containing ``directory``, or None.

    Works for normal clones and linked worktrees alike, and accepts a
    directory that does not exist yet (the nearest existing ancestor is
    probed instead, so ``init`` can check before creating anything)."""
    probe = Path(directory).absolute()
    while not probe.exists():
        if probe.parent == probe:
            return None
        probe = probe.parent
    if not probe.is_dir():
        probe = probe.parent
    result = _run(["rev-parse", "--show-toplevel"], cwd=probe, check=False)
    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top) if top else None


def rel_to_root(root: Path, path: Path | str) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def commit_paths(
    root: Path, paths: list[Path], subject: str, trailers: dict[str, str]
) -> str:
    """Create one commit of exactly ``paths`` and return its SHA.

    The commit message carries machine-readable ``Madp-*`` trailers.
    Values must already be non-secret (round/actor/transport/provider/
    model/session/digests are protocol facts by design)."""
    for key, value in trailers.items():
        if not re.fullmatch(r"[A-Za-z0-9-]+", key):
            raise GitError(f"unsafe Git trailer key: {key!r}")
        if not isinstance(value, str) or "\n" in value or "\r" in value:
            raise GitError(f"unsafe Git trailer value for {key}")
    rels = [rel_to_root(root, path) for path in paths]
    message = subject + "\n\n" + "".join(
        f"{key}: {value}\n" for key, value in trailers.items()
    )
    _run(["add", "-f", "--", *rels], cwd=root)
    _run(
        ["commit", "--quiet", "--no-verify", "-m", message, "--", *rels],
        cwd=root,
    )
    return _run(["rev-parse", "HEAD"], cwd=root).stdout.strip()


# -- provenance queries (read-only) ---------------------------------------


def first_commit_adding(root: Path, rel: str) -> str | None:
    """The commit in which ``rel`` first appeared, or None if never added."""
    result = _run(
        ["log", "--format=%H", "--diff-filter=A", "HEAD", "--", rel],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def committed_bytes(root: Path, commit: str, rel: str) -> bytes | None:
    """The bytes of ``rel`` as committed in ``commit``, or None if absent."""
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{rel}"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def committed_sha256(root: Path, commit: str, rel: str) -> str | None:
    data = committed_bytes(root, commit, rel)
    return hashlib.sha256(data).hexdigest() if data is not None else None


def commit_trailers(root: Path, commit: str) -> list[tuple[str, str]]:
    """Return parsed trailers from one commit without collapsing duplicates."""
    message = _run(["show", "-s", "--format=%B", commit], cwd=root).stdout
    try:
        parsed = subprocess.run(
            ["git", "interpret-trailers", "--parse"],
            cwd=str(root),
            input=message,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git interpret-trailers: {exc}") from exc
    if parsed.returncode != 0:
        raise GitError(
            f"git interpret-trailers failed (exit {parsed.returncode})"
        )
    result: list[tuple[str, str]] = []
    for line in parsed.stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        result.append((key, value))
    return result


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = _run(
        ["merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
    )
    return result.returncode == 0


def history_mutations(root: Path, rels: list[str]) -> list[str]:
    """Commits that modified or deleted any path under ``rels``.

    Published protocol history is append-only: any M/D touch of a
    published turn, evidence record, decision file, frozen definition,
    or the dialogue ignore rules is a history rewrite."""
    result = _run(
        ["log", "--format=%H", "--diff-filter=MD", "HEAD", "--", *rels],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]
