"""Immutable artifact handling: hashing, publication, word counts.

Every byte the protocol reads for validation is read exactly once
through an ``O_NOFOLLOW`` descriptor, and every byte the protocol
writes is written through an exclusive (``O_CREAT | O_EXCL``)
``O_NOFOLLOW`` descriptor after verifying that no path component is a
symlink. Publication takes the already-validated bytes (or an expected
SHA-256), so a file swapped between validation and publication fails
closed instead of silently publishing unvalidated bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path


class ArtifactError(RuntimeError):
    """An artifact cannot be read, published, or trusted."""


_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+|[㐀-鿿]")


def _reject_symlink_components(path: Path, what: str) -> None:
    if path.is_symlink():
        raise ArtifactError(f"{what} must not be a symlink: {path}")
    for parent in path.parents:
        if parent.is_symlink():
            raise ArtifactError(f"{what} must not sit behind a symlink: {parent}")


def read_bytes_nofollow(path: Path | str, what: str = "artifact") -> bytes:
    """Read a regular file once, refusing symlinks at the final component."""
    path = Path(path)
    _reject_symlink_components(path, what)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ArtifactError(f"cannot read {what} {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError(f"{what} is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise ArtifactError(f"cannot read {what} {path}: {exc}") from exc
    finally:
        os.close(fd)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(read_bytes_nofollow(path))


def word_count(text: str) -> int:
    """Count words; CJK characters count individually, like the v1/v2 validator."""
    return len(_WORD_RE.findall(text))


def _write_exclusive_fd(path: Path, what: str) -> int:
    """Open ``path`` for exclusive creation, following no symlink anywhere.

    ``O_EXCL`` fails on ANY existing entry — including a symlink planted
    between a pre-check and this open — so there is no follow window.
    """
    _reject_symlink_components(path, what)
    try:
        return os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
        )
    except FileExistsError as exc:
        raise ArtifactError(f"{what} already exists: {path}") from exc
    except OSError as exc:
        raise ArtifactError(f"cannot create {what} {path}: {exc}") from exc


def _write_all(fd: int, data: bytes, path: Path, what: str) -> None:
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ArtifactError(f"short write to {what} {path}")
            view = view[written:]
        os.fsync(fd)
    except OSError as exc:
        raise ArtifactError(f"cannot write {what} {path}: {exc}") from exc


def write_bytes_exclusive(path: Path | str, data: bytes, what: str = "work file") -> None:
    """Write ``data`` through an exclusive no-follow descriptor.

    A stale regular file from an earlier failed attempt is removed first;
    a symlink (dangling or not) is an error, never silently unlinked, so
    a planted redirection is surfaced instead of papered over.
    """
    path = Path(path)
    if path.is_symlink():
        raise ArtifactError(f"{what} must not be a symlink: {path}")
    if path.exists():
        if not path.is_file():
            raise ArtifactError(f"{what} exists and is not a regular file: {path}")
        try:
            path.unlink()
        except OSError as exc:
            raise ArtifactError(f"cannot replace stale {what} {path}: {exc}") from exc
    fd = _write_exclusive_fd(path, what)
    try:
        _write_all(fd, data, path, what)
    finally:
        os.close(fd)


def publish_bytes(data: bytes, target: Path | str) -> str:
    """Publish exactly ``data`` to ``target`` once and return its SHA-256.

    Published artifacts are immutable: any existing target (file or
    symlink, dangling included) is an error, and no path component may
    be a symlink — publication can never escape the dialogue directory
    through a swapped parent.
    """
    target = Path(target)
    _reject_symlink_components(target, "published artifact")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError(f"cannot create publication directory: {exc}") from exc
    # mkdir may have resolved through a directory swapped in meanwhile;
    # re-check before the exclusive create (the O_NOFOLLOW open still
    # protects the final component either way).
    _reject_symlink_components(target, "published artifact")
    fd = _write_exclusive_fd(target, "published artifact")
    try:
        _write_all(fd, data, target, "published artifact")
    finally:
        os.close(fd)
    return sha256_bytes(data)


def publish(source: Path, target: Path, expected_sha: str | None = None) -> str:
    """Read ``source`` once, optionally pin its SHA-256, publish, return SHA.

    ``expected_sha`` binds publication to previously validated bytes: if
    the file was swapped after validation, the digest no longer matches
    and nothing is published.
    """
    data = read_bytes_nofollow(Path(source), "source artifact")
    digest = sha256_bytes(data)
    if expected_sha is not None and digest != expected_sha:
        raise ArtifactError(
            f"artifact {source} changed between validation and publication: "
            f"validated sha256 {expected_sha}, current {digest}"
        )
    return publish_bytes(data, Path(target))
