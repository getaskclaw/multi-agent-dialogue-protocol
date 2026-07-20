#!/usr/bin/env python3
"""Full repository verification: compile, tests, schemas, secrets, git hygiene.

Run from anywhere: python3 scripts/verify.py
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multi_agent_dialogue import config, evidence  # noqa: E402

# Patterns are assembled from pieces so this file never matches itself.
SECRET_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        "sk-" + "ant-" + "[A-Za-z0-9-]{8,}",
        "AKIA" + "[0-9A-Z]{16}",
        "ghp" + "_[A-Za-z0-9]{36}",
        "gho" + "_[A-Za-z0-9]{36}",
        "xox" + "[bpars]-[A-Za-z0-9-]{10,}",
        "AIza" + "[0-9A-Za-z_-]{35}",
        "-----BEGIN " + "(?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
]


def git_backed(root: Path) -> bool:
    """True in a normal clone (``.git`` directory) AND in a linked Git
    worktree, where ``.git`` is a gitdir-pointer FILE, not a directory."""
    marker = root / ".git"
    return marker.is_dir() or marker.is_file()


def run_step(name: str, argv: list[str]) -> dict:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    ok = result.returncode == 0
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {' '.join(argv)}")
    if not ok:
        sys.stdout.write(result.stdout[-2000:])
        sys.stderr.write(result.stderr[-2000:])
    return {"name": name, "argv": argv, "ok": ok, "returncode": result.returncode}


def check_schemas() -> dict:
    errors: list[str] = []
    protocol_schema = json.loads(
        (ROOT / "schemas" / "protocol.schema.json").read_text(encoding="utf-8")
    )
    evidence_schema = json.loads(
        (ROOT / "schemas" / "runtime-evidence.schema.json").read_text(encoding="utf-8")
    )
    # Every schema example must satisfy the Python validator.
    for position, example in enumerate(protocol_schema.get("examples", [])):
        try:
            config.parse_definition(example)
        except config.ConfigError as exc:
            errors.append(f"protocol.schema.json example {position}: {exc}")
    # Schema required list must agree with the Python evidence contract.
    schema_required = set(evidence_schema.get("required", []))
    python_required = set(evidence.REQUIRED_FIELDS)
    if schema_required != python_required:
        errors.append(
            "runtime-evidence.schema.json required fields diverge from "
            f"evidence.REQUIRED_FIELDS: {sorted(schema_required ^ python_required)}"
        )
    for position, example in enumerate(evidence_schema.get("examples", [])):
        missing = python_required - set(example)
        if missing:
            errors.append(
                f"runtime-evidence example {position} missing fields: {sorted(missing)}"
            )
    # Every committed example definition must parse.
    for example_dir in sorted((ROOT / "examples").iterdir()):
        definition_path = example_dir / "protocol.json"
        if definition_path.is_file():
            try:
                config.load_definition(definition_path)
            except config.ConfigError as exc:
                errors.append(f"{definition_path.relative_to(ROOT)}: {exc}")
    ok = not errors
    print(f"[{'ok' if ok else 'FAIL'}] schema/example cross-check")
    for error in errors:
        print(f"    - {error}", file=sys.stderr)
    return {"name": "schema-cross-check", "ok": ok, "errors": errors}


def check_secrets() -> dict:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    findings: list[str] = []
    if tracked.returncode == 0:
        files = [ROOT / line for line in tracked.stdout.splitlines()]
    else:
        files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: matches {pattern.pattern[:12]}…")
    ok = not findings
    print(f"[{'ok' if ok else 'FAIL'}] secret-pattern scan ({len(files)} files)")
    for finding in findings:
        print(f"    - {finding}", file=sys.stderr)
    return {"name": "secret-scan", "ok": ok, "findings": findings}


def main() -> int:
    results = [
        run_step(
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"],
        ),
        run_step(
            "unittest",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        ),
        check_schemas(),
        check_secrets(),
    ]
    if git_backed(ROOT):
        results.append(run_step("git-diff-check", ["git", "diff", "--check"]))
    ok = all(item["ok"] for item in results)
    print(json.dumps({"ok": ok, "steps": [item["name"] for item in results]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
