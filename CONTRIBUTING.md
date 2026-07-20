# Contributing

Contributions that improve correctness, evidence quality, portability, tests, or documentation are welcome.

## Set up

```bash
git clone https://github.com/getaskclaw/multi-agent-dialogue-protocol.git
cd multi-agent-dialogue-protocol
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

## Verify

Run the complete local gate before opening a pull request:

```bash
python scripts/verify.py
```

For a safe end-to-end check with no credentials or network agent calls:

```bash
bash examples/two-hermes/run.sh
```

## Pull requests

- Keep each change focused.
- Add or update tests for behavior changes.
- Preserve dry-run-by-default and fail-closed behavior.
- Do not weaken runtime-evidence, Git-provenance, hard-stop, or owner-authority checks without an explicit security rationale.
- Never commit credentials, private prompts, real user session databases, or sensitive runtime evidence.
- Treat adapter CLI contracts as version-sensitive and include exact live evidence when changing them.

By contributing, you agree that your contribution is licensed under Apache-2.0.
