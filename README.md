# Multi-Agent Dialogue Protocol (MADP)

[![CI](https://github.com/getaskclaw/multi-agent-dialogue-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/getaskclaw/multi-agent-dialogue-protocol/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

MADP runs a bounded discussion between two or more AI-agent runtimes and records what happened as evidence-backed Git commits. It keeps protocol role, runtime identity, provider, model, session, transport, and human authority separate.

## Why try it?

Most multi-agent demos trust whatever an agent writes about itself. MADP does not. A turn is accepted only when its transport adapter can collect external runtime evidence that matches the frozen protocol definition.

MADP is useful when you need:

- a fixed turn order and hard final stop;
- one explicit agent launch at a time;
- provider/model/session evidence instead of Markdown identity labels;
- one Git commit for initialization, every accepted turn, and the owner decision;
- fail-closed behavior when evidence, process cleanup, state, or Git provenance is wrong.

## Two-minute safe demo

Requirements: Bash, Git, and Python 3.12 or newer. The demo uses deterministic fake runtimes, needs no account or API key, and contacts no agent service.

```bash
git clone https://github.com/getaskclaw/multi-agent-dialogue-protocol.git
cd multi-agent-dialogue-protocol
python3 --version
bash examples/two-hermes/run.sh
```

This zero-install example runs directly from `src/`. It creates a throwaway Git repository, proves a wrong actor is rejected without spawning a runtime, runs a complete bounded dialogue, records an owner decision, validates commit provenance, and exits 0 with `"ok": true` and final `"status": "OWNER_DECIDED"`. Nothing is pushed.

Try the other fake topologies:

```bash
bash examples/two-claude/run.sh
bash examples/three-mixed/run.sh
```

## Install the `madp` command

With `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/madp --help
```

Or with a standard Python environment that includes `venv` and `pip`:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
madp --help
```

## Five production commands

```bash
madp init --definition protocol.json --dialogue DIR
madp status DIR
madp run DIR --actor ACTOR --launch
madp validate DIR --require-git --require-runner-completion
madp owner-decide DIR --decision DECISION.md
```

`madp run` is dry-run by default unless `--launch` is present. One launch processes at most one scheduled turn. Automatic loops are deliberately outside the engine.

## Supported adapters

- **Claude Code via `fable-session`** — external manifest, stream, watcher, and model-audit evidence.
- **Hermes Agent** — provider, model, session, API usage, and final active message derived from the actor profile's `state.db`.
- **Generic command** — requires a separate identity-verifier command; worker self-report is rejected.

Real adapters depend on the exact installed CLI versions. Run the fake demo first, then perform a harmless live smoke and bounded canary on your own machine before relying on a real transport.

## Why trust it?

The repository ships:

- a standard-library core with no runtime Python dependencies;
- unit, CLI, hardening, Git-transaction, recovery, and topology tests;
- deterministic fake implementations of all adapter contracts;
- JSON Schemas for protocol definitions and runtime evidence;
- `scripts/verify.py`, which compiles the tree, runs the suite, checks schemas/examples, scans tracked files for common secret patterns, and checks Git diffs;
- GitHub Actions running the same gates on supported Python versions.

These checks prove the implementation behavior covered by the repository. They do not cryptographically prove an upstream model or provider.

## Intended audience

Use MADP if you operate agent CLIs and need bounded, inspectable decision records. It is especially useful for design reviews, adversarial plan checks, and owner-gated decisions.

MADP is not:

- an autonomous agent scheduler;
- a permission or approval system;
- a distributed-consensus protocol;
- a blockchain;
- cryptographic proof of model, provider, or human identity.

## Current limits

- The canonical ledger is local Git. Remote push/pull and cross-machine writer coordination remain outside the engine.
- Claim locks protect one shared filesystem, not active-active coordinators on separate hosts.
- A user with local code and repository write access can rewrite local history.
- Owner identity is `unverified` unless the definition configures an external owner-proof verifier.
- Recovery can preserve work after a crash, but caller-supplied completions are permanently marked and rejected by the production validation gate.

Read the [technical reference](docs/technical-reference.md) for the exact state machine, adapter contracts, evidence rules, recovery namespace, Git transaction model, and honest limits.

## Development

```bash
python3 scripts/verify.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [PUBLICATION-PROVENANCE.md](PUBLICATION-PROVENANCE.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
