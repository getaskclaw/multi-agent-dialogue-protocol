# Technical reference

This is the exhaustive protocol and runtime reference. New users should start with the [beginner quickstart](../README.md).

MADP is a small, evidence-first protocol engine for bounded decisions between two or more agent runtimes. It coordinates them through plain files in a Git-backed directory where every accepted protocol transition is a local Git commit. A written actor or model label is never treated as runtime identity proof.

## The production path — five commands

After installation, `madp` is the console command. When running directly from an uninstalled source checkout, use `python3 -m multi_agent_dialogue` with `src/` on `PYTHONPATH`. The core uses only the Python 3.12 standard library. The production path is:

```bash
madp init --definition protocol.json --dialogue DIR
madp status DIR
madp run DIR --actor ACTOR --launch
madp validate DIR --require-git --require-runner-completion
madp owner-decide DIR --decision DECISION.md
```

1. **`madp init --definition protocol.json --dialogue DIR`** — freezes
   the protocol definition into `DIR` and creates the initialization
   commit. **A real Git worktree is a hard prerequisite**: `init`
   refuses to run outside a Git repository/worktree, and refuses a
   dialogue path that already contains generated files
   (`state.json`, `definition.json`, `.gitignore`).
2. **`madp status DIR`** — one JSON response naming the current state,
   the next scheduled actor and round, the active claim,
   `blocked_reason` (why nothing can proceed, or `null`),
   `next_legal_action` (the one thing that may legally happen next),
   and `recovered_turns` (how many completed turns carry
   caller-supplied provenance).
3. **`madp run DIR --actor ACTOR --launch`** — executes exactly one
   scheduled turn through its transport adapter and completes it. This
   is the **only production-honest completion door**: it is the only
   path that records `completed_via: runner-launch`. Run it once per
   scheduled turn (dry-run is the default without `--launch`:
   `madp run DIR --actor ACTOR` prints the command packet and starts
   no process, makes no claim, changes no state).
4. **`madp validate DIR --require-git --require-runner-completion`** —
   the production gate (see *Validation levels* below).
5. **`madp owner-decide DIR --decision DECISION.md`** — requires
   `READY_FOR_OWNER`. **Decision grammar:** the decision file must
   contain a line `Decision: <VALUE>` where `<VALUE>` is one of the
   definition's `owner_decisions` (for the examples here:
   `APPROVE`, `REJECT`, `NEED_MORE_EVIDENCE`). Anything else fails
   closed.

`madp next DIR` remains a read-only helper showing the next scheduled
turn; it is not a required transition. Automatic multi-turn loops are
deliberately out of scope; every launch is one explicit turn.

```bash
python3 -m unittest discover -s tests   # full suite (PYTHONPATH=src, or run scripts/verify.py)
python3 scripts/verify.py               # compile + tests + schemas + secret scan + git hygiene
bash examples/two-claude/run.sh         # complete fake-runtime dialogue, start to owner decision
                                        # (creates an isolated throwaway Git repo for the dialogue)
```

## Recovery — the unverified namespace

`claim`, `prepare`, and `complete` are deliberately **not** in the
top-level CLI. They live, together with `release`, in an explicitly
unverified recovery namespace:

```bash
python -m multi_agent_dialogue.unverified claim DIR --actor ACTOR [--revision N]
python -m multi_agent_dialogue.unverified prepare DIR --actor ACTOR --output TASK.md
python -m multi_agent_dialogue.unverified complete DIR --actor ACTOR --turn TURN.md --runtime-evidence EVIDENCE.json
python -m multi_agent_dialogue.unverified release DIR --actor ACTOR
```

They exist to preserve adapter output after an orchestrator crash
between external execution and committed completion, to recover a
stray live `CLAIMED` state, and to regenerate a briefing for a
manually controlled recovery.

**Recovery limits:**

- every operation remains subject to the actor, round, artifact,
  evidence-shape, digest, session-uniqueness, and state checks;
- recovery is explicitly unverified and **cannot manufacture adapter
  provenance**: a completion through this namespace is always recorded
  `completed_via: caller-supplied`, structural validation warns about
  every such round, and the production gate rejects it;
- `release` wires the engine's existing release behavior (only the
  claim holder may release, and only outside `BLOCKED`);
- **no recovery operation releases or completes a `BLOCKED`
  dialogue** — claim, release, and complete all refuse it; recovery
  from `BLOCKED` is a human decision outside the protocol.

## Completion provenance (`completed_via`)

Every completed turn records how it entered the dialogue, with exactly
two values:

- **`runner-launch`** — passed only by `run --launch`'s single
  completion call, after a transport adapter executed the real
  external lifecycle and derived the evidence from external records;
- **`caller-supplied`** — the engine default for every other
  completion door, including the unverified recovery namespace: the
  artifact and evidence passed every structural check, but no adapter
  proved the external lifecycle.

The value is stored in the completed-turn state record and carried as
the `Madp-Completed-Via` commit trailer. `validate --require-git`
cross-checks it between the current state and **the original turn
commit** — the commit in which that turn's artifact and evidence first
appeared. The original commit, not the mutable current state, is the
source of truth: a later state edit (even one that is itself
committed) cannot launder `caller-supplied` into `runner-launch`. A
missing, unknown, or mismatched value is a structural Git-provenance
error and fails closed.

Direct local Python code can still lie *before* committing a turn.
That remains inside the documented local-code and
rewritable-local-Git trust boundary; nothing here claims cryptographic
provenance.

## Validation levels

**Structural** — `madp validate DIR --require-git`:

verifies state, artifacts, evidence shape, digests, commit unity and
order, append-only published history, and the committed
`completed_via` of every turn. A structurally valid caller-supplied
recovery may leave `ok: true`, but the report emits a prominent
warning naming **every** affected round.

**Production** — `madp validate DIR --require-git
--require-runner-completion`:

- `--require-runner-completion` requires `--require-git`; using it
  alone is an error (runner provenance is proven from the turn
  commits);
- it fails unless every completed turn's **Git-proven**
  `completed_via` is `runner-launch`;
- it checks provenance only: it never pretends an incomplete `OPEN` or
  `CLAIMED` dialogue is complete, and it works consistently at
  `OPEN`, `CLAIMED`, `READY_FOR_OWNER`, and `OWNER_DECIDED`, always
  reporting the actual recorded status;
- owner identity remains separately represented by `caller_identity`
  and `owner_proof`; worker completion provenance never replaces owner
  proof.

`madp validate DIR` without flags is convention-only checking (state,
digests, evidence shape) and merely warns that immutability is not
Git-proven.

## Finite execution and continuation

Every dialogue keeps its explicit finite schedule; the configured
final round is a hard stop and extra turns fail closed. A wider
deliberation may continue through a **new bounded instance** with a
higher `version`, a fresh finite schedule, and `evidence_roots`
pointing at the prior closed instance.

In v1 that relationship is an operator convention — **reviewable, not
proven**: the version, evidence roots, prior decision, and Git
ancestry make the continuation reviewable by a human, but the engine
does not authenticate who initialized the continuation and does not
prove a parent–child link across instances. This documentation must
not be read as upgrading that reviewability into authentication or
engine proof.

## Six facts the protocol refuses to conflate

| Fact | Where it lives | Example |
|---|---|---|
| **Protocol role** | definition `actors[].role` | "challenger" — what to argue, nothing more |
| **Runtime identity** | evidence `provider`/`model`/`session_id` | what actually ran, observed at run time |
| **Transport** | definition `actors[].transport` | `fable-session`, `hermes-cli`, `command` |
| **Provider** | constraint `expected_provider` + evidence `provider` | `anthropic`, `nousresearch` |
| **Model** | constraint `expected_model` + evidence `model` | `claude-fable-5`, `hermes-4-405b` |
| **Session evidence** | per-turn `evidence/*.json` | run ID, terminal outcome, artifact SHA-256, proof refs |

The **owner** is a seventh, separate fact: a human named in the
definition who is not an actor, whose `Decision:` line is the only way
a dialogue reaches `OWNER_DECIDED`. And `completed_via` is an eighth:
*how* a completed turn entered the record, independent of who or what
produced it.

A role name never selects a transport, a profile, or a model. A
Markdown `actor:` or `model:` frontmatter field is never accepted as
identity. Completion requires a machine-readable runtime-evidence
record that matches the actor's declared constraints, hashes the exact
artifact bytes, reports a successful terminal outcome, names its
external record family in `proof.kind`, and carries a session ID not
used by any other turn in the dialogue. Evidence is generated by the
adapter from external records — a worker-authored evidence file is
never requested and never trusted.

## Real CLI contracts

The adapters implement the verified invocation shapes of the installed
tools (fakes in `examples/fakes/bin` are contract-faithful).
Structural support is not a production-runtime claim: each transport
earns that claim only after a full live canary.

**Claude via `fable-session` 0.3.0b1** (per-actor settings: `project`,
`registry`, `state_dir`, `tmux_prefix`):

```text
fable-session run --project NAME --task ABS --registry ABS --state-dir ABS --dry-run
fable-session run --project NAME --task ABS --registry ABS --state-dir ABS --launch --tmux PREFIX-UNIQUE
fable-session watch --manifest ABS --follow        # exactly ONE watcher per lane
fable-session audit --manifest ABS --format json   # the model/runtime authority
```

The launch prints `run manifest: /abs/manifest.json (pending)`. The
turn is the FINAL text-bearing assistant event of the manifest's
structured stream; evidence combines manifest, audit JSON, the watch
terminal result, and the adapter-computed artifact hash. The lane must
audit `PURE` with a successful terminal result or completion fails.
The audit proves *model purity* of the observed stream; it does not
independently prove the upstream provider — the evidence records
`provider: anthropic` as the transport family (fable-session launches
the Claude Code CLI), and says so in `proof.provider_basis`.

If anything fails **after** the launch attempt — the launch CLI crashes
after creating the session, the watcher exits nonzero or times out on a
stuck lane, the audit rejects the lane, the stream is unusable — the
adapter stops exactly the generated tmux lane with
`tmux kill-session -t =NAME` (`=` is tmux's exact-match target: no
prefix or fuzzy match, so no unrelated session is ever inspected or
killed; `tmux_command_name`, default `tmux`, names the binary) and
requires `tmux has-session -t =NAME` to exit nonzero — proof the lane
is gone — before the turn claim may be released for a retry. If that
proof cannot be obtained, the dialogue locks `BLOCKED` with the claim
and lock file retained: claim, release, and relaunch all fail closed,
so a retry can never start a duplicate worker beside a surviving lane;
recovery is a human decision. Successful lifecycles never touch tmux
and use exactly one watcher and one audit.

**Hermes** (per-actor settings: `hermes_home`, mandatory and unique per
actor):

```text
HERMES_HOME=/abs/actor/home hermes chat -q PROMPT -Q --source UNIQUE_SOURCE --pass-session-id
```

After the command exits, the adapter opens `${HERMES_HOME}/state.db`
read-only and derives the session ID, provider, observed model set,
and the final ACTIVE assistant message (the turn) from
`sessions`/`messages`/`session_model_usage` — exactly one session with
this turn's unique `--source` inside the invocation window. Stdout is
never the authority.

Compatible Hermes session records may leave
`sessions.ended_at`/`end_reason` both NULL after a successful one-shot.
The adapter therefore proves completion by clean subprocess exit + unique
source attribution + final active assistant message + positive API-call
usage, and records that basis in the proof (`terminal_basis`) instead
of claiming a DB terminal state. If the terminal
fields *are* present they must be consistent (`ended_at` inside the
invocation window; never an `end_reason` without an `ended_at`) and
clean (`end_reason` NULL, empty, or exactly `completed`); ambiguous
sources, missing final messages, zero API calls, mixed models or
providers, out-of-window starts, nonzero exits, and dirty or
inconsistent terminal fields all fail closed.

**Turn briefings**: every actor receives one `## Goal` / `## Checks` /
`## Boundaries` / `## Report` task file (the exact sections
`fable-session` requires; one shape serves every transport). From the
second turn on, the briefing lists every published prior turn AND its
runtime-evidence record by **absolute path inside the dialogue
directory** together with their published sha256 digests, and requires
the worker to read each one in full before producing its challenge or
response — a later actor argues against what was actually written, not
against a relative filename it cannot resolve.

**Generic command**: a command cannot prove its own provider/model by
printing JSON. `settings.identity_verifier_argv` (an external identity
collector run by the adapter after the worker) is REQUIRED; without it
the transport fails closed before any process starts. A fake verifier's
`"fake": true` flag is preserved in the proof, never upgraded to real
identity.

## Git transaction model

A dialogue is a sequence of local Git commits, not just files Git
happens to see:

- **Init requires a worktree.** `init` refuses to run outside a real
  Git repository/worktree (normal clones and linked worktrees both
  work; the dialogue may sit at the repository root or nested anywhere
  below it). It creates exactly one initialization commit containing
  `definition.json`, the initial `state.json`, and the dialogue-local
  `.gitignore` that keeps `work/` (task briefings/prompts and transport
  scratch such as the fable registry, fable state dirs, and per-actor
  `HERMES_HOME`s), the live claim lock, and `*.tmp` files out of
  history forever. Prompts, transport scratch, and live locks are never
  committed.
- **One commit per successful turn.** After evidence validation and
  immutable publication, every completion — `run --launch` and the
  unverified recovery door alike — creates exactly one commit
  containing the updated `state.json`, the published turn, and its
  runtime-evidence record — nothing else. Even nested inside a larger
  repository, unrelated staged or untracked files are never staged and
  user-staged work stays staged. If an earlier failed, released
  attempt bumped only transient state/revision, that state history
  rides along in the next successful turn commit without staging
  anything outside the dialogue's paths.
- **The owner decision is its own terminal commit** (`state.json` +
  `OWNER-DECISION.md`), always distinct from and after the final turn
  commit.
- **Trailers identify the runtime and the completion door.** Commit
  messages carry non-secret machine-readable trailers:
  `Madp-Protocol`, `Madp-Event`, `Madp-Round`, `Madp-Actor`,
  `Madp-Transport`, `Madp-Provider`, `Madp-Model`, `Madp-Session`,
  `Madp-Completed-Via`, `Madp-Artifact-Sha256`,
  `Madp-Evidence-Sha256` (turn commits), `Madp-Definition-Digest`
  (init), and `Madp-Decision`/`Madp-Caller-Identity` (decision).
- **The repository's identity, untouched.** Commits use whatever
  identity the repository already resolves; the engine never runs
  `git config`, never mutates global or local configuration, and skips
  hooks (`--no-verify`) so a host repository's hooks cannot rewrite or
  veto protocol history.

**Recovery and blocking.** If the Git commit fails after a turn was
published (or after the owner decision was recorded on disk, or after
init wrote its files), that transition is NOT complete: the dialogue
locks `BLOCKED` — non-retryable, since claim/release/complete/decide
all refuse BLOCKED dialogues, through the public path and the
unverified recovery namespace alike — and recovery is a human
decision. A turn is never reported complete, and a decision never
reported `OWNER_DECIDED`, without commit proof.

**Validation.** `validate --require-git` proves from local history
alone: the init commit carries definition + initial state + ignore
rules together; each completed turn's artifact and evidence first
appear together in exactly one commit whose committed `state.json`
already contains that exact
round/actor/digest/session/`completed_via` record and whose committed
bytes match the recorded digests; commit order matches the schedule;
and the owner decision, if present, is its own later commit recording
`OWNER_DECIDED`. Any commit that ever modified or deleted published
history is rejected — even when the working tree is clean. The proven
SHAs and the Git-proven `completed_via` of every turn are exposed in
the validation output (`provenance.init_commit`,
`provenance.turn_commits[]`, `provenance.owner_decision_commit`); they
are derived from history, never stored in state, because a commit
cannot contain its own SHA.

**What stays external.** Git hosting, remotes, push/pull, and any
cross-machine coordination remain outside the engine: the transaction
log is strictly local, and publishing it is push/pull discipline for
the humans or schedulers around the protocol.

## Fail-closed guarantees

- claims are atomic (`O_EXCL` lock + compare-and-swap revision); two
  writers can never own one turn;
- wrong actor, wrong round, duplicate claim, stale revision → error;
- published turns are immutable; any byte change flips the dialogue to
  `BLOCKED` before the next completion or decision;
- missing, malformed, or mismatched runtime evidence blocks completion;
- failed/timeout terminal outcomes block completion;
- a missing, unknown, or laundered `completed_via` is a structural
  Git-provenance error; only `run --launch` can record
  `runner-launch`, and the production validation gate rejects every
  other completion door;
- a fable-session failure after launch releases the claim only once the
  exact generated tmux lane is proven dead; unprovable cleanup locks
  the dialogue `BLOCKED` with the claim retained (non-retryable) so a
  second launch cannot create a duplicate worker;
- the configured final round is a hard stop; post-final turns fail
  closed; only `owner-decide` can close a dialogue;
- init outside a Git worktree fails closed; a Git commit failure after
  publication (turn or owner decision) locks the dialogue `BLOCKED`,
  non-retryably — no transition counts without commit proof;
- `BLOCKED` is non-releasable and non-retryable through the public
  path and the recovery namespace alike;
- dry-run is the default and starts no process, writes no state;
- publication and work-file writes are exclusive and no-follow: a
  symlinked target, a symlinked parent directory, or a file swapped
  between validation and publication fails closed — validation and
  publication use the same once-read bytes.

## Layout

```text
src/multi_agent_dialogue/   engine, config, evidence, artifacts, gitops, runner, cli, unverified
src/multi_agent_dialogue/adapters/   command, claude_fable, hermes
schemas/                    protocol + runtime-evidence JSON Schemas
examples/                   two-claude, two-hermes, three-mixed + fake runtimes
tests/                      unit + CLI + end-to-end topology proofs
scripts/verify.py           full verification gate
docs/plans/                 the implementation plan this repo was built from
```

A dialogue directory contains `definition.json` (frozen copy),
`state.json`, `.gitignore` (dialogue-local ignore rules, committed at
init), `turns/`, `evidence/`, `work/` (ignored scratch: briefings,
transport scratch, raw worker output), and finally
`OWNER-DECISION.md`. Init, each completed turn, and the owner decision
are one local Git commit each.

## Honest limits

- **Same model ≠ diverse models.** Two sessions of the same
  provider/model are separate contexts and separate session IDs, but
  they are *not* model-diverse evidence, and the engine does not claim
  otherwise.
- **Model audit ≠ provider proof.** The fable audit is the authority
  for model purity of the observed stream, and the Hermes `state.db`
  records the billing provider it was configured with; neither
  cryptographically proves which upstream provider actually served the
  tokens. Fable evidence records `provider` as the transport family
  (the Claude Code CLI) and labels that basis in the proof instead of
  implying the audit proved it.
- **Provenance is a local-code claim, not cryptography.** The
  `completed_via` cross-check makes a *committed* history honest — a
  later state edit cannot launder a recovery into a runner launch —
  but direct local Python can still lie before the turn commit exists,
  and a local Git history is rewritable by anyone with repository
  access. Publishing/mirroring the history (push) remains the external
  tamper-evidence anchor; nothing here claims cryptographic
  provenance.
- **Evidence is as strong as the backing records.** The adapters derive
  evidence from external records (fable-session manifest + audit +
  stream; the Hermes `state.db`; an explicit verifier command), and the
  engine validates structure, constraints, hashes, and session
  uniqueness — but nothing here cryptographically proves which model
  produced text.
- **Recovery cannot overclaim.** The unverified namespace preserves
  work after a crash, but every completion it makes is permanently
  marked `caller-supplied`, warned about by structural validation, and
  rejected by the production gate.
- **Continuation is reviewable, not proven.** A follow-on instance
  that points its `evidence_roots` at a closed predecessor is
  reviewable through version, roots, prior decision, and Git ancestry;
  the engine neither authenticates who created it nor proves the
  parent–child link.
- **Owner identity is not authenticated.** `owner-decide` is a separate
  terminal transition, but the recorded decision carries
  `caller_identity: "unverified"` unless an external owner-proof
  verifier (`owner_proof_argv`, receiving `{decision_file}`) is
  configured and exits 0 — and even then the claim is exactly "the
  configured verifier accepted it", never cryptographic owner identity.
- **No scheduler.** Real automatic wake-up needs an external scheduler
  (cron, CI, a human); the engine only ever executes one explicit turn.
- **Pinned contracts require local proof.** Adapter invocation shapes are version-sensitive. The repository includes deterministic, contract-faithful fake runtimes for repeatable tests, but it ships no user credentials and makes no production claim for another machine's installed tools. Before relying on a real transport, run a harmless one-turn smoke and a bounded live canary against the exact installed versions, then preserve provider/model/session evidence.
- **Locks are local.** The claim lock protects one shared filesystem;
  coordinating over a Git remote still requires push/pull discipline
  around claims. Same-UID processes are not adversaries in this threat
  model — the no-follow writes stop path tricks, not a hostile user.

## Safety baseline

- The protocol never grants approval; only the owner decides.
- Every runtime must prove provider/model/session identity per turn.
- Each turn runs in a fresh context from one frozen evidence boundary
  (`source_sha`).
- No accidental continuation after the configured final turn.
- Secrets never belong in protocol config, turn files, logs, or Git
  history (`scripts/verify.py` scans for common token patterns).

Built with strict RED → GREEN → REFACTOR and a fail-closed test suite. Historical development plans are intentionally omitted from the sanitized public snapshot; current behavior, schemas, tests, and this reference are the public authority.
