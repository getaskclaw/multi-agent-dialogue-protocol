# Public snapshot provenance

The first public commit is a sanitized source snapshot derived from development commit:

```text
bb3719b4f08492a4b2f4f8c74b83fe55e5e0d8ac
```

The public repository intentionally starts with a fresh Git history. Internal CI runner configuration, machine-specific paths, operator instructions, and private development history were excluded rather than published and then rewritten.

The source behavior was preserved, while the public snapshot adds:

- Apache-2.0 licensing;
- Python package metadata and the `madp` console command;
- a beginner-first README;
- public GitHub Actions CI;
- security and contribution policies;
- this explicit provenance note.

The private development commit identifier is an audit reference, not a claim that the private repository or its complete history is publicly verifiable. The public tree, tests, schemas, commits, and GitHub CI are the authority for the published version.
