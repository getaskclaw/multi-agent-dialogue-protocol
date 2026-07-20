# Security policy

## Supported versions

Security fixes are made on the latest `main` branch. This project has not yet declared a long-term-support release line.

## Report a vulnerability

Use GitHub's private vulnerability reporting for this repository:

https://github.com/getaskclaw/multi-agent-dialogue-protocol/security/advisories/new

Please include affected versions or commits, reproduction steps, impact, and any suggested mitigation. Do not open a public issue for an unpatched vulnerability, and never include real credentials, private prompts, or sensitive runtime evidence.

You should receive an initial response within seven days. Acknowledgement does not promise a particular fix or disclosure date.

## Scope note

MADP validates protocol state and evidence contracts. It does not claim cryptographic proof of upstream model/provider identity, protect against a hostile same-UID local user, or provide distributed consensus. Reports that rely only on violating these documented trust boundaries may be closed as out of scope, but clear boundary-confusion bugs are welcome.
