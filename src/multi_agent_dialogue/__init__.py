"""Transport-neutral, Git-backed decision-dialogue protocol engine.

The core separates six facts that are often conflated:

- protocol role: what an actor is supposed to argue (e.g. "challenger");
- runtime identity: the provider/model/session actually observed at run time;
- transport: how a turn is executed (fable-session, hermes-cli, command);
- provider and model: identity constraints declared per actor and proven
  by runtime evidence, never by Markdown labels;
- session evidence: the machine-readable record captured for each turn.
"""

__version__ = "1.0.1"
