# Four-game reflection pilot

The core is assembled from the source snapshots actually loaded by the
completed GPT-4o mini reflection experiment. The per-game source hashes were
checked against that experiment's manifests before export. See
[source-provenance.json](source-provenance.json) for the imported files, original
hashes, and asset checks. The source used for IPD and RPS is the common core.
Hanabi's action-only partner prompt and Deliberation's closed message-envelope
validation come from the corresponding game snapshots.

The pilot runner makes model routing and dependencies explicit. Its packaged
Deliberation assets replace the historical absolute path; it records the actual
reflector in the memory bank's author metadata. The game rules, memory updates,
and prompts used by the verified pilot are otherwise preserved. Compatibility
fallbacks keep the offline and generic entry points working when no separate
opponent routing is configured, and allow the Hanabi partner to accept legacy
plain-text observations.

## Settings

- Training seeds: `202611070` through `202611073`, grouped into two batches.
- Test seeds: `202611074` through `202611085`. Evaluation does not update the
  final bank.
- IPD: 25 rounds and payoff per round (normalized by 5 for a 0–100 score).
- RPS: 20 rounds and percentage of episodes with strictly positive net reward.
- Hanabi: native two-player terminal score (normalized by 25 for a 0–100 score).
- Deliberation: six parties, opening, 48 discussion utterances, and final
  proposal. The original evaluator scores focal realized utility; an episode
  without agreement scores zero.
- Board-game memory uses E5 embeddings and top-k NLI reranking. Deliberation
  uses E5 encoding, cosine retrieval and automatic review of the full bank.
  Board-game training injection dropout is applied per decision.

This repository runs a fresh pilot for each selected reflector. The historical
paired study reused the two pre-reflection training trajectories across arms
and replayed eligible training requests from a reference cache. This export
contains the source, portable recipe and seed schedule. It does not contain
the historic trajectories, caches, trained banks, or paper-scale results.
Live provider responses can vary even with the same environment seed.

## Outputs

`manifest.json` records a source hash, the configuration, endpoint URLs and
environment-variable names. It does not contain API key values. Deliberation
also records hashes of the scenario assets. Each batch records a verified
checkpoint and the next memory bank. `evaluation/` stores terminal episodes,
`summary.json` aggregates their results, and `api/` records scoped response
caches and usage. These runtime directories can contain prompts and
trajectories, and should be treated as experiment data.

The offline smoke run is a test of the pipeline. It uses synthetic responses
and is labelled `offline_smoke`. The portable pilot uses the extracted
implementation and small sample budget, including the Deliberation review
and per-decision dropout described above. It is distinct from the full
paper-scale implementation and experiment design.
