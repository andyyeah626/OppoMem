# OppoMem

OppoMem adds opponent-aware memory to language model agents in strategic games.
The agent infers an opponent's strategy from observed play, retrieves relevant
memory, and uses it to choose a legal action. Between training batches, the
runner groups behavioral evidence, asks a reflector for reusable advice, and
freezes the resulting memory for evaluation.

## Quick start

Python 3.10 or later is required. These commands work with PowerShell or a
Linux shell after activating the virtual environment.

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# PowerShell:   .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
oppomem smoke --output runs/smoke
python -m pytest -q
```

The smoke command uses a deterministic model substitute and a pure Python RPS
environment. It trains, updates, saves, and evaluates a small memory bank without
API calls or model downloads.

## Four-game pilot

Use **Linux or Ubuntu on WSL** for the full pilot. IPD uses Axelrod. RPS and
Hanabi use OpenSpiel. Deliberation's six-party Base scenario and prompt template
are bundled with their upstream license.

```bash
python -m pip install -e ".[live,ipd,spiel]"
oppomem-pilot --help
oppomem-pilot suite --configs configs/pilot --reflector 4o-mini --output runs/4o-mini
```

Before the live command, export the credential variables in
[.env.example](.env.example) in your shell. Use an OpenAI-compatible endpoint
URL (including the appropriate API prefix) for each gateway. Keep credentials
in the shell environment. The package does not source an environment file.

The pilot needs a cached `intfloat/e5-base-v2` encoder and
`cross-encoder/nli-deberta-v3-xsmall` scorer. To populate a standard Hugging
Face cache, run these once in an environment with network access:

```bash
hf download intfloat/e5-base-v2
hf download cross-encoder/nli-deberta-v3-xsmall
```

Set `OPPOMEM_MODEL_CACHE` to a custom hub cache directory if needed. The live
runner loads weights locally by default. To run just one game or use the
backbone as reflector:

```bash
oppomem-pilot run --config configs/pilot/ipd.json --reflector ds --output runs/ds-ipd
```

The three reflector names in [roles.json](configs/pilot/roles.json) are
`ds`, `sol`, and `4o-mini`. This setting affects only memory reflection
and Deliberation review. Backbone decisions, ToM, and segmentation use
DeepSeek Flash with low reasoning effort. Training counterparts use
Qwen3.6-Flash. Test counterparts use GPT-5.5, except for Hanabi, which uses
two independently stateful OppoMem agents over the same frozen memory bank.
Every game trains two batches of two episodes and evaluates 12 test seeds.
See [the protocol and reproducibility notes](docs/experiment.md) for the
exact seed schedule and scope.

## Repository layout

```text
src/oppomem/
  agent.py, factory.py   ToM, retrieval, and agent assembly
  memory/               segmentation, clustering, retrieval, and reflection
  models/               model clients, response caching, encoder, and NLI
  prompts/              prompts and action output contracts
  games/                IPD, RPS, and Hanabi adapters and self-play
  deliberation/         negotiation, scoring, and memory review
  runtime/              episode and batch orchestration, storage, and audits
  assets/               bundled Deliberation scenario and prompt template
  pilot.py              model roles and the four-game live CLI
configs/pilot/          game settings and endpoint roles
tests/                  offline regression tests and optional native-engine checks
docs/                   experiment notes and source provenance
```

Repeated runs with the same configuration and output directory resume from
verified artifacts. Use a fresh output directory after changing a model,
configuration, or source file. Live runs use paid APIs and write local
episode records and response caches under the chosen output directory.

The project uses the existing [MIT license](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for the bundled scenario.
