"""A single entry point for smoke checks and configured experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ExperimentConfig


def dependencies(config, output_dir=None):
    if config.models.provider == "offline":
        from .runtime.offline import OfflineEncoder, OfflineLLM, OfflineNLI
        return lambda: OfflineLLM(config), OfflineEncoder(), OfflineNLI()
    key = os.environ.get(config.models.api_key_env, "").strip()
    if not key:
        raise ValueError(f"Set the {config.models.api_key_env} environment variable")
    from .models.openai import DeepSeekJsonLLM, OpenAIJsonLLM
    from .models.cache import CachedLLM
    from .models.semantics import DebertaNLIScorer, MiniLMEmbedder
    cls = DeepSeekJsonLLM if config.models.provider == "deepseek" else OpenAIJsonLLM
    client_index = 0
    def factory():
        nonlocal client_index
        journal = (Path(output_dir) / "api" / f"client_{client_index:03d}.jsonl"
                   if output_dir is not None else None)
        client_index += 1
        client = cls(api_key=key, model=config.models.model,
                   base_url=config.models.base_url,
                   reasoning_effort=config.models.reasoning_effort,
                   request_journal=journal)
        return CachedLLM(client, Path(output_dir) / "api/cache",
                         read_only_cache=config.models.read_only_response_cache) if output_dir else client
    options = {"cache_dir": config.models.cache_dir,
               "local_files_only": not config.models.allow_download}
    return (factory, MiniLMEmbedder(config.models.encoder, **options),
            None if config.game == "deliberation" else DebertaNLIScorer(config.models.nli, **options))


def run_one(config, output):
    factory, encoder, nli = dependencies(config, output)
    if config.game == "deliberation":
        from .deliberation.pipeline import run_experiment
    else:
        from .runtime.experiment import run_experiment
    return run_experiment(config, output, llm_factory=factory, embedder=encoder, nli=nli,
                          progress=lambda message: print(message, flush=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Offline complete train/update/test wiring check")
    smoke.add_argument("--output", type=Path, default=Path("runs/smoke"))
    run = commands.add_parser("run", help="Run or resume a JSON-configured experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    suite = commands.add_parser("suite", help="Run/resume the four configured game checks")
    suite.add_argument("--configs", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "suite":
        from .runtime.storage import write
        results = {}
        for game in ("rps", "ipd", "hanabi", "deliberation"):
            print(f"Starting {game}", flush=True)
            try:
                config = ExperimentConfig.load(args.configs / f"{game}.json")
                if config.game != game:
                    raise ValueError(f"{game}.json must configure game={game}")
                results[game] = {"status": "complete", **run_one(config, args.output / game)}
            except Exception as exc:
                results[game] = {"status": "failed", "error_type": type(exc).__name__}
                print(f"{game} failed: {type(exc).__name__}: {exc}", flush=True)
            write(args.output / "suite.json", results)
        print(json.dumps(results, indent=2))
        return int(any(r["status"] != "complete" for r in results.values()))
    config = ExperimentConfig() if args.command == "smoke" else ExperimentConfig.load(args.config)
    result = run_one(config, args.output)
    print(json.dumps(result, indent=2))
    return 0
