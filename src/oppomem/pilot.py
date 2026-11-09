"""Portable live runner for the four-game reflection pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import itertools
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .config import ExperimentConfig, MemoryConfig, ModelConfig
from .models.cache import CachedLLM
from .runtime.storage import read, write


GAMES = ("ipd", "rps", "hanabi", "deliberation")
HANABI_SELFPLAY = "ds-oppomem-selfplay"
REFLECTION_STAGES = frozenset({"memory_reflection", "deliberation_memory_review"})


@dataclass(frozen=True)
class Endpoint:
    model: str
    api_key_env: str
    base_url: str | None = None
    base_url_env: str | None = None
    reasoning_effort: str = "default"

    def resolve(self) -> Endpoint:
        url = os.environ.get(self.base_url_env, "").strip() if self.base_url_env else ""
        url = url or self.base_url
        if not url:
            raise ValueError(f"Set {self.base_url_env or 'base_url'} for {self.model}")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"Invalid endpoint URL for {self.model}")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Endpoint URLs cannot contain credentials, queries or fragments")
        return replace(self, base_url=url.rstrip("/"))

    def key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ValueError(f"Set {self.api_key_env} for {self.model}")
        return value


@dataclass(frozen=True)
class PilotConfig(ExperimentConfig):
    training_opponents: tuple[str, ...] = ()
    training_participant: Endpoint | None = None
    test_participant: Endpoint | None = None
    reflector: Endpoint | None = None

    @property
    def reflection_model(self) -> str:
        return self.reflector.model if self.reflector else self.models.model


def load_config(path: Path, reflector: str) -> PilotConfig:
    raw = read(path)
    roles = read(path.parent / "roles.json")
    if raw.get("game") not in GAMES:
        raise ValueError(f"Pilot game must be one of {', '.join(GAMES)}")
    reflectors = roles["reflectors"]
    if reflector not in reflectors:
        raise ValueError(f"Unknown reflector {reflector!r}; choose from {', '.join(reflectors)}")
    train = Endpoint(**roles["training_participant"]).resolve()
    # Hanabi evaluation is self-play, so it needs no test-opponent credential.
    test = None if raw["game"] == "hanabi" else Endpoint(**roles["test_participant"]).resolve()
    reflection = Endpoint(**reflectors[reflector]).resolve() if reflectors[reflector] else None
    memory = MemoryConfig(**(roles["memory_defaults"] | raw.pop("memory", {})))
    model = ModelConfig(**raw.pop("models"))
    if model.provider != "deepseek":
        raise ValueError("The reflection pilot requires the DeepSeek backbone configuration")
    backbone = Endpoint(model.model, model.api_key_env, model.base_url).resolve()
    cache_dir = os.environ.get("OPPOMEM_MODEL_CACHE") or model.cache_dir
    model = replace(model, base_url=backbone.base_url,
                    cache_dir=str(Path(cache_dir).expanduser().resolve()) if cache_dir else None)
    if raw["game"] == "deliberation":
        from .deliberation.protocol import DEFAULT_GAME_DIR, UPSTREAM_ROOT

        for field, default in (("game_dir", DEFAULT_GAME_DIR), ("prompt_source", UPSTREAM_ROOT)):
            raw[field] = str((path.parent / raw[field]).resolve()) if raw.get(field) else str(default.resolve())
    return PilotConfig(
        **raw, models=model, memory=memory,
        opponents=(test.model,) if test else (HANABI_SELFPLAY,),
        training_opponents=(train.model,), training_participant=train,
        test_participant=test, reflector=reflection,
    )


class ReflectionRouter(CachedLLM):
    """Keep segmentation and play on the backbone and route only review calls."""

    def __init__(self, client, directory, scope, reflector_factory=None):
        super().__init__(client, directory, scope)
        self.reflector_factory = reflector_factory

    def scoped(self, scope):
        return ReflectionRouter(self.client, self.directory, scope, self.reflector_factory)

    def _call(self, method, kwargs):
        if self.reflector_factory is not None and kwargs["stage"] in REFLECTION_STAGES:
            reflector = CachedLLM(self.reflector_factory(), self.directory, self.scope)
            return reflector._call(method, kwargs)
        return super()._call(method, kwargs)


def model_reference(name: str, cache_dir: str | None, allow_download: bool) -> str:
    """Also accept single-snapshot caches that were copied without a refs directory."""
    if cache_dir and not allow_download and not Path(name).is_dir():
        model_dir = Path(cache_dir) / ("models--" + name.replace("/", "--"))
        snapshots = model_dir / "snapshots"
        if snapshots.is_dir() and not (model_dir / "refs" / "main").is_file():
            paths = [path for path in snapshots.iterdir() if (path / "config.json").is_file()]
            if len(paths) == 1:
                return str(paths[0])
    return name


def run_pilot(config: PilotConfig, output: Path):
    from .models.openai import DeepSeekJsonLLM
    from .models.semantics import DebertaNLIScorer, MiniLMEmbedder

    backbone = Endpoint(config.models.model, config.models.api_key_env,
                        config.models.base_url, reasoning_effort=config.models.reasoning_effort)
    roles = (backbone, config.training_participant, config.test_participant, config.reflector)
    keys = {role: role.key() for role in roles if role is not None}
    serial = itertools.count()

    def client(role):
        return DeepSeekJsonLLM(
            api_key=keys[role], model=role.model, base_url=role.base_url,
            reasoning_effort=role.reasoning_effort, max_attempts=5,
            request_journal=output / "api" / f"client_{os.getpid()}_{next(serial):04d}.jsonl",
        )

    def focal_factory():
        reflector_factory = (lambda: client(config.reflector)) if config.reflector else None
        return ReflectionRouter(client(backbone), output / "api/cache", "focal", reflector_factory)

    def participant_factory(training, index, name="partner"):
        role = config.training_participant if training else config.test_participant
        return CachedLLM(client(role), output / "api/cache", f"episode/{index}/{name}")

    model = config.models
    options = {"cache_dir": model.cache_dir, "local_files_only": not model.allow_download}
    encoder = MiniLMEmbedder(model_reference(model.encoder, model.cache_dir, model.allow_download), **options)
    nli = None if config.game == "deliberation" else DebertaNLIScorer(
        model_reference(model.nli, model.cache_dir, model.allow_download), **options,
    )

    def adapter_factory(*, training, index, bank_path, agent_for, client):
        if config.game == "hanabi":
            from .games.hanabi import HanabiAdapter
            from .games.llm_hanabi import LLMHanabiPartner
            from .games.selfplay import OppoMemHanabiPartner

            if training:
                partner = LLMHanabiPartner(participant_factory(True, index))
            else:
                agent = agent_for(bank_path)
                agent.llm = focal_factory().scoped(f"episode/{index}/oppomem_partner")
                partner = OppoMemHanabiPartner(agent)
            return HanabiAdapter(partner)
        if config.game == "ipd":
            from .games.llm_ipd import LLMOpponentIPDAdapter as Adapter
        else:
            from .games.llm_rps import LLMOpponentAdapter as Adapter
        role = config.training_participant if training else config.test_participant
        return Adapter(opponent_id=role.model, llm=participant_factory(training, index), horizon=config.horizon)

    if config.game == "deliberation":
        from .deliberation.pipeline import run_experiment
        kwargs = {"participant_factory": participant_factory}
    else:
        from .runtime.experiment import run_experiment
        kwargs = {"adapter_factory": adapter_factory}
    return run_experiment(
        config, output, llm_factory=focal_factory, embedder=encoder, nli=nli,
        progress=lambda message: print(message, flush=True), **kwargs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train and evaluate one game")
    run.add_argument("--config", type=Path, required=True)
    suite = commands.add_parser("suite", help="Train and evaluate the four games")
    suite.add_argument("--configs", type=Path, required=True)
    for command in (run, suite):
        command.add_argument("--reflector", default="4o-mini", help="Name from roles.json")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_pilot(load_config(args.config, args.reflector), args.output)
    else:
        result = {}
        for game in GAMES:
            config = load_config(args.configs / f"{game}.json", args.reflector)
            if config.game != game:
                raise ValueError(f"{game}.json must configure game={game}")
            result[game] = run_pilot(config, args.output / game)
            write(args.output / "suite.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
