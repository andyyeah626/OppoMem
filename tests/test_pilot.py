"""Exercise portable pilot settings and the reflector-only routing boundary."""

from pathlib import Path

from oppomem.deliberation.protocol import Game
from oppomem.pilot import GAMES, ReflectionRouter, load_config


def test_pilot_profiles_resolve_roles_and_bundled_scenario(monkeypatch):
    monkeypatch.setenv("OPPONENT_BASE_URL", "https://opponent.example/v1")
    monkeypatch.setenv("REFLECTION_BASE_URL", "https://reflector.example/v1")
    configs = Path(__file__).resolve().parents[1] / "configs" / "pilot"
    for game in GAMES:
        cfg = load_config(configs / f"{game}.json", "4o-mini")
        assert cfg.game == game
        assert (cfg.batches, cfg.episodes_per_batch, cfg.test_episodes, cfg.seed) == (2, 2, 12, 202611070)
        assert (cfg.models.model, cfg.models.reasoning_effort) == ("deepseek-flash", "low")
        assert cfg.reflection_model == "gpt-4o-mini-2024-07-18"
        assert cfg.training_participant.model == "qwen3.6-flash"
        if game == "hanabi":
            assert cfg.test_participant is None
            assert cfg.opponents == ("ds-oppomem-selfplay",)
        else:
            assert cfg.test_participant.model == "gpt-5.5"
    cfg = load_config(configs / "deliberation.json", "ds")
    scenario = Game(Path(cfg.game_dir), prompt_source=cfg.prompt_source)
    assert cfg.reflection_model == "deepseek-flash"
    assert len(scenario.agent_names) == 6
    assert len(scenario.speaker_order(cfg.seed, cfg.horizon)) == 50
    assert all(scenario.initial_prompt(name) for name in scenario.agent_names)


def test_only_reflection_stages_use_reflector(tmp_path):
    class Stub:
        reasoning_effort = None

        def __init__(self, model):
            self.model = model

        def complete_json(self, **kwargs):
            return {"model": self.model}

    client = ReflectionRouter(Stub("backbone"), tmp_path, "run",
                              lambda: Stub("reflector")).scoped("episode/0/focal")
    for stage in ("decision", "tom", "memory_windows", "deliberation_action"):
        assert client.complete_json(stage=stage) == {"model": "backbone"}
    for stage in ("memory_reflection", "deliberation_memory_review"):
        assert client.complete_json(stage=stage) == {"model": "reflector"}
