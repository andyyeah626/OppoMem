from __future__ import annotations

from oppomem.prompts.ipd import IPD_RULES, PLAYER_SYSTEM, TOM_SYSTEM, parse_player_action, player_prompt, structured_text_parser, wrap_tom_text
from oppomem.runtime.ipd import PlainTextIPDOpponentAdapter


def test_ipd_action_contract_is_plain_text_and_exact():
    assert parse_player_action("Cooperate") == {"action": "Cooperate"}
    assert parse_player_action("Defect") == {"action": "Defect"}
    for raw in ("FINAL_ACTION: Cooperate", '{"action":"Cooperate"}', "cooperate", ""):
        try:
            parse_player_action(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action: {raw!r}")


def test_ipd_tom_is_locally_wrapped_without_second_model_extraction():
    raw = "TARGET repeats its previous action."
    assert wrap_tom_text(raw) == {
        "opponent_strategy": raw,
        "unresolved": None,
    }
    assert structured_text_parser(PLAYER_SYSTEM) is parse_player_action
    assert structured_text_parser(TOM_SYSTEM) is wrap_tom_text


def test_ipd_prompt_has_simultaneous_rules_and_no_final_action_marker():
    prompt = player_prompt([], horizon=25)
    assert IPD_RULES in PLAYER_SYSTEM
    assert "Round 1 of 25" in prompt
    assert "FINAL_ACTION" not in prompt
    assert "Cooperate or Defect" in PLAYER_SYSTEM


def test_ipd_llm_opponent_uses_plain_action_contract():
    class FakeLLM:
        model = "fake"

        def complete_json(self, **kwargs):
            assert kwargs["system"] == PLAYER_SYSTEM
            assert "Round 1 of 3" in kwargs["prompt"]
            return {"action": "Defect"}

    adapter = PlainTextIPDOpponentAdapter(
        opponent_id="llm",
        llm=FakeLLM(),
        horizon=3,
        source="Axelrod-dev/Axelrod-dev",
    )
    adapter.opponent_id = "llm"
    adapter.reset = lambda **kwargs: None  # type: ignore[method-assign]
    adapter._opponent_view = []
    assert adapter._next_opponent_action() == ("Defect", False)


def test_ipd_llm_opponent_keeps_its_own_perspective_after_asymmetric_rounds():
    class Model:
        def complete_json(self, **kwargs):
            assert "your action = Cooperate; opponent action = Defect; your payoff = 0" in kwargs["prompt"]
            assert "your action = Defect; opponent action = Cooperate; your payoff = 5" in kwargs["prompt"]
            return {"action": "Defect"}
    adapter = PlainTextIPDOpponentAdapter(opponent_id="llm", llm=Model(), horizon=3)
    adapter._opponent_view = [
        {"round": 0, "you": "Cooperate", "them": "Defect", "payoff": 0},
        {"round": 1, "you": "Defect", "them": "Cooperate", "payoff": 5},
    ]
    assert adapter._next_opponent_action() == ("Defect", False)


def test_ipd_reversed_history_uses_non_zero_sum_payoffs():
    from oppomem.prompts.ipd import history_text
    text = history_text([
        {"round": 0, "self": "Cooperate", "opponent": "Defect", "reward": 0},
        {"round": 1, "self": "Defect", "opponent": "Cooperate", "reward": 5},
    ], opponent=True)
    assert "your action = Defect; opponent action = Cooperate; your payoff = 5" in text
    assert "your action = Cooperate; opponent action = Defect; your payoff = 0" in text
