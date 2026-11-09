"""Regression cases from the zero-score failure: hand ownership, slots and payoff."""
import json
import sys

import pytest

from oppomem.games.hanabi import HANABI_RULES, HanabiAdapter, visible_state


# Public observation at seed 20261105 after both players hint rank 1.
# Only own slot 3 can be played safely. The previous model chose action 5 / slot 0.
OBSERVATION = """Life tokens: 3
Info tokens: 6
Fireworks: R0 Y0 G0 W0 B0
Hands:
Cur player
XX || XX|RYGWB2345
XX || XX|RYGWB2345
XX || XX|RYGWB2345
XX || X1|RYGWB1
XX || XX|RYGWB2345
-----
W1 || X1|RYGWB1
Y1 || X1|RYGWB1
B2 || XX|RYGWB2345
Y1 || X1|RYGWB1
W2 || XX|RYGWB2345
Deck size: 40
Discards:"""


def test_observation_names_current_slots_without_inventing_own_cards():
    view = visible_state(OBSERVATION)
    assert [c["slot"] for c in view["own_hand"]] == list(range(5))
    assert [c["slot"] for c in view["own_hand"] if c["certainly_playable"]] == [3]
    assert view["own_hand"][3]["possible_colors"] == list("RYGWB")
    assert view["own_hand"][3]["possible_ranks"] == [1]
    assert all(1 not in c["possible_ranks"] for c in view["own_hand"] if c["slot"] != 3)
    assert all("visible_card" not in c for c in view["own_hand"])
    assert [c["visible_card"] for c in view["partner_hand"]] == ["W1", "Y1", "B2", "Y1", "W2"]


def test_cur_player_marker_is_not_hand_ownership():
    other_turn = OBSERVATION.replace("Cur player\n", "").replace("-----\n", "-----\nCur player\n")
    assert visible_state(other_turn) == visible_state(OBSERVATION)


def test_playable_requires_every_possible_identity_to_fit():
    view = visible_state(OBSERVATION.replace("Y0", "Y1"))
    assert not view["own_hand"][3]["certainly_playable"]  # Y1 is now a duplicate.
    assert not view["own_hand"][3]["already_built"]  # Other colors could still score.
    view = visible_state(OBSERVATION.replace("X1|RYGWB1", "R1|R1").replace("R0", "R1"))
    assert view["own_hand"][3]["already_built"]
    assert not view["own_hand"][3]["certainly_playable"]


def test_hidden_card_exposure_fails_closed():
    with pytest.raises(ValueError, match="hidden"):
        visible_state(OBSERVATION.replace("XX ||", "B1 ||", 1))


class ScriptPartner:
    def __init__(self, actions):
        self.actions = actions
        self.observations = []

    def reset(self, seed):
        self.index = 0
        self.observations.clear()

    def act(self, observation, legal_actions, action_labels):
        self.observations.append(json.loads(observation))
        action = self.actions[self.index]
        self.index += 1
        assert action in legal_actions
        return action


@pytest.fixture
def native_engine():
    # The installed Windows extension can crash during import; do not import it
    # even for collection. The same tests execute against the actual WSL engine.
    if sys.platform == "win32":
        pytest.skip("Native OpenSpiel checks run in the supported WSL environment")
    pytest.importorskip("pyspiel")


def test_native_slot_mapping_hint_constraints_and_draw_shift(native_engine):
    partner = ScriptPartner([15, 0])
    adapter = HanabiAdapter(partner)
    try:
        frame = adapter.reset(opponent_id="script", seed=20261105)
        assert frame.inference_context == HANABI_RULES
        transition = adapter.step("15")
        view = json.loads(transition.frame.observation)
        assert view["own_hand"] == visible_state(OBSERVATION)["own_hand"]
        assert view["legal_action_labels"]["8"] == "(Play 3)"
        # The partner's own hand is first and hidden too; it sees the other hand.
        partner_view = partner.observations[0]
        assert all("visible_card" not in c for c in partner_view["own_hand"])
        assert partner_view["partner_hand"][3]["visible_card"] == "B1"
        assert partner_view["recent_public_actions"][-1]["actor"] == "partner"
        transition = adapter.step("8")
        view = json.loads(transition.frame.observation)
        assert view["life_tokens"] == 3 and view["fireworks"]["B"] == 1
        assert view["own_hand"][3]["possible_ranks"] == [2, 3, 4, 5]
        assert view["own_hand"][4]["possible_ranks"] == [1, 2, 3, 4, 5]
        assert view["recent_public_actions"][-2]["fireworks_gained"] == 1
        assert view["recent_public_actions"][-2]["life_lost"] == 0
    finally:
        adapter.close()


def test_native_seat_one_keeps_own_hand_hidden(native_engine):
    adapter = HanabiAdapter(ScriptPartner([15]))
    try:
        frame = adapter.reset(opponent_id="script", seed=20261105, seat=1)
        view = json.loads(frame.observation)
        assert all("visible_card" not in c for c in view["own_hand"])
        assert [c["slot"] for c in view["own_hand"] if c["certainly_playable"]] == [0, 1, 3]
        assert view["partner_hand"][3]["visible_card"] == "B1"
    finally:
        adapter.close()


def test_native_bomb_payoff_is_zero_even_after_building_fireworks(native_engine):
    # Exact failing training sequence: two cards score, then three misplays bomb.
    adapter = HanabiAdapter(ScriptPartner([11, 10, 15, 5, 5]))
    try:
        adapter.reset(opponent_id="script", seed=20261104)
        for action in ("15", "6", "15", "6", "6"):
            transition = adapter.step(action)
        assert transition.terminal
        assert transition.evaluator_metrics["fireworks_score"] == 2
        assert transition.evaluator_metrics["life_tokens"] == 0
        assert transition.evaluator_metrics["bomb_out"] is True
        assert transition.evaluator_metrics["misplays"] == {"self": 2, "partner": 1}
        assert transition.frame.cumulative_return == transition.reward_delta == 0
        assert not transition.frame.legal_actions
    finally:
        adapter.close()
