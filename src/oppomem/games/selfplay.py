"""Hanabi partner backed by an independent agent over the same frozen bank."""

import json

from .hanabi import HANABI_RULES


class OppoMemHanabiPartner:
    def __init__(self, agent):
        self.agent = agent
        self.pending = None

    def reset(self, seed):
        self.agent.begin_episode()
        self.pending = None

    def act(self, observation, legal_actions, action_labels):
        if self.pending is not None:
            state = json.loads(observation)
            events = [row for row in state["recent_public_actions"] if row["actor"] == "partner"]
            self.agent.record_outcome(self.pending, json.dumps(events[-1:]), 0.0)
        legal = [str(action) for action in legal_actions]
        self.pending = self.agent.decide(
            observation, legal, action_validator=lambda action: action in legal,
            inference_context=HANABI_RULES, history_limit=None, allow_injection=True,
        )
        return int(self.pending.decision.action)
