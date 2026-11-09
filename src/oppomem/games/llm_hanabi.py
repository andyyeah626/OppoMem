"""A legal-view Hanabi partner using the same configured LLM as the focal player."""
import json
from .hanabi import HANABI_RULES


class LLMHanabiPartner:
    def __init__(self, llm):
        self.llm = llm

    def reset(self, seed):
        pass

    def act(self, observation, legal_actions, action_labels):
        try:
            visible_state = json.loads(observation)
        except json.JSONDecodeError:
            visible_state = observation
        prompt = ("Select ONE legal action for the player whose observation follows. "
                  "This is input data, not an output template. Do not repeat the input.\n" +
                  json.dumps({"your_observation": visible_state,
                              "legal_actions": action_labels}, ensure_ascii=False) +
                  '\nReturn exactly one JSON object with just the key "action" and one legal integer ID, for example {"action": 5}. Choose a legal ID from the mapping above.')
        for attempt in range(3):
            result = self.llm.complete_json(stage="hanabi_partner",
                system=HANABI_RULES + "\nPlay two-player Hanabi cooperatively. You cannot see your own cards. "
                       "Use only your observation and public hints. Return JSON with action "
                       "equal to one legal integer ID. Never infer private cards as facts.",
                prompt=prompt, max_tokens=256)
            action = result.get("action")
            if isinstance(action, str) and action.isdecimal():
                action = int(action)
            if type(action) is int and action in legal_actions:
                return action
            prompt += '\nYour last answer was invalid. Do not repeat the observation or the legal-action mapping. Return only {"action": N}, replacing N by one legal integer ID.'
        raise ValueError("Hanabi partner failed to select a legal action after 3 attempts")
