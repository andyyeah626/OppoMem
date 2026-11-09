"""Concise instructions for dedicated opponent modelling and memory reflection.

These contracts are not part of the shared player system prompt. Opponent models
describe overall style; response strategies tell SELF what to do.
"""

SYSTEM_JSON = (
    "Return exactly one valid JSON object matching the requested fields. "
    "Do not include markdown fences, prose outside the object, NaN, or comments."
)

OPPONENT_MODEL_CONTRACT = """You are an experienced player of repeated multi-agent games.
Examine the full provided interaction history in chronological order, attending to
SELF's exact action and the opponent's exact action in every round. Keep the players
and rounds distinct. Use the complete action sequence to identify the opponent's
overall playing style and strategy. Describe its distinguishing features accurately
and concisely in English. In the final description, summarize the strategy rather
than retell individual rounds or list specific action-response rules. Acknowledge
limited evidence briefly. Describe the opponent, not advice for SELF."""

RESPONSE_CONTRACT = """Write a short, direct instruction telling SELF what action to take.
Name the action or specify an actionable way to choose among legal actions, so SELF
can execute the advice at its next decision. Use concise English and an imperative
sentence. Give the instruction itself, without restating the opponent's style,
explaining your reasoning, or giving vague advice. Use only information already
available to SELF; include a brief condition only when needed to make the action clear."""

__all__ = [
    "SYSTEM_JSON",
    "OPPONENT_MODEL_CONTRACT",
    "RESPONSE_CONTRACT",
]
