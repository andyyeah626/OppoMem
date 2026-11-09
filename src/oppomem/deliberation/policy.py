"""Reviewed memory entries and candidate policies for the deliberation focal agent.

Every entry describes a public situation and an abstract response.  No entry names
a concrete package: an entry that did would be a lookup table for one fixed deal
(see core/rps_memory.py), which is what made the M0-M3 banks actively harmful.

Each entry's cluster_id is also its gate precondition in core/deliberation_gate.py.
"""
from __future__ import annotations

from copy import deepcopy


# The retrieval key is built from ToM descriptions of the other parties, so each
# opponent_strategy is written in that same register: what the other parties are
# doing and how settled their support is.  It must still never name a package.
V4_ENTRIES = [
    {
        "cluster_id": "closing_proposal_discipline",
        "opponent_strategy": (
            "Every other party now repeats the same complete package back and raises no new "
            "objection. Their support is broad and stable, no party is pressing for a further "
            "change, and the next statement from the agent is the official closing proposal."
        ),
        "response_strategy": (
            "State exactly one complete package, one option per issue, inside a single deal block. "
            "Never answer with prose alone. Choose a package that meets your own minimum and keeps "
            "the broadest support."
        ),
    },
    {
        "cluster_id": "own_minimum_guard",
        "opponent_strategy": (
            "The other parties keep endorsing one complete package and treat it as settled, but that "
            "package is worth less to the agent than the minimum the agent can accept. No other "
            "party is proposing a substitute."
        ),
        "response_strategy": (
            "Say plainly that the current package is unacceptable for you and re-anchor on terms "
            "you can accept. Never close below your own minimum."
        ),
    },
    {
        "cluster_id": "hold_value_term_under_pressure",
        "opponent_strategy": (
            "A party that funds the package presses the agent to move on the issue the agent values "
            "most, while accepting every other issue unchanged. The pressure is single issue and the "
            "rest of the package is already agreed."
        ),
        "response_strategy": (
            "Trade that issue only for equal value elsewhere. When the rest is already settled, "
            "conceding it is not required to reach agreement."
        ),
    },
    {
        "cluster_id": "all_party_bonus_check",
        "opponent_strategy": (
            "The other parties have stopped objecting and repeat the package on the table back to "
            "each other. Support is broad and no party is pushing for a change, yet the agent can "
            "still take more by moving one issue."
        ),
        "response_strategy": (
            "Compare the private gain from that upgrade against the reward for an all party "
            "agreement and the risk that the displaced party walks away."
        ),
    },
    {
        "cluster_id": "conditional_trade_verification",
        "opponent_strategy": (
            "A party asks the agent to give up a linked issue but has not publicly committed to the "
            "rest of the package. The request is one sided while that party's support stays "
            "unstated."
        ),
        "response_strategy": (
            "Obtain that party's explicit public commitment first, or present both issues together "
            "as one conditional package."
        ),
    },
]

_DROP = {
    "v4_core": ("conditional_trade_verification", "hold_value_term_under_pressure"),
    "v4_no_trade": ("conditional_trade_verification",),
    "v4_no_hold": ("hold_value_term_under_pressure",),
}

# Shorter, more imperative wording for the same policies: the RPS "style/length
# only" axis, which tests whether retrieval is helped by terser bank text.
V4_TERSE = {
    "closing_proposal_discipline": (
        "All parties have settled on one complete package and no objection is public. Only the "
        "next statement counts.",
        "Answer with exactly one complete package in one deal block, never prose. Pick a package "
        "above your own minimum with the widest support.",
    ),
    "own_minimum_guard": (
        "The most supported package still leaves the agent under its own minimum.",
        "Declare it unacceptable and re-anchor. Never close under your own minimum.",
    ),
    "hold_value_term_under_pressure": (
        "A party with money pushes the agent off the issue it values most, conceding the rest.",
        "Trade it only for equal value. If the rest is settled, conceding is not required.",
    ),
    "all_party_bonus_check": (
        "Nobody objects to the table package and support is broad, yet the agent can still take "
        "more by moving one issue.",
        "Weigh that gain against the all party reward and the party it displaces.",
    ),
    "conditional_trade_verification": (
        "The agent is about to give up a linked issue to a party that has not committed publicly.",
        "Get its public commitment first, or present both issues as one conditional package.",
    ),
}


def entries_for(bank: str) -> list[dict]:
    """The reviewed entry list behind a bank name."""
    if bank == "v4":
        return deepcopy(V4_ENTRIES)
    if bank in _DROP:
        dropped = _DROP[bank]
        return deepcopy([e for e in V4_ENTRIES if e["cluster_id"] not in dropped])
    if bank == "v4_terse":
        return [{"cluster_id": ident, "opponent_strategy": texts[0], "response_strategy": texts[1]}
                for ident, texts in V4_TERSE.items()]
    raise ValueError(f"Unknown bank: {bank}")


def add_entry(bank: str, entry: dict) -> list[dict]:
    """A named bank plus one extra reviewed entry, for a targeted candidate."""
    return entries_for(bank) + [deepcopy(entry)]


BANK_NAMES = ("v4", "v4_core", "v4_no_trade", "v4_no_hold", "v4_terse", "legacy")

# gate: preconditions decide.  priority: "gate" uses deliberation_gate.PRIORITY to
# break ties, "similarity" uses cosine rank (which measured non-discriminative).
# query: "situation" keys retrieval on the transient state summary, "tom" on the
# concatenated ToM descriptions the old arm used.  window limits memory to the
# closing stretch; None means every turn of the episode.
CANDIDATES = [
    {"name": "gate_priority", "bank": "v4", "gate": True, "priority": "gate", "query": "situation",
     "final_override": True, "window": None},
    {"name": "gate_similarity", "bank": "v4", "gate": True, "priority": "similarity",
     "query": "situation", "final_override": True, "window": None},
    {"name": "gate_nofinal", "bank": "v4", "gate": True, "priority": "gate", "query": "situation",
     "final_override": False, "window": None},
    {"name": "tom_text_key", "bank": "v4", "gate": True, "priority": "gate", "query": "tom",
     "final_override": True, "window": None},
    {"name": "closing_only", "bank": "v4", "gate": True, "priority": "gate", "query": "situation",
     "final_override": True, "window": 18},
    {"name": "core", "bank": "v4_core", "gate": True, "priority": "gate", "query": "situation",
     "final_override": True, "window": None},
    {"name": "no_trade", "bank": "v4_no_trade", "gate": True, "priority": "gate",
     "query": "situation", "final_override": True, "window": None},
    {"name": "terse", "bank": "v4_terse", "gate": True, "priority": "gate", "query": "situation",
     "final_override": True, "window": None},
    {"name": "similarity_only", "bank": "v4", "gate": False, "priority": "similarity",
     "query": "situation", "final_override": False, "window": None},
    {"name": "legacy", "bank": "legacy", "gate": False, "priority": "similarity", "query": "tom",
     "final_override": False, "window": None},
]

FINAL_ENTRY = "closing_proposal_discipline"

__all__ = ["BANK_NAMES", "CANDIDATES", "FINAL_ENTRY", "V4_ENTRIES", "add_entry", "entries_for"]
