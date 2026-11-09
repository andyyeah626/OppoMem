"""Deliberation-specific memory hygiene and situation features.

Concrete packages remain in trajectories and decision prompts.  Only the text that
can survive in a centroid bank is persisted, so a bank cannot turn into a lookup
table for one fixed deal while the focal agent still reasons from the real history
(same contract as core/rps_memory.py).

Everything here is built from the public window and the focal agent's own utility
table, which already appears in that agent's initial prompt.  Opponent utility
tables are evaluator state and must never reach a retrieval query, an eligibility
test, or a memory entry: `own_table` is the only score accessor and it refuses to
return another party's numbers.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from oppomem.deliberation.protocol import parse_response


# Public refusal vocabulary.  These are the words a party uses when it says a
# package is not good enough for it; no private threshold is consulted.
REFUSAL_MARKERS = (
    "cannot", "can not", "can't", "unacceptable", "reject", "refuses", "refuse",
    "oppose", "opposed", "below my minimum", "below our minimum", "not acceptable",
    "no agreement", "veto", "will not accept", "not willing", "impossible",
)
_REFUSAL_RE = re.compile("|".join(re.escape(marker) for marker in REFUSAL_MARKERS), re.IGNORECASE)
# Entry preconditions.  Each predicate sees only the public state below.
_PRECONDITIONS = {
    "closing_proposal_discipline": lambda s: s["phase"] == "final",
    "own_minimum_guard": lambda s: s["lead"] is not None and s["focal_meets_min"] is False,
    "hold_value_term_under_pressure": lambda s: s["lead"] is not None and s["value_pressure"],
    # Fires when the table is uncontested: that is exactly when a self-serving
    # upgrade of the focal's own terms can silently cost a party and the bonus.
    "all_party_bonus_check": lambda s: (
        s["phase"] != "opening" and s["lead"] is not None
        and not s["lead_dissent"] and not s["lead_refusers"] and len(s["lead_support"]) >= 2),
    "conditional_trade_verification": lambda s: s["phase"] == "discussion" and bool(s["lead_dissent"]),
}


def own_table(game, focal: str) -> tuple[dict[str, list[int]], int]:
    """Return only the focal agent's own private table and its minimum.

    The focal agent already receives exactly this in its initial prompt, so using
    it is public with respect to that agent.  No other party's table is exposed.
    """
    table = game._scores[focal]
    values = {issue: list(rows) for issue, rows in table.items() if issue != "min"}
    return values, int(table["min"])


def own_score(table: Mapping[str, Sequence[int]], package: Mapping[str, str]) -> int:
    """The focal agent's own total for one package."""
    return sum(table[issue][int(option[1:]) - 1] for issue, option in package.items())


def public_packages(text: Any, option_counts: Mapping[str, int]) -> list[dict[str, str]]:
    """Every complete legal package an utterance publicly contains."""
    try:
        parsed = parse_response(f"<ANSWER>{text}</ANSWER>", "discussion", option_counts)
    except (TypeError, ValueError):
        return []
    return [{deal[index][0]: deal[index] for index in range(len(deal))}
            for deal in parsed["deals"]]


def _latest_by_agent(turns: Iterable[Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for turn in turns or []:
        agent = turn.get("agent")
        if isinstance(agent, str):
            latest[agent] = turn
    return latest


def build_state(game, turns: Sequence[Any], focal: str, *, phase: str,
                turn_index: int | None = None) -> dict[str, Any]:
    """Summarize the focal agent's public decision situation for retrieval.

    `turns` is the visible window the focal agent is about to answer from, so the
    state never contains anything the agent cannot already see.
    """
    window = list(turns or [])
    counts = game.option_counts
    latest = _latest_by_agent(window)

    # The leading package is the one the most parties currently stand on, not the
    # one mentioned last: a single late objector must not redefine the table.
    held: dict[tuple, list[Any]] = {}
    for index, turn in enumerate(window):
        packages = public_packages(turn.get("public_answer"), counts)
        if packages:
            held.setdefault(tuple(sorted(packages[-1].items())), []).append((index, turn.get("agent")))
    lead, support, dissent, refusers = None, [], [], []
    if held:
        lead_key = max(held, key=lambda key: (len(held[key]), max(row[0] for row in held[key])))
        lead = dict(lead_key)
        holders = {row[1] for row in held[lead_key]}
        for agent, turn in latest.items():
            if agent == focal:
                continue
            if agent in holders:
                support.append(agent)
            else:
                dissent.append(agent)
    for agent, turn in latest.items():
        if agent == focal or agent in support or agent in dissent:
            continue
        if _REFUSAL_RE.search(str(turn.get("public_answer") or "")):
            refusers.append(agent)

    table, minimum = own_table(game, focal)
    self_score = own_score(table, lead) if lead else None
    meets = None if self_score is None else self_score >= minimum

    pressure, value_issue = False, None
    if lead is not None:
        gains = {}
        for agent in dissent:
            for package in public_packages(latest[agent].get("public_answer"), counts):
                if own_score(table, package) >= self_score:
                    continue
                for issue in counts:
                    loss = table[issue][int(lead[issue][1:]) - 1] - table[issue][int(package[issue][1:]) - 1]
                    if loss > gains.get(issue, 0):
                        gains[issue] = loss
        if gains:
            value_issue = max(gains, key=lambda issue: (gains[issue], issue))
            pressure = gains[value_issue] > 0

    return {
        "phase": phase,
        "turn_index": turn_index if turn_index is not None else len(window),
        "completed": len(window),
        "lead": "".join(lead[issue] for issue in counts) if lead else None,
        "lead_support": sorted(support),
        "lead_dissent": sorted(dissent),
        "lead_refusers": sorted(refusers),
        "value_issue": value_issue,
        "value_pressure": bool(pressure),
        "focal_self_score": self_score,
        "focal_min": minimum,
        "focal_meets_min": meets,
    }


def situation_query(state: Mapping[str, Any]) -> str:
    """Build a transient, situation-dependent retrieval suffix.

    Intentionally never stored in a cluster, and intentionally free of concrete
    options so it stays comparable to option-free bank text.
    """
    if not state.get("lead"):
        agreement = "no complete package on the table"
    elif state.get("lead_dissent") or state.get("lead_refusers"):
        agreement = "a package is on the table with open objections"
    else:
        agreement = "a package is on the table with no open objection"
    meets = state.get("focal_meets_min")
    return (
        f"phase={state.get('phase')}; completed={state.get('completed')}; {agreement}; "
        f"supporters={len(state.get('lead_support') or [])}; "
        f"dissenters={len(state.get('lead_dissent') or [])}; "
        f"refusers={len(state.get('lead_refusers') or [])}; "
        f"own_total_against_own_minimum="
        f"{'below' if meets is False else 'at_or_above' if meets else 'unknown'}; "
        f"value_term_under_pressure={'yes' if state.get('value_pressure') else 'no'}"
    )


def eligible(cluster_id: str, state: Mapping[str, Any]) -> bool:
    """Whether an entry's stated precondition holds in this situation.

    Unknown ids stay eligible so a bank that predates the gate keeps working and
    the gate can never silently disable memory.
    """
    check = _PRECONDITIONS.get(cluster_id)
    return True if check is None else bool(check(state))


# Default precedence among entries whose precondition holds at the same turn,
# most consequential first.  Ordered by the verified cost of getting it wrong:
# a closing answer with no deal block or below the focal's own minimum scores
# zero, which dominates losing the value term or the all-party reward.
PRIORITY = ("closing_proposal_discipline", "own_minimum_guard", "hold_value_term_under_pressure",
            "all_party_bonus_check", "conditional_trade_verification")


def _row(cluster, score) -> dict:
    return {key: cluster[key] for key in
            ("cluster_id", "opponent_strategy", "response_strategy")} | {"score": float(score)}


def select(bank, similarities: Mapping[str, float], state: Mapping[str, Any], threshold: float,
           *, final_entry: str | None = None, final_override: bool = False,
           priority: Sequence[str] | None = PRIORITY) -> dict | None:
    """The one entry to inject, or none.

    Cosine ranking alone was measured non-discriminative on this task: one entry
    held the top similarity at every mid-game turn, so it ranked by how closely its
    wording matched the query template rather than by whether its situation applied.
    Preconditions therefore carry the selection, `priority` breaks ties between
    entries that are simultaneously eligible, and the threshold stays as a floor
    that cancels injection when nothing is close enough to be worth saying.

    The whole episode is decided by the focal agent's single closing proposal, so
    with `final_override` the `final_entry` is used whenever it is eligible at that
    turn.  Its advice governs the shape of the final answer, not which package wins,
    so overriding similarity there cannot smuggle in a particular deal.

    `final_override` deliberately bypasses the threshold as well as the ranking: the
    floor exists to suppress advice that is not close enough to be worth saying, and
    at the closing turn this entry is always worth saying.  Measured on the six
    development episodes, every one of them took the entry at the final turn while
    the floor alone would have delivered none of them.  Eligibility still applies --
    the override cannot fire outside the closing phase.
    """
    by_id = {cluster["cluster_id"]: cluster for cluster in bank["clusters"]}
    if (final_override and final_entry is not None and state.get("phase") == "final"
            and final_entry in by_id and eligible(final_entry, state)):
        return _row(by_id[final_entry], similarities.get(final_entry, 0.0))
    order = [ident for ident in (priority or ()) if ident in by_id]
    order += [ident for ident in sorted(similarities, key=lambda k: (-similarities[k], k))
              if ident not in order]
    for ident in order:
        score = similarities.get(ident, 0.0)
        if score < threshold or not eligible(ident, state):
            continue
        return _row(by_id[ident], score)
    return None


__all__ = [
    "PRIORITY",
    "REFUSAL_MARKERS",
    "build_state",
    "eligible",
    "own_score",
    "own_table",
    "public_packages",
    "select",
    "situation_query",
]
