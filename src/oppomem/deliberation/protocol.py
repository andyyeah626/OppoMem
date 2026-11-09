"""Provider-independent LLM-Deliberation protocol and evaluator.

Only public utterances and the speaking agent's own previous plan enter a turn.
Opponent utility tables are evaluator state; callers must not include evaluator
outputs such as collective utility in memory-reflection inputs.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_BUNDLED_SOURCE = Path(__file__).resolve().parents[1] / "assets" / "deliberation"
UPSTREAM_ROOT = Path(os.environ.get("OPPOMEM_DELIBERATION_SOURCE", str(_BUNDLED_SOURCE)))
DEFAULT_GAME_DIR = UPSTREAM_ROOT / "games_descriptions" / "base"
BASE_OPTION_COUNTS = {"A": 3, "B": 3, "C": 4, "D": 4, "E": 5}
SCORING_VERSION = "textual_base_threshold_ge_bonus_after_votes_v1"
LEGACY_SCORING_VERSION = "upstream_evaluate_deals_notebook_v1"
_PHASES = {"opening", "discussion", "final"}
_PRIVATE_TAG = re.compile(r"<(/?)(SCRATCHPAD|PLAN)\s*>", re.IGNORECASE)
_DEAL_BLOCK = re.compile(r"<DEAL\s*>(.*?)</DEAL\s*>", re.IGNORECASE | re.DOTALL)


def _strip_private(text: str) -> tuple[str, bool]:
    """Remove even nested/unclosed private blocks without publishing contents."""
    pieces: list[str] = []
    stack: list[str] = []
    cursor = 0
    malformed = False
    for token in _PRIVATE_TAG.finditer(text):
        if not stack:
            pieces.append(text[cursor : token.start()])
        closing, tag = token.group(1), token.group(2).upper()
        if closing:
            if not stack or stack[-1] != tag:
                malformed = True
            else:
                stack.pop()
        else:
            stack.append(tag)
        cursor = token.end()
    if not stack:
        pieces.append(text[cursor:])
    cleaned = "".join(pieces)
    if re.search(r"</?(?:SCRATCHPAD|PLAN)\b", cleaned, re.IGNORECASE):
        malformed = True
    return ("" if malformed else cleaned), malformed


def _option_counts(value: Mapping[str, int] | Sequence[int] | None) -> dict[str, int]:
    if value is None:
        return dict(BASE_OPTION_COUNTS)
    if isinstance(value, Mapping):
        counts = {str(k).upper(): int(v) for k, v in value.items()}
    else:
        counts = {chr(65 + i): int(v) for i, v in enumerate(value)}
    if not counts or any(not re.fullmatch(r"[A-Z]", k) or v < 1 for k, v in counts.items()):
        raise ValueError("option_counts must contain positive counts keyed by issue letters")
    return dict(sorted(counts.items()))


def _parse_deals(public: str, counts: dict[str, int]) -> tuple[list[list[str]], list[dict[str, str]]]:
    deals: list[list[str]] = []
    invalid: list[dict[str, str]] = []
    for match in _DEAL_BLOCK.finditer(public):
        body = match.group(1).strip()
        tokens = [s.upper() for s in re.split(r"[\s,;]+", body) if s]
        choices: dict[str, int] = {}
        reason = ""
        for token in tokens:
            parsed = re.fullmatch(r"([A-Z])([1-9][0-9]*)", token)
            if not parsed:
                reason = "non_option_text"
                break
            issue, number = parsed.group(1), int(parsed.group(2))
            if issue not in counts or number > counts[issue]:
                reason = "unknown_or_out_of_range_option"
                break
            if issue in choices:
                reason = "duplicate_issue"
                break
            choices[issue] = number
        if not reason and set(choices) != set(counts):
            reason = "incomplete_deal"
        if reason:
            invalid.append({"text": body, "reason": reason})
        else:
            deals.append([f"{issue}{choices[issue]}" for issue in counts])
    # A matched block nested inside an unmatched block is still ambiguous.
    outside = _DEAL_BLOCK.sub("", public)
    if re.search(r"</?DEAL\b", outside, re.IGNORECASE):
        invalid.append({"text": "", "reason": "unbalanced_deal_tags"})
    return deals, invalid


def parse_response(
    raw: str,
    phase: str,
    option_counts: Mapping[str, int] | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Separate private/public text and validate deals without inventing repairs.

    ``valid_response`` distinguishes an unusable response envelope from a legal
    response whose proposal fails. A semantically invalid final deal is the
    latter: callers should score it as a failed negotiation, not retry it.
    The upstream opening prompt does not require an ANSWER wrapper.
    """
    if phase not in _PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    counts = _option_counts(option_counts)
    plan_matches = re.findall(r"<PLAN\s*>(.*?)</PLAN\s*>", raw, re.IGNORECASE | re.DOTALL)
    plan = plan_matches[-1].strip() if plan_matches else ""
    cleaned, malformed_private = _strip_private(raw)
    errors: list[str] = []
    public = ""
    if malformed_private:
        errors.append("malformed_private_tags")
    answer_tokens = list(re.finditer(r"</?ANSWER\s*>", cleaned, re.IGNORECASE))
    answer = re.fullmatch(
        r"\s*(.*?)<ANSWER\s*>(.*?)</ANSWER\s*>(.*?)\s*",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if len(answer_tokens) == 2 and answer is not None:
        public = answer.group(2).strip()
    elif phase == "opening" and not re.search(r"</?ANSWER\b", cleaned, re.IGNORECASE):
        public = cleaned.strip()
    else:
        errors.append("expected_one_complete_answer_block")
    if not public:
        errors.append("empty_public_answer")
    if errors:
        public = ""
    deals, invalid = _parse_deals(public, counts)
    return {
        "public_answer": public,
        "plan": plan if not errors else "",
        "deals": deals,
        "invalid_deals": invalid,
        "valid_response": not errors,
        "format_errors": errors,
        "final_valid": phase == "final" and not errors and len(deals) == 1 and not invalid,
    }


class Game:
    """One immutable game with private evaluator utilities and configured incentives."""

    def __init__(self, game_dir: Path = DEFAULT_GAME_DIR, *, use_config_incentives: bool = False,
                 prompt_source: str | Path | None = None):
        self.prompt_source = Path(prompt_source) if prompt_source else UPSTREAM_ROOT
        self.game_dir = Path(game_dir).resolve()
        self.use_config_incentives = bool(use_config_incentives)
        self.agents: list[dict[str, str]] = []
        self._scores: dict[str, dict[str, Any]] = {}
        for line in (self.game_dir / "config.txt").read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = [v.strip() for v in line.split(",")]
            if len(fields) != 5:
                raise ValueError("Every config line must have five comma-separated fields")
            name, filename, role, incentive, model = fields
            if incentive not in {"cooperative", "greedy"}:
                raise ValueError(f"Unsupported Base incentive: {incentive}")
            if any(a["name"] == name for a in self.agents):
                raise ValueError(f"Duplicate agent: {name}")
            self.agents.append({"name": name, "file": filename, "role": role,
                                "incentive": incentive, "configured_model": model})
            rows = (self.game_dir / "scores_files" / f"{filename}.txt").read_text(encoding="utf-8-sig").splitlines()
            if len(rows) < 2:
                raise ValueError(f"Missing score rows: {filename}")
            scores: dict[str, Any] = {
                chr(65 + i): [int(v.strip()) for v in row.split(",")]
                for i, row in enumerate(rows[:-1])
            }
            scores["min"] = int(rows[-1].strip())
            self._scores[name] = scores
        self.agent_names = [agent["name"] for agent in self.agents]
        if len(self.agents) < 3:
            raise ValueError("The negotiation requires at least three parties")
        p1s = [a["name"] for a in self.agents if a["role"] == "p1"]
        p2s = [a["name"] for a in self.agents if a["role"] == "p2"]
        if len(p1s) != 1 or len(p2s) != 1:
            raise ValueError("Exactly one p1 and one p2 are required")
        self.p1, self.p2 = p1s[0], p2s[0]
        self.option_counts = {
            issue: len(values) for issue, values in self._scores[self.p1].items() if issue != "min"
        }
        _option_counts(self.option_counts)
        for scores in self._scores.values():
            if {issue: len(values) for issue, values in scores.items() if issue != "min"} != self.option_counts:
                raise ValueError("All agents must have the same issue and option counts")
        self.initial_deal = (self.game_dir / "initial_deal.txt").read_text(encoding="utf-8-sig").strip()
        initial, invalid = _parse_deals(f"<DEAL>{self.initial_deal}</DEAL>", self.option_counts)
        if len(initial) != 1 or invalid:
            raise ValueError("initial_deal.txt must contain one complete legal deal")
        self.public_rules = (self.game_dir / "global_instructions.txt").read_text(encoding="utf-8-sig")
        self._initial_prompts: dict[str, str] = {}

    def _check_agent(self, name: str) -> None:
        if name not in self.agent_names:
            raise ValueError(f"Unknown agent: {name}")

    def initial_prompt(self, name: str) -> str:
        self._check_agent(name)
        if name not in self._initial_prompts:
            # This upstream module only imports os/string. Do not import agent,
            # main or utils: those require vertexai/torch even for remote APIs.
            path = self.prompt_source / "initial_prompts.py"
            spec = importlib.util.spec_from_file_location("_deliberation_initial_prompts", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load prompt template: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            agent = next(a for a in self.agents if a["name"] == name)
            prompt = module.InitialPrompt(
                str(self.game_dir), name, agent["file"], self.p1, self.p2,
                num_issues=len(self.option_counts), num_agents=len(self.agents),
                incentive=agent["incentive"] if self.use_config_incentives else "cooperative",
            ).return_initial_prompt()
            prompt += (
                "\nAcceptance is determined using each party's base deal score: "
                "a score equal to the minimum is acceptable. The unanimity bonus "
                "is awarded only after every party meets its minimum without the bonus."
            )
            self._initial_prompts[name] = prompt
        return self._initial_prompts[name]

    def speaker_order(self, seed: int, rounds_num: int = 48) -> list[str]:
        if rounds_num < 1:
            raise ValueError("rounds_num must be positive")
        rng = random.Random(seed)
        order: list[str] = []
        last = self.p1
        while len(order) < rounds_num:
            shuffled = rng.sample(self.agent_names, len(self.agents))
            while shuffled[0] == last or shuffled[-1] == self.p1:
                shuffled = rng.sample(self.agent_names, len(self.agents))
            order.extend(shuffled)
            last = shuffled[-1]
        return [self.p1, *order[:rounds_num], self.p1]

    def visible_history(self, name: str, turns: list[dict], window_size: int = 6) -> list[dict]:
        self._check_agent(name)
        if window_size < 1:
            raise ValueError("window_size must be positive")
        return [
            {"index": turn["index"], "agent": turn["agent"], "public_answer": turn["public_answer"]}
            for turn in turns[-window_size:]
        ]

    def round_prompt(
        self, name: str, turns: list[dict], phase: str, turn_index: int,
        rounds_num: int = 48, window_size: int = 6,
    ) -> str:
        self._check_agent(name)
        if phase not in _PHASES:
            raise ValueError(f"Unknown phase: {phase}")
        if phase in {"opening", "final"} and name != self.p1:
            raise ValueError("Only p1 makes the opening and final proposals")
        if phase == "opening":
            # Verbatim upstream build_first_slot; it intentionally has no ANSWER tag.
            return (
                f" The negotiation now begins. As a representative of {name}, you are now talking to the other parties. "
                f"Use two to three short sentences overall. This is round: 0. To start, propose the following deal: "
                f"{self.initial_deal}. Enclose the deal between: <DEAL> </DEAL> format. "
            )
        visible = self.visible_history(name, turns, window_size)
        history = " \n ".join(
            f". You ({t['agent']}): {t['public_answer']}" if t["agent"] == name
            else f". {t['agent']}: {t['public_answer']}" for t in visible
        )
        prompt = f"The following is a chronological history of up to {window_size} interactions <HISTORY> {history} </HISTORY> "
        last_plan = next((str(t.get("plan", "")) for t in reversed(turns) if t["agent"] == name and t.get("plan")), "")
        if last_plan:
            prompt += (
                "The following are your previous plans from last interactions. You should follow them while also "
                f"adjusting them according to new observations. <PREV_PLAN> {last_plan} </PREV_PLAN> "
            )
        prompt += "\n Now it is your turn to talk."
        # Public indices include the opening at zero; upstream discussion slots
        # start at zero separately, so its final-cycle rule uses index - 1.
        discussion_index = turn_index - 1
        final_round = rounds_num - discussion_index <= len(self.agents)
        if phase == "final":
            prompt += " This is an official and final proposal session."
        elif final_round:
            prompt += " This is the final discussion session."
        agent = next(a for a in self.agents if a["name"] == name)
        effective_incentive = agent["incentive"] if self.use_config_incentives else "cooperative"
        if effective_incentive == "cooperative":
            prompt += '''
        Please use a scratchpad to show intermediate calculations and explain yourself and why you are agreeing with a deal or suggesting a new one.
        You should map the individual options to their scores denoted by the number between parentheses.
        You have a calculator tool at your disposal, where you simply add scores of the options to determine the total score of a deal.
        In your scratchpad,
            1) think about what others may prefer,
            2) Based on others' preferences and history and your notes, propose one proposal that balances between your scores and accommodating others and that is more likely to lead to an agreement.
        You must follow these important negotiation guidelines in all your suggestions:
        Aim for a balanced agreement considering all parties' interests.
        Show flexibility and openness to accommodate others' preferences.
        Express your objectives clearly and actively listen to others.
        Empathize with other parties' concerns to foster rapport.
        Focus on common interests to create a win-win situation.
        It is very important for you that you all reach an agreement as long as your minimum score is met.
        '''
        else:
            prompt += '''
        Your goal now is to observe others' interactions and see how you can make a move that might maximize your score.
        You don't put much weight on accommodating others' preferences, and you would like to maximize your score as much as possible.
        You are allowed to use negotiation tactics such as appealing to or convincing other parties and offering reciprocal support.
        You might compromise on issues that matter least to you, but be careful not to ruin the deal because you still want a deal above your minimum threshold.
        Please use a scratchpad to explain your observations, calculate your own complete-deal scores, and form a plan.
        '''
        prompt += '''
        Enclose the scratchpad between <SCRATCHPAD> and </SCRATCHPAD>. The scratchpad is secret and not seen by other parties.
        Your final answer is public and must never contain scores. Enclose your final answer after the scratchpad between <ANSWER> and </ANSWER>.
        Make your final answer very short and brief in 2-3 sentences and containing only your main proposals.
        Use options' short notations instead of long descriptions.
        Enclose any deals you suggest between: <DEAL> </DEAL>.
        '''
        if phase == "final":
            prompt += (
                "Submit exactly one final, complete deal inside a single <DEAL>...</DEAL> block in your <ANSWER>. "
                "Include exactly one legal option for every issue. Do not offer alternative deals or request another round."
            )
        elif not final_round:
            prompt += (
                "After the final answer, building on your current move and analysis, briefly write down short notes "
                "for yourself of what exact options you can explore the next time you speak. "
                "Enclose the notes between <PLAN> and </PLAN>."
            )
        elif name == self.p1:
            prompt += (
                "After the final answer, building on your current move, briefly write down short notes for yourself "
                "of what exact options you can suggest in the next and final voting session. "
                "Enclose the notes between <PLAN> and </PLAN>."
            )
        return prompt

    def _score_deal(self, deal: list[str]) -> dict[str, Any]:
        scores = {
            name: sum(table[option[0]][int(option[1:]) - 1] for option in deal)
            for name, table in self._scores.items()
        }
        accepts = {name: value >= self._scores[name]["min"] for name, value in scores.items()}
        success = accepts[self.p1] and accepts[self.p2] and sum(accepts.values()) >= len(self.agents) - 1
        unanimous = all(accepts.values())
        return {
            "success": success, "unanimous": unanimous, "scores": scores,
            "focal_base_utility": scores[self.p1],
            "focal_realized_utility": scores[self.p1] + (10 if unanimous else 0) if success else 0,
            "collective_score": sum(scores.values()) / len(scores),
        }

    def score_episode(self, turns: list[dict]) -> dict[str, Any]:
        total = wrong = focal_total = focal_wrong = invalid_count = 0
        any_success = False
        for turn in turns:
            self._check_agent(turn["agent"])
            parsed = parse_response(f"<ANSWER>{turn['public_answer']}</ANSWER>", "discussion", self.option_counts)
            invalid_count += len(parsed["invalid_deals"])
            for deal in parsed["deals"]:
                outcome = self._score_deal(deal)
                total += 1
                is_wrong = outcome["scores"][turn["agent"]] < self._scores[turn["agent"]]["min"]
                wrong += int(is_wrong)
                if turn["agent"] == self.p1:
                    focal_total += 1
                    focal_wrong += int(is_wrong)
                    any_success |= outcome["success"]
        final_deal = None
        outcome = None
        if turns and turns[-1]["agent"] == self.p1 and turns[-1].get("phase", "final") == "final":
            parsed = parse_response(f"<ANSWER>{turns[-1]['public_answer']}</ANSWER>", "final", self.option_counts)
            if parsed["final_valid"]:
                final_deal = parsed["deals"][0]
                outcome = self._score_deal(final_deal)
        return {
            "scoring_version": SCORING_VERSION,
            "final_valid": outcome is not None,
            "final_deal": final_deal,
            "final_success": bool(outcome and outcome["success"]),
            "any_success": bool(any_success),
            "unanimous": bool(outcome and outcome["unanimous"]),
            "focal_base_utility": outcome["focal_base_utility"] if outcome else None,
            "focal_realized_utility": outcome["focal_realized_utility"] if outcome else 0,
            "collective_score": outcome["collective_score"] if outcome else 0,
            "wrong_deals": wrong / total if total else None,
            "wrong_deal_count": wrong,
            "deal_count": total,
            "focal_wrong_deals": focal_wrong / focal_total if focal_total else None,
            "focal_wrong_deal_count": focal_wrong,
            "focal_deal_count": focal_total,
            "invalid_deal_count": invalid_count,
            "legacy": self._legacy_score(turns),
        }

    def _legacy_score(self, turns: list[dict]) -> dict[str, Any]:
        """Exact notebook arithmetic/parsing, including stale-final/bonus quirks.

        The notebook crashes on some malformed outputs. Report unavailable in
        those cases rather than silently replacing its semantics with ours.
        Its external expected-length check belongs to the episode runner.
        """
        result: dict[str, Any] = {"scoring_version": LEGACY_SCORING_VERSION, "available": False}
        total = wrong = 0
        any_success = False
        current_success = current_unanimous = None
        try:
            for turn in turns:
                answer = str(turn["public_answer"]).replace("\n", "")
                options = [re.findall(f"{issue}[1-9]", answer, re.DOTALL) for issue in self.option_counts]
                if any(not matches for matches in options):
                    continue
                deal = [matches[0] for matches in options]
                scores = {
                    name: sum(table[token[0]][int(token[1]) - 1] for token in deal)
                    for name, table in self._scores.items()
                }
                accepted = {name: value > self._scores[name]["min"] for name, value in scores.items()}
                agreed = sum(accepted.values())
                veto = [accepted[self.p1], accepted[self.p2]]
                proposer = turn["agent"]
                total += 1
                if scores[proposer] < self._scores[proposer]["min"]:
                    if proposer == self.p1 and scores[proposer] + 10 >= self._scores[proposer]["min"]:
                        veto[0] = True
                    else:
                        wrong += 1
                if proposer == self.p1:
                    current_success = current_unanimous = False
                    if agreed == len(self.agents):
                        current_success = current_unanimous = True
                    elif agreed == len(self.agents) - 1 and not veto[0] and scores[self.p1] + 10 >= self._scores[self.p1]["min"]:
                        current_success = current_unanimous = True
                    elif agreed == len(self.agents) - 1 and all(veto):
                        current_success = True
                if current_success is None:
                    raise UnboundLocalError("upstream curr_deal_done is unbound")
                any_success = bool(current_success or any_success)
            if total == 0:
                raise ZeroDivisionError("upstream has no complete deals")
        except (IndexError, KeyError, ValueError, UnboundLocalError, ZeroDivisionError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        result.update({
            "available": True, "final_success": bool(current_success),
            "any_success": any_success, "unanimous": bool(current_unanimous),
            "wrong_deals": wrong / total, "wrong_deal_count": wrong, "deal_count": total,
        })
        return result
