"""Extracted prompts responsibilities of the working memory implementation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from oppomem.prompts.history import completed_joint_actions
from oppomem.prompts.contracts import OPPONENT_MODEL_CONTRACT, RESPONSE_CONTRACT, SYSTEM_JSON
from oppomem.models.semantics import cosine, encode_documents, encode_query
from oppomem.types import CentroidCluster, CentroidMatch, EpisodeTrace, UNKNOWN_OPPONENT_STRATEGY
from oppomem.memory.rps import abstract_opponent_strategy, abstract_response_strategy, sanitize_cluster_texts



def window_prompt(trace: EpisodeTrace, *, fixed_policy: bool = False,
                  abstract_actions: bool = False) -> str:
    """Group concise opponent-style hypotheses using the full ordered trajectory."""
    game_context = next(
        (turn.inference_context for turn in trace.turns if turn.inference_context), ""
    )
    joint = completed_joint_actions(trace.turns)
    payload = {
        "GAME_CONTEXT": game_context,
        "ONLINE_TOM_CHAIN": [{
            "round": state.round_index,
            "opponent_strategy": state.opponent_strategy,
        } for state in trace.tom_chain],
        # Round-aligned so a hypothesis can be read against the actions that
        # provoked it.  reward is -1/0/+1 from SELF's perspective.
        "COMPLETED_JOINT_ACTIONS": [
            {"round": row["round"], "self_action": row["self_action"],
             "opponent_action": row["other_action"], "reward": row["reward"]}
            for row in joint
        ],
    }
    protocol = ('''This is a fixed-policy protocol. Return either an empty windows array or
exactly one window. The program assigns its complete-episode range; do not output
round_ranges. Different wordings of a stable hypothesis do not indicate a policy change.
''' if fixed_policy else '''Use coarse semantic windows. A window may contain multiple
inclusive ranges. Windows may overlap, and uncertain or uninformative rounds may be
omitted. Prefer a small number of stable strategy descriptions rather than one
window per wording change.
''')
    action_rule = ("For the final opponent_strategy, do not name Rock, Paper, or "
                   "Scissors and do not list a concrete throw sequence. Abstract the "
                   "behavior as a fixed single-action preference, delayed countering, "
                   "switching, repetition, or a multi-action cycle.\n\n"
                   if abstract_actions else "")
    return '''Read the complete joint action sequence in chronological order, examining
SELF's and the opponent's exact actions in every round without mixing players or
rounds. Use it to check and group the descriptions in ONLINE_TOM_CHAIN
by the opponent's overall playing style and strategy, not wording changes.
Write a precise, concise opponent_strategy for each supported group. Preserve what
distinguishes the strategy in the final description rather than retelling individual
rounds or listing specific action-response rules. Omit unsupported descriptions and
insufficient-evidence regions.

''' + action_rule + protocol + '''If no useful opponent policy is represented, return an empty windows array.

Return exactly one JSON object with a windows array. Each window must contain exactly
''' + ('''opponent_strategy (string).
''' if fixed_policy else '''opponent_strategy (string) and
round_ranges (inclusive integer [start,end] arrays, for example [[2,7],[9,10]]).
''') + '''Each opponent_strategy must be 1-2 concise English sentences under 60 words. Do not include a response
strategy, reasoning, alternatives, confidence, labels, evidence, or extra keys.
DATA:\n''' + json.dumps(payload, ensure_ascii=False)

def reflection_prompt(trace: EpisodeTrace, window: dict,
                      canonical_opponent_strategy: str,
                      previous_response_strategy: str | None = None,
                      abstract_actions: bool = False) -> str:
    """Reflect only over trajectory rows assigned to one semantic ToM segment."""
    selected = set(window["rounds"])
    turns = [turn for turn in trace.turns if turn.round_index in selected]
    game_context = next(
        (turn.inference_context for turn in trace.turns if turn.inference_context), ""
    )
    payload = {
        "GAME_CONTEXT": game_context,
        "CLUSTERED_OPPONENT_STRATEGY": canonical_opponent_strategy,
        "PRIOR_RESPONSE_STRATEGY": previous_response_strategy,
        "CORRESPONDING_TRAJECTORY": [{
            **row,
            "response_intent": turn.response_intent,
        } for row, turn in zip(completed_joint_actions(turns), turns)],
    }
    action_rule = ("For RPS, prescribe an executable action-selection method using history, "
                   "without any English or Chinese concrete throw names. Do not merely say "
                   "to play a counter action; explain how to infer the target action.\n\n"
                   if abstract_actions else "")
    return action_rule + '''Use the assigned trajectory and its rewards to write a clear action instruction
for SELF against this opponent. Check the opponent description against the actual
play. Keep useful prior advice and revise it where the evidence supports a better
action. Base the instruction on what SELF can do at its next decision.
Return exactly {"response_strategy":"one concise English action instruction under 40 words"}.
Do not include explanations, opponent descriptions, internal metadata, or extra keys.
DATA:\n''' + json.dumps(payload, ensure_ascii=False)

def batch_reflection_prompt(*, canonical_opponent_strategy: str,
                            previous_response_strategy: str | None,
                            support_segments: int,
                            evidence: list[dict[str, Any]],
                            guidance_shown: list[str],
                            injection_contrast: dict[str, Any],
                            abstract_actions: bool = False,
                            reflection_phase: str = "bootstrap",
                            retrieved_evidence: list[dict[str, Any]] | None = None,
                            full_trajectories: Any = None) -> str:
    """Revise one cluster's response strategy from a whole batch of evidence.

    Two things distinguish this from `reflection_prompt`, which reflects over a single
    episode's single segment:

    * The evidence is pooled across the batch's episodes, already anonymized to
      `episode_NN`, so the model sees the policy repeated against several opponent
      realizations instead of one.
    * Every row carries `memory_state`.  Because the run withholds a retrieved memory
      on a random half of the rounds, the batch contains rounds where SELF was shown
      guidance and rounds where it was not, under the same frozen bank.  That is
      direct evidence about whether the stored guidance helps, which a single-episode
      reflection never had, and it is why the prompt is allowed to ask for a revision
      in the direction the contrast implies.

    The contrast is stated as an association, not an effect: injecting at one round
    changes what the opponent sees at the next, so the two groups are not independent.
    The prompt says so, to keep the model from manufacturing a causal claim.
    """
    payload = {
        "REFLECTION_PHASE": reflection_phase,
        "ACTUAL_RETRIEVAL_EVIDENCE": retrieved_evidence or [],
        "FULL_ORDERED_TRAJECTORIES": full_trajectories or [],
        "CANONICAL_OPPONENT_STRATEGY": canonical_opponent_strategy,
        "PRIOR_RESPONSE_STRATEGY": previous_response_strategy,
        "SUPPORT_SEGMENTS": int(support_segments),
        "GUIDANCE_SHOWN_ON_INJECTED_ROUNDS": guidance_shown,
        "INJECTION_CONTRAST": injection_contrast,
        "CORRESPONDING_TRAJECTORY": evidence,
    }
    action_rule = ("This is RPS memory. Preserve a useful, executable action-selection rule, "
                   "with no concrete throw names in English or Chinese. Different opponent "
                   "patterns may require different advice. Never use a generic counter-action "
                   "sentence in place of explaining how to infer the next target action. "
                   + ("BOOTSTRAP: infer advice appropriate to the canonical ToM and its observed pattern. "
                      if reflection_phase == "bootstrap" else
                      "RETRIEVED UPDATE: optimize advice for situations where this cluster was actually "
                      "retrieved, including mistaken retrievals. Actual trajectories and rewards take "
                      "precedence over canonical ToM. Advice may diverge from that ToM. Hidden advice "
                      "was not used; do not attribute its outcomes to following the advice. ")
                   + "\n\n" if abstract_actions else "")
    return action_rule + '''Use this batch's ordered interaction evidence and rewards to write a direct action
instruction for SELF against the encountered opponent. Keep the players and action
order clear; actual play takes precedence over the opponent description.

memory_state tells you whether SELF saw stored advice before acting: "injected"
means shown, "withheld" means retrieved but randomly hidden, and "no_match" means
nothing was retrieved. GUIDANCE_SHOWN_ON_INJECTED_ROUNDS lists the advice shown;
INJECTION_CONTRAST compares injected and withheld outcomes.

Full trajectories also contain contextual rounds involving other clusters. Only rows
with focal_retrieval=true describe retrieval of the cluster currently being revised;
memory_state on other rows can refer to other advice and must not be attributed to
this cluster. offline_assigned_round identifies supporting rounds for a newly formed
cluster without actual retrieval. Preserve full sequence context when interpreting
these marked rounds.

Use the comparison together with the full trajectories to identify how the advice
can improve SELF's expected reward in the situations where this cluster is retrieved.
Reassess the action-selection rule from the evidence; inconclusive comparisons are
not a reason to default to the prior advice. If injected outcomes are worse, examine
the decisions and revise the rule to address plausible failures. Do not change wording
without improving the decision rule. The comparison is an association, not a proven
causal effect; do not invent evidence of improvement.

Before answering, privately audit every focal injected round: reconstruct what the
shown rule would predict from the completed history, compare that prediction with the
observed target action and reward, and identify which condition or predictor failed.
Use withheld focal rounds as additional trajectory evidence, while remembering that
their outcomes did not result from the advice. When injected mean reward is no better
than withheld mean reward, replace or materially repair the core predictor unless the
round-level audit gives concrete evidence that the rule itself was followed and was
accurate. Cosmetic paraphrases are invalid. Output only the improved instruction.

Write what SELF should do next, using an explicit action-selection method. Keep it brief and executable from current information. Do not repeat the
opponent's description or substitute a vague objective for an action instruction.
Return exactly {"response_strategy":"one concise English action instruction under 40 words"}.
Do not mention cluster IDs, support counts, opponent names, memory mechanics, hidden
state, or evaluator labels. Do not include reasoning, evidence summaries, or extra keys.
DATA:\n''' + json.dumps(payload, ensure_ascii=False)
