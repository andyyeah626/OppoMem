from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


UNKNOWN_OPPONENT_STRATEGY = "Insufficient evidence for a reliable opponent strategy."


@dataclass(slots=True)
class Turn:
    round_index: int
    observation: str
    legal_actions: list[str]
    agent_action: str
    opponent_action: str
    reward: float
    inference_context: str = ""
    # Short pre-action strategy intent used only by post-episode reflection.
    response_intent: str = ""


@dataclass(slots=True)
class ToMState:
    round_index: int
    opponent_strategy: str
    next_action_probabilities: dict[str, float] = field(default_factory=dict)
    # SELF's most recent completed throw.  Carried as trace telemetry only: it is
    # deliberately NOT part of retrieval_text().  See that method for why.
    self_last_action: str = ""

    def retrieval_text(self, *, situation: str | None = None) -> str:
        """The query text every cluster is scored against.

        Opponent description only.  It used to append "SELF just played <action>",
        on the theory that SELF's throw is the dimension on which clusters ought to
        differ -- added after measuring that 78% of the 2026-09-20 clusters never
        referenced SELF's throw.  That fixed the discrimination problem by
        prescribing its answer: every query then asserted a SELF-conditioned
        relation, so an opponent whose rule does not involve SELF had to compete on
        a dimension that does not describe it, and the store was pushed toward
        SELF-reactive keys no matter what the trajectory showed.  Whether a rule is
        conditioned on SELF is a finding the memory is supposed to reach from the
        trajectory.  It cannot also be the frame the query imposes on retrieval.
        """
        base = f"Opponent strategy: {self.opponent_strategy}"
        return base if not situation else base + f"\nCurrent situation: {situation}"


@dataclass(slots=True)
class MemoryUse:
    round_index: int
    cluster_id: str
    selected: bool
    opponent_strategy: str
    response_strategy: str
    score: float
    metric: str
    agent_action: str = ""
    opponent_action: str = ""
    reward: float = 0.0


@dataclass(slots=True)
class EpisodeTrace:
    episode_id: str
    game: str
    variant: str
    turns: list[Turn] = field(default_factory=list)
    tom_chain: list[ToMState] = field(default_factory=list)
    memory_uses: list[MemoryUse] = field(default_factory=list)
    learning_audit: dict[str, Any] = field(default_factory=dict)
    initial_reward: float = 0.0
    initial_events: list[str] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return float(self.initial_reward + sum(turn.reward for turn in self.turns))

    def trajectory_text(self) -> str:
        initial = (
            "Before our first decision: " + repr(self.initial_events) + "\n"
            if self.initial_events else ""
        )
        return initial + "\n".join(
            f"r{turn.round_index}: observation={turn.observation}; "
            f"self={turn.agent_action}; opponent={turn.opponent_action}; "
            f"reward={turn.reward:+g}"
            for turn in self.turns
        )


@dataclass(slots=True)
class CentroidCluster:
    """One opponent ToM, one response strategy, and its retrieval prototype.

    The prototype fields are optional so schema-7 banks written before prototype
    retrieval remain readable and continue to use their stored centroid.
    """

    cluster_id: str
    opponent_strategy: str
    response_strategy: str
    centroid: list[float]
    # Required only to update the exact running mean without storing members.
    centroid_size: int = 1
    prototype_strategy: str | None = None
    prototype_embedding: list[float] | None = None
    # The retained member ToM set for capped/medoid banks: a list of
    # {"opponent_strategy": str, "embedding": [...]} rows.  Only banks built with
    # `max_cluster_members` populate it; on every legacy bank it stays None and the
    # centroid/prototype fields keep their previous meaning.  Optional so a bank
    # written without members (and every earlier schema-7 bank) still loads.
    members: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CentroidCluster":
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in raw.items() if key in known})


@dataclass(slots=True)
class CentroidMatch:
    cluster_id: str
    opponent_strategy: str
    response_strategy: str
    score: float
    metric: str = "cosine"

    @property
    def cosine(self) -> float:
        """Compatibility alias for older callers; new logs use score + metric."""
        return self.score


@dataclass(slots=True)
class Decision:
    action: str
    selected_memory_ids: list[str] = field(default_factory=list)
    response_intent: str = ""
