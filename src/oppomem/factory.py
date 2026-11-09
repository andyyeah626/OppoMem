from __future__ import annotations

from pathlib import Path
from typing import Any

from oppomem.agent import AgentVariant, ToMMemoryAgent
from oppomem.memory import MECHANISM, CentroidMemoryBank, CentroidMemoryLearner, CentroidRetriever, CosineTopKNLIReranker, CosineTop1NLIGatedRetriever, NLITop1Retriever
from oppomem.models.protocol import JsonLLM
from oppomem.models.semantics import Embedder, NLIScorer, RetrievalEmbedder


def build_agent(
    *,
    variant: AgentVariant,
    llm: JsonLLM,
    memory_path: str | Path,
    embedder: Embedder,
    nli: NLIScorer,
    semantic_config: dict[str, Any],
    load_existing_memory: bool = False,
    game: str = "repeated_rps",
    summarizer_llm: JsonLLM | None = None,
) -> ToMMemoryAgent:
    """Build one agent.  `summarizer_llm` serves only the two memory-summary stages
    (`memory_windows`, `memory_reflection`); every action-deciding call stays on
    `llm`.  Omitting it makes the two models the same object, which is the
    behaviour every existing caller already had."""
    variant = AgentVariant(variant)
    embedding_mode = str(semantic_config.get("embedding_mode", "none"))
    use_prototype = bool(semantic_config.get("use_prototype", False))
    if embedding_mode != "none" and not isinstance(embedder, RetrievalEmbedder):
        embedder = RetrievalEmbedder(embedder, embedding_mode)
    tom_prompt_variant = str(semantic_config.get("tom_prompt_variant", "A")).upper()
    if semantic_config.get("memory_mechanism") != MECHANISM:
        raise ValueError(f"Only {MECHANISM} is supported")
    if variant is AgentVariant.VANILLA:
        return ToMMemoryAgent(
            game=game, variant=variant, llm=llm,
            tom_prompt_variant=tom_prompt_variant,
        )

    bank = CentroidMemoryBank(memory_path)
    if load_existing_memory:
        bank.load()
    scope = semantic_config.get("memory_scope", "default")
    protocol_id = semantic_config.get("protocol_id", scope)
    if bank.metadata and bank.metadata.get("game") != game:
        raise ValueError("Memory bank belongs to a different game")
    if bank.metadata and bank.metadata.get("scope") != scope:
        raise ValueError("Memory bank belongs to a different visibility scope")
    if bank.metadata and bank.metadata.get("protocol_id") != protocol_id:
        raise ValueError("Memory bank belongs to a different game protocol")
    if bank.metadata and bank.metadata.get("tom_prompt_variant") != tom_prompt_variant:
        raise ValueError("Memory bank belongs to a different ToM prompt variant")
    if bank.metadata and bank.metadata.get("embedding_mode", "none") != embedding_mode:
        raise ValueError("Memory bank belongs to a different embedding mode")
    if bank.metadata and "use_prototype" in bank.metadata \
            and bool(bank.metadata["use_prototype"]) != use_prototype:
        raise ValueError("Memory bank belongs to a different prototype mode")
    if not bank.metadata:
        bank.metadata = {
            "mechanism": MECHANISM,
            "game": game,
            "scope": scope,
            "protocol_id": protocol_id,
            "tom_prompt_variant": tom_prompt_variant,
            "granularity": (
                "episode_policy"
                if semantic_config.get("fixed_policy_protocol", False)
                else "semantic_tom_window"
            ),
        }
    retrieval_backend = str(semantic_config.get("retrieval_backend", "cosine"))
    stored_backend = str(bank.metadata.get("retrieval_backend", retrieval_backend))
    if stored_backend != retrieval_backend:
        raise ValueError("Memory bank belongs to a different retrieval backend")
    if retrieval_backend == "cosine_topk_nli_rerank":
        similarity_threshold = float(semantic_config.get("retrieval_min_similarity", .55))
        entailment_threshold = float(semantic_config.get("retrieval_min_entailment", .0))
        top_k = int(semantic_config.get("retrieval_top_k", 8))
        retriever = CosineTopKNLIReranker(
            embedder, nli, min_similarity=similarity_threshold,
            min_entailment=entailment_threshold, top_k=top_k,
            use_prototype=use_prototype,
        )
        bank.metadata["retrieval_min_similarity"] = similarity_threshold
        bank.metadata["retrieval_min_entailment"] = entailment_threshold
        bank.metadata["retrieval_top_k"] = top_k
    elif retrieval_backend == "cosine_top1_nli_gate":
        similarity_threshold = float(
            semantic_config.get("retrieval_min_similarity", .55)
        )
        entailment_threshold = float(
            semantic_config.get("retrieval_min_entailment", .80)
        )
        retriever = CosineTop1NLIGatedRetriever(
            embedder, nli,
            min_similarity=similarity_threshold,
            min_entailment=entailment_threshold,
        )
        bank.metadata["retrieval_min_similarity"] = similarity_threshold
        bank.metadata["retrieval_min_entailment"] = entailment_threshold
    elif retrieval_backend == "nli_entailment_top1":
        retrieval_threshold = float(semantic_config.get("retrieval_min_entailment", .97))
        retriever = NLITop1Retriever(nli, min_entailment=retrieval_threshold)
        bank.metadata["retrieval_min_entailment"] = retrieval_threshold
    elif retrieval_backend == "cosine":
        retrieval_threshold = float(semantic_config.get("retrieval_min_similarity", .75))
        retriever = CentroidRetriever(
            embedder, min_similarity=retrieval_threshold,
            use_prototype=use_prototype,
        )
        bank.metadata["retrieval_min_similarity"] = retrieval_threshold
    else:
        raise ValueError(f"Unknown retrieval backend {retrieval_backend!r}")
    bank.metadata["retrieval_backend"] = retrieval_backend
    bank.metadata["embedding_mode"] = embedding_mode
    bank.metadata["use_prototype"] = use_prototype
    fixed_policy = bool(semantic_config.get("fixed_policy_protocol", False))
    bank.metadata["fixed_policy_protocol"] = fixed_policy
    bank.metadata["tom_prompt_variant"] = tom_prompt_variant
    learner = CentroidMemoryLearner(
        bank,
        llm,
        embedder,
        nli,
        game,
        scope,
        similarity_threshold=semantic_config.get("merge_similarity_threshold", .82),
        contradiction_threshold=semantic_config.get("nli_contradiction_threshold", .70),
        max_centroid_size=semantic_config.get("max_centroid_size"),
        max_cluster_members=semantic_config.get("max_cluster_members"),
        fixed_policy=fixed_policy,
        summarizer_llm=summarizer_llm,
        abstract_rps=bool(semantic_config.get("abstract_rps_memory", False)),
    )
    bank.metadata["max_cluster_members"] = semantic_config.get("max_cluster_members")
    return ToMMemoryAgent(
        game=game,
        variant=variant,
        llm=llm,
        bank=bank,
        retriever=retriever,
        learner=learner,
        tom_prompt_variant=tom_prompt_variant,
        tom_nl_horizon=int(semantic_config.get("tom_nl_horizon", 20)),
        injection_min_margin=float(
            semantic_config.get("injection_min_margin", 0.0)
        ),
        injection_tom_overlap_min=float(
            semantic_config.get("injection_tom_overlap_min", 0.0)
        ),
        abstract_rps=bool(semantic_config.get("abstract_rps_memory", False)),
    )
