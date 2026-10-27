"""Evaluation: a SEPARATE Groq judge model scores the RAG answer on 13 metrics.
RAGAS: Context Precision, Context Recall, Correctness, Relevance, Faithfulness.
Judge prompts: Completeness, Coherence, Conciseness, Citation Accuracy.
Plain maths: Precision, Recall, Hit Rate (framework.py) and Hallucination Rate (1 - Faithfulness)."""
# Compatibility patch: ragas 0.2.x imports langchain_community's Vertex AI classes,
# which newer langchain-community versions no longer ship. Register harmless stand-ins
# BEFORE ragas is imported.
import sys
import types

_m = types.ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _m
_m = types.ModuleType("langchain_community.llms.vertexai")
_m.VertexAI = type("VertexAI", (), {})
sys.modules["langchain_community.llms.vertexai"] = _m

import asyncio
import json
import re
import time
from typing import Callable, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

import framework
import rag

_IDK = ("don't know", "do not know", "dont know", "cannot find", "can't find",
        "not mentioned", "not provided", "no information", "not in the context",
        "not contain", "insufficient")


def is_idk(text: str) -> bool:
    """True if an answer is a refusal such as "I don't know"."""
    t = rag.strip_citations(text).lower()
    return len(t) < 220 and any(p in t for p in _IDK)


# ------------------------------------------------------------------ RAGAS
class FastEmbedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return rag.embed_texts(list(texts)).tolist()

    def embed_query(self, text):
        return rag.embed_texts([text])[0].tolist()


def _build():
    judge = LangchainLLMWrapper(
        ChatGroq(model=rag.judge_model(), temperature=0, max_retries=5)
    )
    emb = LangchainEmbeddingsWrapper(FastEmbedEmbeddings())
    return judge, emb


async def _score(metric, sample):
    try:
        v = float(await metric.single_turn_ascore(sample))
        return v if v == v else "ERROR: RAGAS returned NaN"
    except Exception as e:  # keep going if one metric fails (rate limit / bad JSON)
        return f"ERROR: {type(e).__name__}: {str(e)[:150]}"


async def _ragas(question, answer, contexts, reference, refused, on_progress) -> Dict[str, object]:
    judge, emb = _build()
    sample = SingleTurnSample(
        user_input=question, response=answer,
        retrieved_contexts=contexts, reference=reference or None,
    )
    plan = {}  # canonical metric name -> RAGAS metric (Groq has no n>1 completions -> strictness=1)
    if not refused:
        plan["Faithfulness"] = Faithfulness(llm=judge)
        plan["Relevance"] = ResponseRelevancy(llm=judge, embeddings=emb, strictness=1)
    if reference:
        plan["Context Precision"] = LLMContextPrecisionWithReference(llm=judge)
        plan["Context Recall"] = LLMContextRecall(llm=judge)
        if not refused:
            plan["Correctness"] = FactualCorrectness(llm=judge)
    elif not refused:
        plan["Context Precision"] = LLMContextPrecisionWithoutReference(llm=judge)
    out: Dict[str, object] = {}
    for name, metric in plan.items():  # sequential: friendlier to free-tier rate limits
        if on_progress:
            on_progress(name)
        out[name] = await _score(metric, sample)
    return out


# -------------------------------------------------------- judge-prompt metrics
def _chat(system: str, user: str) -> str:
    delay = 3
    for attempt in range(5):
        try:
            return rag.chat(rag.judge_model(), system, user)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return ""


def _ask_json(prompt: str):
    out = _chat("You output only valid JSON, with no other text.", prompt)
    m = re.search(r"(\{.*\}|\[.*\])", out, re.S)
    if not m:
        raise ValueError("The judge did not return JSON.")
    return json.loads(m.group(1))


def rubric_scores(question: str, answer: str, reference: str) -> Dict[str, Optional[float]]:
    """One judge call scores Coherence, Conciseness and (with a correct answer) Completeness."""
    ref_part = (f"\nCorrect answer (for completeness):\n{reference}\n" if reference else "")
    d = _ask_json(
        "You are a strict QA evaluator. Score the ANSWER below from 1 (very poor) to 5 (excellent).\n"
        "- coherence: is it clear, logical and well structured?\n"
        "- conciseness: is it to the point, without filler or repetition?\n"
        + ("- completeness: does it cover ALL the key points of the correct answer?\n" if reference else "")
        + f"\nQuestion:\n{question}\n{ref_part}\nAnswer:\n{answer}\n\n"
        'Reply with ONLY this JSON: {"coherence": 1-5, "conciseness": 1-5'
        + (', "completeness": 1-5' if reference else "") + "}")
    if not isinstance(d, dict):
        raise ValueError("The judge returned an unexpected format.")
    return {
        "Coherence": framework.likert_to_unit(d.get("coherence")),
        "Conciseness": framework.likert_to_unit(d.get("conciseness")),
        "Completeness": framework.likert_to_unit(d.get("completeness")) if reference else None,
    }


def citation_accuracy(answer: str, chunks: List[rag.Chunk]) -> Optional[float]:
    """Share of cited statements whose cited chunk(s) really support them.
    An answer that cites nothing scores 0."""
    if not framework.cited_numbers(answer):
        return 0.0
    data = _ask_json(
        f"Numbered context chunks:\n{rag.format_context(chunks)}\n\nAnswer with citations:\n{answer}\n\n"
        "For every statement in the answer that has a citation like [1], decide whether the cited chunk(s) "
        "really support that statement. Reply with ONLY a JSON array: "
        '[{"statement": "...", "cited": [1], "supported": true}]')
    rows = [d for d in (data if isinstance(data, list) else []) if isinstance(d, dict)]
    if not rows:
        raise ValueError("The judge returned no citations to check.")
    good = 0
    for d in rows:
        cited = [int(x) for x in re.findall(r"\d+", " ".join(str(c) for c in d.get("cited", [])))]
        valid = bool(cited) and all(1 <= x <= len(chunks) for x in cited)
        good += 1 if valid and bool(d.get("supported")) else 0
    return good / len(rows)


# ------------------------------------------------------------ one question
def evaluate_row(question: str, answer: str, chunks: List[rag.Chunk], reference: str = "",
                 pages: Optional[List[int]] = None, texts: Optional[List[str]] = None,
                 negative: bool = False,
                 on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Score one answer on the 13 metrics.
    Returns {"refused": bool, "metrics": {name: float or None}, "errors": {name: message}}.
    - A refusal ("I don't know") has nothing to grade, so answer metrics are skipped.
    - A question marked as not in the document never uses a reference."""
    refused = is_idk(answer)
    clean = rag.strip_citations(answer)
    ref = "" if negative else (reference or "").strip()
    metrics: Dict[str, Optional[float]] = {m: None for m in framework.METRIC_NAMES}
    errors: Dict[str, str] = {}

    # 1) plain maths: Precision, Recall, Hit Rate
    metrics.update(framework.retrieval_metrics([(c.page, c.text) for c in chunks], pages or [], texts or []))

    # 2) RAGAS
    for name, val in asyncio.run(_ragas(question, clean, [c.text for c in chunks], ref, refused, on_progress)).items():
        if framework.is_num(val):
            metrics[name] = float(val)
        else:
            errors[name] = str(val)

    # 3) Hallucination Rate = 1 - Faithfulness
    if metrics["Faithfulness"] is not None:
        metrics["Hallucination Rate"] = max(0.0, min(1.0, 1.0 - metrics["Faithfulness"]))

    # 4) judge prompts (only when there is an answer to grade)
    if not refused:
        if on_progress:
            on_progress("Coherence, Conciseness, Completeness")
        try:
            for k, v in rubric_scores(question, clean, ref).items():
                if v is not None:
                    metrics[k] = v
        except Exception as e:
            for k in ("Coherence", "Conciseness") + (("Completeness",) if ref else ()):
                errors[k] = f"{type(e).__name__}: {str(e)[:150]}"
        if on_progress:
            on_progress("Citation Accuracy")
        try:
            metrics["Citation Accuracy"] = citation_accuracy(answer, chunks)
        except Exception as e:
            errors["Citation Accuracy"] = f"{type(e).__name__}: {str(e)[:150]}"

    return {"refused": refused, "metrics": metrics, "errors": errors}
