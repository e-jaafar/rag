import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from rag.corpus import Passage, load_corpus
from rag.models import Answerability, Generator
from rag.search import TOP_K, Hit, Search, tokens

MIN_RELEVANCE = 0.0
logger = logging.getLogger("rag.events")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(logging.StreamHandler())


def event(name: str, **fields):
    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": name,
        **fields,
    }))


def citation(passage: Passage) -> dict:
    return {**asdict(passage), "url": f"/sources/{passage.id}"}


@dataclass(frozen=True)
class Answer:
    answer: str | None
    reason: str | None = None


def check_answer(draft: str, excerpt: str, *, grounded: bool) -> Answer:
    answer_tokens = tokens(draft)
    if not answer_tokens or answer_tokens in (["unknown"], ["unanswerable"]):
        return Answer(None, "model_abstained")
    # Match whole words in sequence and tolerate differences in punctuation and case.
    if grounded and f" {' '.join(answer_tokens)} " not in f" {' '.join(tokens(excerpt))} ":
        return Answer(None, "answer_not_in_source")
    return Answer(draft)


class Pipeline:
    def __init__(self, corpus: Path):
        torch.set_num_threads(2)
        passages = load_corpus(corpus)
        self.passages = {passage.id: passage for passage in passages}
        self.search = Search(passages)
        self.generator = Generator()
        self.answerability = Answerability()

    def generate(self, question: str, hits: list[Hit], *, grounded: bool) -> Answer:
        if not hits:
            return Answer(None, "no_candidates")
        best = hits[0]
        if grounded and best.score < MIN_RELEVANCE:
            return Answer(None, "low_relevance")
        excerpt = best.passage.excerpt
        if grounded and not self.answerability.supports(question, excerpt):
            return Answer(None, "passage_not_answerable")
        draft = self.generator.answer(question, excerpt)
        return check_answer(draft, excerpt, grounded=grounded)

    def ask(self, question: str) -> dict:
        request_id = uuid4().hex
        started = time.perf_counter()
        stage = "lexical_search"
        hits = []
        try:
            lexical = self.search.bm25(question)
            stage = "hybrid_search"
            candidates = self.search.hybrid(question, lexical)
            stage = "reranking"
            hits = self.search.rerank(question, candidates)[:TOP_K]
            stage = "generation"
            result = self.generate(question, hits, grounded=True)
        except Exception as error:
            event("request_failed", request_id=request_id, stage=stage, error_type=type(error).__name__)
            raise PipelineError(request_id) from error

        fields = {
            "request_id": request_id,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "retrieved": [{"id": hit.passage.id, "score": round(hit.score, 4)} for hit in hits],
        }
        if result.reason:
            event("abstention", reason=result.reason, **fields)
        return {
            "request_id": request_id,
            "status": "answered" if result.answer else "abstained",
            "answer": result.answer,
            "reason": result.reason,
            "citations": [citation(hits[0].passage)] if result.answer else [],
        }


class PipelineError(Exception):
    pass
