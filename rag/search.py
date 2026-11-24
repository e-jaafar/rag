import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.corpus import Passage
from rag.models import MODELS

TOP_K = 3
CANDIDATES = 12


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Search:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        texts = [passage.search_text for passage in passages]
        self.lexical = BM25Okapi([tokens(text) for text in texts])
        name, revision = MODELS["embedding"]
        self.embedder = SentenceTransformer(name, revision=revision, device="cpu")
        self.vectors = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        name, revision = MODELS["reranker"]
        self.reranker = CrossEncoder(name, revision=revision, device="cpu", max_length=384)

    def bm25(self, question: str) -> list[Hit]:
        scores = self.lexical.get_scores(list(dict.fromkeys(tokens(question))))
        order = np.argsort(-scores, kind="stable")
        return [
            Hit(self.passages[index], float(scores[index]))
            for index in order[:CANDIDATES]
            if scores[index] > 0
        ]

    def hybrid(self, question: str, lexical: list[Hit]) -> list[Passage]:
        vector = self.embedder.encode(question, normalize_embeddings=True, show_progress_bar=False)
        order = np.argsort(-(self.vectors @ vector), kind="stable")[:CANDIDATES]
        dense = [self.passages[index] for index in order]
        fused = Counter()
        for ranking in ([hit.passage for hit in lexical], dense):
            for rank, passage in enumerate(ranking, 1):
                fused[passage] += 1 / (60 + rank)
        return [passage for passage, _ in fused.most_common(CANDIDATES)]

    def rerank(self, question: str, candidates: list[Passage]) -> list[Hit]:
        pairs = [(question, passage.search_text) for passage in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        hits = [Hit(passage, float(score)) for passage, score in zip(candidates, scores, strict=True)]
        return sorted(hits, key=lambda hit: -hit.score)
