import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from rag.corpus import DEFAULT_CORPUS
from rag.models import MIN_ANSWERABILITY, MODELS
from rag.pipeline import MIN_RELEVANCE, Pipeline, check_answer
from rag.search import TOP_K, Hit, tokens

DEFAULT_QUESTIONS = DEFAULT_CORPUS.parent / "questions.json"


def label(hit: Hit) -> str:
    return f"{hit.passage.source}#{hit.passage.heading}"


def grade(case: dict, hits: list[Hit], answer: str | None, reason: str | None) -> dict:
    relevant = set(case["relevant"])
    retrieved = [label(hit) for hit in hits]
    precision = {
        k: sum(source in relevant for source in retrieved[:k]) / k
        for k in (1, TOP_K)
    } if relevant else None
    cited = retrieved[0] if answer else None
    failure = None
    if answer is None:
        if relevant:
            failure = "missed_answer"
    elif not relevant:
        failure = "unanswerable_claim"
    elif cited not in relevant:
        failure = "irrelevant_citation"
    elif tokens(answer) not in [tokens(alias) for alias in case["answers"]]:
        failure = "answer_mismatch"
    elif check_answer(answer, hits[0].passage.excerpt, grounded=True).reason:
        failure = "answer_not_in_source"
    return {
        "retrieved": [
            {"source": label(hit), "passage_id": hit.passage.id, "score": round(hit.score, 4)}
            for hit in hits
        ],
        "precision": precision,
        "answer": answer,
        "citation": cited,
        "abstention_reason": reason,
        "failure": failure,
    }


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(results: list[dict]) -> dict:
    reference_rows = [row for row in results if row["precision"] is not None]
    answered = [row for row in results if row["answer"] is not None]
    hallucinations = sum(row["failure"] is not None for row in answered)
    correct = sum(row["failure"] is None for row in answered)
    return {
        "precision_at_1": round(mean(row["precision"][1] for row in reference_rows), 4),
        "precision_at_3": round(mean(row["precision"][TOP_K] for row in reference_rows), 4),
        "answered": len(answered),
        "coverage": rate(len(answered), len(results)),
        "proxy_hallucinations": hallucinations,
        "proxy_hallucination_rate_among_answers": rate(hallucinations, len(answered)),
        "proxy_hallucination_rate_among_questions": rate(hallucinations, len(results)),
        "answerable_accuracy": rate(correct, len(reference_rows)),
        "correct_abstentions": sum(row["answer"] is None for row in results if row["precision"] is None),
        "failure_counts": dict(Counter(row["failure"] for row in results if row["failure"])),
    }


def evaluate() -> dict:
    fixtures = json.loads(DEFAULT_QUESTIONS.read_text(encoding="utf-8"))
    pipeline = Pipeline(DEFAULT_CORPUS)
    labels = Counter(f"{p.source}#{p.heading}" for p in pipeline.passages.values())
    ids = set()
    cases = []
    for case in fixtures:
        if case["id"] in ids or bool(case["relevant"]) != bool(case["answers"]):
            raise ValueError(f"Invalid fixture: {case['id']}")
        if any(labels[source] != 1 for source in case["relevant"]):
            raise ValueError(f"Fixture references a missing or ambiguous passage: {case['id']}")
        ids.add(case["id"])
        question = case["question"]
        lexical = pipeline.search.bm25(question)
        candidates = pipeline.search.hybrid(question, lexical)
        reranked = pipeline.search.rerank(question, candidates)[:TOP_K]
        stages = {}
        for name, hits, gate in (
            ("bm25", lexical[:TOP_K], False),
            ("hybrid_reranked", reranked, False),
            ("grounded", reranked, True),
        ):
            result = pipeline.generate(question, hits, grounded=gate)
            stages[name] = grade(case, hits, result.answer, result.reason)
        cases.append({**case, "stages": stages})
    return {
        "models": MODELS,
        "passages": len(pipeline.passages),
        "questions": len(cases),
        "answerable_questions": sum(bool(case["relevant"]) for case in cases),
        "relevance_threshold_logit": MIN_RELEVANCE,
        "answerability_threshold": MIN_ANSWERABILITY,
        "summary": {
            stage: summarize([case["stages"][stage] for case in cases])
            for stage in ("bm25", "hybrid_reranked", "grounded")
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RAG configurations on labeled questions")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
