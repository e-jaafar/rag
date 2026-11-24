# Grounded RAG

A local question-answering API for Markdown knowledge bases. It retrieves evidence using BM25 and dense search, reranks candidates, and returns a short answer with a source citation. The service abstains and records why when the top passage fails the relevance or answerability checks, or when it lacks the generated phrase.

## Run locally

Use Python 3.13. The first run downloads about 1.3 GB of pretrained model files from Hugging Face. Inference runs on CPU without an API key. After the download, `HF_HUB_OFFLINE=1` keeps model loading offline.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
# Linux only: install the CPU build of PyTorch first.
# python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --limit-concurrency 16
```

```sh
curl -sS http://127.0.0.1:8000/health

curl -sS http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many days of production database snapshots are kept?"}'
```

The example returns `35 days` and cites the **Production snapshots** passage in [backup.md](data/corpus/backup.md#L12-L13). Each citation includes its file, heading, line range, excerpt, and a URL you can query on the running service. An abstention returns HTTP 200 with `status: "abstained"`, a `null` answer, and a reason. Inference errors return HTTP 503 with a request ID. Interactive API documentation is available at `/docs`.

Point `RAG_CORPUS` to a directory of UTF-8 Markdown files and restart to load your own documents. The bundled corpus contains fictional operations policies. The service indexes passages in memory at startup and runs one inference at a time per process. Use it with an English corpus of short, independently answerable passages. The local service exposes indexed excerpts via the citation endpoint, so restrict access to the service according to the visibility of its corpus.

## Pipeline

1. Parse Markdown paragraphs into bounded passages with file and line provenance.
2. Rank passages by BM25 and by cosine similarity over [MiniLM embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Merge the rankings with reciprocal rank fusion and keep up to 12 candidates.
3. Score each question–passage pair with the [MS MARCO MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). Keep the top three for evaluation and use the first for generation.
4. Check whether the top passage can answer the question using [Deepset MiniLM trained on SQuAD 2.0](https://huggingface.co/deepset/minilm-uncased-squad2). Require a reranker logit of at least 0 and a nonempty reader span with a score of at least 0.30.
5. Ask [FLAN-T5 Base](https://huggingface.co/google/flan-t5-base) for a short answer from that passage. Return it with a citation if the answer appears as a contiguous token sequence in the excerpt.

Abstentions and inference errors emit JSON events to standard error with a request ID and cause. Abstention events include the retrieved passage IDs, scores, and elapsed time. Runtime event records omit raw questions and passages. The offline benchmark includes the public fixture questions and outputs for inspection.

## Benchmark

```sh
python -m rag.evaluate
python -m unittest discover -s tests -v
```

The benchmark runs three configurations on the same [45 annotated questions](data/questions.json), including 35 answerable questions and 10 whose answers are absent. All three use the same generator and top-passage context budget. The baseline uses BM25. The intermediate configuration adds dense retrieval, rank fusion, and reranking. The final configuration also applies the relevance, answerability, and quote checks.

**Precision@k** averages the fraction of the top `k` positions occupied by labeled relevant passages over answerable questions. Each question has one relevant passage, so precision@3 tops out at 33.33%. **Hallucination proxy** counts emitted answers to unanswerable questions, answers with an irrelevant citation, answers outside the annotated aliases, and answers missing from the cited passage. The benchmark reports both the rate among emitted answers and the rate across all questions. Coverage and correct abstentions show the cost of withholding answers.

The benchmark writes the model revisions, per-question rankings, cited sources, abstention reasons, and failure labels to [reports/benchmark.json](reports/benchmark.json). The fixture measures short factual answers on a single synthetic corpus. The answerability cutoff was selected after inspecting this fixture, so the reported results are an in-sample comparison. The alias-based score can penalize valid wording, and the quote check can accept the wrong fact if it appears in a passage.

| Metric | BM25 baseline | Hybrid + rerank | Final with answer checks |
| --- | ---: | ---: | ---: |
| Precision@1 | 74.29% | 97.14% | 97.14% |
| Precision@3 | 29.52% | 33.33% | 33.33% |
| Hallucination proxy among emitted answers | 26.47% (9/34) | 23.81% (10/42) | 12.12% (4/33) |
| Hallucination proxy across all questions | 20.00% | 22.22% | 8.89% |
| Coverage | 75.56% | 93.33% | 73.33% |
| Correct answers on answerable questions | 71.43% | 91.43% | 82.86% |
| Correct abstentions on unanswerable questions | 3/10 | 3/10 | 6/10 |

The hybrid stage improved the first result from 26 to 34 relevant passages across 35 answerable questions. It also answered more often, increasing the number of unsupported answers from nine to ten. The answer checks removed six of those unsupported answers and withheld three correct answers from the hybrid stage.

The four remaining unsupported answers all used a real passage that described a nearby policy. For example, the model quoted the recovery **point** objective when asked for the recovery **time** objective (q36), and returned the response time for **normal** premium requests when the question asked about **critical** ones (q42). The report also identifies five answerable questions where the relevance check withheld an answer and one where the generated phrase dropped an article and failed the quote check (q26).
