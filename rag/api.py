import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from rag.corpus import DEFAULT_CORPUS
from rag.pipeline import Pipeline, PipelineError, citation, event


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipeline = Pipeline(Path(os.getenv("RAG_CORPUS", str(DEFAULT_CORPUS))))
    except Exception as error:
        event("startup_failed", error_type=type(error).__name__)
        raise
    app.state.inference_lock = Lock()
    yield


app = FastAPI(title="Grounded RAG", lifespan=lifespan)


class Question(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


@app.get("/health")
def health():
    return {"status": "ready", "passages": len(app.state.pipeline.passages)}


@app.get("/sources/{passage_id}")
def source(passage_id: str):
    passage = app.state.pipeline.passages.get(passage_id)
    if passage is None:
        raise HTTPException(404, "Passage not found")
    return citation(passage)


@app.post("/ask")
def ask(request: Question):
    try:
        with app.state.inference_lock:
            return app.state.pipeline.ask(request.question)
    except PipelineError as error:
        raise HTTPException(
            503, {"message": "Answer pipeline unavailable", "request_id": str(error)}
        ) from error
