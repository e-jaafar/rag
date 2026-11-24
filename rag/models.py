import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

MODELS = {
    "embedding": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "reranker": (
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ),
    "answerability": (
        "deepset/minilm-uncased-squad2",
        "934656cdda79824eabf503ed56e15c01ddbdbe3f",
    ),
    "generator": (
        "google/flan-t5-base",
        "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    ),
}


class Generator:
    def __init__(self):
        name, revision = MODELS["generator"]
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name, revision=revision, use_safetensors=True
        ).eval()

    @torch.inference_mode()
    def answer(self, question: str, passage: str) -> str:
        prompt = (
            "Answer the question with the shortest exact phrase from the passage. "
            "If the passage does not answer the question, respond UNKNOWN.\n"
            f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


MIN_ANSWERABILITY = 0.30


class Answerability:
    def __init__(self):
        name, revision = MODELS["answerability"]
        model = AutoModelForQuestionAnswering.from_pretrained(
            name, revision=revision, use_safetensors=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.reader = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)

    def supports(self, question: str, excerpt: str) -> bool:
        best = self.reader(question=question, context=excerpt, handle_impossible_answer=True)
        return bool(best["answer"]) and best["score"] >= MIN_ANSWERABILITY
