import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from rag.corpus import Passage, load_corpus
from rag.evaluate import grade, summarize
from rag.pipeline import Pipeline, PipelineError, check_answer
from rag.search import Hit

PASSAGE = Passage("example", "runbook.md", "Retention", 3, 3, "Snapshots are kept for 35 days.")


class CitationTests(unittest.TestCase):
    def test_citation_points_to_markdown_lines(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runbook.md").write_text(
                "# Retention\n\nSnapshots are kept for 35 days.\nThey expire automatically.\n",
                encoding="utf-8",
            )
            [passage] = load_corpus(root)
        self.assertEqual((passage.source, passage.start_line, passage.end_line), ("runbook.md", 3, 4))
        self.assertEqual(passage.excerpt, "Snapshots are kept for 35 days. They expire automatically.")

    def test_grounding_rejects_unsupported_phrase(self):
        self.assertEqual(check_answer("35 days", PASSAGE.excerpt, grounded=True).answer, "35 days")
        self.assertEqual(
            check_answer("5 days", PASSAGE.excerpt, grounded=True).reason,
            "answer_not_in_source",
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.search = Mock()
        self.pipeline.search.rerank.return_value = [Hit(PASSAGE, 5.0)]
        self.pipeline.answerability = Mock()
        self.pipeline.answerability.supports.return_value = True
        self.pipeline.generator = Mock()
        self.pipeline.generator.answer.return_value = "35 days"

    def test_answer_cites_the_passage_used_for_generation(self):
        result = self.pipeline.ask("How long are snapshots retained?")
        self.assertEqual(result["answer"], "35 days")
        self.assertEqual(result["citations"][0]["url"], "/sources/example")
        self.pipeline.generator.answer.assert_called_once_with(
            "How long are snapshots retained?", PASSAGE.excerpt
        )

    def test_low_relevance_is_logged_and_skips_generation(self):
        self.pipeline.search.rerank.return_value = [Hit(PASSAGE, -2.0)]
        with patch("rag.pipeline.event") as log:
            result = self.pipeline.ask("private question")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["reason"], "low_relevance")
        self.pipeline.generator.answer.assert_not_called()
        self.assertEqual(log.call_args.kwargs["request_id"], result["request_id"])
        self.assertNotIn("private question", json.dumps(log.call_args.kwargs))

    def test_reader_can_reject_a_highly_ranked_passage(self):
        self.pipeline.answerability.supports.return_value = False
        with patch("rag.pipeline.event"):
            result = self.pipeline.ask("What is the retention for billing transactions?")
        self.assertEqual(result["reason"], "passage_not_answerable")
        self.assertEqual(result["citations"], [])
        self.pipeline.generator.answer.assert_not_called()

    def test_inference_failure_logs_stage_without_exception_message(self):
        self.pipeline.generator.answer.side_effect = RuntimeError("private prompt")
        with patch("rag.pipeline.event") as log, self.assertRaises(PipelineError) as error:
            self.pipeline.ask("private question")
        self.assertEqual(log.call_args.args, ("request_failed",))
        self.assertEqual(log.call_args.kwargs["stage"], "generation")
        self.assertEqual(log.call_args.kwargs["request_id"], str(error.exception))
        self.assertNotIn("private", json.dumps(log.call_args.kwargs))


class EvaluationTests(unittest.TestCase):
    def test_denominators_include_abstentions_only_in_question_rate(self):
        hits = [Hit(PASSAGE, 4.0)]
        known = {"relevant": ["runbook.md#Retention"], "answers": ["35 days"]}
        unknown = {"relevant": [], "answers": []}
        report = summarize([
            grade(known, hits, "35 days", None),
            grade(unknown, hits, "35 days", None),
            grade(unknown, hits, None, "model_abstained"),
        ])
        self.assertEqual(report["precision_at_3"], 0.3333)
        self.assertEqual(report["proxy_hallucination_rate_among_answers"], 0.5)
        self.assertEqual(report["proxy_hallucination_rate_among_questions"], 0.3333)
        self.assertEqual(report["correct_abstentions"], 1)
