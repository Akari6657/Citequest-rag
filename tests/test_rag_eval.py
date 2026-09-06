"""RAG citation metrics distinguish cited IDs from whole-answer validation."""

import json

import pytest

from app.core.schemas import AskResponse, CitationInfo
from app.eval import rag_eval
from app.rag.citation import verify_citations
from app.rag.llm_provider import MockLLMProvider


def evaluate_answers(monkeypatch, tmp_path, answers):
    citations = [{"citation_id": i, "paper_id": f"P{i}", "chunk_id": f"C{i}",
                  "title": f"Paper {i}"} for i in (1, 2)]
    pending = iter(answers)

    def answer_question(question, **kwargs):
        text = MockLLMProvider(next(pending)).generate(user=question).text
        verification = verify_citations(text, citations)
        return AskResponse(
            question=question, answer=text, citation_valid=verification.valid,
            citation_warnings=verification.warnings,
            citations=[CitationInfo(**c) for c in citations], latency_ms=10,
        )

    monkeypatch.setattr(rag_eval, "answer_question", answer_question)
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps({"question": f"Q{i}"}) for i in range(len(answers))))
    return rag_eval.run_rag_eval(path)


def test_precision_counts_cited_ids_not_passing_answers(monkeypatch, tmp_path):
    report = evaluate_answers(monkeypatch, tmp_path, ["Supported [1][2].", "Mixed [1][99]."])
    summary = report["summary"]
    assert summary["citation_precision"] == 0.75
    assert summary["citation_validation_pass_rate"] == 0.5
    assert summary["total_citations"] == 4 and summary["valid_citations"] == 3
    assert report["details"][1]["valid_citations"] == 1
    assert report["details"][1]["invalid_citations"] == 1


def test_repeated_ids_count_once_per_answer(monkeypatch, tmp_path):
    report = evaluate_answers(monkeypatch, tmp_path, ["Repeated [1][1][99].", "Again [1]."])
    assert report["summary"]["total_citations"] == 3
    assert report["summary"]["valid_citations"] == 2
    assert report["summary"]["citation_precision"] == 0.6667
    assert report["summary"]["avg_citations_per_answer"] == 1.5


@pytest.mark.parametrize("answer,precision,no_citation_rate", [
    ("No citations.", None, 1.0), ("Unknown [99].", 0.0, 0.0),
])
def test_absent_and_invalid_citations_are_distinguished(
    monkeypatch, tmp_path, answer, precision, no_citation_rate,
):
    report = evaluate_answers(monkeypatch, tmp_path, [answer])
    assert report["summary"]["citation_precision"] == precision
    assert report["summary"]["no_citation_rate"] == no_citation_rate
    assert report["summary"]["citation_validation_pass_rate"] == 0.0


def test_empty_evaluation_returns_no_metrics(monkeypatch, tmp_path):
    assert evaluate_answers(monkeypatch, tmp_path, []) == {}
