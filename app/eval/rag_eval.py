"""
RAG evaluation: measure citation quality.

Metrics:
- citation_precision  — valid cited IDs / all cited IDs, deduplicated per answer;
                        null when no citations were produced
- citation_validation_pass_rate — fraction of answers with citation_valid=true
- no_citation_rate    — fraction of answers with zero citations
- avg_citations       — average number of citation markers per answer
- avg_latency_ms      — average end-to-end /ask response time

Usage:
    python -m app.eval.rag_eval --eval data/eval/rag_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.answer import answer_question
from app.rag.citation import extract_citations

logger = logging.getLogger(__name__)

DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "metadata.sqlite"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "faiss"


def run_rag_eval(
    eval_path: Path,
    db_path: Path = DEFAULT_DB,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict:
    """Run RAG evaluation on a set of question-answer pairs.

    Returns a dict with per-metric aggregates and per-question details.
    """
    questions = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            questions.append(rec["question"])

    if not questions:
        logger.warning("No eval questions found")
        return {}

    logger.info("Evaluating %d RAG questions ...", len(questions))

    results_detail: list[dict] = []
    total_citations = 0
    total_valid_citations = 0
    no_citation_count = 0
    total_invalid = 0
    latencies: list[float] = []

    for q in questions:
        logger.info("  Q: %s", q[:60])
        r = answer_question(q, db_path=db_path, index_dir=index_dir)

        cited = extract_citations(r.answer)
        n_cited = len(cited)
        evidence_ids = {citation.citation_id for citation in r.citations}
        n_valid = sum(citation_id in evidence_ids for citation_id in cited)
        total_citations += n_cited
        total_valid_citations += n_valid
        latencies.append(r.latency_ms)

        if n_cited == 0:
            no_citation_count += 1
        if not r.citation_valid:
            total_invalid += 1

        results_detail.append({
            "question": q,
            "answer": r.answer[:300],
            "citations_used": n_cited,
            "valid_citations": n_valid,
            "invalid_citations": n_cited - n_valid,
            "evidence_count": len(r.citations),
            "citation_valid": r.citation_valid,
            "warnings": r.citation_warnings,
            "latency_ms": round(r.latency_ms, 2),
        })

    n = len(questions)
    summary = {
        "queries": n,
        "citation_precision": (
            round(total_valid_citations / total_citations, 4)
            if total_citations else None
        ),
        "citation_validation_pass_rate": round(1.0 - total_invalid / n, 4),
        "total_citations": total_citations,
        "valid_citations": total_valid_citations,
        "no_citation_rate": round(no_citation_count / n, 4) if n > 0 else 0,
        "avg_citations_per_answer": round(total_citations / n, 2) if n > 0 else 0,
        "avg_latency_ms": round(np.mean(latencies), 2) if latencies else 0,
    }

    return {"summary": summary, "details": results_detail}


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--eval",
        default=str(PROJECT_ROOT / "data" / "eval" / "rag_eval.jsonl"),
        help="Path to RAG eval JSONL file",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    eval_path = Path(args.eval)
    if not eval_path.exists():
        sys.exit(f"Eval file not found: {eval_path}")

    results = run_rag_eval(eval_path, db_path=Path(args.db), index_dir=Path(args.index_dir))

    if not results:
        sys.exit("No results.")

    print("\n" + "=" * 60)
    print(f"RAG Evaluation — {results['summary']['queries']} questions")
    print("=" * 60)
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")
    print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
