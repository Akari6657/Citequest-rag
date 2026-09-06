"""
Download CS papers from arXiv API (free, no key needed).

Fetches recent papers from major CS categories and saves as JSONL
in our standard format. Much more reliable than UnarXive streaming.

Usage:
    python scripts/download_arxiv.py --size 5000

Categories covered:
    cs.AI, cs.LG, cs.CL, cs.CV, cs.IR, cs.CR, cs.DB, cs.DS,
    cs.NE, cs.SE, cs.RO, cs.DC
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

COMBINED_STRATEGY = "combined"
CATEGORY_BALANCED_STRATEGY = "category-balanced"
DOWNLOAD_CATEGORY_FIELD = "_download_category"
ARXIV_QUERY_RESULT_LIMIT = 10_000

# Major CS categories to fetch from
CS_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language (NLP)
    "cs.CV",   # Computer Vision
    "cs.IR",   # Information Retrieval
    "cs.CR",   # Cryptography and Security
    "cs.DB",   # Databases
    "cs.DS",   # Data Structures and Algorithms
    "cs.NE",   # Neural and Evolutionary Computing
    "cs.SE",   # Software Engineering
    "cs.RO",   # Robotics
    "cs.DC",   # Distributed, Parallel, and Cluster Computing
]

# Map arXiv categories to concept labels
CATEGORY_LABELS = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.IR": "Information Retrieval",
    "cs.CR": "Cryptography and Security",
    "cs.DB": "Databases",
    "cs.DS": "Data Structures and Algorithms",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.SE": "Software Engineering",
    "cs.RO": "Robotics",
    "cs.SY": "Systems and Control",
    "cs.DC": "Distributed Parallel and Cluster Computing",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IT": "Information Theory",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NI": "Networking and Internet Architecture",
    "cs.SI": "Social and Information Networks",
    "stat.ML": "Machine Learning",
}


def _balanced_category_targets(categories: list[str], size: int) -> dict[str, int]:
    """Distribute an exact corpus size across categories deterministically."""
    if not categories:
        raise ValueError("At least one category is required")
    if len(set(categories)) != len(categories):
        raise ValueError("Categories must be unique")
    if size <= 0:
        raise ValueError("Size must be positive")

    base, remainder = divmod(size, len(categories))
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def _year_windows(min_year: int, end_date: date) -> list[tuple[str, str]]:
    """Return inclusive arXiv submittedDate windows, newest year first."""
    if min_year <= 0:
        raise ValueError("Minimum year must be positive")
    if min_year > end_date.year:
        raise ValueError("Minimum year must not be later than the end date")

    windows = []
    for year in range(end_date.year, min_year - 1, -1):
        start = f"{year:04d}01010000"
        end = (
            f"{end_date:%Y%m%d}2359"
            if year == end_date.year
            else f"{year:04d}12312359"
        )
        windows.append((start, end))
    return windows


def _load_existing_state(
    output_path: Path,
    *,
    strategy: str,
    categories: list[str],
) -> tuple[set[str], Counter[str]]:
    """Load and strictly validate resumable download state."""
    existing_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    if not output_path.exists():
        return existing_ids, category_counts

    allowed_categories = set(categories)
    with open(output_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc.msg}"
                ) from exc

            paper_id = record.get("paper_id")
            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(f"Missing paper_id at line {line_number}")
            if paper_id in existing_ids:
                raise ValueError(f"Duplicate paper_id {paper_id!r} at line {line_number}")
            existing_ids.add(paper_id)

            if strategy == CATEGORY_BALANCED_STRATEGY:
                category = record.get(DOWNLOAD_CATEGORY_FIELD)
                if category not in allowed_categories:
                    raise ValueError(
                        f"Invalid or missing {DOWNLOAD_CATEGORY_FIELD} at line "
                        f"{line_number}; this file cannot resume a category-balanced run"
                    )
                if record.get("primary_category") != category:
                    raise ValueError(
                        f"primary_category does not match {DOWNLOAD_CATEGORY_FIELD} "
                        f"at line {line_number}"
                    )
                category_counts[category] += 1

    return existing_ids, category_counts


def _map_to_paper(result) -> dict | None:
    """Convert an arxiv.Result to our standard paper dict."""
    try:
        arxiv_id = result.entry_id.split("/")[-1]
        # Remove version suffix from ID (e.g. "2301.00001v2" → "2301.00001")
        if "v" in arxiv_id and arxiv_id.split("v")[-1].isdigit():
            arxiv_id = arxiv_id[: arxiv_id.rindex("v")]
    except (AttributeError, ValueError):
        return None

    title = (result.title or "").strip()
    if not title:
        return None

    abstract = (result.summary or "").strip()
    authors = [a.name for a in (result.authors or []) if a.name]
    year = result.published.year if result.published else None

    # Venue from journal_ref or comment
    venue = None
    if result.journal_ref:
        venue = str(result.journal_ref).strip()
    elif result.comment:
        comment = str(result.comment).strip()
        if len(comment) < 120:
            venue = comment

    # Map categories to concept labels
    concepts = []
    seen = set()
    for cat in (result.categories or []):
        cat_str = str(cat)
        label = CATEGORY_LABELS.get(cat_str, cat_str)
        if label not in seen:
            concepts.append(label)
            seen.add(label)

    doi = str(result.doi).strip() if result.doi else None
    url = str(result.pdf_url).strip() if result.pdf_url else f"https://arxiv.org/abs/{arxiv_id}"
    citation_count = 0  # arXiv API doesn't provide this

    return {
        "paper_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "primary_category": str(getattr(result, "primary_category", "") or ""),
        "year": year,
        "venue": venue,
        "authors": authors,
        "concepts": concepts,
        "doi": doi,
        "url": url,
        "citation_count": citation_count,
        "open_access": True,  # arXiv papers are OA
    }


def main():
    parser = argparse.ArgumentParser(description="Download CS papers from arXiv API")
    parser.add_argument(
        "--size", type=int, default=5000, help="Target number of papers (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "arxiv_cs_sample.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(CS_CATEGORIES),
        help="Comma-separated arXiv categories",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2024,
        help="Minimum publication year (default: 2024)",
    )
    parser.add_argument(
        "--strategy",
        choices=[COMBINED_STRATEGY, CATEGORY_BALANCED_STRATEGY],
        default=COMBINED_STRATEGY,
        help=(
            "Sampling strategy. Use category-balanced for corpora over 10,000 "
            "papers (default: combined)."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Inclusive corpus freeze date in YYYY-MM-DD format (default: UTC today)",
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.min_year <= 0:
        parser.error("--min-year must be positive")
    try:
        end_date = date.fromisoformat(args.end_date)
    except ValueError:
        parser.error("--end-date must use YYYY-MM-DD format")
    if end_date > datetime.now(timezone.utc).date():
        parser.error("--end-date must not be in the future")
    if args.min_year > end_date.year:
        parser.error("--min-year must not be later than --end-date")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    categories = [c.strip() for c in args.categories.split(",")]
    if not all(categories):
        parser.error("--categories must contain non-empty category names")
    if len(set(categories)) != len(categories):
        parser.error("--categories must not contain duplicates")
    if args.strategy == COMBINED_STRATEGY and args.size > ARXIV_QUERY_RESULT_LIMIT:
        parser.error(
            f"The combined arXiv query is limited to {ARXIV_QUERY_RESULT_LIMIT:,} "
            "results; use --strategy category-balanced for a larger corpus"
        )

    category_targets: dict[str, int] = {}
    if args.strategy == CATEGORY_BALANCED_STRATEGY:
        category_targets = _balanced_category_targets(categories, args.size)
        largest_target = max(category_targets.values())
        if largest_target > ARXIV_QUERY_RESULT_LIMIT:
            parser.error(
                "The requested corpus is too large for category-balanced API "
                f"pagination: largest category target is {largest_target:,}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing papers for dedup and resumable target-total semantics.
    try:
        existing_ids, category_counts = _load_existing_state(
            output_path,
            strategy=args.strategy,
            categories=categories,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if existing_ids:
        logger.info("Existing papers in output: %d", len(existing_ids))

    logger.info("Fetching from arXiv: %s", ", ".join(categories))
    logger.info("Target total: %d papers (min year: %d)", args.size, args.min_year)
    logger.info("Sampling strategy: %s", args.strategy)
    logger.info("Corpus freeze date: %s", end_date.isoformat())

    if len(existing_ids) >= args.size:
        logger.info(
            "Target already satisfied: %d papers in %s",
            len(existing_ids),
            output_path,
        )
        return

    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3)
    written = 0
    skipped_old = 0
    skipped_dup = 0
    skipped_crosslist = 0
    checked = 0

    if args.strategy == CATEGORY_BALANCED_STRATEGY:
        for category, count in category_counts.items():
            if count > category_targets[category]:
                parser.error(
                    f"Existing category {category} has {count:,} records, exceeding "
                    f"the current target of {category_targets[category]:,}"
                )
        query_plan = [
            (category, category_targets[category])
            for category in categories
            if category_targets[category] > 0
        ]
    else:
        category_query = " OR ".join(f"cat:{category}" for category in categories)
        query_plan = [(None, args.size)]

    with open(output_path, "a" if existing_ids else "w", encoding="utf-8") as f:
        try:
            for source_category, category_target in query_plan:
                accepted = (
                    category_counts[source_category]
                    if source_category is not None
                    else len(existing_ids)
                )
                if accepted >= category_target:
                    continue

                if source_category is not None:
                    logger.info(
                        "Category %s: %d/%d already present",
                        source_category,
                        accepted,
                        category_target,
                    )

                if source_category is not None:
                    search_queries = [
                        (
                            f"cat:{source_category} AND "
                            f"submittedDate:[{start} TO {end}]"
                        )
                        for start, end in _year_windows(args.min_year, end_date)
                    ]
                    max_results = ARXIV_QUERY_RESULT_LIMIT
                else:
                    search_queries = [category_query]
                    max_results = args.size * 3

                for query in search_queries:
                    if accepted >= category_target:
                        break
                    logger.info("Query window: %s", query)
                    search = arxiv.Search(
                        query=query,
                        max_results=max_results,
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                    )

                    for result in client.results(search):
                        checked += 1

                        if result.published and result.published.year < args.min_year:
                            skipped_old += 1
                            continue

                        paper = _map_to_paper(result)
                        if paper is None:
                            continue

                        if (
                            source_category is not None
                            and paper["primary_category"] != source_category
                        ):
                            skipped_crosslist += 1
                            continue

                        if paper["paper_id"] in existing_ids:
                            skipped_dup += 1
                            continue

                        if source_category is not None:
                            paper[DOWNLOAD_CATEGORY_FIELD] = source_category
                        f.write(json.dumps(paper, ensure_ascii=False) + "\n")
                        existing_ids.add(paper["paper_id"])
                        written += 1

                        if source_category is not None:
                            category_counts[source_category] += 1
                            accepted = category_counts[source_category]
                        else:
                            accepted = len(existing_ids)

                        if written % 100 == 0:
                            category_detail = (
                                f", {source_category}={accepted}/{category_target}"
                                if source_category is not None
                                else ""
                            )
                            logger.info(
                                "  %d written (checked %d, skipped %d old + %d dup + "
                                "%d crosslist%s)",
                                written,
                                checked,
                                skipped_old,
                                skipped_dup,
                                skipped_crosslist,
                                category_detail,
                            )

                        if accepted >= category_target or len(existing_ids) >= args.size:
                            break

                if accepted < category_target:
                    raise RuntimeError(
                        f"Category {source_category or 'combined'} produced only "
                        f"{accepted:,}/{category_target:,} unique accepted papers "
                        "across all configured date windows"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted. Progress saved.")

    logger.info(
        "Done! Wrote %d new papers; output now contains %d "
        "(checked %d, skipped %d old + %d dup + %d crosslist)",
        written,
        len(existing_ids),
        checked,
        skipped_old,
        skipped_dup,
        skipped_crosslist,
    )
    logger.info("Output: %s (%.0f KB)", output_path, output_path.stat().st_size / 1024)
    if len(existing_ids) < args.size:
        sys.exit(
            f"Target not reached: {len(existing_ids)}/{args.size} papers. "
            "The partial JSONL is preserved; re-run the same command to continue."
        )


if __name__ == "__main__":
    main()
