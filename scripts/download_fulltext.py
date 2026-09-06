"""
Download full-text papers from peS2o (open-access S2ORC derivative).

peS2o is a cleaned, filtered version of the Semantic Scholar Open Research
Corpus. Shards 10-19 contain full-text papers (s2orc source).

This script:
1. Streams one peS2o v2 training shard (already cached after first run)
2. Keeps papers with full text (source='s2orc', text > 500 chars)
3. Extracts title from first paragraph, abstract from second paragraph
4. Saves as JSONL

No external API needed — title and abstract are extracted directly from
the structured full text.

Usage:
    python scripts/download_fulltext.py --size 5000

Output format:
{
  "paper_id": "s2_211829000",
  "title": "Subfunctionalization of phytochrome B1/B2...",
  "abstract": "Gene duplication and polyploidization are...",
  "full_text": "Subfunctionalization...\\n\\nGene duplication...\\n\\n...",
  "year": null,
  "authors": [],
  "venue": null,
  "concepts": []
}
"""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# peS2o v2: shards 10-19 are s2orc (full text)
PES2O_REPO = "allenai/peS2o"
S2ORC_SHARD = 10  # first full-text shard


def _strip_abstract_prefix(text: str) -> str:
    """Remove common prefixes from abstract paragraphs."""
    for prefix in ("Abstract ", "Abstract:", "Abstract\n", "Abstract."):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def extract_metadata(full_text: str) -> dict:
    """Extract title and abstract from peS2o full text.

    peS2o text structure:
      paragraph 0 → title
      paragraph 1 → abstract (sometimes prefixed with 'Abstract')
      everything  → body
    """
    paragraphs = full_text.split("\n\n")

    title = paragraphs[0].strip() if paragraphs else ""

    abstract = ""
    if len(paragraphs) > 1:
        abstract = _strip_abstract_prefix(paragraphs[1].strip())

    return {
        "title": title,
        "abstract": abstract,
    }


def build_record(s2_id: str, full_text: str) -> dict:
    """Build a single output record from a peS2o paper."""
    meta = extract_metadata(full_text)
    return {
        "paper_id": f"s2_{s2_id}",
        "title": meta["title"],
        "abstract": meta["abstract"],
        "full_text": full_text,
        "year": None,
        "authors": [],
        "venue": None,
        "concepts": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Download full-text papers from peS2o")
    parser.add_argument("--size", type=int, default=5000,
                        help="Number of papers to download (default: 5000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--min-chars", type=int, default=500,
                        help="Minimum full-text length (default: 500)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        DATA_DIR / f"peS2o_fulltext_{args.size}.jsonl"
    )

    # Locate the shard in HF cache
    shard_name = f"train-{S2ORC_SHARD:05d}-of-00020.json.gz"
    cache_base = PROJECT_ROOT / "data" / ".hf_cache"

    # Try project-local cache first, then global HF cache
    local_path = cache_base / "datasets--allenai--peS2o" / "snapshots"
    global_path = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--allenai--peS2o" / "snapshots"

    shard_path = None
    for base in [local_path, global_path]:
        if base.exists():
            # Find any snapshot
            for snap in base.iterdir():
                candidate = snap / "data" / "v2" / shard_name
                if candidate.exists():
                    shard_path = candidate
                    break
        if shard_path:
            break

    if not shard_path:
        # Download it
        logger.info("Shard not cached; downloading...")
        try:
            from huggingface_hub import hf_hub_download
            shard_path = Path(hf_hub_download(
                PES2O_REPO, f"data/v2/{shard_name}", repo_type="dataset",
                cache_dir=str(cache_base),
            ))
        except Exception as e:
            logger.error("Download failed: %s", e)
            sys.exit(1)

    logger.info("Reading shard: %s", shard_path)

    collected = []
    scanned = 0
    fulltext_seen = 0

    with gzip.open(shard_path) as f:
        for line in f:
            scanned += 1
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not paper.get("source", "").startswith("s2orc"):
                continue

            full_text = paper.get("text", "")
            if len(full_text) < args.min_chars:
                continue

            fulltext_seen += 1
            record = build_record(str(paper["id"]), full_text)
            collected.append(record)

            if len(collected) % 500 == 0:
                logger.info("  collected %d / %d papers...", len(collected), args.size)

            if len(collected) >= args.size:
                break

            if scanned % 200_000 == 0:
                logger.info("  scanned %d, full-text: %d, collected: %d",
                            scanned, fulltext_seen, len(collected))

    with open(output_path, "w", encoding="utf-8") as f:
        for record in collected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done! %d papers saved to %s", len(collected), output_path)
    logger.info("  Scanned: %d, full-text found: %d", scanned, fulltext_seen)


if __name__ == "__main__":
    main()
