"""
Ingestion pipeline for General Conference Parent-Child RAG.

What this script does
---------------------
1) Loads talks from a CSV file (expects columns like: title, speaker, year, season, url, text).
2) Standardizes metadata fields.
3) Splits each talk into:
   - Parent chunks (~1536 tokens, context-rich), prefixed with metadata header
   - Child chunks (~256 tokens, search-optimized), metadata-aware for header budget
4) Embeds child chunks with OpenAI embeddings.
5) Upserts child vectors into Pinecone.
6) Exports parent chunks to local JSON (free "docstore" for recursive retrieval).

Why this structure
------------------
- Child chunks are better for retrieval precision.
- Parent chunks preserve context for final answer generation.
- Keeping parent text in JSON avoids managed DB costs.

Environment variables required
------------------------------
- OPENAI_API_KEY
- PINECONE_API_KEY

Optional (used by app.py for querying, not by this script)
---------------------------------------------------------
- COHERE_API_KEY

Optional environment variables
------------------------------
- PINECONE_INDEX (default: conference-rag)
- PINECONE_NAMESPACE (default: talks-v1)
- EMBEDDING_MODEL (default: text-embedding-3-small)

Usage examples
--------------
python ingest.py
python ingest.py --input-csv SCRAPED_TALKS.csv --max-talks 500
python ingest.py --namespace talks-v2 --parent-store storage/parent_store_v2.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core.node_parser import TokenTextSplitter
from openai import OpenAI
from pinecone import Pinecone


@dataclass
class IngestConfig:
    """Runtime configuration for ingestion."""

    input_csv: Path = Path("SCRAPED_TALKS.csv")
    parent_store_path: Path = Path("storage/parent_store.json")
    embedding_model: str = "text-embedding-3-small"
    pinecone_index: str = "conference-rag"
    pinecone_namespace: str = "talks-v1"
    parent_chunk_size: int = 1536
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 256
    child_chunk_overlap: int = 50
    embedding_batch_size: int = 128
    upsert_batch_size: int = 200
    max_talks: int | None = None


def normalize_whitespace(text: str) -> str:
    """Collapse excessive spaces while preserving paragraph boundaries."""
    if not text:
        return ""
    # Normalize line endings first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep paragraph breaks intact, but clean each paragraph.
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)


def normalize_metadata(row: dict[str, Any]) -> dict[str, str]:
    """
    Standardize metadata from CSV row into canonical fields.

    The scraper already outputs these fields, but this function guards against
    missing keys and accidental whitespace/typing drift.
    """
    title = str(row.get("title", "")).strip()
    speaker = str(row.get("speaker", "")).strip()
    year = str(row.get("year", "")).strip()
    season = str(row.get("season", "")).strip()
    url = str(row.get("url", "")).strip()
    text = normalize_whitespace(str(row.get("text", "")))

    return {
        "title": title,
        "speaker": speaker,
        "year": year,
        "season": season,
        "url": url,
        "text": text,
    }


def stable_talk_id(url: str, title: str, speaker: str, year: str, season: str) -> str:
    """
    Build a stable talk ID.

    Preference:
    1) URL slug (human-readable, stable)
    2) hash fallback if URL is missing
    """
    if url:
        slug = url.rstrip("/").split("/")[-1]
        slug = slug.split("?")[0]
        slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", slug).strip("-").lower()
        if slug:
            return f"{year}-{season.lower()}-{slug}"

    raw = f"{title}|{speaker}|{year}|{season}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"talk-{digest}"


def format_talk_header(speaker: str, year: str, season: str, title: str) -> str:
    """
    Build the metadata prefix used for parent text and child embedding text.

    Example:
    [Speaker: Thomas S. Monson, Date: Oct 1994, Title: The Path to Peace]
    """
    season_clean = season.strip().lower()
    if season_clean == "october":
        month_token = "Oct"
    elif season_clean == "april":
        month_token = "Apr"
    else:
        # Fallback for unexpected season values.
        month_token = season.strip()[:3] if season.strip() else "Unknown"
    return f"[Speaker: {speaker}, Date: {month_token} {year}, Title: {title}]"


def batched(items: list[Any], size: int) -> list[list[Any]]:
    """Return fixed-size list batches."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_talk_rows(input_csv: Path, max_talks: int | None = None) -> list[dict[str, str]]:
    """
    Load talks from CSV.

    Uses csv.DictReader (not pandas) to safely handle multiline quoted text.
    """
    talks: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            normalized = normalize_metadata(row)
            if not normalized["text"]:
                continue
            talks.append(normalized)
            if max_talks and idx >= max_talks:
                break
    return talks


def build_parent_child_records(
    talks: list[dict[str, str]],
    cfg: IngestConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split talks into parent and child records.

    Returns:
      parents: records saved locally in JSON for recursive retrieval.
      children: records embedded + upserted to Pinecone.
    """
    parent_splitter = TokenTextSplitter(
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=cfg.parent_chunk_overlap,
    )
    child_splitter = TokenTextSplitter(
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.child_chunk_overlap,
    )

    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    for talk in talks:
        talk_id = stable_talk_id(
            url=talk["url"],
            title=talk["title"],
            speaker=talk["speaker"],
            year=talk["year"],
            season=talk["season"],
        )
        parent_chunks = parent_splitter.split_text(talk["text"])
        header = format_talk_header(
            speaker=talk["speaker"],
            year=talk["year"],
            season=talk["season"],
            title=talk["title"],
        )

        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"{talk_id}::p{p_idx:03d}"
            parent_record = {
                "parent_id": parent_id,
                "talk_id": talk_id,
                "title": talk["title"],
                "speaker": talk["speaker"],
                "year": talk["year"],
                "season": talk["season"],
                "url": talk["url"],
                "text": f"{header}\n{parent_text}",
            }
            parents.append(parent_record)

            # LlamaIndex metadata-aware split: reserve token budget for metadata
            # so the final "metadata + child_chunk" text stays near child_chunk_size.
            child_chunks = child_splitter.split_text_metadata_aware(parent_text, header)
            for c_idx, child_text in enumerate(child_chunks):
                child_id = f"{parent_id}::c{c_idx:03d}"
                child_record = {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "talk_id": talk_id,
                    "title": talk["title"],
                    "speaker": talk["speaker"],
                    "year": talk["year"],
                    "season": talk["season"],
                    "url": talk["url"],
                    "text": child_text,
                    # This field is used for embeddings and prepends metadata context.
                    # The chunk text is created with split_text_metadata_aware above.
                    "embedding_text": f"{header}\n{child_text}",
                }
                children.append(child_record)

    return parents, children


def embed_child_batch(
    openai_client: OpenAI,
    model: str,
    child_batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Embed a child batch and convert into Pinecone vector payload objects.
    """
    texts = [item["embedding_text"] for item in child_batch]
    response = openai_client.embeddings.create(
        model=model,
        input=texts,
    )

    vectors: list[dict[str, Any]] = []
    for item, emb in zip(child_batch, response.data):
        vectors.append(
            {
                "id": item["child_id"],
                "values": emb.embedding,
                "metadata": {
                    "parent_id": item["parent_id"],
                    "talk_id": item["talk_id"],
                    "title": item["title"],
                    "speaker": item["speaker"],
                    "year": item["year"],
                    "season": item["season"],
                    "url": item["url"],
                    # Store raw child text for direct evidence display if needed.
                    "text": item["text"],
                },
            }
        )
    return vectors


def upsert_children(
    children: list[dict[str, Any]],
    cfg: IngestConfig,
    openai_client: OpenAI,
    pinecone_index: Any,
) -> None:
    """
    Embed and upsert children in batches.

    This keeps API calls and payload sizes bounded for reliability and cost control.
    """
    if not children:
        print("No child records found; skipping Pinecone upsert.")
        return

    total = len(children)
    print(f"Embedding/upserting {total} child chunks...")

    embedded_count = 0
    started = time.time()
    for embed_batch in batched(children, cfg.embedding_batch_size):
        vectors = embed_child_batch(openai_client, cfg.embedding_model, embed_batch)
        for upsert_batch in batched(vectors, cfg.upsert_batch_size):
            pinecone_index.upsert(vectors=upsert_batch, namespace=cfg.pinecone_namespace)
        embedded_count += len(embed_batch)

        elapsed = time.time() - started
        print(f"  Progress: {embedded_count}/{total} children embedded+upserted ({elapsed:.1f}s)")


def write_parent_store(parents: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save parent chunk records to local JSON docstore.

    We save as an object keyed by parent_id for quick lookup during retrieval.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = {p["parent_id"]: p for p in parents}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"Wrote {len(parents)} parent records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest General Conference talks into Parent-Child RAG stores.")
    parser.add_argument("--input-csv", default="SCRAPED_TALKS.csv", help="Path to input talks CSV.")
    parser.add_argument("--parent-store", default="storage/parent_store.json", help="Path to parent JSON store output.")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX", "conference-rag"), help="Pinecone index name.")
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", "talks-v1"),
        help="Pinecone namespace for child vectors.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model name.",
    )
    parser.add_argument("--max-talks", type=int, default=None, help="Optional cap for test runs.")
    parser.add_argument("--parent-size", type=int, default=1536, help="Parent chunk token size.")
    parser.add_argument("--parent-overlap", type=int, default=150, help="Parent chunk overlap tokens.")
    parser.add_argument("--child-size", type=int, default=256, help="Child chunk token size.")
    parser.add_argument("--child-overlap", type=int, default=50, help="Child chunk overlap tokens.")
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Number of children per embedding request.")
    parser.add_argument("--upsert-batch-size", type=int, default=200, help="Number of vectors per Pinecone upsert.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY. Add it to your environment or .env file.")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY. Add it to your environment or .env file.")

    args = parse_args()
    cfg = IngestConfig(
        input_csv=Path(args.input_csv),
        parent_store_path=Path(args.parent_store),
        embedding_model=args.embedding_model,
        pinecone_index=args.index_name,
        pinecone_namespace=args.namespace,
        parent_chunk_size=args.parent_size,
        parent_chunk_overlap=args.parent_overlap,
        child_chunk_size=args.child_size,
        child_chunk_overlap=args.child_overlap,
        embedding_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        max_talks=args.max_talks,
    )

    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {cfg.input_csv}")

    print("Loading talks...")
    talks = load_talk_rows(cfg.input_csv, max_talks=cfg.max_talks)
    print(f"Loaded talks: {len(talks)}")

    print("Building parent/child chunks...")
    parents, children = build_parent_child_records(talks, cfg)
    print(f"Parent chunks: {len(parents)}")
    print(f"Child chunks: {len(children)}")

    print("Connecting to OpenAI + Pinecone...")
    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pinecone_client.Index(cfg.pinecone_index)

    upsert_children(
        children=children,
        cfg=cfg,
        openai_client=openai_client,
        pinecone_index=pinecone_index,
    )
    write_parent_store(parents, cfg.parent_store_path)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
