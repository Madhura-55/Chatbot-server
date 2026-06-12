"""
Policy Ingestion Script

Reads markdown policy/FAQ files from data/policies/, chunks them,
generates embeddings via Gemini, and upserts them into ChromaDB.

Run this script:
- Once initially to populate the vector store
- Again whenever policy documents are added or updated

Usage:
    python scripts/ingest_policies.py [--reset]
"""

import argparse
import re
from pathlib import Path

from loguru import logger

from config import get_settings
from services import VectorStoreService, GeminiService


POLICIES_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


def chunk_markdown(text: str, source_title: str, max_chars: int = 1000) -> list[dict]:
    """
    Split a markdown document into chunks by section headers (## ...).

    Each chunk includes the section heading for context and metadata
    pointing back to the source document and section title.
    """
    # Split on level-2 headers, keeping the header with its content
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section title (first line)
        first_line = section.split("\n", 1)[0]
        section_title = first_line.lstrip("# ").strip() or source_title

        # Further split overly long sections by paragraph
        if len(section) <= max_chars:
            chunks.append({"text": section, "title": f"{source_title} - {section_title}"})
        else:
            paragraphs = section.split("\n\n")
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) > max_chars and buffer:
                    chunks.append({"text": buffer.strip(), "title": f"{source_title} - {section_title}"})
                    buffer = ""
                buffer += para + "\n\n"
            if buffer.strip():
                chunks.append({"text": buffer.strip(), "title": f"{source_title} - {section_title}"})

    return chunks


def load_all_policy_chunks() -> list[dict]:
    """Load and chunk all markdown files in data/policies/."""
    all_chunks = []

    for file_path in sorted(POLICIES_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        # Use the top-level H1 as the source title, fallback to filename
        h1_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        source_title = h1_match.group(1).strip() if h1_match else file_path.stem

        chunks = chunk_markdown(text, source_title)
        for chunk in chunks:
            chunk["source_file"] = file_path.name
        all_chunks.extend(chunks)

        logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest policy documents into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing documents before ingesting (use for full re-index)",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Please configure your .env file.")

    logger.info("Initializing services...")
    gemini = GeminiService()
    vector_store = VectorStoreService()
    vector_store.connect()

    if args.reset:
        logger.info("Resetting vector store (--reset flag passed)...")
        vector_store.delete_all()

    logger.info("Loading and chunking policy documents...")
    chunks = load_all_policy_chunks()

    if not chunks:
        logger.warning(f"No policy documents found in {POLICIES_DIR}")
        return

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    ids = []
    documents = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{chunk['source_file']}::{idx}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "title": chunk["title"],
            "source_file": chunk["source_file"],
        })

    embeddings = gemini.embed_batch(documents, task_type="retrieval_document")

    vector_store.upsert_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Ingestion complete. Vector store now has {vector_store.count()} documents.")


if __name__ == "__main__":
    main()
