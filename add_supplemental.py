"""
Add supplemental text files to the existing Kenny Robinson ChromaDB vector database.
Handles replacements (original_cin, imdb) and new additions (comedy_history_101, google_doc_catalog).
"""

import os
import re
import logging
import chromadb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPLEMENTAL_DIR = os.path.join(BASE_DIR, "kenny_robinson_corpus", "supplemental_corpus")
CHROMA_DIR = os.path.join(BASE_DIR, "kenny_robinson_corpus", "chromadb")
COLLECTION_NAME = "kenny_robinson"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# filename -> (source_name, replaces_existing_source_name_or_None)
FILES = {
    "comedy_history_101.txt": ("comedy_history_101", None),           # new — previously failed fetch
    "original_cin_full.txt":  ("original_cin", "original_cin"),       # replaces thin 576-char version
    "imdb_bio_full.txt":      ("imdb_bio", "imdb"),                   # replaces thin 26-char version
    "google_doc_catalog.txt": ("google_doc_catalog", None),           # entirely new source
}


def parse_file(filepath: str) -> tuple[str, str]:
    """Read a supplemental file and extract its URL + body text.
    Returns (source_url, body_text).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    source_url = ""
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("URL:"):
            source_url = stripped.split(":", 1)[1].strip()
        # Body starts after the first blank line following the header block
        if stripped == "" and i > 0:
            body_start = i + 1
            break

    body = "".join(lines[body_start:]).strip()
    return source_url, body


def chunk_text(text: str) -> list[str]:
    """Split text into chunks of ~CHUNK_SIZE tokens with ~CHUNK_OVERLAP token overlap."""
    words = text.split()
    if not words:
        return []

    words_per_chunk = int(CHUNK_SIZE / 1.33)
    words_overlap = int(CHUNK_OVERLAP / 1.33)

    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - words_overlap
        if start >= len(words):
            break

    return chunks


def main():
    # Open existing ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    count_before = collection.count()
    logger.info(f"Opened collection '{COLLECTION_NAME}' — {count_before} existing documents")

    # Gather IDs of sources being replaced
    existing = collection.get(include=["metadatas"])
    sources_to_replace = {v[1] for v in FILES.values() if v[1] is not None}
    ids_to_delete = [
        id for id, m in zip(existing["ids"], existing["metadatas"])
        if m["source_name"] in sources_to_replace
    ]

    if ids_to_delete:
        logger.info(f"Deleting {len(ids_to_delete)} old chunks for replaced sources: {sorted(sources_to_replace)}")
        collection.delete(ids=ids_to_delete)
        logger.info(f"  Deleted IDs: {ids_to_delete}")

    # Process each supplemental file
    total_added = 0

    for filename, (source_name, _replaces) in FILES.items():
        filepath = os.path.join(SUPPLEMENTAL_DIR, filename)

        if not os.path.exists(filepath):
            logger.error(f"  File not found: {filepath} — skipping")
            continue

        source_url, body = parse_file(filepath)
        if not body:
            logger.warning(f"  Empty body in {filename} — skipping")
            continue

        chunks = chunk_text(body)

        ids = [f"{source_name}_chunk_{i:04d}" for i in range(len(chunks))]
        metadatas = [
            {"source_url": source_url, "source_name": source_name, "chunk_index": i}
            for i in range(len(chunks))
        ]

        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        total_added += len(chunks)
        logger.info(f"  {filename} -> {source_name}: {len(chunks)} chunks added ({len(body):,} chars)")

    # Summary
    count_after = collection.count()
    logger.info("")
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Old chunks deleted:   {len(ids_to_delete)}")
    logger.info(f"  New chunks added:     {total_added}")
    logger.info(f"  Collection before:    {count_before}")
    logger.info(f"  Collection after:     {count_after}")
    logger.info(f"  Net change:           +{count_after - count_before}")


if __name__ == "__main__":
    main()
