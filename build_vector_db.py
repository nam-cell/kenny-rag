"""
Build a ChromaDB vector database from Kenny Robinson articles.
Fetches articles, saves raw text, chunks, and indexes into ChromaDB.
"""

import os
import re
import time
import logging
import requests
from bs4 import BeautifulSoup, Comment
import chromadb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "kenny_robinson_corpus", "raw")
CHROMA_DIR = os.path.join(BASE_DIR, "kenny_robinson_corpus", "chromadb")
COLLECTION_NAME = "kenny_robinson"

CHUNK_SIZE = 500       # approximate tokens per chunk
CHUNK_OVERLAP = 50     # approximate token overlap

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

URLS = [
    ("comedy_green_room", "https://www.thecomedygreenroom.com/post/kenny-robinson"),
    ("now_toronto", "https://nowtoronto.com/culture/40-at-40-kenny-robinson-godfather-of-canadian-comedy/"),
    ("original_cin", "https://www.original-cin.ca/posts/2025/4/8/original-cin-chat-nubian-show-founder-kenny-robinson-on-30-years-of-comedy-and-a-crave-debut"),
    ("caribbean_camera", "https://thecaribbeancamera.com/kenny-robinson-nubian-show-30-years/"),
    ("globe_and_mail_1", "https://www.theglobeandmail.com/news/toronto/kenny-robinson-comedian/article565003/"),
    ("wikipedia", "https://en.wikipedia.org/wiki/Kenny_Robinson_(comedian)"),
    ("kenny_robinson_bio", "https://www.kennyrobinson.com/bio.html"),
    ("imdb", "https://www.imdb.com/name/nm0732811/"),
    ("cbc_elbows_up", "https://www.cbc.ca/arts/q/kenny-robinson-is-elbows-up-for-canadian-comedians-1.7503763"),
    ("cbc_godfather", "https://www.cbc.ca/comedy/the-godfather-of-comedy-kenny-robinson-celebrates-25-years-of-his-successful-nubian-comedy-revue-1.5804743"),
    ("deadline_eh_list", "https://deadline.com/2025/07/eh-list-standup-specials-kenny-robinson-new-metric-media-1236464843/"),
    ("globe_and_mail_2", "https://www.theglobeandmail.com/arts/colour-him-funny/article4152789/"),
    ("comedy_history_101", "http://www.comedyhistory101.com/comedy-history-101/2025/6/6/the-def-jam-comedy-of-canada-the-nubian-show"),
    ("parton_and_pearl", "https://www.partonandpearl.com/blog/new-doc-people-of-comedy-premieres-april-9"),
    ("playback_online", "https://playbackonline.ca/2025/07/23/new-metric-media-introduces-eh-list-standup-comedy-slate/"),
    ("ted_woloshyn", "https://www.tedwoloshyn.ca/season-4-episode-28-kenny-robinson/"),
    ("funny_business", "https://www.funnybusiness.ca/comedians.php?standup=kenny-robinson"),
]


# ── Extraction ─────────────────────────────────────────────────────────────────

STRIP_TAGS = [
    "nav", "header", "footer", "aside", "script", "style", "noscript",
    "iframe", "form", "button", "svg",
]


def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove boilerplate tags and comments from soup."""
    # Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    # Remove unwanted tags
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    return soup


def _get_text_from_container(container) -> str:
    """Extract text from a container, preferring structured elements then fallback."""
    # Try structured extraction first
    lines = []
    for el in container.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "td", "dd", "dt", "span", "div"]):
        text = el.get_text(separator=" ", strip=True)
        # Only keep non-trivial text that isn't just a single word/label
        if text and len(text) > 20:
            lines.append(text)

    if lines:
        # Deduplicate (parent elements contain child text)
        seen = set()
        deduped = []
        for line in lines:
            # Skip if this line is a substring of something already added
            if any(line in s for s in seen):
                continue
            # Remove previously added lines that are substrings of this one
            deduped = [s for s in deduped if s not in line]
            seen = {s for s in seen if s not in line}
            deduped.append(line)
            seen.add(line)
        text = "\n\n".join(deduped)
    else:
        # Fallback: just get all text from the container
        text = container.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_article_text(html: str, url: str) -> str:
    """Extract main article content from HTML, stripping boilerplate."""
    soup = BeautifulSoup(html, "lxml")
    soup = _clean_soup(soup)

    # Try to find the main content container (site-specific first, then generic)
    container = None

    # Wikipedia
    if "wikipedia.org" in url:
        container = soup.find("div", id="mw-content-text")

    # Squarespace-based sites (original-cin, comedygreenroom, comedyhistory101, partonandpearl)
    if not container:
        container = soup.find("div", class_=re.compile(r"sqs-block-content", re.I))
    if not container:
        container = soup.find("div", class_=re.compile(r"blog-item-content", re.I))

    # Generic article/main patterns
    if not container:
        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"role": "main"})
            or soup.find("div", class_=re.compile(r"(article[_-]?body|article[_-]?content|post[_-]?content|entry[_-]?content|story[_-]?body)", re.I))
            or soup.find("div", id=re.compile(r"(article|content|post|entry|story)", re.I))
            or soup.find("div", class_=re.compile(r"(article|content|post|entry|story)", re.I))
        )

    # Last resort: body
    if not container:
        container = soup.body or soup

    return _get_text_from_container(container)


# ── Fetching ───────────────────────────────────────────────────────────────────

def fetch_article(name: str, url: str) -> str | None:
    """Fetch a single URL and extract article text. Returns None on failure."""
    try:
        logger.info(f"Fetching: {name} -> {url}")
        resp = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        text = extract_article_text(resp.text, url)
        char_count = len(text)
        if char_count < 50:
            logger.warning(f"  Short content ({char_count} chars) — may be JS-rendered/paywalled")
        else:
            logger.info(f"  Extracted {char_count:,} chars")
        return text if char_count > 0 else None
    except requests.RequestException as e:
        logger.error(f"  FAILED {name}: {e}")
        return None


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of ~chunk_size tokens with ~overlap token overlap."""
    words = text.split()
    if not words:
        return []

    # Convert token counts to approximate word counts (~1.33 tokens per word)
    words_per_chunk = int(chunk_size / 1.33)
    words_overlap = int(overlap / 1.33)

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


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Step 1: Fetch articles
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching articles")
    logger.info("=" * 60)

    fetched = {}
    failed = []

    for name, url in URLS:
        text = fetch_article(name, url)
        if text:
            fetched[name] = {"url": url, "text": text}
        else:
            failed.append((name, url))
        time.sleep(1)  # polite delay

    # Step 2: Save raw text files
    logger.info("=" * 60)
    logger.info("STEP 2: Saving raw text files")
    logger.info("=" * 60)

    for name, data in fetched.items():
        filepath = os.path.join(RAW_DIR, f"{name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {data['url']}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(data["text"])
        logger.info(f"  Saved: {name}.txt ({len(data['text']):,} chars)")

    # Step 3: Chunk all texts
    logger.info("=" * 60)
    logger.info("STEP 3: Chunking texts")
    logger.info("=" * 60)

    all_chunks = []

    for name, data in fetched.items():
        chunks = chunk_text(data["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{name}_chunk_{i:04d}",
                "text": chunk,
                "metadata": {
                    "source_url": data["url"],
                    "source_name": name,
                    "chunk_index": i,
                },
            })
        logger.info(f"  {name}: {len(chunks)} chunks")

    # Step 4: Build ChromaDB
    logger.info("=" * 60)
    logger.info("STEP 4: Building ChromaDB vector database")
    logger.info("=" * 60)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present to rebuild cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("  Deleted existing collection for clean rebuild")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Kenny Robinson comedy corpus"},
    )

    # Add in batches (ChromaDB handles embedding via default function)
    BATCH_SIZE = 100
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        logger.info(f"  Added batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    # Step 5: Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total URLs attempted:  {len(URLS)}")
    logger.info(f"  Successfully fetched:  {len(fetched)}")
    logger.info(f"  Failed:                {len(failed)}")
    logger.info(f"  Total chunks created:  {len(all_chunks)}")
    logger.info(f"  ChromaDB collection:   {collection.count()} documents")
    logger.info(f"  ChromaDB path:         {CHROMA_DIR}")

    if failed:
        logger.info("  Failed URLs:")
        for name, url in failed:
            logger.info(f"    - {name}: {url}")

    # Quick sanity check: query the collection
    if collection.count() > 0:
        logger.info("")
        logger.info("Sanity check — querying 'Nubian Show comedy':")
        results = collection.query(query_texts=["Nubian Show comedy"], n_results=3)
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            logger.info(f"  Result {i+1} [{meta['source_name']}]: {doc[:120]}...")


if __name__ == "__main__":
    main()
