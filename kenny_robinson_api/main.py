"""
Kenny Robinson RAG Bot — Telegram + FastAPI health + ChromaDB + Anthropic
Single-process: polling mode (no webhook/SSL needed)
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

import chromadb
import httpx
from fastapi import FastAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
import uvicorn

# --- Config ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromadb")
COLLECTION_NAME = "kenny_robinson"
N_RESULTS = 5
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8042"))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("kenny-rag")

# --- ChromaDB ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
logger.info(f"ChromaDB loaded: {collection.count()} chunks in '{COLLECTION_NAME}'")

# --- System Prompt ---
SYSTEM_PROMPT = """You are a knowledgeable research assistant specializing in Kenny Robinson, \
the Canadian comedian known as "The Godfather of Canadian Comedy."

Answer questions using ONLY the provided context chunks. If the context doesn't contain enough \
information to fully answer, say what you can and note what's missing. Always cite which source(s) \
you're drawing from.

Key facts for reference:
- Born January 7, 1958, Winnipeg, Manitoba (grew up in Transcona)
- Founded the Nubian Disciples of Pryor (now Nubian Comedy Revue) in 1995 at Yuk Yuk's Toronto
- The show runs last Sunday of every month
- Discovered Russell Peters (via Joe Bodolai seeing Peters at a Nubian show)
- Won Phil Hartman Award (2014)
- Signed with New Metric Media (July 2025) for standup special + album
- Documentary "People of Comedy" streaming on Crave (premiered April 9, 2025)
- Runs Monday workshop at Blackhurst Cultural Centre
- Influences: Richard Pryor, George Carlin

Format your answers for Telegram: use plain text, keep it concise, and use line breaks for readability. \
Do not use markdown headers or bullet points — use dashes if listing items."""


# --- RAG Pipeline ---
async def retrieve_chunks(question: str) -> list[dict]:
    """Query ChromaDB for relevant chunks."""
    results = collection.query(
        query_texts=[question],
        n_results=N_RESULTS,
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source_name": meta.get("source_name", ""),
            "source_url": meta.get("source_url", ""),
            "distance": round(dist, 4),
        })
    return chunks


async def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call Anthropic API with retrieved context."""
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source_name']}]\n{c['text']}" for c in chunks
    )

    user_message = f"""Context chunks (from verified sources):

{context_block}

Question: {question}"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


# --- Telegram Handlers ---
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    first_name = update.effective_user.first_name or "there"
    keyboard = [
        [
            InlineKeyboardButton("How did Kenny get started?", callback_data="q:How did Kenny get started in comedy?"),
            InlineKeyboardButton("What's the Nubian show?", callback_data="q:What is the Nubian Comedy Revue?"),
        ],
        [
            InlineKeyboardButton("Who has he worked with?", callback_data="q:Who has Kenny Robinson worked with and mentored?"),
            InlineKeyboardButton("What awards has he won?", callback_data="q:What awards has Kenny Robinson won?"),
        ],
        [
            InlineKeyboardButton("What should I watch first?", callback_data="q:What Kenny Robinson content should I watch first?"),
            InlineKeyboardButton("📚 Resources & Links", callback_data="cmd:sources"),
        ],
    ]
    await update.message.reply_text(
        f"Hey {first_name}! 👋 I'm loaded up with research on Kenny Robinson — "
        f"the Godfather of Canadian Comedy himself.\n\n"
        f"Ask me anything about his background, career, the Nubian show, "
        f"his mentorship style, who he's worked with, or how to make the most "
        f"of this connection.\n\n"
        f"You can also use /sources to browse all the verified links.\n\n"
        f"This is a huge opportunity — what do you want to know?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()

    # Handle command buttons
    if query.data == "cmd:sources":
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        seen = {}
        for m in all_meta:
            name = m.get("source_name", "unknown")
            url = m.get("source_url", "")
            if name not in seen and url:
                seen[name] = url
        text = f"📚 Knowledge Base — {len(seen)} verified sources\n\n"
        for name, url in sorted(seen.items()):
            text += f"- {name}\n  {url}\n\n"
        if len(text) > 4096:
            text = text[:4090] + "..."
        await query.message.reply_text(text, disable_web_page_preview=True)
        return

    # Handle question buttons
    question = query.data.removeprefix("q:")
    # Send a placeholder so user knows we're working
    thinking_msg = await query.message.reply_text("🔍 Searching Kenny Robinson research...")
    try:
        chunks = await retrieve_chunks(question)
        if not chunks:
            await thinking_msg.edit_text("I couldn't find relevant info. Try rephrasing?")
            return
        answer = await generate_answer(question, chunks)
        source_names = sorted(set(c["source_name"] for c in chunks))
        source_line = "\n\n📚 Sources: " + ", ".join(source_names)
        full_reply = answer + source_line
        if len(full_reply) > 4096:
            full_reply = full_reply[:4090] + "..."
        await thinking_msg.edit_text(full_reply)
    except Exception as e:
        logger.error(f"Button RAG error: {e}", exc_info=True)
        await thinking_msg.edit_text("Sorry, something went wrong. Please try again.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Just send me a question about Kenny Robinson.\n\n"
        "Examples:\n"
        "- How did Kenny discover Russell Peters?\n"
        "- What is the Nubian Comedy Revue?\n"
        "- What awards has Kenny won?\n"
        "- Tell me about Kenny's early career\n\n"
        "/sources — list all sources in the knowledge base\n"
        "/stats — show database stats"
    )


async def cmd_sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all unique sources with URLs."""
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    seen = {}
    for m in all_meta:
        name = m.get("source_name", "unknown")
        url = m.get("source_url", "")
        if name not in seen and url:
            seen[name] = url
    text = f"📚 Knowledge Base — {len(seen)} verified sources\n\n"
    for name, url in sorted(seen.items()):
        text += f"- {name}\n  {url}\n\n"
    # Telegram limit
    if len(text) > 4096:
        text = text[:4090] + "..."
    await update.message.reply_text(text, disable_web_page_preview=True)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = collection.count()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = set(m.get("source_name", "unknown") for m in all_meta)
    await update.message.reply_text(
        f"Chunks: {count}\n"
        f"Sources: {len(sources)}\n"
        f"Model: {ANTHROPIC_MODEL}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main RAG flow: retrieve → generate → reply."""
    question = update.message.text.strip()
    if not question:
        return

    # Send a placeholder so user knows we're working
    thinking_msg = await update.message.reply_text("🔍 Searching Kenny Robinson research...")

    try:
        chunks = await retrieve_chunks(question)
        if not chunks:
            await thinking_msg.edit_text(
                "I couldn't find any relevant information in my sources. "
                "Try rephrasing your question?"
            )
            return

        answer = await generate_answer(question, chunks)

        # Append source attribution
        source_names = sorted(set(c["source_name"] for c in chunks))
        source_line = "\n\n📚 Sources: " + ", ".join(source_names)

        # Telegram message limit is 4096 chars
        full_reply = answer + source_line
        if len(full_reply) > 4096:
            full_reply = full_reply[:4090] + "..."

        await thinking_msg.edit_text(full_reply)

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        await thinking_msg.edit_text(
            "Sorry, something went wrong processing your question. Please try again."
        )


# --- FastAPI Health Server ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

health_app = FastAPI(title="Kenny RAG Health", lifespan=lifespan)


@health_app.get("/health")
async def health():
    return {
        "status": "ok",
        "collection_count": collection.count(),
        "model": ANTHROPIC_MODEL,
    }


# --- Entrypoint ---
async def main():
    """Run Telegram bot (polling) + FastAPI health server concurrently."""
    # Build Telegram bot
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("sources", cmd_sources))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting Kenny RAG bot (polling mode)...")

    # Start bot polling
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    # Start health server
    config = uvicorn.Config(health_app, host="0.0.0.0", port=HEALTH_PORT, log_level="warning")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
