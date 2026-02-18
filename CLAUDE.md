# Kenny Robinson Research Bot

## Product
AI-powered Telegram RAG bot for Kenny Robinson ("The Godfather of Canadian Comedy").
ChromaDB vector search → Anthropic API → sourced answers via Telegram.

## Owner
Nam (product) · Dhonam (deployment/review)

## Architecture
```
User → Telegram Bot → main.py (polling)
                       ├── ChromaDB (50 chunks, 18 sources)
                       └── Anthropic API (claude-sonnet-4)
                       └── FastAPI /health (port 8042)
```

## Run Locally
```bash
cp .env.example .env   # fill in tokens
docker compose up --build
```

## Deploy to VPS
```bash
bash deploy.sh
```

## Bot Commands
- `/start` — welcome message
- `/help` — usage examples
- `/sources` — list knowledge base sources
- `/stats` — chunk count, model info

## Key Paths
| Item | Path |
|------|------|
| Bot + API source | `kenny_robinson_api/main.py` |
| Vector DB | `kenny_robinson_api/chromadb/` |
| Raw corpus | `kenny_robinson_corpus/raw/` + `supplemental_corpus/` |
| Build scripts | `build_vector_db.py`, `add_supplemental.py` |
| Docker config | `docker-compose.yml` + `kenny_robinson_api/Dockerfile` |

## Adding Sources
1. Save `.txt` to `kenny_robinson_corpus/supplemental_corpus/`
2. Run `add_supplemental.py` to ingest into ChromaDB
3. Copy updated `chromadb/` → `kenny_robinson_api/chromadb/`
4. Rebuild: `docker compose up --build`

## Env Vars (required)
- `TELEGRAM_BOT_TOKEN` — from @BotFather
- `ANTHROPIC_API_KEY` — Anthropic API key
- `ANTHROPIC_MODEL` — default: claude-sonnet-4-20250514
- `HEALTH_PORT` — default: 8042

## Conventions
- Docker required (no bare installs on VPS)
- 3-stage pipeline: Wireframe → Nam + Claude → Dhonam Review
