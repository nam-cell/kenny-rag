# Kenny Robinson RAG — Session Context

## Product
Telegram bot (@KRobin_bot) that answers questions about Canadian comedian Kenny Robinson using RAG (Retrieval-Augmented Generation) over a curated corpus of articles and interviews.

## Status: ✅ DEPLOYED
- **VPS:** 143.14.200.238 (user: namd)
- **Path:** /opt/kenny-rag
- **Container:** kenny-rag-bot
- **Health:** http://localhost:8042/health
- **Bot:** @KRobin_bot on Telegram

## Architecture
- **Frontend:** Telegram bot (polling mode, no webhook/SSL needed)
- **Backend:** Python 3.12 / FastAPI health endpoint / python-telegram-bot 21.0+
- **RAG:** ChromaDB (50 chunks, 18 sources, all-MiniLM-L6-v2 embeddings)
- **LLM:** Anthropic API (claude-sonnet-4-20250514)
- **Infra:** Single Docker container, docker-compose

## Repo
- **GitHub:** https://github.com/nam-cell/kenny-rag (public)
- **Local:** C:\Users\NamD\SeederWorks\products\kenny
- **VPS:** /opt/kenny-rag (cloned from GitHub)

## Key Files
| File | Purpose |
|------|---------|
| kenny_robinson_api/main.py | Bot + RAG pipeline (252 lines) |
| kenny_robinson_api/Dockerfile | Python 3.12-slim container |
| docker-compose.yml | Service definition |
| deploy.sh | Git-based deploy: push → pull on VPS → docker build |
| kenny_robinson_api/chromadb/ | Vector DB (~1.5MB, committed to repo) |
| kenny_robinson_corpus/raw/ | 20 source text files |
| CLAUDE.md | Product conventions |
| .env | Secrets (gitignored) |

## Secrets (gitignored, in .env on both local + VPS)
- TELEGRAM_BOT_TOKEN — @KRobin_bot via @BotFather
- ANTHROPIC_API_KEY — sk-ant-api03-...
- ANTHROPIC_MODEL — claude-sonnet-4-20250514
- HEALTH_PORT — 8042

## Deploy Workflow
```bash
# From local (PowerShell):
cd C:\Users\NamD\SeederWorks\products\kenny
git add -A && git commit -m "message" && git push origin main

# On VPS:
ssh namd@143.14.200.238
cd /opt/kenny-rag && git pull origin main
sudo docker compose up -d --build
```

Or use `bash deploy.sh` which automates all steps.

## Port Allocation (VPS)
| Product | Port |
|---------|------|
| Cerebro | 3101 |
| Kenny RAG | 8042 |
| Wireframe Studio | 8001/3001 (if deployed) |

## Registry
- Product tracker spreadsheet: SeederWorks/admin/SW_Product_Registry.xlsx
- 3 tabs: Products, Keys & Tokens, VPS

## Decisions Made This Session
1. **React → Telegram bot** — eliminated SSL/mixed-content issues
2. **Polling mode** — no webhook, no domain/cert needed
3. **Git-based deploy** — GitHub as transfer mechanism instead of scp
4. **Public repo** — no secrets in repo, avoids VPS auth hassle
5. **Single container** — bot + health endpoint in one process

## Monitoring
```bash
# Health check
ssh namd@143.14.200.238 'curl -s localhost:8042/health'

# Logs
ssh namd@143.14.200.238 'sudo docker logs -f kenny-rag-bot'

# Restart
ssh namd@143.14.200.238 'cd /opt/kenny-rag && sudo docker compose restart'
```
