#!/bin/bash
# Kenny Robinson RAG Bot — Git-based VPS Deployment
# Usage: bash deploy.sh

set -e

REPO="git@github.com:nam-cell/kenny-rag.git"
VPS="143.14.200.238"
VPS_USER="namd"
REMOTE_DIR="/opt/kenny-rag"

echo "=== Kenny RAG Bot Deployment ==="
echo "Target: $VPS_USER@$VPS:$REMOTE_DIR"
echo ""

# Step 1: Push local changes
echo "[1/4] Pushing to GitHub..."
git add -A
git status
read -p "Commit message: " MSG
git commit -m "$MSG" || echo "Nothing to commit"
git push origin main

# Step 2: Pull on VPS
echo "[2/4] Pulling on VPS..."
ssh $VPS_USER@$VPS "
  if [ ! -d $REMOTE_DIR/.git ]; then
    sudo mkdir -p $REMOTE_DIR
    sudo chown -R namd:namd $REMOTE_DIR
    git clone $REPO $REMOTE_DIR
  else
    cd $REMOTE_DIR && git pull origin main
  fi
"

# Step 3: Build and start
echo "[3/4] Building container..."
ssh $VPS_USER@$VPS "cd $REMOTE_DIR && sudo docker compose up -d --build"

# Step 4: Verify
echo "[4/4] Verifying..."
sleep 5
ssh $VPS_USER@$VPS "curl -s http://localhost:8042/health | python3 -m json.tool"

echo ""
echo "=== Deployment complete ==="
echo "Logs: ssh $VPS_USER@$VPS 'sudo docker logs -f kenny-rag-bot'"
echo "Now test @KRobin_bot on Telegram!"
