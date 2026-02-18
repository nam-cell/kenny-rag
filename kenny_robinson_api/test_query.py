"""Quick test script for the Kenny Robinson RAG API."""

import sys
import io
import json
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://localhost:8042"

# Test 1: Health
print("=" * 50)
print("TEST: /health")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# Test 2: Query
queries = [
    "How did Russell Peters get discovered?",
    "What films has Kenny Robinson appeared in?",
    "What is the Eh-List?",
    "What award did Kenny win in 2014?",
    "What is the People of Comedy documentary?",
]

for q in queries:
    print("=" * 50)
    print(f"QUERY: {q}")
    r = requests.post(f"{BASE}/query", json={"question": q, "n_results": 5})
    data = r.json()
    for c in data["chunks"]:
        print(f"  [{c['source_name']}] dist={c['distance']:.4f} | {c['text'][:100]}...")
    print()
