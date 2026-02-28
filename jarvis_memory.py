# jarvis_memory.py — Jarvis Semantic Memory Engine v6.0
# Jarvis remembers HOW Paul codes, not just facts.
# Every conversation, review, skill, and pattern is embedded + retrievable.
# Architecture: local vector store (no external DB needed)
# Embedding: sentence-transformers (local) or DeepSeek API fallback

import asyncio
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)
VECTOR_STORE = MEMORY_DIR / "vectors.jsonl"
EPISODIC_STORE = MEMORY_DIR / "episodic.jsonl"
PROFILE_STORE = MEMORY_DIR / "paul_profile.json"
CODING_PATTERNS = MEMORY_DIR / "coding_patterns.json"


class LocalEmbedder:
    """Embed text into vectors. Tries local model first, falls back to TF-IDF-style."""

    def __init__(self):
        self.model = None
        self.method = "tfidf"
        self._try_load_transformer()

    def _try_load_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.method = "transformer"
            print("  Embedder: sentence-transformers (all-MiniLM-L6-v2)")
        except ImportError:
            print("  Embedder: TF-IDF fallback (pip install sentence-transformers for better memory)")

    def embed(self, text: str) -> list[float]:
        if self.method == "transformer" and self.model:
            vec = self.model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        else:
            return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str, dims: int = 256) -> list[float]:
        """Lightweight deterministic embedding using character n-grams.
        Not as powerful as transformers but works offline with zero deps.
        """
        text = text.lower().strip()
        # Build character trigram frequency vector
        vec = [0.0] * dims
        tokens = re.findall(r'\w+', text)
        for token in tokens:
            for i in range(len(token) - 2):
                gram = token[i:i+3]
                idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % dims
                vec[idx] += 1.0
        # L2 normalize
        magnitude = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / magnitude for x in vec]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a)) or 1.0
        mag_b = math.sqrt(sum(x*x for x in b)) or 1.0
        return dot / (mag_a * mag_b)


class SemanticMemoryStore:
    """Store and retrieve memories by semantic similarity."""

    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        self.memories: list[dict] = []
        self._load()

    def _load(self):
        if VECTOR_STORE.exists():
            try:
                with open(VECTOR_STORE) as f:
                    self.memories = [json.loads(line) for line in f if line.strip()]
                print(f"  Memory: loaded {len(self.memories)} vectors")
            except Exception as e:
                print(f"  Memory load error: {e}")
                self.memories = []

    def store(self, content: str, metadata: dict = None) -> str:
        """Embed and store a memory. Returns memory ID."""
        mem_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        vector = self.embedder.embed(content)
        entry = {
            "id": mem_id,
            "content": content[:2000],
            "vector": vector,
            "metadata": metadata or {},
            "stored_at": datetime.now().isoformat()
        }
        self.memories.append(entry)
        # Append to disk
        with open(VECTOR_STORE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return mem_id

    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> list[dict]:
        """Find top-k most semantically similar memories."""
        if not self.memories:
            return []
        query_vec = self.embedder.embed(query)
        scored = []
        for mem in self.memories:
            score = self.embedder.cosine_similarity(query_vec, mem["vector"])
            if score >= min_score:
                scored.append({**mem, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_recent(self, n: int = 10) -> list[dict]:
        return sorted(self.memories, key=lambda x: x.get("stored_at", ""), reverse=True)[:n]

    def forget(self, mem_id: str) -> bool:
        before = len(self.memories)
        self.memories = [m for m in self.memories if m["id"] != mem_id]
        if len(self.memories) < before:
            self._rewrite_store()
            return True
        return False

    def _rewrite_store(self):
        with open(VECTOR_STORE, "w") as f:
            for m in self.memories:
                f.write(json.dumps(m) + "\n")

    def stats(self) -> dict:
        types = {}
        for m in self.memories:
            t = m.get("metadata", {}).get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        return {"total": len(self.memories), "by_type": types}


class PaulProfileEngine:
    """Builds and maintains a deep profile of how Paul codes.
    Learns his patterns, preferences, strengths, growth areas.
    """

    DEFAULT_PROFILE = {
        "name": "Paul",
        "coding_style": [],
        "preferred_patterns": [],
        "recurring_mistakes": [],
        "strengths": [],
        "growth_areas": [],
        "languages": {},
        "frameworks": {},
        "commit_velocity": [],  # commits per day
        "review_verdicts": [],  # SHIP_IT/MINOR_ISSUES/etc history
        "last_updated": None
    }

    def __init__(self):
        self.profile = self._load()

    def _load(self) -> dict:
        if PROFILE_STORE.exists():
            try:
                return json.loads(PROFILE_STORE.read_text())
            except Exception:
                pass
        return dict(self.DEFAULT_PROFILE)

    def _save(self):
        self.profile["last_updated"] = datetime.now().isoformat()
        PROFILE_STORE.write_text(json.dumps(self.profile, indent=2))

    def ingest_review(self, review: dict):
        """Learn from a code review verdict."""
        verdict = review.get("verdict", "")
        if verdict:
            self.profile["review_verdicts"].append({
                "verdict": verdict,
                "repo": review.get("commit", {}).get("repo", ""),
                "ts": datetime.now().isoformat()
            })
            # Keep last 100
            self.profile["review_verdicts"] = self.profile["review_verdicts"][-100:]

        # Extract patterns from review text
        review_text = review.get("review", "")
        if "STRENGTHS" in review_text:
            match = re.search(r'STRENGTHS:(.*?)(?:ISSUES|ONE_THING|$)', review_text, re.DOTALL)
            if match:
                strength = match.group(1).strip()[:200]
                if strength and strength not in self.profile["strengths"]:
                    self.profile["strengths"].append(strength)
                    self.profile["strengths"] = self.profile["strengths"][-20:]

        self._save()

    def ingest_commit(self, commit: dict):
        """Track commit patterns."""
        # Track languages from file extensions in message/repo
        repo = commit.get("repo", "")
        if ".py" in commit.get("message", "") or "python" in repo.lower():
            self.profile["languages"]["Python"] = self.profile["languages"].get("Python", 0) + 1
        self.profile["commit_velocity"].append(datetime.now().isoformat())
        self.profile["commit_velocity"] = self.profile["commit_velocity"][-200:]
        self._save()

    def get_summary(self) -> str:
        """Generate a human-readable profile summary."""
        verdicts = self.profile.get("review_verdicts", [])
        verdict_counts = {}
        for v in verdicts:
            verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

        ship_rate = 0
        if verdicts:
            ship_rate = round(verdict_counts.get("SHIP_IT", 0) / len(verdicts) * 100)

        top_lang = max(self.profile.get("languages", {"Python": 1}).items(),
                       key=lambda x: x[1], default=("Python", 1))[0]

        return (
            f"Paul Profile:\n"
            f"  Total reviews: {len(verdicts)} | Ship rate: {ship_rate}%\n"
            f"  Top language: {top_lang}\n"
            f"  Commits tracked: {len(self.profile.get('commit_velocity', []))}\n"
            f"  Strengths identified: {len(self.profile.get('strengths', []))}\n"
            f"  Last updated: {self.profile.get('last_updated', 'never')}"
        )


class JarvisMemoryEngine:
    """Full memory system. Semantic store + episodic log + Paul profile."""

    def __init__(self):
        print("\n🜏 JARVIS MEMORY ENGINE v6.0")
        self.embedder = LocalEmbedder()
        self.semantic = SemanticMemoryStore(self.embedder)
        self.profile = PaulProfileEngine()
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    def remember(self, content: str, mem_type: str = "conversation",
                 extra: dict = None) -> str:
        """Store a memory with type tagging."""
        metadata = {"type": mem_type, **(extra or {})}
        mem_id = self.semantic.store(content, metadata)
        # Also append to episodic log
        with open(EPISODIC_STORE, "a") as f:
            f.write(json.dumps({
                "id": mem_id, "type": mem_type,
                "content": content[:500],
                "ts": datetime.now().isoformat()
            }) + "\n")
        return mem_id

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve most relevant memories for a query."""
        return self.semantic.search(query, top_k=top_k)

    def recall_as_context(self, query: str, top_k: int = 3) -> str:
        """Return recalled memories formatted as context for a prompt."""
        memories = self.recall(query, top_k)
        if not memories:
            return ""
        parts = ["Relevant memories:"]
        for m in memories:
            parts.append(f"[{m['metadata'].get('type', '?')} | score:{m['score']}] {m['content'][:300]}")
        return "\n".join(parts)

    def learn_from_review(self, review: dict):
        """Ingest a code review into memory + profile."""
        self.profile.ingest_review(review)
        self.remember(
            review.get("review", ""),
            mem_type="code_review",
            extra={"verdict": review.get("verdict"), "repo": review.get("commit", {}).get("repo", "")}
        )

    def learn_from_commit(self, commit: dict):
        """Learn from a commit."""
        self.profile.ingest_commit(commit)
        content = f"Commit: {commit.get('message', '')} in {commit.get('repo', '')}"
        self.remember(content, mem_type="commit",
                      extra={"repo": commit.get("repo", "), "sha": commit.get("sha", "")[:8]})

    async def ask_with_memory(self, question: str) -> str:
        """Answer a question using relevant memories as context."""
        context = self.recall_as_context(question)
        profile_summary = self.profile.get_summary()

        messages = [
            {"role": "system", "content": f"""You are Jarvis, Paul's personal AI.
You have deep memory of his coding history and patterns.

{profile_summary}

{context}"""},
            {"role": "user", "content": question}
        ]
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        # Store this conversation too
        self.remember(f"Q: {question}\nA: {answer[:500]}", mem_type="conversation")
        return answer

    def stats(self) -> dict:
        return {
            "memory": self.semantic.stats(),
            "profile": self.profile.get_summary(),
            "embedder": self.embedder.method
        }


if __name__ == "__main__":
    engine = JarvisMemoryEngine()
    print(engine.stats())
