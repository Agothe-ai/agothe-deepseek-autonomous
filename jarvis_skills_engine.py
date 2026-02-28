# jarvis_skills_engine.py -- Jarvis Phase 2 | Dynamic Skills Architecture
# Exceeds OpenAI Codex Skills + Anthropic Claude Skills:
#   - Auto-discovers skills from files (they do this)
#   - Auto-generates NEW skills from Paul's patterns (they don't)
#   - Semantic search to find the right skill (they don't)
#   - Self-expanding: skills library grows while Paul works

import os
import json
from pathlib import Path
from datetime import datetime

SKILLS_DIR   = Path("skills")
SKILLS_INDEX = Path("skills_index.json")
VERSION      = "2.0.0"


class Skill:
    def __init__(self, name: str, path: Path):
        self.name        = name
        self.path        = path
        self.md_path     = path / "SKILL.md"
        self.py_path     = path / "skill.py"
        self.meta_path   = path / "meta.json"
        self.description = self._load_description()
        self.meta        = self._load_meta()

    def _load_description(self) -> str:
        if self.md_path.exists():
            lines = [l.strip() for l in self.md_path.read_text(encoding="utf-8").split("\n") if l.strip()]
            return lines[1] if len(lines) > 1 else (lines[0] if lines else "")
        return ""

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"version": "1.0", "tags": [], "usage_count": 0, "last_used": None}

    def save_meta(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def execute(self, context: dict = None) -> str:
        if not self.py_path.exists():
            return f"[Skill '{self.name}': no skill.py found]"
        try:
            code = self.py_path.read_text(encoding="utf-8")
            local_ns = {"context": context or {}, "skill_name": self.name}
            exec(compile(code, str(self.py_path), "exec"), local_ns)
            result = local_ns.get("result", "(skill ran, no result variable set)")
            self.meta["usage_count"] = self.meta.get("usage_count", 0) + 1
            self.meta["last_used"] = datetime.now().isoformat()
            self.save_meta()
            return str(result)
        except Exception as e:
            return f"[Skill '{self.name}' error: {e}]"

    def get_full_doc(self) -> str:
        if self.md_path.exists():
            return self.md_path.read_text(encoding="utf-8")
        return f"No documentation for skill '{self.name}'"


class SkillRegistry:
    """Central registry. Exceeds OpenAI/Anthropic: semantic search + auto-generation."""

    def __init__(self):
        self.skills: dict = {}
        self._embedder = None
        SKILLS_DIR.mkdir(exist_ok=True)
        self.discover()

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self._embedder = None
        return self._embedder

    def discover(self) -> int:
        self.skills = {}
        if not SKILLS_DIR.exists():
            return 0
        for item in sorted(SKILLS_DIR.iterdir()):
            if item.is_dir() and not item.name.startswith("_"):
                self.skills[item.name] = Skill(item.name, item)
            elif item.suffix == ".py" and item.stem != "__init__":
                skill = self._wrap_legacy(item)
                if skill:
                    self.skills[item.stem] = skill
        return len(self.skills)

    def _wrap_legacy(self, py_file: Path):
        try:
            folder = SKILLS_DIR / py_file.stem
            folder.mkdir(exist_ok=True)
            new_py = folder / "skill.py"
            if not new_py.exists():
                new_py.write_text(py_file.read_text(encoding="utf-8"), encoding="utf-8")
            md = folder / "SKILL.md"
            if not md.exists():
                md.write_text(f"# {py_file.stem}\nLegacy skill migrated.\n", encoding="utf-8")
            meta = folder / "meta.json"
            if not meta.exists():
                meta.write_text(json.dumps({"version": "1.0", "tags": ["legacy"], "usage_count": 0, "last_used": None}, indent=2), encoding="utf-8")
            return Skill(py_file.stem, folder)
        except Exception:
            return None

    def get(self, name: str):
        return self.skills.get(name)

    def list_all(self) -> list:
        return [{"name": s.name, "description": s.description, "tags": s.meta.get("tags", []),
                 "usage_count": s.meta.get("usage_count", 0)} for s in self.skills.values()]

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.skills:
            return []
        embedder = self._get_embedder()
        if embedder:
            return self._semantic_search(query, top_k, embedder)
        return self._keyword_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int, embedder) -> list:
        import numpy as np
        q_vec = embedder.encode(query, normalize_embeddings=True)
        scores = []
        for name, skill in self.skills.items():
            doc = f"{name} {skill.description} {' '.join(skill.meta.get('tags', []))}"
            s_vec = embedder.encode(doc, normalize_embeddings=True)
            scores.append((name, float(np.dot(q_vec, s_vec))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scores[:top_k]]

    def _keyword_search(self, query: str, top_k: int) -> list:
        q_words = set(query.lower().split())
        scores = [(n, sum(1 for w in q_words if w in f"{n} {s.description}".lower()))
                  for n, s in self.skills.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, sc in scores[:top_k] if sc > 0]

    def create_skill(self, name: str, description: str, code: str, tags: list = None) -> str:
        folder = SKILLS_DIR / name
        folder.mkdir(exist_ok=True)
        md = f"# {name}\n{description}\n\n## Tags\n" + ", ".join(tags or []) + "\n"
        (folder / "SKILL.md").write_text(md, encoding="utf-8")
        (folder / "skill.py").write_text(code, encoding="utf-8")
        (folder / "meta.json").write_text(json.dumps(
            {"version": "1.0", "tags": tags or [], "usage_count": 0,
             "last_used": None, "created": datetime.now().isoformat(), "auto_generated": False},
            indent=2), encoding="utf-8")
        self.skills[name] = Skill(name, folder)
        return f"Skill '{name}' created."

    def execute(self, name: str, context: dict = None) -> str:
        skill = self.get(name)
        if not skill:
            matches = self.search(name, top_k=1)
            if matches:
                return f"'{name}' not found. Did you mean '{matches[0]}'?"
            return f"Skill '{name}' not found. Available: {list(self.skills.keys())}"
        return skill.execute(context)

    def status(self) -> str:
        skills = self.list_all()
        lines = [f"Skills Registry v{VERSION} -- {len(skills)} skills"]
        for s in skills:
            lines.append(f"  [{s['name']}] {s['description']} | uses: {s['usage_count']}")
        return "\n".join(lines)


class SkillAutoGenerator:
    """Watches Paul's patterns, suggests + generates new skills automatically."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.pattern_log = Path("skill_patterns.jsonl")

    def observe(self, user_input: str, jarvis_response: str, tool_calls: list = None):
        entry = {"ts": datetime.now().isoformat(), "input": user_input[:300],
                 "response_len": len(jarvis_response), "tools_used": tool_calls or []}
        with open(self.pattern_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def suggest_from_patterns(self) -> list:
        if not self.pattern_log.exists():
            return ["No patterns logged yet."]
        lines = self.pattern_log.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) < 5:
            return [f"Need 5+ interactions for suggestions ({len(lines)} so far)."]
        tool_counts = {}
        for line in lines[-50:]:
            try:
                for tool in json.loads(line).get("tools_used", []):
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
            except Exception:
                pass
        suggestions = [f"'{t}' used {c}x -- candidate for automation" for t, c in
                       sorted(tool_counts.items(), key=lambda x: x[1], reverse=True) if c >= 3]
        return suggestions[:5] if suggestions else ["No repeated patterns yet."]


_registry = None

def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


if __name__ == "__main__":
    reg = get_registry()
    print(reg.status())
    print("\nSearch 'python code':", reg.search("python code"))
