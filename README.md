# 🜏 JARVIS — Paulk Autonomous AI System

> **The best coder AI. Yours. Runs locally. No subscriptions.**  
> Built by Paul + Future (ALEXION PRIME) | Architecture: Claude Code + Cursor + Devin + OpenAI Codex

---

## ⚡ Quick Start

```bat
git clone https://github.com/gtsgob/agothe-deepseek-autonomous
cd agothe-deepseek-autonomous
set DEEPSEEK_API_KEY=your_key_here
set GITHUB_TOKEN=your_github_token
boot_paulk.bat
```

Pick a mode. That's it. Jarvis is running.

---

## 🧠 What Jarvis Can Do

### 8 Operational Modes

| Mode | Command | What it does |
|------|---------|------|
| **1** CLI Chat | `paul_core.py` | Full AI assistant, 15 tools, DeepSeek + R1 |
| **2** Voice PTT | `jarvis_voice.py` | Press Enter → speak → Jarvis responds aloud |
| **3** Voice Wake | `jarvis_voice.py --wake` | Say "Hey Jarvis" from anywhere in the room |
| **4** Web Dashboard | `jarvis_api.py` | Dark UI at localhost:8000, chat + memory live |
| **5** Coder Engine | `jarvis_evolve.py` | Planner→Executor→Verifier autonomous coding |
| **6** Self-Heal | `jarvis_self_heal.py` | Watches all .py files, auto-patches errors |
| **7** GitHub Watch | `jarvis_github_watcher.py` | Live commit watcher, R1 review every push |
| **8** GitHub+Voice | Mode 8 in boot | Push code → hear review spoken in <30s |

---

## 🏗️ Architecture

```
jarvis/
├── paul_core.py              ← Main brain: 15 tools, memory, multi-model
├── jarvis_voice.py           ← Whisper STT + pyttsx3 TTS + wake word
├── jarvis_api.py             ← FastAPI web dashboard
├── jarvis_evolve.py          ← World-Class Coder Engine
│   ├── ShadowWorkspace       ← Cursor-style: edit in memory, diff before apply
│   ├── TestHarness           ← Codex-style: TDD loop, AST validation
│   ├── Planner               ← Devin-style: blueprint before touching anything
│   ├── Executor              ← Claude Code-style: read→write→verify loop
│   ├── Verifier              ← R1 reasoner: deep skeptical code review
│   ├── GödelModifier         ← Self-reads own source, proposes improvements
│   └── IntelligenceScraper   ← Absorbs patterns from top AI agents on GitHub
├── jarvis_self_heal.py       ← Daemon: scan→detect→patch→verify every 30s
├── jarvis_github_watcher.py  ← Live GitHub: poll→diff→R1 review→speak
├── jarvis_memory.py          ← Semantic memory: embeddings + Paul profile
├── jarvis_taskqueue.py       ← Autonomous worker: runs tasks while Paul sleeps
├── notion_bridge.py          ← Notion DB sync
├── skills/                   ← Loadable skill modules
│   ├── morning_brief.py
│   ├── system_scan.py
│   ├── notion_sync.py
│   ├── code_review.py
│   ├── world_coder.py
│   └── github_intel.py
├── protocols/vault/          ← Intelligence vault (learned agent architectures)
│   └── agent_architectures.json
└── memory/                   ← Semantic memory store (auto-created)
    ├── vectors.jsonl
    ├── episodic.jsonl
    └── paul_profile.json
```

---

## 🔑 Environment Variables

```bat
set DEEPSEEK_API_KEY=sk-...       # Required — get at platform.deepseek.com
set GITHUB_TOKEN=ghp_...          # Recommended — 5000 req/hr vs 60
set GITHUB_USERNAME=gtsgob        # Your GitHub username
set WHISPER_MODEL=base            # tiny/base/small/medium (voice quality)
set JARVIS_VOICE=1                # Enable voice in any mode
```

To make permanent:
```bat
setx DEEPSEEK_API_KEY sk-...
setx GITHUB_TOKEN ghp_...
```

---

## 🜏 The Architecture Secret

Every top coding agent — Claude Code, Cursor, Devin, Codex — has the same core secret:

> **The model is just the brain. The loop is the intelligence.**

Jarvis implements all four loops:
- **Claude Code**: `think → tool_call → observe → repeat` (never summarize, always read full files)
- **Cursor**: Shadow workspace — ALL edits in memory first, diff before touching disk
- **Devin**: Planner + Executor + Verifier as separate roles so the executor can't rationalize its own mistakes  
- **Codex**: TDD — write failing tests first, make them pass, verify, ship

All four. On DeepSeek. On your machine. Zero subscription.

---

## 📦 Dependencies

Core (auto-installed):
```
openai fastapi uvicorn
```

Voice (Mode 2/3/8):
```
pyttsx3 openai-whisper pyaudio numpy
```

Better memory (optional):
```
sentence-transformers
```

Tests:
```
pytest
```

---

## 🚀 Version History

| Version | What shipped |
|---------|--------------|
| v1.0 | Basic Jarvis skeleton |
| v2.0 | 15 tools, multi-model, web dashboard, skills system |
| v3.0 | World-Class Coder Engine (Planner/Executor/Verifier/Gödelmodifier) |
| v4.0 | Voice: Whisper STT + pyttsx3 TTS + wake word detection |
| v5.0 | GitHub Live Watcher: R1 review every push, spoken in <30s |
| v6.0 | Semantic Memory Engine: embeddings + Paul profile engine |
| v7.0 | Autonomous Task Queue: Jarvis works while Paul sleeps |
| **v8.0** | **Living README + Master architecture** |

---

*Session δ_H: 0.07 | Ω: 0.99 | Field: accelerating* 🜏⚛️
