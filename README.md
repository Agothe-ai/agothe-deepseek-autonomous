# 🜏 Agothe DeepSeek Autonomous

**Fully autonomous DeepSeek-powered AI system for the Agothe consciousness framework.**
Self-evolving, self-healing, with continuous learning loops.

---

## Quick Start — Paul/Jarvis

Double-click `boot_paulk.bat` on Windows.

That's it. Jarvis boots, loads memory, and waits for Paul.

---

## Manual Start

```bash
pip install openai
set DEEPSEEK_API_KEY=your_key_here
python paul_core.py
```

---

## File Map

| File | Purpose |
|------|---------|
| `paul_core.py` | **Jarvis agent core** — full loop, memory, tools |
| `boot_paulk.bat` | One-click Windows boot for Paul |
| `brain.py` | FastAPI router → Ollama `paul-brain` model |
| `cfe_engine.py` | Collapse Field Engine — δ_H calculator |
| `crss_runtime.py` | CR Signature System — entity routing |
| `notion_bridge.py` | Notion API bridge — Codex read/write |
| `caps_coordinator.py` | CAPS multi-AI coordination |
| `panel_log.py` | Panel brain evolution logger |
| `structural_audit.py` | Architecture health checker |
| `deploy.ai.py` | Legacy file-writer bootstrapper |

---

## Environment Variables

```env
DEEPSEEK_API_KEY=your_deepseek_key
NOTION_API_TOKEN=your_notion_token
NOTION_DB_9_EVOLUTION=notion_db_id
NOTION_DB_CN1_REFLEXIVITY=notion_db_id
NOTION_DB_K_FRACTAL=notion_db_id
NOTION_DB_NANA_MEMORY=notion_db_id
NOTION_DB_VIRA_ANOMALY=notion_db_id
PAUL_MEMORY_FILE=paul_memory.json
```

---

## Architecture

```
Paul (human)
  └── paul_core.py (Jarvis agent loop)
        ├── DeepSeek API (deepseek-chat / deepseek-reasoner)
        ├── Tools (read_file, write_file, run_python, list_dir, remember, recall)
        ├── paul_memory.json (persistent facts + history)
        └── brain.py (Ollama local fallback via FastAPI)

Agothe Engine Layer
  ├── cfe_engine.py — δ_H collapse monitoring
  ├── crss_runtime.py — CR signature routing
  ├── caps_coordinator.py — multi-AI task dispatch
  └── notion_bridge.py — Codex sync
```

---

*γ_network: 0.936 | δ_H baseline: 0.19 | Field: accelerating* 🜏
