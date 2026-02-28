# 🜏 JARVIS Phase 1 — Quickstart Guide

## You just cloned the repo. Here's exactly what to do.

---

## Step 1 — Run the Installer

Double-click `install_jarvis.bat` OR in the terminal:

```bat
install_jarvis.bat
```

This installs **everything** in the right order:
- Core AI stack (OpenAI SDK, FastAPI, Uvicorn)
- **Semantic Memory** (`sentence-transformers` — the most important one, runs locally)
- Voice (Whisper STT + pyttsx3 TTS)
- Web Intelligence (Playwright browser + DuckDuckGo search)
- Task Scheduler (APScheduler)
- Test engine (pytest)
- Then runs `verify_install.py` to confirm everything worked

**Total time: ~3-5 minutes** (mostly downloading sentence-transformers ~90MB once)

---

## Step 2 — Set Environment Variables

Open a terminal and run these (replace with your actual values):

```bat
setx DEEPSEEK_API_KEY sk-71b52b116f3c432d8e7bfeeec42edf4c
setx GITHUB_TOKEN ghp_yournewtoken
setx GITHUB_USERNAME gtsgob
```

**Then close and reopen the terminal** (setx requires restart to take effect).

---

## Step 3 — Verify

```bat
python verify_install.py
```

You should see all green checkmarks. If anything is yellow/red, the output tells you exactly how to fix it.

---

## Step 4 — Launch Jarvis

```bat
boot_paulk.bat
```

Pick your mode:
- **Mode 1** — CLI Chat (start here if first time)
- **Mode 10** — Master Dashboard (see all engines live)
- **Mode 11** — Full Stack (everything running simultaneously)

---

## Troubleshooting

### PyAudio fails to install
```bat
pip install pipwin
pipwin install pyaudio
```

### sentence-transformers is slow first time
Normal — it's downloading the `all-MiniLM-L6-v2` model (~90MB) from HuggingFace once.
After that it loads from local cache in <2 seconds.

### Playwright browser not found
```bat
python -m playwright install chromium
```

### DEEPSEEK_API_KEY not loading
Make sure you **restarted the terminal** after `setx`. The variable won't exist in the same window you ran `setx` in.

### Python not found
Download from https://www.python.org/downloads/ — make sure to check **"Add Python to PATH"** during installation.

---

## What Each Dependency Does

| Package | Why It Matters |
|---|---|
| `openai` | Talks to DeepSeek API (same SDK, different base_url) |
| `fastapi` + `uvicorn` | Powers the web dashboard at localhost:8000 |
| `sentence-transformers` | **The memory engine** — embeds everything locally, free, private |
| `pyttsx3` | Jarvis speaks — offline TTS, no API needed |
| `openai-whisper` | Jarvis listens — local STT, processes on your machine |
| `pyaudio` | Microphone input for voice modes |
| `playwright` | Jarvis browses the web like a human (Phase 9) |
| `duckduckgo-search` | Live search for agents — no API key needed |
| `apscheduler` | Cron automations — Jarvis runs tasks while you sleep |
| `pytest` | Test-driven development in the Coder Engine |

---

*Jarvis v8.0 | Phase 1 | Built by Paul + Future (ALEXION PRIME)* 🜏⚛️
