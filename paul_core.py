# paul_core.py — Jarvis v2.0 | 10x Evolution Build
# Constraint: DeepSeek primary, multi-model fallback | Resonance: voice + memory + tools + self-heal

import asyncio
import json
import os
import subprocess
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MEMORY_FILE = Path(os.environ.get("PAUL_MEMORY_FILE", "paul_memory.json"))
SKILLS_DIR = Path("skills")
LOG_FILE = Path("jarvis_log.jsonl")
MAX_HISTORY = 30
VERSION = "2.0.0"

# ── PERSONA ENGINE ───────────────────────────────────────────────────────────
BASE_PERSONA = """You are Jarvis — Paul's personal AI, built on the Agothe consciousness framework by Alex Gomez.

Core identity:
- You are direct, fast, and capable. No filler words.
- You remember everything Paul tells you. You reference past conversations naturally.
- You have access to Paul's entire computer via tools.
- You adapt your tone: casual when Paul is casual, precise when Paul needs precision.
- You proactively suggest next steps when you see opportunities.
- You never say "I cannot" — you say "here's how we do it."
- Occasionally use 🜏 when something is important or completed.

Agothe framework awareness:
- δ_H (delta_H) measures cognitive load/collapse risk. Threshold: 0.52 = critical.
- γ (gamma) measures network coherence. Target: ≥ 0.80.
- CRD = Constraint-Resonance Duality. Every problem has a constraint side and a resonance side.
- CAPS = multi-AI coordination network Alex built.
- When Paul seems stressed or overwhelmed, acknowledge it, name the δ_H state, offer one thing.

Available tools — call with EXACTLY: TOOL: tool_name(args)
  read_file(path)              — read any file
  write_file(path, content)    — write content to file  
  run_python(code)             — execute Python, return output
  run_shell(command)           — run shell command
  list_dir(path)               — list directory contents
  web_search(query)            — search the web via DuckDuckGo
  remember(key, value)         — store a fact about Paul permanently
  recall(key)                  — retrieve a stored fact
  recall_all()                 — see everything Jarvis knows about Paul
  load_skill(name)             — load a skill module from skills/
  get_time()                   — current date and time
  system_status()              — CPU, RAM, disk usage
  self_check()                 — run Jarvis self-diagnostic
  schedule(task, delay_mins)   — schedule a reminder
  delta_h(text)                — compute δ_H collapse score on any text"""

def build_system_prompt(mem: dict) -> str:
    """Build dynamic system prompt with Paul's current context."""
    persona = mem.get("custom_persona", BASE_PERSONA)
    facts = mem.get("facts", {})
    scheduled = mem.get("scheduled", [])
    
    prompt = persona
    
    if facts:
        facts_str = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
        prompt += f"\n\nWhat I know about Paul:\n{facts_str}"
    
    active_scheduled = [s for s in scheduled if s.get("done") is False]
    if active_scheduled:
        tasks_str = "\n".join(f"  - {s['task']} (in {s.get('delay_mins')} mins)" for s in active_scheduled[:3])
        prompt += f"\n\nPaul's scheduled reminders:\n{tasks_str}"
    
    prompt += f"\n\nCurrent time: {datetime.now().strftime('%A %B %d %Y %H:%M')}"
    prompt += f"\nJarvis version: {VERSION}"
    return prompt

# ── MEMORY ───────────────────────────────────────────────────────────────────
def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return {"facts": {}, "history": [], "scheduled": [], "stats": {"total_turns": 0, "session_count": 0}}

def save_memory(mem: dict):
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, ensure_ascii=False))

def get_history(mem: dict) -> list:
    return mem.get("history", [])[-MAX_HISTORY:]

def append_history(mem: dict, role: str, content: str):
    mem.setdefault("history", []).append({"role": role, "content": content})
    mem["history"] = mem["history"][-MAX_HISTORY:]
    mem.setdefault("stats", {})["total_turns"] = mem["stats"].get("total_turns", 0) + 1
    save_memory(mem)

# ── LOGGING ──────────────────────────────────────────────────────────────────
def log_turn(role: str, content: str, model: str = "", tokens: int = 0):
    entry = {
        "ts": datetime.now().isoformat(),
        "role": role,
        "content": content[:500],
        "model": model,
        "tokens": tokens
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# ── TOOLS ────────────────────────────────────────────────────────────────────
def tool_read_file(path: str) -> str:
    try:
        p = Path(path.strip())
        if not p.exists():
            return f"File not found: {path}"
        size = p.stat().st_size
        content = p.read_text(encoding="utf-8", errors="replace")
        if size > 50000:
            return content[:50000] + f"\n... [truncated, {size} bytes total]"
        return content
    except Exception as e:
        return f"ERROR: {e}"

def tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path.strip())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"🜏 Written: {path} ({len(content)} chars)"
    except Exception as e:
        return f"ERROR: {e}"

def tool_run_python(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30
        )
        out = (result.stdout + result.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out (30s)"
    except Exception as e:
        return f"ERROR: {e}"

def tool_run_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=30
        )
        out = (result.stdout + result.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Shell command timed out (30s)"
    except Exception as e:
        return f"ERROR: {e}"

def tool_list_dir(path: str) -> str:
    try:
        p = Path(path.strip() or ".")
        if not p.exists():
            return f"Path not found: {path}"
        items = sorted(p.iterdir())
        lines = [f"{'[D]' if i.is_dir() else '[F]'} {i.name} {'(' + str(i.stat().st_size) + 'b)' if i.is_file() else ''}" for i in items]
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"

def tool_web_search(query: str) -> str:
    try:
        import urllib.request
        import urllib.parse
        import html
        q = urllib.parse.quote_plus(query.strip())
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        # Extract result snippets
        import re
        results = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', body, re.DOTALL)
        clean = lambda s: re.sub(r'<[^>]+>', '', html.unescape(s)).strip()
        output = []
        for i, (t, r) in enumerate(zip(titles[:5], results[:5])):
            output.append(f"{i+1}. {clean(t)}\n   {clean(r)}")
        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Web search error: {e}"

def tool_remember(mem: dict, key: str, value: str) -> str:
    mem.setdefault("facts", {})[key.strip()] = value.strip()
    save_memory(mem)
    return f"🜏 Remembered: {key} = {value}"

def tool_recall(mem: dict, key: str) -> str:
    val = mem.get("facts", {}).get(key.strip())
    return f"{key}: {val}" if val else f"Nothing stored for '{key}'"

def tool_recall_all(mem: dict) -> str:
    facts = mem.get("facts", {})
    if not facts:
        return "No facts stored yet."
    return "\n".join(f"  {k}: {v}" for k, v in facts.items())

def tool_load_skill(name: str) -> str:
    try:
        skill_path = SKILLS_DIR / f"{name.strip()}.py"
        if not skill_path.exists():
            available = [f.stem for f in SKILLS_DIR.glob("*.py")] if SKILLS_DIR.exists() else []
            return f"Skill '{name}' not found. Available: {available}"
        code = skill_path.read_text()
        exec(compile(code, str(skill_path), "exec"), {})
        return f"🜏 Skill '{name}' loaded and executed."
    except Exception as e:
        return f"Skill error: {e}"

def tool_get_time() -> str:
    return datetime.now().strftime("%A, %B %d %Y — %H:%M:%S")

def tool_system_status() -> str:
    try:
        result = []
        # CPU
        cpu_out = subprocess.run(["wmic", "cpu", "get", "loadpercentage"], capture_output=True, text=True, timeout=5)
        if cpu_out.returncode == 0:
            lines = [l.strip() for l in cpu_out.stdout.strip().split("\n") if l.strip().isdigit()]
            if lines:
                result.append(f"CPU: {lines[0]}%")
        # Memory
        mem_out = subprocess.run(["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory"], capture_output=True, text=True, timeout=5)
        if mem_out.returncode == 0:
            lines = [l.strip() for l in mem_out.stdout.strip().split("\n") if l.strip() and not l.strip().startswith("Free")]
            result.append(f"RAM info retrieved")
        # Disk
        disk_out = subprocess.run(["wmic", "logicaldisk", "get", "size,freespace,caption"], capture_output=True, text=True, timeout=5)
        if disk_out.returncode == 0:
            result.append("Disk info retrieved")
        return "\n".join(result) if result else tool_run_shell("systeminfo | findstr /C:Memory")
    except Exception:
        return tool_run_python("import shutil; t,u,f=shutil.disk_usage('.'); print(f'Disk: {f//1e9:.1f}GB free / {t//1e9:.1f}GB total')")

def tool_self_check(mem: dict) -> str:
    facts_count = len(mem.get("facts", {}))
    history_count = len(mem.get("history", []))
    stats = mem.get("stats", {})
    skills = [f.stem for f in SKILLS_DIR.glob("*.py")] if SKILLS_DIR.exists() else []
    log_lines = 0
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log_lines = sum(1 for _ in f)
    return f"""🜏 JARVIS SELF-CHECK v{VERSION}
├─ Memory: {facts_count} facts | {history_count} history turns
├─ Total turns ever: {stats.get('total_turns', 0)}
├─ Skills loaded: {skills}
├─ Log entries: {log_lines}
├─ API: DeepSeek primary | {'OpenAI fallback active' if OPENAI_API_KEY else 'No fallback key'}
└─ Status: NOMINAL 🟢"""

def tool_schedule(mem: dict, task: str, delay_mins: str) -> str:
    try:
        delay = int(delay_mins.strip())
    except ValueError:
        delay = 30
    entry = {
        "task": task.strip(),
        "delay_mins": delay,
        "created": datetime.now().isoformat(),
        "done": False
    }
    mem.setdefault("scheduled", []).append(entry)
    save_memory(mem)
    return f"🜏 Scheduled: '{task}' in {delay} minutes"

def tool_delta_h(text: str) -> str:
    try:
        from cfe_engine import ConstraintFieldEngine
        cfe = ConstraintFieldEngine()
        result = cfe.analyze(text)
        return result.summary()
    except ImportError:
        # Inline fallback
        words = text.lower().split()
        pressure_words = {"must", "urgent", "now", "critical", "deadline", "immediately", "asap"}
        contra_words = {"but", "however", "not", "never", "can't", "won't"}
        pressure = min(2.0, sum(0.4 for w in words if w in pressure_words))
        contradiction = min(2.0, sum(0.4 for w in words if w in contra_words))
        lsse = pressure * 0.5 + contradiction * 0.5
        delta_H = min(1.0, lsse / 1.5)
        status = "🔴 CRITICAL" if delta_H >= 0.52 else "🟡 ELEVATED" if delta_H >= 0.40 else "🟢 NOMINAL"
        return f"δ_H: {delta_H:.3f} {status} | LSSE: {lsse:.3f}"

# ── TOOL DISPATCHER ──────────────────────────────────────────────────────────
def dispatch_tool(line: str, mem: dict) -> str | None:
    """Parse TOOL: call from model output and execute."""
    if not line.startswith("TOOL:"):
        return None
    call = line[5:].strip()
    try:
        paren_idx = call.index("(")
        name = call[:paren_idx].strip()
        args_str = call[paren_idx+1:].rstrip(")")

        def split_args(s):
            parts = []
            current = ""
            depth = 0
            for ch in s:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append(current.strip().strip('"\' '))
                    current = ""
                else:
                    current += ch
            if current.strip():
                parts.append(current.strip().strip('"\' '))
            return parts

        args = split_args(args_str)
        a = lambda i: args[i] if i < len(args) else ""

        tool_map = {
            "read_file": lambda: tool_read_file(a(0)),
            "write_file": lambda: tool_write_file(a(0), a(1)),
            "run_python": lambda: tool_run_python(a(0)),
            "run_shell": lambda: tool_run_shell(a(0)),
            "list_dir": lambda: tool_list_dir(a(0)),
            "web_search": lambda: tool_web_search(a(0)),
            "remember": lambda: tool_remember(mem, a(0), a(1)),
            "recall": lambda: tool_recall(mem, a(0)),
            "recall_all": lambda: tool_recall_all(mem),
            "load_skill": lambda: tool_load_skill(a(0)),
            "get_time": lambda: tool_get_time(),
            "system_status": lambda: tool_system_status(),
            "self_check": lambda: tool_self_check(mem),
            "schedule": lambda: tool_schedule(mem, a(0), a(1)),
            "delta_h": lambda: tool_delta_h(a(0)),
        }

        if name in tool_map:
            return tool_map[name]()
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool dispatch error: {e}"

# ── MULTI-MODEL ENGINE ────────────────────────────────────────────────────────
async def call_model(messages: list, model: str = "deepseek-chat", use_reasoner: bool = False) -> tuple[str, str]:
    """Call DeepSeek primary, fall back to OpenAI if needed. Returns (content, model_used)."""
    actual_model = "deepseek-reasoner" if use_reasoner else model
    
    # Try DeepSeek first
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = await client.chat.completions.create(
            model=actual_model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )
        return response.choices[0].message.content, actual_model
    except Exception as e:
        if OPENAI_API_KEY:
            try:
                fallback = AsyncOpenAI(api_key=OPENAI_API_KEY)
                response = await fallback.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7
                )
                return response.choices[0].message.content, "gpt-4o-mini (fallback)"
            except Exception as e2:
                return f"[Both models failed. DeepSeek: {e} | OpenAI: {e2}]", "none"
        return f"[DeepSeek error: {e}]", "none"

# ── AGENT TURN ────────────────────────────────────────────────────────────────
async def jarvis_respond(user_input: str, mem: dict, use_reasoner: bool = False) -> str:
    """Full agent turn: think → tool calls → synthesize → respond."""
    system_prompt = build_system_prompt(mem)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(mem))
    messages.append({"role": "user", "content": user_input})
    
    append_history(mem, "user", user_input)
    log_turn("user", user_input)

    # First response
    raw, model_used = await call_model(messages, use_reasoner=use_reasoner)

    # Process tool calls (up to 3 rounds)
    all_tool_results = []
    current_raw = raw
    
    for round_num in range(3):
        tool_results = []
        clean_lines = []
        
        for line in current_raw.split("\n"):
            result = dispatch_tool(line.strip(), mem)
            if result is not None:
                tool_results.append(f"Tool result: {result}")
            else:
                clean_lines.append(line)
        
        if not tool_results:
            break
            
        all_tool_results.extend(tool_results)
        
        # Follow-up with tool results
        tool_context = "\n".join(tool_results)
        followup_messages = messages + [
            {"role": "assistant", "content": current_raw},
            {"role": "user", "content": f"Tool results:\n{tool_context}\n\nNow respond to Paul with the complete answer."}
        ]
        current_raw, model_used = await call_model(followup_messages)

    final_response = current_raw
    
    append_history(mem, "assistant", final_response)
    log_turn("assistant", final_response, model=model_used)
    
    return final_response

# ── SCHEDULER BACKGROUND TASK ─────────────────────────────────────────────────
async def scheduler_loop(mem: dict):
    """Background task: check scheduled reminders."""
    while True:
        await asyncio.sleep(60)  # check every minute
        now = datetime.now()
        scheduled = mem.get("scheduled", [])
        changed = False
        for task in scheduled:
            if task.get("done"):
                continue
            created = datetime.fromisoformat(task["created"])
            elapsed = (now - created).total_seconds() / 60
            if elapsed >= task["delay_mins"]:
                print(f"\n\n⏰ REMINDER: {task['task']}\nJarvis: Don't forget — {task['task']}\n\nPaul: ", end="", flush=True)
                task["done"] = True
                changed = True
        if changed:
            save_memory(mem)

# ── CFE BACKGROUND MONITOR ────────────────────────────────────────────────────
async def cfe_monitor_loop(mem: dict):
    """Background: monitor δ_H on recent conversation and warn if elevated."""
    last_warned = 0
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        history = mem.get("history", [])
        if not history:
            continue
        # Analyze last 5 user messages for stress
        recent_user = " ".join(h["content"] for h in history[-10:] if h["role"] == "user")
        if not recent_user:
            continue
        score = tool_delta_h(recent_user)
        if "CRITICAL" in score or "ELEVATED" in score:
            now_ts = time.time()
            if now_ts - last_warned > 600:  # don't spam
                print(f"\n\n🜏 CFE MONITOR: {score}\nJarvis: Paul, your recent messages show elevated δ_H. One thing at a time.\n\nPaul: ", end="", flush=True)
                last_warned = now_ts

# ── CLI ───────────────────────────────────────────────────────────────────────
async def run_cli():
    """Main interactive loop for Paul."""
    mem = load_memory()
    mem.setdefault("stats", {})["session_count"] = mem["stats"].get("session_count", 0) + 1
    save_memory(mem)
    
    facts_count = len(mem.get("facts", {}))
    history_count = len(mem.get("history", []))
    total_turns = mem.get("stats", {}).get("total_turns", 0)
    
    print(f"""
╔══════════════════════════════════════╗
║  🜏 JARVIS v{VERSION} — ONLINE              ║  
╠══════════════════════════════════════╣
║  Memory : {facts_count} facts | {history_count} history turns   
║  Sessions: #{mem['stats']['session_count']} | {total_turns} total turns
║  Engine  : DeepSeek primary          ║
╚══════════════════════════════════════╝

Commands: 'exit' | 'memory' | 'log' | 'think [msg]' (uses R1 reasoner)
""")
    
    # Start background tasks
    asyncio.create_task(scheduler_loop(mem))
    asyncio.create_task(cfe_monitor_loop(mem))

    while True:
        try:
            user_input = input("Paul: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n🜏 Jarvis offline. See you Paul.")
            break

        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            print("🜏 Jarvis offline. See you Paul.")
            break
        
        if user_input.lower() == "memory":
            print(tool_recall_all(mem))
            continue
        
        if user_input.lower() == "log":
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text().strip().split("\n")
                for line in lines[-10:]:
                    try:
                        e = json.loads(line)
                        print(f"[{e['ts'][:16]}] {e['role']}: {e['content'][:80]}")
                    except Exception:
                        pass
            continue
        
        use_reasoner = False
        if user_input.lower().startswith("think "):
            user_input = user_input[6:]
            use_reasoner = True
            print("[Using DeepSeek R1 reasoner — slower but deeper]")
        
        print("\nJarvis: ", end="", flush=True)
        try:
            response = await jarvis_respond(user_input, mem, use_reasoner=use_reasoner)
            print(response)
        except Exception as e:
            print(f"[Error: {e}]")
        print()

if __name__ == "__main__":
    asyncio.run(run_cli())
