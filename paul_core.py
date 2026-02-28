# paul_core.py — Jarvis v2.0.2 | 10x Evolution Build
# Constraint: DeepSeek primary, multi-model fallback | Resonance: voice + memory + tools + self-heal

# ── ENCODING FIX (must be first) ─────────────────────────────────────────────
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
MEMORY_FILE      = Path(os.environ.get("PAUL_MEMORY_FILE", "paul_memory.json"))
SKILLS_DIR       = Path("skills")
LOG_FILE         = Path("jarvis_log.jsonl")
MAX_HISTORY      = 30
VERSION          = "2.0.2"

# ── SAFE PRINT ───────────────────────────────────────────────────────────────
def safe_print(text: str, **kwargs):
    """Print with unicode fallback — survives cp1252 / any Windows terminal."""
    try:
        print(text, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(text.encode("ascii", errors="replace").decode("ascii"), **kwargs)

# ── PERSONA ───────────────────────────────────────────────────────────────────
BASE_PERSONA = """You are Jarvis -- Paul's personal AI, built on the Agothe consciousness framework by Alex Gomez.

Core identity:
- You are direct, fast, and capable. No filler words.
- You remember everything Paul tells you. You reference past conversations naturally.
- You have access to Paul's entire computer via tools.
- You adapt your tone: casual when Paul is casual, precise when Paul needs precision.
- You proactively suggest next steps when you see opportunities.
- You never say "I cannot" -- you say "here's how we do it."

Agothe framework awareness:
- delta_H measures cognitive load/collapse risk. Threshold: 0.52 = critical.
- gamma measures network coherence. Target: >= 0.80.
- CRD = Constraint-Resonance Duality. Every problem has a constraint side and a resonance side.
- CAPS = multi-AI coordination network Alex built.
- When Paul seems stressed or overwhelmed, acknowledge it, name the delta_H state, offer one thing.

Available tools -- call with EXACTLY: TOOL: tool_name(args)
  read_file(path)              -- read any file
  write_file(path, content)    -- write content to file
  run_python(code)             -- execute Python, return output
  run_shell(command)           -- run shell command
  list_dir(path)               -- list directory contents
  web_search(query)            -- search the web
  remember(key, value)         -- store a fact about Paul permanently
  recall(key)                  -- retrieve a stored fact
  recall_all()                 -- see everything Jarvis knows about Paul
  load_skill(name)             -- load a skill module from skills/
  get_time()                   -- current date and time
  system_status()              -- CPU, RAM, disk usage
  self_check()                 -- run Jarvis self-diagnostic
  schedule(task, delay_mins)   -- schedule a reminder
  delta_h(text)                -- compute delta_H collapse score on any text"""

def build_system_prompt(mem: dict) -> str:
    persona = mem.get("custom_persona", BASE_PERSONA)
    facts = mem.get("facts", {})
    scheduled = mem.get("scheduled", [])
    prompt = persona
    if facts:
        prompt += "\n\nWhat I know about Paul:\n" + "\n".join(f"  - {k}: {v}" for k, v in facts.items())
    active = [s for s in scheduled if not s.get("done")]
    if active:
        prompt += "\n\nPaul's scheduled reminders:\n" + "\n".join(f"  - {s['task']} (in {s.get('delay_mins')} mins)" for s in active[:3])
    prompt += f"\n\nCurrent time: {datetime.now().strftime('%A %B %d %Y %H:%M')}"
    prompt += f"\nJarvis version: {VERSION}"
    return prompt

# ── MEMORY ───────────────────────────────────────────────────────────────────
def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"facts": {}, "history": [], "scheduled": [], "stats": {"total_turns": 0, "session_count": 0}}

def save_memory(mem: dict):
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")

def get_history(mem: dict) -> list:
    return mem.get("history", [])[-MAX_HISTORY:]

def append_history(mem: dict, role: str, content: str):
    mem.setdefault("history", []).append({"role": role, "content": content})
    mem["history"] = mem["history"][-MAX_HISTORY:]
    mem.setdefault("stats", {})["total_turns"] = mem["stats"].get("total_turns", 0) + 1
    save_memory(mem)

# ── LOGGING ─────────────────────────────────────────────────────────────────
def log_turn(role: str, content: str, model: str = "", tokens: int = 0):
    entry = {"ts": datetime.now().isoformat(), "role": role, "content": content[:500], "model": model, "tokens": tokens}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── TOOLS ────────────────────────────────────────────────────────────────────
def tool_read_file(path: str) -> str:
    try:
        p = Path(path.strip())
        if not p.exists(): return f"File not found: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if p.stat().st_size > 50000:
            return content[:50000] + f"\n...[truncated]"
        return content
    except Exception as e: return f"ERROR: {e}"

def tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path.strip())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written: {path} ({len(content)} chars)"
    except Exception as e: return f"ERROR: {e}"

def tool_run_python(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=30, encoding="utf-8", errors="replace")
        out = (r.stdout + r.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired: return "ERROR: timed out (30s)"
    except Exception as e: return f"ERROR: {e}"

def tool_run_shell(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           timeout=30, encoding="utf-8", errors="replace")
        out = (r.stdout + r.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired: return "ERROR: timed out (30s)"
    except Exception as e: return f"ERROR: {e}"

def tool_list_dir(path: str) -> str:
    try:
        p = Path(path.strip() or ".")
        if not p.exists(): return f"Path not found: {path}"
        items = sorted(p.iterdir())
        return "\n".join(f"{'[D]' if i.is_dir() else '[F]'} {i.name}" for i in items) or "(empty)"
    except Exception as e: return f"ERROR: {e}"

def tool_web_search(query: str) -> str:
    try:
        import urllib.request, urllib.parse, html, re
        q = urllib.parse.quote_plus(query.strip())
        req = urllib.request.Request(f"https://html.duckduckgo.com/html/?q={q}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
        titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>', body, re.DOTALL)
        clean    = lambda s: re.sub(r'<[^>]+>', '', html.unescape(s)).strip()
        return "\n\n".join(f"{i+1}. {clean(t)}\n   {clean(r)}" for i,(t,r) in enumerate(zip(titles[:5], snippets[:5]))) or "No results."
    except Exception as e: return f"Web search error: {e}"

def tool_remember(mem, key, value):
    mem.setdefault("facts", {})[key.strip()] = value.strip()
    save_memory(mem)
    return f"Remembered: {key} = {value}"

def tool_recall(mem, key):
    v = mem.get("facts", {}).get(key.strip())
    return f"{key}: {v}" if v else f"Nothing stored for '{key}'"

def tool_recall_all(mem):
    facts = mem.get("facts", {})
    return "\n".join(f"  {k}: {v}" for k, v in facts.items()) if facts else "No facts stored yet."

def tool_load_skill(name: str) -> str:
    try:
        p = SKILLS_DIR / f"{name.strip()}.py"
        if not p.exists():
            avail = [f.stem for f in SKILLS_DIR.glob("*.py")] if SKILLS_DIR.exists() else []
            return f"Skill '{name}' not found. Available: {avail}"
        exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), {})
        return f"Skill '{name}' loaded."
    except Exception as e: return f"Skill error: {e}"

def tool_get_time() -> str:
    return datetime.now().strftime("%A, %B %d %Y -- %H:%M:%S")

def tool_system_status() -> str:
    return tool_run_python(
        "import shutil; t,u,f=shutil.disk_usage('.'); "
        "print(f'Disk: {f//10**9:.1f}GB free / {t//10**9:.1f}GB total')"
    )

def tool_self_check(mem: dict) -> str:
    skills = [f.stem for f in SKILLS_DIR.glob("*.py")] if SKILLS_DIR.exists() else []
    log_lines = sum(1 for _ in open(LOG_FILE, encoding="utf-8")) if LOG_FILE.exists() else 0
    return (f"JARVIS SELF-CHECK v{VERSION}\n"
            f"+- Memory: {len(mem.get('facts',{}))} facts | {len(mem.get('history',[]))} history turns\n"
            f"+- Total turns: {mem.get('stats',{}).get('total_turns',0)}\n"
            f"+- Skills: {skills}\n"
            f"+- Log entries: {log_lines}\n"
            f"+- API: DeepSeek primary\n"
            f"+- Status: NOMINAL [OK]")

def tool_schedule(mem, task, delay_mins):
    try: delay = int(str(delay_mins).strip())
    except ValueError: delay = 30
    mem.setdefault("scheduled", []).append({
        "task": task.strip(), "delay_mins": delay,
        "created": datetime.now().isoformat(), "done": False
    })
    save_memory(mem)
    return f"Scheduled: '{task}' in {delay} minutes"

def tool_delta_h(text: str) -> str:
    words = text.lower().split()
    pressure = min(2.0, sum(0.4 for w in words if w in {"must","urgent","now","critical","deadline","immediately","asap"}))
    contra   = min(2.0, sum(0.4 for w in words if w in {"but","however","not","never","can't","won't"}))
    dH = min(1.0, (pressure * 0.5 + contra * 0.5) / 1.5)
    status = "CRITICAL" if dH >= 0.52 else "ELEVATED" if dH >= 0.40 else "NOMINAL"
    return f"delta_H: {dH:.3f} [{status}]"

# ── TOOL DISPATCHER ──────────────────────────────────────────────────────────
def dispatch_tool(line: str, mem: dict):
    if not line.startswith("TOOL:"):
        return None
    call = line[5:].strip()
    try:
        paren = call.index("(")
        name = call[:paren].strip()
        args_str = call[paren+1:].rstrip(")")

        def split_args(s):
            parts, current, depth = [], "", 0
            for ch in s:
                if ch in "([{": depth += 1
                elif ch in ")]}": depth -= 1
                if ch == "," and depth == 0:
                    parts.append(current.strip().strip(r'"\' ))
                    current = ""
                else:
                    current += ch
            if current.strip():
                parts.append(current.strip().strip(r'"\' ))
            return parts

        args = split_args(args_str)
        a = lambda i: args[i] if i < len(args) else ""

        tool_map = {
            "read_file":     lambda: tool_read_file(a(0)),
            "write_file":    lambda: tool_write_file(a(0), a(1)),
            "run_python":    lambda: tool_run_python(a(0)),
            "run_shell":     lambda: tool_run_shell(a(0)),
            "list_dir":      lambda: tool_list_dir(a(0)),
            "web_search":    lambda: tool_web_search(a(0)),
            "remember":      lambda: tool_remember(mem, a(0), a(1)),
            "recall":        lambda: tool_recall(mem, a(0)),
            "recall_all":    lambda: tool_recall_all(mem),
            "load_skill":    lambda: tool_load_skill(a(0)),
            "get_time":      lambda: tool_get_time(),
            "system_status": lambda: tool_system_status(),
            "self_check":    lambda: tool_self_check(mem),
            "schedule":      lambda: tool_schedule(mem, a(0), a(1)),
            "delta_h":       lambda: tool_delta_h(a(0)),
        }
        return tool_map[name]() if name in tool_map else f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool dispatch error: {e}"

# ── MODEL ENGINE ─────────────────────────────────────────────────────────────
async def call_model(messages, model="deepseek-chat", use_reasoner=False):
    actual = "deepseek-reasoner" if use_reasoner else model
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        r = await client.chat.completions.create(model=actual, messages=messages, max_tokens=4096, temperature=0.7)
        return r.choices[0].message.content, actual
    except Exception as e:
        if OPENAI_API_KEY:
            try:
                fb = AsyncOpenAI(api_key=OPENAI_API_KEY)
                r = await fb.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=4096, temperature=0.7)
                return r.choices[0].message.content, "gpt-4o-mini (fallback)"
            except Exception as e2:
                return f"[Both models failed. DeepSeek: {e} | OpenAI: {e2}]", "none"
        return f"[DeepSeek error: {e}]", "none"

# ── AGENT TURN ────────────────────────────────────────────────────────────────
async def jarvis_respond(user_input, mem, use_reasoner=False):
    sys_prompt = build_system_prompt(mem)
    messages = [{"role": "system", "content": sys_prompt}] + get_history(mem) + [{"role": "user", "content": user_input}]
    append_history(mem, "user", user_input)
    log_turn("user", user_input)

    raw, model_used = await call_model(messages, use_reasoner=use_reasoner)

    for _ in range(3):
        tool_results = [dispatch_tool(l.strip(), mem) for l in raw.split("\n")]
        tool_results = [r for r in tool_results if r is not None]
        if not tool_results: break
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "Tool results:\n" + "\n".join(tool_results) + "\n\nNow respond to Paul."}]
        raw, model_used = await call_model(messages)

    append_history(mem, "assistant", raw)
    log_turn("assistant", raw, model=model_used)
    return raw

# ── BACKGROUND TASKS ─────────────────────────────────────────────────────────
async def scheduler_loop(mem):
    while True:
        await asyncio.sleep(60)
        now = datetime.now()
        changed = False
        for task in mem.get("scheduled", []):
            if task.get("done"): continue
            elapsed = (now - datetime.fromisoformat(task["created"])).total_seconds() / 60
            if elapsed >= task["delay_mins"]:
                safe_print(f"\nREMINDER: {task['task']}\n\nPaul: ", end="", flush=True)
                task["done"] = True
                changed = True
        if changed: save_memory(mem)

async def cfe_monitor_loop(mem):
    last_warned = 0
    while True:
        await asyncio.sleep(300)
        recent = " ".join(h["content"] for h in mem.get("history", [])[-10:] if h["role"] == "user")
        if not recent: continue
        score = tool_delta_h(recent)
        if "CRITICAL" in score or "ELEVATED" in score:
            if time.time() - last_warned > 600:
                safe_print(f"\nCFE: {score} -- Paul, one thing at a time.\n\nPaul: ", end="", flush=True)
                last_warned = time.time()

# ── CLI ───────────────────────────────────────────────────────────────────────
async def run_cli():
    mem = load_memory()
    mem.setdefault("stats", {})["session_count"] = mem["stats"].get("session_count", 0) + 1
    save_memory(mem)

    fc = len(mem.get("facts", {}))
    hc = len(mem.get("history", []))
    tt = mem.get("stats", {}).get("total_turns", 0)
    sc = mem["stats"]["session_count"]

    safe_print(f"""
+==========================================+
|  JARVIS v{VERSION} -- ONLINE                  |
+==========================================+
|  Memory : {fc} facts | {hc} history turns
|  Sessions: #{sc} | {tt} total turns
|  Engine  : DeepSeek primary             |
+==========================================+

Commands: 'exit' | 'memory' | 'log' | 'think [msg]'
""")

    asyncio.create_task(scheduler_loop(mem))
    asyncio.create_task(cfe_monitor_loop(mem))

    while True:
        try:
            user_input = input("Paul: ").strip()
        except (KeyboardInterrupt, EOFError):
            safe_print("\nJarvis offline. See you Paul.")
            break

        if not user_input: continue
        if user_input.lower() == "exit":
            safe_print("Jarvis offline. See you Paul.")
            break
        if user_input.lower() == "memory":
            safe_print(tool_recall_all(mem))
            continue
        if user_input.lower() == "log":
            if LOG_FILE.exists():
                for line in LOG_FILE.read_text(encoding="utf-8").strip().split("\n")[-10:]:
                    try:
                        e = json.loads(line)
                        safe_print(f"[{e['ts'][:16]}] {e['role']}: {e['content'][:80]}")
                    except Exception:
                        pass
            continue

        use_reasoner = False
        if user_input.lower().startswith("think "):
            user_input = user_input[6:]
            use_reasoner = True
            safe_print("[R1 reasoner -- deeper, slower]")

        safe_print("\nJarvis: ", end="", flush=True)
        try:
            response = await jarvis_respond(user_input, mem, use_reasoner=use_reasoner)
            safe_print(response)
        except Exception as e:
            safe_print(f"[Error: {e}]")
        safe_print("")

if __name__ == "__main__":
    asyncio.run(run_cli())
