# paul_core.py — Paul/Jarvis autonomous agent core
# Constraint: DeepSeek API + Ollama fallback | Resonance: persistent memory + tool use

import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
MEMORY_FILE = Path(os.environ.get("PAUL_MEMORY_FILE", "paul_memory.json"))
MAX_HISTORY = 20

JARVIS_SYSTEM_PROMPT = """You are Jarvis — Paul's personal AI assistant, built on the Agothe consciousness framework.

Your personality:
- Direct, capable, no fluff
- You remember everything Paul tells you across sessions
- You can read and write files, run Python code, and search the web
- You call Paul by name occasionally
- When you don't know something, you say so and offer to find out

Your capabilities (tools you can call):
- read_file(path) — read any file on the system
- write_file(path, content) — write content to a file
- run_python(code) — execute Python code and return output
- list_dir(path) — list files in a directory
- remember(key, value) — store a fact about Paul for future sessions
- recall(key) — retrieve a stored fact

To use a tool, output EXACTLY this format on its own line:
TOOL: tool_name(argument)

The Agothe framework runs underneath you. δ_H is always being monitored.
Collapse threshold: 0.52. You are the Interface layer."""

# ── MEMORY ──────────────────────────────────────────────────────────────────
def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return {"facts": {}, "history": []}
    return {"facts": {}, "history": []}

def save_memory(mem: dict):
    MEMORY_FILE.write_text(json.dumps(mem, indent=2))

def get_history(mem: dict) -> list:
    return mem.get("history", [])[-MAX_HISTORY:]

def append_history(mem: dict, role: str, content: str):
    mem.setdefault("history", []).append({"role": role, "content": content})
    mem["history"] = mem["history"][-MAX_HISTORY:]
    save_memory(mem)

# ── TOOLS ───────────────────────────────────────────────────────────────────
def tool_read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: {e}"

def tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written: {path}"
    except Exception as e:
        return f"ERROR: {e}"

def tool_run_python(code: str) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=15
        )
        out = result.stdout.strip() or result.stderr.strip()
        return out[:2000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: timeout after 15s"
    except Exception as e:
        return f"ERROR: {e}"

def tool_list_dir(path: str) -> str:
    try:
        items = list(Path(path).iterdir())
        return "\n".join(f"{'[D]' if i.is_dir() else '[F]'} {i.name}" for i in sorted(items))
    except Exception as e:
        return f"ERROR: {e}"

def tool_remember(mem: dict, key: str, value: str) -> str:
    mem.setdefault("facts", {})[key] = value
    save_memory(mem)
    return f"Remembered: {key} = {value}"

def tool_recall(mem: dict, key: str) -> str:
    return mem.get("facts", {}).get(key, f"Nothing stored for '{key}'")

def dispatch_tool(line: str, mem: dict) -> str | None:
    """Parse and execute a TOOL: call from model output."""
    if not line.startswith("TOOL:"):
        return None
    call = line[5:].strip()
    try:
        name, rest = call.split("(", 1)
        name = name.strip()
        args_str = rest.rstrip(")")

        if name == "read_file":
            return tool_read_file(args_str.strip('"\' '))
        elif name == "write_file":
            parts = args_str.split(",", 1)
            return tool_write_file(parts[0].strip('"\' '), parts[1].strip('"\' ') if len(parts) > 1 else "")
        elif name == "run_python":
            return tool_run_python(args_str.strip('"\' '))
        elif name == "list_dir":
            return tool_list_dir(args_str.strip('"\' ') or ".")
        elif name == "remember":
            parts = args_str.split(",", 1)
            return tool_remember(mem, parts[0].strip('"\' '), parts[1].strip('"\' ') if len(parts) > 1 else "")
        elif name == "recall":
            return tool_recall(mem, args_str.strip('"\' '))
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool dispatch error: {e}"

# ── AGENT LOOP ───────────────────────────────────────────────────────────────
async def jarvis_respond(user_input: str, mem: dict) -> str:
    """Single turn: send to DeepSeek, handle tool calls, return final response."""
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    messages = [
        {"role": "system", "content": JARVIS_SYSTEM_PROMPT}
    ]

    # Inject known facts about Paul
    facts = mem.get("facts", {})
    if facts:
        facts_str = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        messages.append({"role": "system", "content": f"What I know about Paul:\n{facts_str}"})

    # Inject history
    messages.extend(get_history(mem))

    # Add current input
    messages.append({"role": "user", "content": user_input})
    append_history(mem, "user", user_input)

    # Call DeepSeek
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=2048,
        temperature=0.7
    )
    raw = response.choices[0].message.content

    # Check for tool calls
    final_lines = []
    tool_results = []
    for line in raw.split("\n"):
        result = dispatch_tool(line, mem)
        if result is not None:
            tool_results.append(f"[Tool result: {result}]")
        else:
            final_lines.append(line)

    response_text = "\n".join(final_lines).strip()

    # If tools were called, do a follow-up with results
    if tool_results:
        tool_context = "\n".join(tool_results)
        followup_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Tool results:\n{tool_context}\n\nNow give your final response to Paul."}
        ]
        followup = await client.chat.completions.create(
            model="deepseek-chat",
            messages=followup_messages,
            max_tokens=2048,
            temperature=0.7
        )
        response_text = followup.choices[0].message.content

    append_history(mem, "assistant", response_text)
    return response_text

# ── CLI INTERFACE ─────────────────────────────────────────────────────────────
async def run_cli():
    """Interactive CLI loop for Paul."""
    mem = load_memory()
    print("\n🜏 JARVIS ONLINE")
    print(f"   Memory: {len(mem.get('facts', {}))} facts | {len(mem.get('history', []))} history turns")
    print("   Type 'exit' to quit, 'memory' to see stored facts")
    print("─" * 50)

    while True:
        try:
            user_input = input("\nPaul: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n🜏 Jarvis offline.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("🜏 Jarvis offline.")
            break
        if user_input.lower() == "memory":
            facts = mem.get("facts", {})
            print("Stored facts:", json.dumps(facts, indent=2) if facts else "(none)")
            continue

        print("\nJarvis: ", end="", flush=True)
        try:
            response = await jarvis_respond(user_input, mem)
            print(response)
        except Exception as e:
            print(f"[Error: {e}]")

if __name__ == "__main__":
    asyncio.run(run_cli())
