# jarvis_api.py — Jarvis FastAPI server for web dashboard + external access
# Run: uvicorn jarvis_api:app --host 0.0.0.0 --port 8000

import asyncio
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Jarvis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MEMORY_FILE = Path("paul_memory.json")
LOG_FILE = Path("jarvis_log.jsonl")

class ChatRequest(BaseModel):
    message: str
    use_reasoner: bool = False

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🜏 Jarvis Dashboard</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0a0a0f; color: #e0e0ff; font-family: 'Courier New', monospace; min-height: 100vh; }
    .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px 30px; border-bottom: 1px solid #00ff9944; }
    .header h1 { color: #00ff99; font-size: 1.8rem; letter-spacing: 3px; }
    .header p { color: #888; font-size: 0.85rem; margin-top: 4px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 30px; max-width: 1200px; }
    .card { background: #111120; border: 1px solid #00ff9922; border-radius: 12px; padding: 20px; }
    .card h2 { color: #00ff99; font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 15px; text-transform: uppercase; }
    .chat-box { grid-column: 1 / -1; }
    .messages { height: 400px; overflow-y: auto; background: #080810; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .msg { margin-bottom: 12px; line-height: 1.5; }
    .msg.paul { color: #88aaff; } .msg.paul::before { content: 'Paul: '; font-weight: bold; }
    .msg.jarvis { color: #00ff99; } .msg.jarvis::before { content: '🜏 Jarvis: '; font-weight: bold; }
    .input-row { display: flex; gap: 10px; }
    input[type=text] { flex: 1; background: #080810; border: 1px solid #00ff9944; border-radius: 8px; padding: 12px; color: #e0e0ff; font-family: monospace; font-size: 1rem; outline: none; }
    input[type=text]:focus { border-color: #00ff99; }
    button { background: #00ff9922; border: 1px solid #00ff99; color: #00ff99; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-family: monospace; font-size: 0.9rem; transition: all 0.2s; }
    button:hover { background: #00ff9944; }
    .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #ffffff11; }
    .stat:last-child { border-bottom: none; }
    .stat-label { color: #888; font-size: 0.85rem; }
    .stat-value { color: #00ff99; font-weight: bold; }
    .fact-item { padding: 6px 0; border-bottom: 1px solid #ffffff08; font-size: 0.85rem; }
    .fact-key { color: #88aaff; } .fact-val { color: #e0e0ff; margin-left: 10px; }
    .status-dot { width: 8px; height: 8px; background: #00ff99; border-radius: 50%; display: inline-block; margin-right: 8px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    #think-toggle { margin-left: 10px; background: #ff990022; border-color: #ff9900; color: #ff9900; }
    #think-toggle.active { background: #ff990044; }
  </style>
</head>
<body>
  <div class="header">
    <h1><span class="status-dot"></span>JARVIS v2.0</h1>
    <p>Agothe Autonomous Agent — Paul's Interface Layer</p>
  </div>
  <div class="grid">
    <div class="card" id="stats-card">
      <h2>System Status</h2>
      <div id="stats-content"><div class="stat"><span class="stat-label">Loading...</span></div></div>
    </div>
    <div class="card" id="memory-card">
      <h2>Paul's Memory</h2>
      <div id="memory-content"><div class="fact-item">Loading...</div></div>
    </div>
    <div class="card chat-box">
      <h2>Chat with Jarvis</h2>
      <div class="messages" id="messages"></div>
      <div class="input-row">
        <input type="text" id="input" placeholder="Ask Jarvis anything..." onkeydown="if(event.key==='Enter')send()" autofocus>
        <button onclick="send()">Send</button>
        <button id="think-toggle" onclick="toggleThink()" title="Use DeepSeek R1 for deep reasoning">R1</button>
      </div>
    </div>
  </div>
  <script>
    let useReasoner = false;
    function toggleThink() {
      useReasoner = !useReasoner;
      document.getElementById('think-toggle').classList.toggle('active', useReasoner);
    }
    function addMsg(role, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      const box = document.getElementById('messages');
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }
    async function send() {
      const input = document.getElementById('input');
      const msg = input.value.trim();
      if (!msg) return;
      input.value = '';
      addMsg('paul', msg);
      addMsg('jarvis', '...');
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: msg, use_reasoner: useReasoner})
        });
        const data = await res.json();
        const msgs = document.querySelectorAll('.msg.jarvis');
        msgs[msgs.length-1].textContent = data.response;
      } catch(e) {
        const msgs = document.querySelectorAll('.msg.jarvis');
        msgs[msgs.length-1].textContent = '[Error: ' + e + ']';
      }
    }
    async function loadStats() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        const html = Object.entries(data).map(([k,v]) =>
          `<div class="stat"><span class="stat-label">${k}</span><span class="stat-value">${v}</span></div>`
        ).join('');
        document.getElementById('stats-content').innerHTML = html;
      } catch(e) {}
    }
    async function loadMemory() {
      try {
        const res = await fetch('/memory');
        const data = await res.json();
        const facts = data.facts || {};
        const html = Object.entries(facts).length > 0
          ? Object.entries(facts).map(([k,v]) =>
              `<div class="fact-item"><span class="fact-key">${k}</span><span class="fact-val">${v}</span></div>`
            ).join('')
          : '<div class="fact-item" style="color:#666">No facts stored yet.</div>';
        document.getElementById('memory-content').innerHTML = html;
      } catch(e) {}
    }
    loadStats(); loadMemory();
    setInterval(() => { loadStats(); loadMemory(); }, 30000);
  </script>
</body>
</html>
"""

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        from paul_core import load_memory, jarvis_respond
        mem = load_memory()
        response = await jarvis_respond(req.message, mem, use_reasoner=req.use_reasoner)
        return {"response": response, "ts": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    mem_file = Path("paul_memory.json")
    mem = json.loads(mem_file.read_text()) if mem_file.exists() else {}
    log_lines = sum(1 for _ in open(LOG_FILE)) if LOG_FILE.exists() else 0
    return {
        "status": "🟢 NOMINAL",
        "version": "2.0.0",
        "facts": len(mem.get("facts", {})),
        "history_turns": len(mem.get("history", [])),
        "total_turns": mem.get("stats", {}).get("total_turns", 0),
        "sessions": mem.get("stats", {}).get("session_count", 0),
        "log_entries": log_lines,
        "delta_H": "0.19 🟢"
    }

@app.get("/memory")
async def get_memory():
    mem_file = Path("paul_memory.json")
    if not mem_file.exists():
        return {"facts": {}, "history_count": 0}
    mem = json.loads(mem_file.read_text())
    return {
        "facts": mem.get("facts", {}),
        "history_count": len(mem.get("history", [])),
        "scheduled": mem.get("scheduled", [])
    }

@app.delete("/memory/{key}")
async def delete_memory(key: str):
    mem_file = Path("paul_memory.json")
    if not mem_file.exists():
        raise HTTPException(404, "No memory file")
    mem = json.loads(mem_file.read_text())
    if key in mem.get("facts", {}):
        del mem["facts"][key]
        mem_file.write_text(json.dumps(mem, indent=2))
        return {"deleted": key}
    raise HTTPException(404, f"Key '{key}' not found")

@app.get("/log")
async def get_log(limit: int = 20):
    if not LOG_FILE.exists():
        return {"entries": []}
    with open(LOG_FILE) as f:
        lines = f.readlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return {"entries": entries}
